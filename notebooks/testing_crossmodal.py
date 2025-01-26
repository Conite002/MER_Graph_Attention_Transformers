import pickle
import torch
import os
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
import sys

# Charger les fonctions externes
sys.path.append(os.path.abspath(os.path.join('..')))
from data.loaddata import load_data
from models.GraphModel import GraphModel
from processing.preprocessing import prepare_graph_data, prepare_dataloaders
from models.GNN import GNN


### Définition du Modèle Multimodal ###
class MultimodalModel(nn.Module):
    def __init__(self, args):
        super(MultimodalModel, self).__init__()
        self.graph_model = GraphModel(
            args['g_dim'], args['h1_dim'], args['h2_dim'], args['device'], args
        )
        self.crossmodal = CrossModalModule(args['g_dim'], args['h1_dim'])
        self.feature_projector = nn.Linear(96, args['h1_dim'])

    def forward(self, data):
        multimodal_features = [data['audio'], data['text'], data['video']]
        lengths = data['text_len_tensor']

        # Sortie du graphe
        node_features = self.graph_model.feature_packing(multimodal_features, lengths)
        graph_out = self.graph_model(node_features, lengths)

        # Sortie cross-modalité
        crossmodal_out = self.crossmodal(multimodal_features)

        # Projection de graph_out pour correspondre à crossmodal_out
        graph_out = self.feature_projector(graph_out)

        # Concaténation des deux sorties
        print(graph_out.size(), crossmodal_out.size())
        combined_out = torch.cat([graph_out, crossmodal_out], dim=-1)

        return combined_out


class CrossModalModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CrossModalModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, multimodal_features):
        combined_features = torch.stack(multimodal_features, dim=1)  # (batch_size, n_modalities, input_dim)
        attended_features, _ = self.attention(combined_features, combined_features, combined_features)
        projected_features = self.projection(attended_features.mean(dim=1))
        return projected_features


### Fonction pour Entraîner le Modèle ###
def train(model, train_loader, optimizer, criterion, device, modality_lengths):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        features, labels = batch

        # Découpage Dynamique des Modalités
        start_idx = 0
        audio_features = features[:, start_idx:start_idx + modality_lengths['audio']]
        start_idx += modality_lengths['audio']
        text_features = features[:, start_idx:start_idx + modality_lengths['text']]
        start_idx += modality_lengths['text']
        video_features = features[:, start_idx:start_idx + modality_lengths['video']]

        # Calculer dynamiquement les longueurs
        multimodal_features = [audio_features, text_features, video_features]
        lengths = torch.tensor([feature.size(0) for feature in multimodal_features], dtype=torch.long)

        # Préparer les données pour le modèle
        multimodal_data = {
            'audio': audio_features.to(device),
            'text': text_features.to(device),
            'video': video_features.to(device),
            'text_len_tensor': lengths.to(device)
        }
        labels = labels.to(device)

        # Forward pass
        outputs = model(multimodal_data)

        # Calcul de la perte
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss


### Fonction pour Tester le Modèle ###
def test(model, test_loader, device, modality_lengths):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            features, labels = batch

            # Découpage Dynamique des Modalités
            start_idx = 0
            audio_features = features[:, start_idx:start_idx + modality_lengths['audio']]
            start_idx += modality_lengths['audio']
            text_features = features[:, start_idx:start_idx + modality_lengths['text']]
            start_idx += modality_lengths['text']
            video_features = features[:, start_idx:start_idx + modality_lengths['video']]

            # Préparer les données pour le modèle
            multimodal_data = {
                'audio': audio_features.to(device),
                'text': text_features.to(device),
                'video': video_features.to(device),
                'text_len_tensor': torch.ones(labels.size(0), dtype=torch.long).to(device)
            }
            labels = labels.to(device)

            # Forward pass
            outputs = model(multimodal_data)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


### Préparer les Données et Lancer l'Entraînement ###
args = {
    'g_dim': 768,
    'h1_dim': 64,
    'h2_dim': 32,
    'num_classes': 7,
    'edge_type': ['temp', 'multi'],
    'modalities': ['audio', 'text', 'video'],
    'wp': 2,
    'wf': 2,
    'crossmodal_nheads': 2,
    'num_crossmodal': 2,
    'self_att_nheads': 2,
    'num_self_att': 2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

data = load_data(data_type='loaders_datasets.pt')
train_features, train_labels, train_lengths = prepare_graph_data(data['train'], args)
dev_features, dev_labels, dev_lengths = prepare_graph_data(data['val'], args)
test_features, test_labels, test_lengths = prepare_graph_data(data['test'], args)

modality_lengths = {
    'audio': 768,
    'text': 768,
    'video': 768
}

train_loader = prepare_dataloaders(train_features, train_labels, train_lengths, batch_size=32, equilibrate=False)
val_loader = prepare_dataloaders(dev_features, dev_labels, dev_lengths, batch_size=32)
test_loader = prepare_dataloaders(test_features, test_labels, test_lengths, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalModel(args).to(device)

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    train_loss = train(model, train_loader, optimizer, criterion, device, modality_lengths)

print("Testing:")
test(model, test_loader, device, modality_lengths)
