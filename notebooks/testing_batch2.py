import pickle
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, TransformerConv
from sklearn.metrics import accuracy_score, f1_score
import random
from torch.utils.data import DataLoader, Dataset

# Load dataset
data_dir = os.path.join('..', "outputs", "embeddings")
data = torch.load(os.path.join(data_dir, "loaders_datasets.pt"))
# data = torch.load(os.path.join(data_dir, "loaders_datasets_reduced_label_dim_4.pt"))

# Preprocess data
for split in data.keys():
    for modal in ['audio', 'text', 'video']:
        modal_tensors = data[split][modal].tensors
        data[split][modal] = {
            'features': modal_tensors[0],
            'labels': modal_tensors[1]
        }

    if 'text' in data[split]:
        text_features = data[split]['text']['features']
        text_len_tensor = torch.sum((text_features != 0).long(), dim=1)
        data[split]['text']['text_len_tensor'] = text_len_tensor

# Validation helper functions
def validate_edge_type(edge_type, num_relations):
    if not torch.all((edge_type >= 0) & (edge_type < num_relations)):
        raise ValueError(f"Invalid edge_type values: {edge_type.unique()}")

def validate_edge_index(edge_index, num_nodes):
    if not torch.all((edge_index >= 0) & (edge_index < num_nodes)):
        raise ValueError(f"Invalid edge_index values: {edge_index}")

# GraphModel
class GraphModel(torch.nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, device, args):
        super(GraphModel, self).__init__()
        self.device = device
        self.feature_projection = torch.nn.Linear(g_dim, h1_dim)
        self.rgcn = RGCNConv(h1_dim, h2_dim, num_relations=num_relations)
        self.transformer = TransformerConv(h2_dim, h2_dim, heads=1)
        self.final_layer = torch.nn.Linear(h2_dim, args['num_classes'])

    def forward(self, x, lengths, edge_index, edge_type):
        device = x.device
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)

        # Validate edge_index and edge_type
        validate_edge_index(edge_index, x.size(0))
        validate_edge_type(edge_type, self.rgcn.num_relations)

        x = self.feature_projection(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.transformer(x, edge_index)
        x = torch.relu(x)
        return self.final_layer(x)

    def batch_graphify(self, lengths, n_modals, edge_multi=True, edge_temp=True, P=2, F=2, temp_sample_rate=0.5, multi_modal_sample_rate=0.5, edge_reduction_rate=0.75):
        """
        Constructs the graph structure for the entire dataset.
        """
        edge_indices, edge_types = [], []
        start_idx = 0

        for length in lengths:
            end_idx = start_idx + length

            # Multi-modal edges
            if edge_multi:
                for i in range(start_idx, end_idx):
                    for j in range(start_idx, end_idx):
                        if i != j and random.random() < multi_modal_sample_rate:
                            edge_indices.append([i, j])
                            edge_types.append(random.randint(0, 5))

            # Temporal edges
            if edge_temp:
                for i in range(start_idx, end_idx):
                    past_nodes = list(range(max(start_idx, i - P), i))
                    if past_nodes:
                        sampled_past = random.sample(past_nodes, min(len(past_nodes), max(1, int(len(past_nodes) * temp_sample_rate))))
                        for j in sampled_past:
                            edge_indices.append([j, i])
                            edge_types.append(6)

                    future_nodes = list(range(i + 1, min(end_idx, i + F + 1)))
                    if future_nodes:
                        sampled_future = random.sample(future_nodes, min(len(future_nodes), max(1, int(len(future_nodes) * temp_sample_rate))))
                        for j in sampled_future:
                            edge_indices.append([i, j])
                            edge_types.append(7)

            start_idx = end_idx

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        # Validate edge_type
        validate_edge_type(edge_type, 9)  # Assuming num_relations = 9
        return edge_index, edge_type

# GraphDataset : Une classe pour gérer les batchs
class GraphDataset(Dataset):
    def __init__(self, features, labels, edge_index, edge_type, batch_size):
        self.features = features
        self.labels = labels
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.labels))

        # Indices locaux pour ce batch
        batch_nodes = torch.arange(start, end, device=self.features.device)
        batch_mask = (
            torch.isin(self.edge_index[0], batch_nodes) &
            torch.isin(self.edge_index[1], batch_nodes)
        )
        batch_edge_index = self.edge_index[:, batch_mask]
        batch_edge_type = self.edge_type[batch_mask]

        # Re-mappage des indices globaux vers les indices locaux
        global_to_local = {int(n): i for i, n in enumerate(batch_nodes.tolist())}
        batch_edge_index = torch.stack([
            torch.tensor([global_to_local[src.item()] for src in batch_edge_index[0]]),
            torch.tensor([global_to_local[dst.item()] for dst in batch_edge_index[1]])
        ], dim=0).to(self.features.device)

        return {
            'features': self.features[start:end],
            'labels': self.labels[start:end],
            'edge_index': batch_edge_index,
            'edge_type': batch_edge_type
        }


# Fonction de préparation des datasets avec DataLoader
def create_dataloaders(data, split, args, batch_size):
    video_features = torch.mean(data[split]['video']['features'], dim=1)
    features = torch.cat([
        data[split]['audio']['features'],
        data[split]['text']['features'],
        video_features
    ], dim=0)

    labels = torch.cat([
        data[split]['audio']['labels'],
        data[split]['text']['labels'],
        data[split]['video']['labels']
    ], dim=0)

    lengths = [
        data[split]['audio']['features'].size(0),
        data[split]['text']['features'].size(0),
        video_features.size(0)
    ]

    model = GraphModel(
        g_dim=args['g_dim'],
        h1_dim=args['h1_dim'],
        h2_dim=args['h2_dim'],
        num_relations=args['num_relations'],
        device=args['device'],
        args=args
    )

    edge_index, edge_type = model.batch_graphify(
        lengths=lengths,
        n_modals=args['n_modals'],
        edge_multi=args['edge_multi'],
        edge_temp=args['edge_temp'],
        P=args['wp'],
        F=args['wf'],
        temp_sample_rate=0.2,
        multi_modal_sample_rate=0.2,
        edge_reduction_rate=0.75
    )

    dataset = GraphDataset(features, labels, edge_index, edge_type, batch_size)
    return DataLoader(dataset, batch_size=1, shuffle=True)

# Entraînement avec DataLoader
def train_with_dataloader(model, train_loader, val_loader, test_loader, optimizer, criterion, args, epochs=50, patience=10):
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch['features'].to(args['device']),
                        None,  # Les longueurs ne sont pas utilisées
                        batch['edge_index'].to(args['device']),
                        batch['edge_type'].to(args['device']))
            loss = criterion(out, batch['labels'].to(args['device']))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[INFO] Epoch {epoch + 1} | Train Loss: {total_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['features'].to(args['device']),
                            None,
                            batch['edge_index'].to(args['device']),
                            batch['edge_type'].to(args['device']))
                val_loss += criterion(out, batch['labels'].to(args['device'])).item()

        val_loss /= len(val_loader)
        print(f"[INFO] Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"[INFO] Epoch {epoch + 1}: Model saved.")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("[INFO] Early stopping triggered!")
            break

    # Test
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_acc, test_f1 = 0, 0
    with torch.no_grad():
        all_preds, all_labels = [], []
        for batch in test_loader:
            out = model(batch['features'].to(args['device']),
                        None,
                        batch['edge_index'].to(args['device']),
                        batch['edge_type'].to(args['device']))
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(batch['labels'].cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"[INFO] Test Accuracy: {test_acc * 100:.2f}%, Test F1-Score: {test_f1 * 100:.2f}")
    return test_acc, test_f1

args = {
    'g_dim': 768,
    'h1_dim': 64,
    'h2_dim': 32,
    'num_classes': 4,
    'num_relations': 9,
    'n_modals': 3,
    'wp': 2,
    'wf': 2,
    'edge_multi': True,
    'edge_temp': True,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Pipeline
batch_size = 128
train_loader = create_dataloaders(data, 'train', args, batch_size)
val_loader = create_dataloaders(data, 'val', args, batch_size)
test_loader = create_dataloaders(data, 'test', args, batch_size)

model = GraphModel(
    g_dim=args['g_dim'],
    h1_dim=args['h1_dim'],
    h2_dim=args['h2_dim'],
    num_relations=args['num_relations'],
    device=args['device'],
    args=args
).to(args['device'])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

train_with_dataloader(model, train_loader, val_loader, test_loader, optimizer, criterion, args)
