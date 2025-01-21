import pickle
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


data_dir = os.path.join('..', "outputs", "embeddings")
with open(os.path.join(data_dir, "loaders_datasets.pkl"), 'rb') as f:
    data = pickle.load(f)

#------------------------------------------------------------------------------------------
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


for split in data.keys():
    print(f"{split} audio features shape: {data[split]['audio']['features'].shape}")
    print(f"{split} text features shape: {data[split]['text']['features'].shape}")
    print(f"{split} video features shape: {data[split]['video']['features'].shape}")
    print(f"{split} text length tensor shape: {data[split]['text']['text_len_tensor'].shape}")
#------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv
from torch.nn.utils.rnn import pad_sequence

class GraphModel(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, device, args):
        super(GraphModel, self).__init__()
        self.device = device
        self.n_modals = args['n_modals']
        self.wp = args['wp']
        self.wf = args['wf']
        self.edge_multi = args['edge_multi']
        self.edge_temp = args['edge_temp']

        # GNN Layers: RGCN followed by TransformerConv
        self.rgcn = RGCNConv(g_dim, h1_dim, num_relations=num_relations)
        self.transformer = TransformerConv(h1_dim, h2_dim, heads=4)
        self.final_layer = nn.Linear(h2_dim, args['num_classes'])

    def forward(self, x, lengths, edge_index, edge_type):
        """
        Forward pass of the GraphModel.
        Args:
            x (torch.Tensor): Node features.
            lengths (list): Lengths of sequences in the batch.
            edge_index (torch.Tensor): Graph edge indices.
            edge_type (torch.Tensor): Graph edge types.
        Returns:
            torch.Tensor: Final class predictions.
        """
        # Apply RGCN
        x = self.rgcn(x, edge_index, edge_type)
        x = torch.relu(x)

        # Apply TransformerConv
        x = self.transformer(x, edge_index)
        x = torch.relu(x)

        # Apply final classification layer
        output = self.final_layer(x)
        return output

    def batch_graphify(self, lengths):
        """
        Constructs the graph structure for the batch.
        """
        edge_indices, edge_types, node_types = [], [], []
        total_length = sum(lengths)

        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length

            # Node types
            node_types += [j % self.n_modals for j in range(start_idx, end_idx)]

            # Temporal and multi-modal edges
            for i in range(start_idx, end_idx):
                for j in range(start_idx, end_idx):
                    if i != j:
                        if self.edge_temp:
                            edge_indices.append([i, j])
                            edge_types.append(0 if i < j else 1)  # 0 = future, 1 = past
                        if self.edge_multi and node_types[i] != node_types[j]:
                            edge_indices.append([i, j])
                            edge_types.append(2)  # Multi-modal edge type
            start_idx = end_idx

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device)
        edge_type = torch.tensor(edge_types, dtype=torch.long).to(self.device)

        return edge_index, edge_type

    def feature_packing(self, x):
        """
        Packs node features for batch processing.
        """
        return pad_sequence(x, batch_first=True).to(self.device)
#------------------------------------------------------------------------------------------
# Training 
def train_and_evaluate(model, train_data, val_data, optimizer, criterion, epochs=50, patience=10):
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.lengths, train_data.edge_index, train_data.edge_type)
        loss = criterion(out, train_data.y)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x, val_data.lengths, val_data.edge_index, val_data.edge_type)
            val_loss = criterion(val_out, val_data.y)

        print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping triggered!")
            break

    # Test
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        test_out = model(val_data.x, val_data.lengths, val_data.edge_index, val_data.edge_type)
        test_pred = test_out.argmax(dim=1)
        test_acc = accuracy_score(val_data.y.cpu(), test_pred.cpu())
        test_f1 = f1_score(val_data.y.cpu(), test_pred.cpu(), average='macro')

    return test_acc, test_f1
#------------------------------------------------------------------------------------------
# Data preparation

def prepare_graph_data(data, split, args):
    """
    Prepares data for the GraphModel by extracting features, labels, and constructing the graph structure.
    Args:
        data (dict): Input data dictionary.
        split (str): Dataset split ('train', 'val', 'test').
        args (dict): Additional arguments for graph configuration.
    Returns:
        dict: Prepared graph data including node features, labels, edge indices, and edge types.
    """
    # Prepare features
    video_features = torch.mean(data[split]['video']['features'], dim=1)  # Average over temporal dimension
    features = torch.cat([
        data[split]['audio']['features'],
        data[split]['text']['features'],
        video_features
    ], dim=0)

    # Prepare labels
    labels = torch.cat([
        data[split]['audio']['labels'],
        data[split]['text']['labels'],
        data[split]['video']['labels']
    ], dim=0)

    # Calculate lengths for graph batching
    lengths = [
        data[split]['audio']['features'].size(0),
        data[split]['text']['features'].size(0),
        video_features.size(0)
    ]

    # Create edge index and edge types
    model = GraphModel(
        g_dim=args['g_dim'],
        h1_dim=args['h1_dim'],
        h2_dim=args['h2_dim'],
        num_relations=args['num_relations'],
        device=args['device'],
        args=args
    )
    edge_index, edge_type = model.batch_graphify(lengths)

    # Return prepared data
    return {
        'x': features.to(args['device']),
        'y': labels.to(args['device']),
        'lengths': lengths,
        'edge_index': edge_index,
        'edge_type': edge_type
    }
#------------------------------------------------------------------------------------------
# Pipeline

def main_pipeline(data, args):
    # Prepare data
    train_data = prepare_graph_data(data, 'train', args)
    val_data = prepare_graph_data(data, 'val', args)
    test_data = prepare_graph_data(data, 'test', args)

    # Initialize model
    model = GraphModel(
        g_dim=args['g_dim'],
        h1_dim=args['h1_dim'],
        h2_dim=args['h2_dim'],
        num_relations=args['num_relations'],
        device=args['device'],
        args=args
    ).to(args['device'])

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    print("Training the GraphModel...")
    acc, f1 = train_and_evaluate(model, train_data, val_data, optimizer, criterion)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print(f"Test F1-Score: {f1 * 100:.2f}%")

#------------------------------------------------------------------------------------------
args = {
    'g_dim': 128,  # Input feature dimension
    'h1_dim': 64,  # Hidden layer 1 dimension
    'h2_dim': 32,  # Hidden layer 2 dimension
    'num_classes': len(torch.unique(data['train']['audio']['labels'])),
    'num_relations': 3,  # Temporal and multi-modal edges
    'n_modals': 3,  # Number of modalities (audio, text, video)
    'wp': 2,  # Past window
    'wf': 2,  # Future window
    'edge_multi': True,  # Enable multi-modal edges
    'edge_temp': True,  # Enable temporal edges
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

main_pipeline(data, args)
