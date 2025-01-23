# import pickle
# import torch
# import os
# import numpy as np
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv, GATConv, RGCNConv
# from sklearn.metrics import accuracy_score, f1_score
# import matplotlib.pyplot as plt
# import random

# data_dir = os.path.join('..', "outputs", "embeddings")
# # with open(os.path.join(data_dir, "loaders_datasets.pkl"), 'rb') as f:
# #     data = pickle.load(f)
# # load with torch
# data = torch.load(os.path.join(data_dir, "loaders_datasets_reduced_label_dim_4.pt"))
# #------------------------------------------------------------------------------------------
# for split in data.keys():
#     for modal in ['audio', 'text', 'video']:
#         modal_tensors = data[split][modal].tensors
#         data[split][modal] = {
#             'features': modal_tensors[0], 
#             'labels': modal_tensors[1]   
#         }

#     if 'text' in data[split]:
#         text_features = data[split]['text']['features']
#         text_len_tensor = torch.sum((text_features != 0).long(), dim=1) 
#         data[split]['text']['text_len_tensor'] = text_len_tensor


# # for split in data.keys():
# #     print(f"{split} audio features shape: {data[split]['audio']['features'].shape}")
# #     print(f"{split} text features shape: {data[split]['text']['features'].shape}")
# #     print(f"{split} video features shape: {data[split]['video']['features'].shape}")
# #     print(f"{split} text length tensor shape: {data[split]['text']['text_len_tensor'].shape}")

# # ------------------------------------------------------------------------------------------
# def validate_edge_type(edge_type, num_relations):
#     """
#     Validate edge_type to ensure all values are within the valid range.
#     """
#     if not torch.all((edge_type >= 0) & (edge_type < num_relations)):
#         raise ValueError(f"Invalid edge_type values: {edge_type.unique()}")

# def validate_edge_index(edge_index, num_nodes):
#     """
#     Validate edge_index to ensure all node indices are within bounds.
#     """
#     if not torch.all((edge_index >= 0) & (edge_index < num_nodes)):
#         raise ValueError(f"Invalid edge_index values: {edge_index}")

# #------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# from torch_geometric.nn import RGCNConv, TransformerConv
# from torch.nn.utils.rnn import pad_sequence

# class GraphModel(nn.Module):
#     def __init__(self, g_dim, h1_dim, h2_dim, num_relations, device, args):
#         super(GraphModel, self).__init__()
#         self.device = device
#         self.n_modals = args['n_modals']
#         self.wp = args['wp']
#         self.wf = args['wf']
#         self.edge_multi = args['edge_multi']
#         self.edge_temp = args['edge_temp']

#         # GNN Layers: RGCN followed by TransformerConv
#         self.feature_projection = nn.Linear(g_dim, h1_dim)

#         self.rgcn = RGCNConv(h1_dim, h2_dim, num_relations=num_relations)
#         self.transformer = TransformerConv(h2_dim, h2_dim, heads=1)
#         self.final_layer = nn.Linear(h2_dim, args['num_classes'])

#     def forward(self, x, lengths, edge_index, edge_type):
#         device = x.device
#         edge_index = edge_index.to(device)
#         edge_type = edge_type.to(device)
        
#         if edge_index.dim() == 3 and edge_index.size(0) == 1:
#             edge_index = edge_index.squeeze(0)

#         # Validate edge_index
#         if edge_index.dim() != 2 or edge_index.size(0) != 2:
#             raise ValueError(f"[DEBUG] Invalid edge_index shape in forward: {edge_index.shape}")

#         # Validate edge_type
#         if not torch.all((edge_type >= 0) & (edge_type < self.rgcn.num_relations)):
#             raise ValueError(f"[DEBUG] Invalid edge_type values in forward: {edge_type.unique()}")

#         print(f"[DEBUG] forward | edge_index.shape: {edge_index.shape}, edge_type.shape: {edge_type.shape}")


#         x = self.feature_projection(x)
#         x = self.rgcn(x, edge_index, edge_type)
#         x = torch.relu(x)
#         x = self.transformer(x, edge_index)
#         x = torch.relu(x)
#         return self.final_layer(x)






#     def batch_graphify(self, lengths, n_modals, edge_multi=True, edge_temp=True, P=2, F=2, temp_sample_rate=0.5, multi_modal_sample_rate=0.5, edge_reduction_rate=0.75):
#         edge_indices, edge_types = [], []
#         start_idx = 0

#         total_multi_edges, total_temp_edges = 0, 0

#         for length in lengths:
#             end_idx = start_idx + length

#             # Multi-modal edges (R_multi)
#             if edge_multi:
#                 for i in range(start_idx, end_idx):
#                     for j in range(start_idx, end_idx):
#                         if i != j and random.random() < multi_modal_sample_rate:
#                             edge_indices.append([i, j])
#                             edge_types.append(random.randint(0, 5))  # Example multi-modal edge types

#             # Temporal edges (R_temp)
#             if edge_temp:
#                 for i in range(start_idx, end_idx):
#                     past_nodes = list(range(max(start_idx, i - P), i))
#                     if past_nodes:
#                         sampled_past = random.sample(past_nodes, min(len(past_nodes), max(1, int(len(past_nodes) * temp_sample_rate))))
#                         for j in sampled_past:
#                             edge_indices.append([j, i])
#                             edge_types.append(6)

#                     future_nodes = list(range(i + 1, min(end_idx, i + F + 1)))
#                     if future_nodes:
#                         sampled_future = random.sample(future_nodes, min(len(future_nodes), max(1, int(len(future_nodes) * temp_sample_rate))))
#                         for j in sampled_future:
#                             edge_indices.append([i, j])
#                             edge_types.append(7)

#             start_idx = end_idx

#         edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
#         edge_type = torch.tensor(edge_types, dtype=torch.long)

#         # Validate edge_type
#         if not torch.all((edge_type >= 0) & (edge_type < 9)):  # Assuming num_relations = 9
#             raise ValueError(f"[DEBUG] Invalid edge_type values: {edge_type.unique().tolist()}")

#         return edge_index, edge_type



#     def feature_packing(self, x):
#         """
#         Packs node features for batch processing.
#         """
#         return pad_sequence(x, batch_first=True).to(self.device)
# #------------------------------------------------------------------------------------------
# # Training 
# def train_and_evaluate(model, train_data, val_data, test_data, optimizer, criterion, epochs=50, patience=10, batch_size=512):
#     best_val_loss = float('inf')
#     no_improve_epochs = 0
#     train_loader = create_dataloader(train_data, batch_size=batch_size)
#     val_loader = create_dataloader(val_data, batch_size=batch_size)

#     for epoch in range(epochs):
#         # Training
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             print(f"[DEBUG] train_loader | batch.x.shape: {batch.x.shape}")
#             print(f"[DEBUG] train_loader | batch.edge_index.shape: {batch.edge_index.shape}, edge_type.shape: {batch.edge_type.shape}")

#             # Validate edge_index
#             if not torch.all((batch.edge_index[0] >= 0) & (batch.edge_index[0] < batch.x.size(0))):
#                 raise ValueError(f"[DEBUG] Out-of-bounds src indices in edge_index: {batch.edge_index[0]}")
#             if not torch.all((batch.edge_index[1] >= 0) & (batch.edge_index[1] < batch.x.size(0))):
#                 raise ValueError(f"[DEBUG] Out-of-bounds dst indices in edge_index: {batch.edge_index[1]}")


#             optimizer.zero_grad()
#             out = model(batch.x, batch.lengths, batch.edge_index, batch.edge_type)
#             loss = criterion(out, batch.y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
            
#         print(f"[INFO] Epoch {epoch + 1} | Train Loss: {total_loss:.4f}")


#         # Validation
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 out = model(batch.x, batch.lengths, batch.edge_index, batch.edge_type)
#                 val_loss += criterion(out, batch.y).item()

#         val_loss /= len(val_loader)
#         print(f"[INFO] Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")


#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             no_improve_epochs = 0
#             torch.save(model.state_dict(), 'best_model.pth')
#             print(f"[INFO] Epoch {epoch + 1}: Model saved.")
#         else:
#             no_improve_epochs += 1

#         if no_improve_epochs >= patience:
#             print("[INFO] Early stopping triggered!")
#             break
#     # Test
#     model.load_state_dict(torch.load('best_model.pth'))
#     model.eval()
#     test_loader = create_dataloader(test_data, batch_size=batch_size)
#     test_acc, test_f1 = 0, 0
#     with torch.no_grad():
#         all_preds, all_labels = [], []
#         for batch in test_loader:
#             out = model(batch.x, batch.lengths, batch.edge_index, batch.edge_type)
#             preds = out.argmax(dim=1)
#             all_preds.append(preds.cpu())
#             all_labels.append(batch.y.cpu())
#         all_preds = torch.cat(all_preds)
#         all_labels = torch.cat(all_labels)

#         test_acc = accuracy_score(all_labels, all_preds)
#         test_f1 = f1_score(all_labels, all_preds, average='macro')
#     print(f"[INFO] Test Accuracy: {test_acc * 100:.2f}%, Test F1-Score: {test_f1 * 100:.2f}%")
#     return test_acc, test_f1

# #------------------------------------------------------------------------------------------
# # Data preparation

# from collections import namedtuple

# # Define a named tuple for prepared graph data
# GraphData = namedtuple('GraphData', ['x', 'y', 'lengths', 'edge_index', 'edge_type'])

# def prepare_graph_data(data, split, args):
#     """
#     Prepares data for the GraphModel by extracting features, labels, and constructing the graph structure.
#     Args:
#         data (dict): Input data dictionary.
#         split (str): Dataset split ('train', 'val', 'test').
#         args (dict): Additional arguments for graph configuration.
#     Returns:
#         GraphData: Encapsulated graph data with node features, labels, and graph structure.
#     """
#     # Prepare features
#     video_features = torch.mean(data[split]['video']['features'], dim=1)  # Average over temporal dimension
#     features = torch.cat([
#         data[split]['audio']['features'],
#         data[split]['text']['features'],
#         video_features
#     ], dim=0)

#     # Prepare labels
#     labels = torch.cat([
#         data[split]['audio']['labels'],
#         data[split]['text']['labels'],
#         data[split]['video']['labels']
#     ], dim=0)

#     # Calculate lengths for graph batching
#     lengths = [
#         data[split]['audio']['features'].size(0),
#         data[split]['text']['features'].size(0),
#         video_features.size(0)
#     ]

#     # Create edge index and edge type
#     model = GraphModel(
#         g_dim=args['g_dim'],
#         h1_dim=args['h1_dim'],
#         h2_dim=args['h2_dim'],
#         num_relations=args['num_relations'],
#         device=args['device'],
#         args=args
#     )

#     edge_index, edge_type = model.batch_graphify(
#             lengths=lengths,
#             n_modals=args['n_modals'],
#             edge_multi=args['edge_multi'],
#             edge_temp=args['edge_temp'],
#             P=args['wp'],
#             F=args['wf'],
#             temp_sample_rate=0.2,  # Example sampling rate for temporal edges
#             multi_modal_sample_rate=0.2,  # Example sampling rate for multi-modal edges
#             edge_reduction_rate=0.75  # Retain 75% of total edges
#         )

#     print(f" edge_index: {edge_index.shape}, edge_type: {edge_type.shape}")
#     edge_index = edge_index.to(args['device'])
#     edge_type = edge_type.to(args['device'])

#     # Return encapsulated graph data
#     return GraphData(
#         x=features.to(args['device']),
#         y=labels.to(args['device']),
#         lengths=lengths,
#         edge_index=edge_index,
#         edge_type=edge_type
#     )

# #------------------------------------------------------------------------------------------
# from torch_geometric.data import DataLoader
# def create_dataloader(graph_data, batch_size):
#     device = graph_data.edge_index.device  # Ensure all tensors are on the same device
#     node_count = graph_data.x.size(0)
#     indices = torch.arange(node_count, device=device)  # Create indices on the correct device
#     split_indices = indices.split(batch_size)  # Split node indices into batches

#     batches = []
#     for idx in split_indices:
#         edge_mask = (
#             torch.isin(graph_data.edge_index[0], idx) & torch.isin(graph_data.edge_index[1], idx)
#         )

#         # Filter edge_index and edge_type
#         edge_index = graph_data.edge_index[:, edge_mask]
#         edge_type = graph_data.edge_type[edge_mask]
#         node_map = {global_id.item(): local_id for local_id, global_id in enumerate(idx)}
#         edge_index = torch.tensor(
#             [[node_map[src.item()], node_map[dst.item()]] for src, dst in edge_index.t()],
#             dtype=torch.long,
#             device=device
#         ).t()
#         if not torch.all((edge_index >= 0) & (edge_index < len(idx))):
#             raise ValueError(f"[DEBUG] Invalid edge_index values in create_dataloader: {edge_index}")

#         # Validate edge_type
#         if not torch.all((edge_type >= 0) & (edge_type < 9)):  # Assuming 9 relations
#             raise ValueError(f"[DEBUG] Invalid edge_type values in create_dataloader: {edge_type.unique()}")

#         print(f"[DEBUG] create_dataloader | edge_index.shape: {edge_index.shape}, edge_type.shape: {edge_type.shape}")

#         # Create a batch of graph data
#         batch = GraphData(
#             x=graph_data.x[idx],
#             y=graph_data.y[idx],
#             lengths=graph_data.lengths,
#             edge_index=edge_index,
#             edge_type=edge_type
#         )
#         batches.append(batch)

#     return DataLoader(batches, batch_size=1, shuffle=True)






# #------------------------------------------------------------------------------------------
# # Pipeline
# def main_pipeline(data, args):
#     # Prepare data
#     print("[INFO] Preparing train_data...")
#     train_data = prepare_graph_data(data, 'train', args)
#     print("[INFO] Preparing val_data...")
#     val_data = prepare_graph_data(data, 'val', args)
#     print("[INFO] Preparing test_data...")
#     test_data = prepare_graph_data(data, 'test', args)

#     # Initialize model
#     model = GraphModel(
#         g_dim=args['g_dim'],
#         h1_dim=args['h1_dim'],
#         h2_dim=args['h2_dim'],
#         num_relations=args['num_relations'],
#         device=args['device'],
#         args=args
#     ).to(args['device'])

#     # Optimizer and criterion
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
#     criterion = nn.CrossEntropyLoss()

#     # Train and evaluate
#     print("[INFO] Training the GraphModel...")
#     acc, f1 = train_and_evaluate(model, train_data, val_data, test_data, optimizer, criterion)
#     print(f"[INFO] Final Test Accuracy: {acc * 100:.2f}%, Final Test F1-Score: {f1 * 100:.2f}%")



# #------------------------------------------------------------------------------------------
# args = {
#     'g_dim': 768,  # Reduced input feature dimension
#     'h1_dim': 64,  # Smaller hidden layer
#     'h2_dim': 32,  # Smaller second hidden layer
#     'num_classes': 4,  # Number of output classes
#     'num_relations': 9,  # Total edge types (multi-modal + temporal)
#     'n_modals': 3,  # Number of modalities
#     'wp': 2,  # Past window size
#     'wf': 2,  # Future window size
#     'edge_multi': True,  # Enable multi-modal edges
#     'edge_temp': True,  # Enable temporal edges
#     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# }

# main_pipeline(data, args)

#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
import pickle
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, TransformerConv
from sklearn.metrics import accuracy_score, f1_score
import random

# Load dataset
data_dir = os.path.join('..', "outputs", "embeddings")
# data = torch.load(os.path.join(data_dir, "loaders_datasets_reduced_label_dim_4.pt"))
data = torch.load(os.path.join(data_dir, "loaders_datasets.pt"))

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

# Training and evaluation
def train_and_evaluate(model, train_data, val_data, test_data, optimizer, criterion, epochs=50, patience=10):
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.lengths, train_data.edge_index, train_data.edge_type)
        loss = criterion(out, train_data.y)
        loss.backward()
        optimizer.step()
        print(f"[INFO] Epoch {epoch + 1} | Train Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x, val_data.lengths, val_data.edge_index, val_data.edge_type)
            val_loss = criterion(val_out, val_data.y).item()
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

    # Test evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        test_out = model(test_data.x, test_data.lengths, test_data.edge_index, test_data.edge_type)
        test_preds = test_out.argmax(dim=1).cpu()
        test_labels = test_data.y.cpu()
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='macro')
    print(f"[INFO] Test Accuracy: {test_acc:.4f}, Test F1-Score: {test_f1:.4f}")
    return test_acc, test_f1

# Prepare data and run pipeline
def prepare_graph_data(data, split, args):
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

    print(f" edge_index: {edge_index.shape}, edge_type: {edge_type.shape}")
    edge_index = edge_index.to(args['device'])
    edge_type = edge_type.to(args['device'])

    return GraphData(
        x=features.to(args['device']),
        y=labels.to(args['device']),
        lengths=lengths,
        edge_index=edge_index,
        edge_type=edge_type
    )

from collections import namedtuple
GraphData = namedtuple('GraphData', ['x', 'y', 'lengths', 'edge_index', 'edge_type'])

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

train_data = prepare_graph_data(data, 'train', args)
val_data = prepare_graph_data(data, 'val', args)
test_data = prepare_graph_data(data, 'test', args)

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

train_and_evaluate(model, train_data, val_data, test_data, optimizer, criterion)
