import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm   
from sklearn.metrics import confusion_matrix, classification_report
from models.GNN import GNN
import numpy as np
from utils.debug import debug_message

class GraphModel(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, device, args):
        super(GraphModel, self).__init__()

        self.n_modals = len(args['modalities'])
        self.wp = args['wp']
        self.wf = args['wf']
        self.device = device
        self.gfp = args['gfpush']
        self.d_nodes = args['drop_nodes']
        print(f"GraphModel --> Edge type: {args['edge_type']}")
        print(f"GraphModel --> Window past: {args['wp']}")
        print(f"GraphModel --> Window future: {args['wf']}")
        self.num_classes = args['num_classes']
        edge_temp = "temp" in args['edge_type']
        edge_multi = "multi" in args['edge_type']

        edge_type_to_idx = {}

        if edge_temp:
            temporal = [-1, 1, 0]
            for j in temporal:
                for k in range(self.n_modals):
                    edge_type_to_idx[f"{j}{k}{k}"] = len(edge_type_to_idx)
        else:
            for j in range(self.n_modals):
                edge_type_to_idx[f"0{j}{j}"] = len(edge_type_to_idx)

        if edge_multi:
            for j in range(self.n_modals):
                for k in range(self.n_modals):
                    if j != k:
                        edge_type_to_idx[f"0{j}{k}"] = len(edge_type_to_idx)

        self.edge_type_to_idx = edge_type_to_idx
        self.num_relations = len(edge_type_to_idx)
        self.edge_multi = edge_multi
        self.edge_temp = edge_temp

        self.gnn = GNN(g_dim, h1_dim, h2_dim, self.num_relations, self.n_modals, args)
        self.fc = nn.Sequential(
            nn.Linear(h2_dim, 128),  # Intermediate hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization
            nn.Linear(128, self.num_classes)  # Output layer
        )
    def forward(self, x, lengths):
        if self.d_nodes:
            x = drop_nodes(x, drop_prob=0.2)
        node_features = self.feature_packing(x, lengths)

        node_type, edge_index, edge_type, edge_index_lengths = self.batch_graphify(lengths)
        if self.gfp:
            node_features, edge_index, node_type = gfpush(node_features, edge_index, node_type, top_k=0.5)

        out_gnn = self.gnn(node_features, edge_index, edge_type)
        out_gnn = self.multi_concat(out_gnn, lengths)
        out_gnn = self.fc(out_gnn)
        return out_gnn

    def batch_graphify(self, lengths):
        node_type, edge_index, edge_type, edge_index_lengths = [], [], [], []
        edge_type_lengths = [0] * len(self.edge_type_to_idx)

        lengths = lengths.tolist()

        sum_length = 0
        total_length = sum(lengths)
        batch_size = len(lengths)

        for k in range(self.n_modals):
            for j in range(batch_size):
                cur_len = lengths[j]
                node_type.extend([k] * cur_len)

        for j in range(batch_size):
            cur_len = lengths[j]

            perms = self.edge_perms(cur_len, total_length)
            edge_index_lengths.append(len(perms))

            for item in perms:
                vertices = item[0]
                neighbor = item[1]
                edge_index.append([vertices + sum_length, neighbor + sum_length])

                if vertices % total_length > neighbor % total_length:
                    temporal_type = 1
                elif vertices % total_length < neighbor % total_length:
                    temporal_type = -1
                else:
                    temporal_type = 0

                edge_type.append(self.edge_type_to_idx[f"{temporal_type}{node_type[vertices + sum_length]}{node_type[neighbor + sum_length]}"])

            sum_length += cur_len

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        if edge_index.max() >= total_length or edge_index.min() < 0:
            raise ValueError(f"Invalid edge_index values: max = {edge_index.max()}, min = {edge_index.min()}, total_length = {total_length}")

        node_type = torch.tensor(node_type).long().to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        edge_index_lengths = torch.tensor(edge_index_lengths).long().to(self.device)

        return node_type, edge_index, edge_type, edge_index_lengths

    def edge_perms(self, length, total_lengths):
        all_perms = set()
        array = np.arange(length)

        for j in range(length):
            if self.wp == -1 and self.wf == -1:
                eff_array = array
            elif self.wp == -1:
                eff_array = array[: min(length, j + self.wf)]
            elif self.wf == -1:
                eff_array = array[max(0, j - self.wp) :]
            else:
                eff_array = array[max(0, j - self.wp) : min(length, j + self.wf)]

            for k in range(self.n_modals):
                node_index = j + k * total_lengths
                if self.edge_temp:
                    for item in eff_array:
                        all_perms.add((node_index, item + k * total_lengths))
                else:
                    all_perms.add((node_index, node_index))
                if self.edge_multi:
                    for l in range(self.n_modals):
                        if l != k:
                            all_perms.add((node_index, j + l * total_lengths))

        all_perms = [(src, dst) for src, dst in all_perms if 0 <= src < total_lengths and 0 <= dst < total_lengths]
        return list(all_perms)

    def feature_packing(self, x, lengths):
        if isinstance(x, (list, tuple)):
            packed = torch.cat(x, dim=0)
        else:
            packed = x
        
        if packed.size(0) != lengths.sum():
            raise ValueError(f"Mismatch in packed features size: packed.size(0) = {packed.size(0)}, lengths.sum() = {lengths.sum().item()}")
        
        return packed



    def multi_concat(self, out_gnn, lengths):
        split_sizes = lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths
        
        if sum(split_sizes) != out_gnn.size(0):
            raise ValueError(f"Mismatch: lengths.sum() = {sum(split_sizes)}, out_gnn.size(0) = {out_gnn.size(0)}")
        
        split_features = torch.split(out_gnn, split_sizes, dim=0)
        if any(f.size(1) != split_features[0].size(1) for f in split_features):
            raise ValueError("Inconsistent tensor shapes in split features")

        return torch.cat(split_features, dim=1)


import torch

def drop_nodes(x, drop_prob=0.2):
    """
    Drop a subset of nodes by masking their features.
    Args:
        x: Node feature matrix of shape (num_nodes, feature_dim).
        drop_prob: Probability of dropping each node.
    Returns:
        Modified node feature matrix with some nodes dropped.
    """
    if drop_prob <= 0.0:
        return x

    mask = torch.rand(x.size(0), device=x.device) > drop_prob
    x = x * mask.unsqueeze(1)  
    return x

import torch.nn.functional as F

def gfpush(x, edge_index, batch, top_k=0.5):
    """
    Apply GFPush to reduce graph size by selecting important nodes.
    Args:
        x: Node feature matrix (num_nodes, feature_dim).
        edge_index: Edge indices (2, num_edges).
        batch: Batch assignment vector for each node (num_nodes,).
        top_k: Proportion of nodes to keep (e.g., 0.5 means keep top 50%).
    Returns:
        Reduced graph with selected nodes.
    """
    scores = F.relu(x).sum(dim=1)  

    num_nodes = x.size(0)
    num_keep = int(top_k * num_nodes)
    top_nodes = scores.topk(num_keep).indices

    new_x = x[top_nodes]
    new_batch = batch[top_nodes]

    mask = torch.isin(edge_index[0], top_nodes) & torch.isin(edge_index[1], top_nodes)
    new_edge_index = edge_index[:, mask]

    return new_x, new_edge_index, new_batch



# -----------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import RGCNConv, TransformerConv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

class FusionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FusionLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.attention = nn.Linear(output_dim, 1)

    def forward(self, audio_feat, text_feat, video_feat):
        combined_feat = torch.cat([audio_feat, text_feat, video_feat], dim=-1)

        fused_feat = torch.relu(self.fc1(combined_feat))
        fused_feat = self.fc2(fused_feat)

        attention_weights = torch.sigmoid(self.attention(fused_feat))
        weighted_feat = fused_feat * attention_weights

        return weighted_feat

class CrossModalGNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, n_modals, args):
        super(CrossModalGNN, self).__init__()

        # RGCN for relational learning
        self.rgcn = RGCNConv(g_dim, h1_dim, num_relations=num_relations)

        # Cross-modal attention with TransformerConv
        self.cross_modal_attention = TransformerConv(h1_dim, h2_dim, heads=4, dropout=0.3)

        # Residual connection
        self.residual = nn.Linear(h1_dim, h2_dim*4)

        # Batch normalization
        self.bn = nn.BatchNorm1d(h2_dim*4)

    def forward(self, x, edge_index, edge_type):
        if edge_index.max() >= x.size(0) or edge_index.min() < 0:
            raise ValueError(f"Invalid edge_index values: max = {edge_index.max()}, min = {edge_index.min()}, x.size(0) = {x.size(0)}")

        # RGCN forward pass
        x = self.rgcn(x, edge_index, edge_type)
        x = torch.relu(x)

        # Cross-modal attention with residual connection
        attention_out = self.cross_modal_attention(x, edge_index)
        x = torch.relu(attention_out + self.residual(x))
        # print(f"x shape before TransformerConv: {x.shape}")
        # print(f"attention_out shape: {attention_out.shape}")  # Should match residual connection
        # print(f"x shape after residual connection: {x.shape}")

        # Apply batch normalization
        x = self.bn(x)

        return x

class GraphModel_CFusion(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, device, args, num_classes=7):
        super(GraphModel_CFusion, self).__init__()

        self.n_modals = len(args['modalities'])
        self.wp = args['wp']
        self.wf = args['wf']
        self.device = device
        self.gfp = args['gfpush']
        self.d_nodes = args['drop_nodes']

        # Initialize edge type mapping
        self.edge_type_to_idx = {}
        self.edge_temp = "temp" in args['edge_type']
        self.edge_multi = "multi" in args['edge_type']

        if self.edge_temp:
            temporal = [-1, 1, 0]
            for j in temporal:
                for k in range(self.n_modals):
                    self.edge_type_to_idx[f"{j}{k}{k}"] = len(self.edge_type_to_idx)
        else:
            for j in range(self.n_modals):
                self.edge_type_to_idx[f"0{j}{j}"] = len(self.edge_type_to_idx)

        if self.edge_multi:
            for j in range(self.n_modals):
                for k in range(self.n_modals):
                    if j != k:
                        self.edge_type_to_idx[f"0{j}{k}"] = len(self.edge_type_to_idx)

        self.edge_type_to_idx = self.edge_type_to_idx
        self.num_relations = len(self.edge_type_to_idx)

        # Initialize cross-modal GNN
        self.gnn = CrossModalGNN(g_dim, h1_dim, h2_dim, num_relations=self.num_relations, n_modals=self.n_modals, args=args)

        # Fusion layer for combining modalities
        self.fusion_layer = FusionLayer(
            input_dim=256 * self.n_modals,  # Combined feature size of all modalities
            hidden_dim=256,
            output_dim=128
        )

        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        if self.d_nodes:
            x = drop_nodes(x, drop_prob=0.2)

        node_features = self.feature_packing(x, lengths)
        node_type, edge_index, edge_type, edge_index_lengths = self.batch_graphify(lengths)
        if self.gfp:
            node_features, edge_index, node_type = gfpush(node_features, edge_index, node_type, top_k=0.5)

        out_gnn = self.gnn(node_features, edge_index, edge_type)

        fused_features = self.fusion_layer(out_gnn, out_gnn, out_gnn)  # Replace with actual modality features

        out = self.fc(fused_features)
        return out
    def edge_perms(self, length, total_lengths):
        """
        Compute all valid edge permutations for a given graph.
        Args:
            length (int): Number of nodes in the current graph.
            total_lengths (int): Total number of nodes across all graphs in the batch.
        Returns:
            List of valid edge permutations as tuples (source, destination).
        """
        all_perms = set()
        array = np.arange(length)

        for j in range(length):
            if self.wp == -1 and self.wf == -1:
                eff_array = array  # Full temporal connectivity
            elif self.wp == -1:
                eff_array = array[: min(length, j + self.wf)]
            elif self.wf == -1:
                eff_array = array[max(0, j - self.wp) :]
            else:
                eff_array = array[max(0, j - self.wp) : min(length, j + self.wf)]

            for k in range(self.n_modals):
                node_index = j + k * total_lengths
                if self.edge_temp:
                    for item in eff_array:
                        all_perms.add((node_index, item + k * total_lengths))
                else:
                    all_perms.add((node_index, node_index))
                if self.edge_multi:
                    for l in range(self.n_modals):
                        if l != k:
                            all_perms.add((node_index, j + l * total_lengths))

        # Filter out invalid edges
        all_perms = [(src, dst) for src, dst in all_perms if 0 <= src < total_lengths and 0 <= dst < total_lengths]
        return list(all_perms)

    def batch_graphify(self, lengths):
        node_type, edge_index, edge_type, edge_index_lengths = [], [], [], []
        edge_type_lengths = [0] * len(self.edge_type_to_idx)

        lengths = lengths.tolist()

        sum_length = 0
        total_length = sum(lengths)
        batch_size = len(lengths)

        for k in range(self.n_modals):
            for j in range(batch_size):
                cur_len = lengths[j]
                node_type.extend([k] * cur_len)

        for j in range(batch_size):
            cur_len = lengths[j]

            perms = self.edge_perms(cur_len, total_length)
            edge_index_lengths.append(len(perms))

            for item in perms:
                vertices = item[0]
                neighbor = item[1]
                edge_index.append([vertices + sum_length, neighbor + sum_length])

                if vertices % total_length > neighbor % total_length:
                    temporal_type = 1
                elif vertices % total_length < neighbor % total_length:
                    temporal_type = -1
                else:
                    temporal_type = 0

                edge_type.append(self.edge_type_to_idx[f"{temporal_type}{node_type[vertices + sum_length]}{node_type[neighbor + sum_length]}"])

            sum_length += cur_len

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        if edge_index.max() >= total_length or edge_index.min() < 0:
            raise ValueError(f"Invalid edge_index values: max = {edge_index.max()}, min = {edge_index.min()}, total_length = {total_length}")

        node_type = torch.tensor(node_type).long().to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        edge_index_lengths = torch.tensor(edge_index_lengths).long().to(self.device)

        return node_type, edge_index, edge_type, edge_index_lengths

    def feature_packing(self, x, lengths):
        if isinstance(x, (list, tuple)):
            packed = torch.cat(x, dim=0)
        else:
            packed = x
        
        if packed.size(0) != lengths.sum():
            raise ValueError(f"Mismatch in packed features size: packed.size(0) = {packed.size(0)}, lengths.sum() = {lengths.sum().item()}")
        
        return packed



    def multi_concat(self, out_gnn, lengths):
        split_sizes = lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths
        
        if sum(split_sizes) != out_gnn.size(0):
            raise ValueError(f"Mismatch: lengths.sum() = {sum(split_sizes)}, out_gnn.size(0) = {out_gnn.size(0)}")
        
        split_features = torch.split(out_gnn, split_sizes, dim=0)
        if any(f.size(1) != split_features[0].size(1) for f in split_features):
            raise ValueError("Inconsistent tensor shapes in split features")

        return torch.cat(split_features, dim=1)



def train_model(model, train_loader, val_loader, optimizer, criterion, args, epochs=10):
    best_val_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1} Training") as t:
            for features, labels in train_loader:
                features = features.to(args['device'])
                labels = labels.to(args['device'])

                optimizer.zero_grad()
                out = model(features, lengths=torch.tensor([len(features)]).to(args['device']))
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(args['device'])
                labels = labels.to(args['device'])

                out = model(features, lengths=torch.tensor([len(features)]).to(args['device']))
                val_loss += criterion(out, labels).item()

        val_loss /= len(val_loader)
        debug_message(f"Epoch {epoch + 1} train loss {train_loss / len(train_loader):.4f}, val loss {val_loss:.4f}")

        # Learning rate adjustment
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'cross_modal_best_model.pth')
            debug_message(f"Epoch {epoch + 1}: Best model saved")