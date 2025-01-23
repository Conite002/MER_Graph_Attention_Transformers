import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm   
from sklearn.metrics import confusion_matrix, classification_report


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, n_modals, args):
        super(GNN, self).__init__()
        self.rgcn = RGCNConv(g_dim, h1_dim, num_relations=num_relations)
        self.transformer = TransformerConv(h1_dim, h2_dim, heads=1)

    def forward(self, x, edge_index, edge_type):
        # Validate edge_index
        debug_message("Validating edge_index", edge_index.shape)
        if edge_index.max() >= x.size(0) or edge_index.min() < 0:
            raise ValueError(f"Invalid edge_index values: max = {edge_index.max()}, min = {edge_index.min()}, x.size(0) = {x.size(0)}")

        # Apply GNN layers
        x = self.rgcn(x, edge_index, edge_type)
        debug_message("RGCN output shape", x.shape)
        x = torch.relu(x)
        x = self.transformer(x, edge_index)
        x = torch.relu(x)
        debug_message("Transformer output shape", x.shape)
        return x