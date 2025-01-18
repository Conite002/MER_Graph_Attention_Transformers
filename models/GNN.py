import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv

class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, num_modals, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_modals = num_modals
        print(f"GNN -> Use RGCNConv & TransformerConv")
        self.conv1 = RGCNConv(g_dim, h1_dim, num_relations)
        self.conv2 = TransformerConv(h1_dim, h2_dim, heads=args.graph_transformers_nheads, dropout=args.graph_transformer_dropout, concat=True)
        self.bn = nn.BatchNorm1d(h2_dim * args.graph_transformers_nheads)

    def forward(self, node_features, node_type, edge_index, edge_type):
        x = self.conv1(node_features, edge_index, edge_type)
        x = nn.functional.leaky_relu(self.conv2(x, edge_index))
        return x