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

        print(f"GraphModel --> Edge type: {args['edge_type']}")
        print(f"GraphModel --> Window past: {args['wp']}")
        print(f"GraphModel --> Window future: {args['wf']}")
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

    def forward(self, x, lengths):
        node_features = self.feature_packing(x, lengths)

        node_type, edge_index, edge_type, edge_index_lengths = self.batch_graphify(lengths)

        out_gnn = self.gnn(node_features, edge_index, edge_type)
        out_gnn = self.multi_concat(out_gnn, lengths)

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

