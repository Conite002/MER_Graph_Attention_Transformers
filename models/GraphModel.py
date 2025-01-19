import torch
import torch.nn as nn
import numpy as np
from .GNN import GNN
from .FeatureFunctions import features_packing, multi_concat

class GraphModel(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, device, args):
        super(GraphModel, self).__init__()
        self.n_modals = len(args.modalities)
        self.wp = args.wp
        self.wf = args.wf
        self.device = device

        edge_temp = "temp" in args.edge_type
        edge_multi = "multi" in args.edge_type

        edge_type_to_idx = {}
        if edge_temp:
            temporal = [-1, 0, 1]
            for j in temporal:
                for k in range(self.n_modals):
                    edge_type_to_idx[str(j) + str(k) + str(k)] = len(edge_type_to_idx)
        else:
            for j in range(self.n_modals):
                edge_type_to_idx['0' + str(j) + str(j)] = len(edge_type_to_idx)
        if edge_multi:
            for j in range(self.n_modals):
                for k in range(self.n_modals):
                    if j != k:
                        edge_type_to_idx['0' + str(j) + str(k)] = len(edge_type_to_idx)
        
        self.edge_type_to_idx = edge_type_to_idx
        self.num_relations = len(edge_type_to_idx)
        self.edge_temp = edge_temp
        self.edge_multi = edge_multi

        self.gnn = GNN(g_dim, h1_dim, h2_dim, self.num_relations, self.n_modals, args)

    def forward(self, node_features, lengths):
        note_type, edge_index, edge_type, edge_index_lengths = self.batch_graphify(lengths)
        out_gnn = self.gnn(node_features, note_type, edge_index, edge_type)
        out_gnn = multi_concat(out_gnn, edge_index_lengths, self.n_modals)
        return out_gnn
    
    def batch_graphify(self, lengths):
        node_type, edge_index, edge_type = [], [], []
        lengths = lengths.tolist()

        sum_lengths = 0
        total_lengths = sum(lengths)
        batch_size = len(lengths)

        # Populate node_type
        for k in range(self.n_modals):
            for j in range(batch_size):
                cur_len = lengths[j]
                node_type.extend([k] * cur_len)

        # Populate edge_index and edge_type
        for j in range(batch_size):
            cur_len = lengths[j]
            perms = self.edge_perms(cur_len, total_lengths)
            for item in perms:
                vertices, neighbor = item
                edge_index.append([vertices + sum_lengths, neighbor + sum_lengths])
                temporal_type = 0 if vertices == neighbor else (1 if vertices > neighbor else -1)
                edge_type.append(self.edge_type_to_idx[f"{temporal_type}{node_type[vertices]}{node_type[neighbor]}"])
            sum_lengths += cur_len

        # Validate edge_index
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            assert edge_index.max() < total_lengths, (
                f"Invalid index in edge_index: max={edge_index.max()}, total_lengths={total_lengths}"
            )
            assert edge_index.min() >= 0, "Negative index found in edge_index"
        else:
            edge_index = torch.tensor([], dtype=torch.long).to(self.device)

        # Convert to tensors and move to device
        node_type = torch.tensor(node_type, dtype=torch.long).to(self.device)
        edge_type = torch.tensor(edge_type, dtype=torch.long).to(self.device)
        edge_index_lengths = torch.tensor(lengths, dtype=torch.long).to(self.device)

        return node_type, edge_index, edge_type, edge_index_lengths


    def edge_perms(self, length, total_length):
        array = np.arange(length)
        all_perms = set()
        for j in range(length):
            eff_array = array[max(0, j - self.wp):min(j + self.wf + 1, length)]
            for k in range(self.n_modals):
                node_index = j + k * total_length
                for item in eff_array:
                    neighbor_index = item + k * total_length
                    if neighbor_index < total_length:  # Ensure valid index
                        all_perms.add((node_index, neighbor_index))
        return all_perms
