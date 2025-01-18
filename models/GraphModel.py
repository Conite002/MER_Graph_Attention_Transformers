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

        print(f"GraphModel -> Edge type: {args.edge_type}")
        print(f"GraphModel -> Window past : {args.wp}")
        print(f"GraphModel -> Window future : {args.wf}")
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
                    if(j != k):
                        edge_type_to_idx['0' + str(j) + str(k)] = len(edge_type_to_idx)
        
        self.edge_type_to_idx = edge_type_to_idx
        self.num_relations = len(edge_type_to_idx)
        self.edge_temp = edge_temp
        self.edge_multi = edge_multi

        self.gnn = GNN(g_dim, h1_dim, h2_dim, self.num_relations, self.n_modals, args)

    def forward(self, x, lengths):
        node_features = features_packing(x, lengths)
        note_type, edge_index, edge_type, edge_index_lenghts = self.batch_graphify(lengths)
        out_gnn = self.gnn(node_features, note_type, edge_index, edge_type)
        out_gnn = multi_concat(out_gnn, edge_index_lenghts, self.n_modals)
        return out_gnn
    
    def batch_graphify(self, lengths):
        node_type, edge_index, edge_type, edge_index_lengths = [], [], [], []
        edge_type_lengths = [0] * len(self.edge_type_to_idx)
        lengths = lengths.tolist()

        sum_lengths = 0
        total_lengths = sum(lengths)
        batch_size = len(lengths)

        for k in range(self.n_modals):
            for j in range(batch_size):
                cur_len= lengths[j]
                node_type.extend([k] * cur_len)
        
        for j in range(batch_size):
            cur_len = lengths[j]
            perms = self.edge_perms(cur_len, total_lengths)
            edge_index_lengths.append(len(perms))

            for item in perms:
                vertices = item[0]
                neighbor = item[1]
                edge_index.append(torch.tensor([vertices + sum_lengths, neighbor + sum_lengths]))
                if vertices % total_lengths > neighbor % total_lengths:
                    temporal_type = 1
                elif vertices % total_lengths < neighbor % total_lengths:
                    temporal_type = -1
                else:
                    temporal_type = 0
                edge_type.append(self.edge_type_to_idx[str(temporal_type) +  str(node_type[vertices + sum_lengths]) + str(node_type[neighbor + sum_lengths])])

            sum_lengths += cur_len

        node_type = torch.tensor(node_type, dtype=torch.long, device=self.device)
        edge_index = torch.stack(edge_index).t().contiguous().to(self.device)
        edge_type = torch.tensor(edge_type, dtype=torch.long, device=self.device)
        edge_index_lengths = torch.tensor(edge_index_lengths, dtype=torch.long, device=self.device)
        return node_type, edge_index, edge_type, edge_index_lengths
    

    def edge_perms(self, length, total_length):
        all_perms = set()
        array = np.arange(length)
        
        for j in range(length):
            if self.wp == -1 and self.wf == -1:
                eff_array = array
            elif self.wp == -1:
                eff_array = array[ :min(j + self.wf + 1, length)]
            elif self.wf == -1:
                eff_array = array[max(0, j - self.wp) : ]
            else:
                eff_array = array[
                    max(0, j - self.wp) : min(j + self.wf + 1, length)
                ]
            perms = set()

            for k in range(self.n_modals):
                node_index = j + k * total_length
                if self.edge_temp:
                    for item in eff_array:
                        perms.add((node_index, item + k * total_length))
                else:
                    perms.add((node_index, node_index))

                if self.edge_multi:
                    for l in range(self.n_modals):
                        if k != l:
                            perms.add((node_index, j + l * total_length))
            all_perms = all_perms.union(perms)

        return all_perms