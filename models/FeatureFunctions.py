import torch
import numpy as np

def features_packing(multimodal_feature, lengths):
    batch_size = lengths.size(0)
    node_features = []

    for feature in multimodal_feature:
        for j in range(batch_size):
            node_features.append(feature[j,  :lengths[j].item()])
    node_features = torch.cat(node_features, dim=0)
    return node_features

def multi_concat(node_feature, lengths, n_modals):
    sum_length = lengths.sum().item()
    feature = []
    for j in range(n_modals):
        feature.append(node_feature[j * sum_length:(j + 1) * sum_length])
    feature = torch.cat(feature, dim=-1)
    return feature