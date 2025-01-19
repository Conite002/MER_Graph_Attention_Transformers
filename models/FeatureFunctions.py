import torch

def features_packing(multimodal_feature, lengths):
    node_features = []
    for feature in multimodal_feature:
        for j in range(lengths.size(0)):
            sliced_feature = feature[j, :lengths[j].item()]
            if feature.dim() == 3:
                sliced_feature = sliced_feature.mean(dim=0, keepdim=True)
            node_features.append(sliced_feature)
    return torch.cat(node_features, dim=0)

def multi_concat(node_feature, lengths, n_modals):
    sum_length = lengths.sum().item()
    features = [node_feature[j * sum_length:(j + 1) * sum_length] for j in range(n_modals)]
    return torch.cat(features, dim=-1)
