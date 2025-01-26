import pickle
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
from sklearn.metrics import accuracy_score, f1_score


def load_data(data_type='loaders_datasets_reduced_label_dim_4.pt'):
    data_dir = os.path.join('..', "outputs", "embeddings")
    data = torch.load(os.path.join(data_dir, data_type))

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

    return data