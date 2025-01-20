import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import torch
import argparse
from models.Framework import Framework
from utils.args import get_test_args
from data.dummy import create_dummy_data
import pickle




data_dir = os.path.join(project_root, "outputs", "embeddings")
with open(os.path.join(data_dir, "loaders_datasets.pkl"), 'rb') as f:
    data = pickle.load(f)

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


for split in data.keys():
    print(f"{split} audio features shape: {data[split]['audio']['features'].shape}")
    print(f"{split} text features shape: {data[split]['text']['features'].shape}")
    print(f"{split} video features shape: {data[split]['video']['features'].shape}")
    print(f"{split} text length tensor shape: {data[split]['text']['text_len_tensor'].shape}")

#------------------------------------------------------------------------------------------------------
# Reduce the size of the data 
#------------------------------------------------------------------------------------------------------