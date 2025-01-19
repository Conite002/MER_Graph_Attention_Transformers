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
import torch

# Mock Arguments
class Args:
    modalities = ['audio', 'text', 'video']
    wp = 1
    wf = 1
    edge_type = "temp_multi"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 64
    graph_transformers_nheads = 4
    graph_transformer_dropout = 0.1
    num_classes = 3

args = Args()

# Mock Data
data = {
    'audio': {'features': torch.rand(10, 15, 768)},
    'text': {'features': torch.rand(10, 20, 768), 'text_len_tensor': torch.randint(1, 20, (10,))},
    'video': {'features': torch.rand(10, 30, 768)},
    'label_tensor': torch.randint(0, 3, (10,))
}

# Model
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Framework(args)

# Forward Pass
logits = model(data)
print(f"Logits shape: {logits.shape}")

# Loss
loss = model.get_loss(data)
print(f"Loss: {loss.item()}")

#------------------------------------------------------------------------------------------------------

args = get_test_args()
framework = Framework(args)
output = framework(data)
print(f"Framework Forward Output Shape: {output.shape}")

