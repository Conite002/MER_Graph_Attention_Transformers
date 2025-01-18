import argparse
import torch

def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modalities', type=list, default=['audio', 'text', 'video'])
    parser.add_argument('--wp', type=int, default=1)
    parser.add_argument('--wf', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hidden_size', type=int, default=64, help="Hidden size")
    parser.add_argument('--graph_transformers_nheads', type=int, default=4, help="Number of heads in the graph transformer")
    parser.add_argument('--graph_transformer_dropout', type=float, default=0.2, help="Dropout in the graph transformer")
    parser.add_argument('--cross_modal', type=bool, default=True, help="Cross modal")
    parser.add_argument("--edge_type", default="temp_multi", choices=("temp_multi", "multi", "temp"), help="Choices edge contruct type", type=str)
    parser.add_argument('--dim_modals', type=dict, default={'a': 128, 't': 256, 'v': 64})
    return parser.parse_args()
