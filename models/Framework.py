import torch
import torch.nn as nn
from .GraphModel import GraphModel
from .FeatureFunctions import features_packing, multi_concat

class Framework(nn.Module):
    def __init__(self, args):
        super(Framework, self).__init__()
        self.device = args.device
        g_dim = args.hidden_size
        h_dim = args.hidden_size
        ic_dim = h_dim * len(args.modalities) * args.graph_transformers_nheads

        self.graph_model = GraphModel(g_dim, h_dim, h_dim, self.device, args)
        self.clf = nn.Linear(ic_dim, args.num_classes).to(self.device)

    def forward(self, data):
        graph_out = self.represent(data)
        logits = self.clf(graph_out)
        return logits

    def represent(self, data):
        multimodal_features = [
            data['audio']['features'].to(self.device),
            data['text']['features'].to(self.device),
            data['video']['features'].to(self.device)
        ]
        text_len_tensor = data['text']['text_len_tensor'].to(self.device)
        packed_features = features_packing(multimodal_features, text_len_tensor)
        return self.graph_model(packed_features, text_len_tensor)

    def get_loss(self, data):
        logits = self.forward(data)
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, data['label_tensor'].to(self.device))
