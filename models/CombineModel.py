import torch.nn as nn
import torch
import numpy as np


class CombinedModel(nn.Module):
    def __init__(self, graph_model, crossmodal_model):
        super(CombinedModel, self).__init__()
        self.graph_model = graph_model
        self.crossmodal_model = crossmodal_model

    def forward(self, features, lengths):
        graph_output = self.graph_model(features, lengths)
    
        modalities = [graph_output[:, :, :256], graph_output[:, :, 256:512], graph_output[:, :, 512:]] 
        crossmodal_output = self.crossmodal_model(modalities)

        return crossmodal_output
