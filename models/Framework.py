import torch
import torch.nn as nn
from .GraphModel import GraphModel
from .FeatureFunctions import features_packing, multi_concat

class Framework(nn.Module):
    def __init__(self, args):
        super(Framework, self).__init__()
        self.n_modals = len(args.modalities)
        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device
        g_dim = args.hidden_size
        h_dim = args.hidden_size

        ic_dim = h_dim * self.n_modals * args.graph_transformers_nheads
        if args.cross_modal:
            ic_dim += h_dim * self.n_modals * (self.n_modals - 1)

        a_dim = args.dim_modals['a']
        t_dim = args.dim_modals['t']
        v_dim = args.dim_modals['v']

        self.graph_model = GraphModel(g_dim, h_dim, h_dim, self.device, args)
        print(f"Build_graph -> {self.graph_model}")

    def represent(self, data):
        a = data['audio']
        t = data['text']
        v = data['video']

        multimodal_features = [a, t, v]
        out_encode = features_packing(multimodal_feature=multimodal_features, lengths=data['text_len_tensor'])
        out = []
        out_graph = self.graph_model(out_encode, data['text_len_tensor'])
        out.append(out_graph)
        # cross-modal
        out_cr = self.crossmodal(multimodal_features)
        out_cr = out_cr.permute(1, 0, 2)
        lengths = data['text_len_tensor']
        batch_size = lengths.size(0)

        cr_feat = []
        for j in range(batch_size):
            cr_feat.append(out_cr[j, :lengths[j].item()])
        cr_feat = torch.cat(cr_feat, dim=0).to(self.device)
        out.append(cr_feat)

        out = torch.cat(out, dim=-1)
        return out
    
    def forward(self, data):
        graph_out = self.represent(data)
        out = self.clf(graph_out, data['text_len_tensor'])

        return out
    
    def get_loss(self, data):
        graph_out = self.represent(data)
        loss = self.clf.get_loss(graph_out, data['label_tensor'], data['text_len_tensor'])
        return loss
    
    def get_log(self):
        return self.rlog
