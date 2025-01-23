from utils.debug import debug_message
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class GraphDataset(Dataset):
    def __init__(self, features, labels, lengths):
        self.features = features
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def prepare_dataloaders(features, labels, lengths, batch_size):
    dataset = GraphDataset(features, labels, lengths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
    
def prepare_graph_data(data, args):
    """
    Prépare les données pour le modèle GraphModel.
    :param data: Dictionnaire contenant les caractéristiques et les étiquettes.
    :param args: Arguments de configuration.
    :return: Tenseur pour les caractéristiques, les étiquettes, et les longueurs.
    """
    video_features = torch.mean(data['video']['features'], dim=1)
    features = torch.cat([
        data['audio']['features'],
        data['text']['features'],
        video_features
    ], dim=0)

    labels = torch.cat([
        data['audio']['labels'],
        data['text']['labels'],
        data['video']['labels']
    ], dim=0)

    lengths = [
        data['audio']['features'].size(0),
        data['text']['features'].size(0),
        video_features.size(0)
    ]

    debug_message("Prepared features shape", features.shape)
    debug_message("Prepared labels shape", labels.shape)
    debug_message("Prepared lengths", lengths)

    return features, labels, torch.tensor(lengths, dtype=torch.long)


