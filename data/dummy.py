import torch



def create_dummy_data(batch_size=2, seq_length=10):
    data = {
        'audio': torch.randn(batch_size, seq_length, 128),
        'text': torch.randn(batch_size, seq_length, 256),
        'video': torch.randn(batch_size, seq_length, 64),
        'text_len_tensor': torch.tensor([seq_length, seq_length - 2]),
        'label_tensor': torch.tensor([1, 0])  # Dummy labels for classification
    }
    return data
