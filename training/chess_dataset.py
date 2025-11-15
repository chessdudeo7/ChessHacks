import torch
from torch.utils.data import Dataset
from encoding import fen_to_vector

class ChessDataset(Dataset):
    def __init__(self, positions, vocab):
        self.positions = positions
        self.vocab = vocab

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        fen, move = self.positions[idx]
        x = fen_to_vector(fen)
        y = self.vocab.encode(move)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

