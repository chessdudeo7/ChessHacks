import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self, input_size=773, output_size=10000):
        """
        input_size: 64*12 + 5 = 773 features from fen_to_vector
        output_size: number of unique moves in your MoveVocab
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits for CrossEntropyLoss