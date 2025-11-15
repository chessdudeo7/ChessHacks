import torch
from model import ChessNet

# Load saved vocab
vocab = torch.load("move_vocab.pt")

# Create model with correct output size
model = ChessNet(output_size=len(vocab.idx_to_move))
model.load_state_dict(torch.load("chess_model.pt"))
model.eval()  # set to evaluation mode

# Example: predict a move from a FEN
from encoding import fen_to_vector
fen = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
x = torch.tensor(fen_to_vector(fen)).unsqueeze(0)  # batch dimension
with torch.no_grad():
    preds = model(x)
    move_idx = preds.argmax(dim=1).item()
    move = vocab.decode(move_idx)
    print("Predicted move:", move)
