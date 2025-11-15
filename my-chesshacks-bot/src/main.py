from .utils import chess_manager, GameContext
import torch
import chess
import random
import os
import sys

# ----------------------------------------------------
# Add training folder (where Modal model is downloaded)
# ----------------------------------------------------
TRAINING_DIR = r"c:\Users\snowy\Documents\ChessHacks\trained_model"
sys.path.insert(0, TRAINING_DIR)  # allow importing local training code

# Import model components
from chess_dataset import fen_to_vector
from move_vocab import MoveVocab
from model import ChessNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------
# Load the latest trained vocab
# ----------------------------------------------------
VOCAB_PATH = os.path.join(TRAINING_DIR, "trained_model_vocab.pt")
vocab = torch.load(VOCAB_PATH, map_location=DEVICE)
print("Loaded vocab with", len(vocab.idx_to_move), "moves.")


# ----------------------------------------------------
# Load the latest trained model
# ----------------------------------------------------
MODEL_PATH = os.path.join(TRAINING_DIR, "trained_model.pt")
model = ChessNet(output_size=len(vocab.idx_to_move))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print("Ready to play chess.\n")


# ----------------------------------------------------
# AI Move Logic
# ----------------------------------------------------
@chess_manager.entrypoint
def play_move(ctx: GameContext):
    legal_moves = list(ctx.board.generate_legal_moves())

    if not legal_moves:
        ctx.logProbabilities({})
        return None

    # Encode board to tensor
    x = torch.tensor(
        fen_to_vector(ctx.board.fen()),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    # Predict move probabilities
    with torch.no_grad():
        logits = model(x).squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Map probabilities to legal moves
    move_probs = {}
    for move in legal_moves:
        uci = move.uci()
        if uci in vocab.move_to_idx:
            idx = vocab.move_to_idx[uci]
            move_probs[move] = float(probs[idx])

    # Fallback: uniform distribution
    if not move_probs:
        for move in legal_moves:
            move_probs[move] = 1.0 / len(legal_moves)

    # Log for the API
    ctx.logProbabilities({
        m.uci(): p for m, p in move_probs.items()
    })

    # Choose move based on probability weights
    moves = list(move_probs.keys())
    weights = list(move_probs.values())
    chosen_move = random.choices(moves, weights, k=1)[0]

    return chosen_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    pass