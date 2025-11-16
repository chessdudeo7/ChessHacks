import torch
import chess
from torch.utils.data import Dataset

# Mapping from piece symbol to the index in the 12-plane tensor
PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, # White pieces
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11 # Black pieces
}

def fen_to_tensors(fen_string):
    """
    Converts a FEN string into two tensors:
    1. Spatial Tensor: (12, 8, 8) for piece locations
    2. Scalar Tensor: (5,) for game state (turn, castling)
    """
    board = chess.Board(fen_string)
    
    # 1. Spatial Tensor (12 planes, 8x8)
    spatial_tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank) # Note: chess.square uses 0-7
            piece = board.piece_at(square)
            
            if piece:
                plane_idx = PIECE_TO_PLANE[piece.symbol()]
                # (7-rank) flips the board vertically to match FEN order (rank 8 is first)
                spatial_tensor[plane_idx, 7 - rank, file] = 1.0

    # 2. Scalar Tensor (5 features)
    # (Side to move, W_KS, W_QS, B_KS, B_QS)
    scalar_tensor = torch.tensor([
        1.0 if board.turn == chess.WHITE else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
    ], dtype=torch.float32)

    return spatial_tensor, scalar_tensor

class ChessDataset(Dataset):
    """
    Dataset that loads (FEN, move) pairs and converts them to tensors.
    """
    def __init__(self, positions, vocab):
        self.positions = positions
        self.vocab = vocab

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        fen, move_uci = self.positions[idx]
        
        # Convert FEN to the two tensors
        spatial_tensor, scalar_tensor = fen_to_tensors(fen)
        
        # Convert move to target index
        # Use a default index (e.g., 0) if move is not in vocab (should be rare)
        move_idx = self.vocab.move_to_idx.get(move_uci, 0)
        policy_target = torch.tensor(move_idx, dtype=torch.long)

        # --- FIX ---
        # Removed trailing colon (:) which was a SyntaxError
        return spatial_tensor, scalar_tensor, policy_target