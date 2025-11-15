import chess
import numpy as np

# Map FEN pieces to indices
PIECE_MAP = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}

def fen_to_vector(fen):
    board = chess.Board(fen)
    vec = np.zeros(64 * 12 + 5, dtype=np.float32)

    # Encode pieces
    for square, piece in board.piece_map().items():
        idx = square * 12 + PIECE_MAP[piece.symbol()]
        vec[idx] = 1.0

    # Extra features
    offset = 64 * 12
    vec[offset + 0] = board.turn
    vec[offset + 1] = board.fullmove_number
    vec[offset + 2] = board.halfmove_clock
    vec[offset + 3] = board.has_kingside_castling_rights(board.turn)
    vec[offset + 4] = board.has_queenside_castling_rights(board.turn)

    return vec