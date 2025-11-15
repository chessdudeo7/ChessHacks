class MoveVocab:
    def __init__(self):
        self.move_to_idx = {}
        self.idx_to_move = {}
    
    def build_from_positions(self, positions):
        """Build vocabulary from all positions"""
        moves = set()
        for fen, move in positions:
            moves.add(move)
        
        for i, move in enumerate(sorted(moves)):
            self.move_to_idx[move] = i
            self.idx_to_move[i] = move
    
    def encode(self, move):
        """Convert move to index"""
        if move not in self.move_to_idx:
            raise ValueError(f"Move {move} not in vocabulary")
        return self.move_to_idx[move]
    
    def decode(self, idx):
        """Convert index back to move"""
        return self.idx_to_move.get(idx, None)