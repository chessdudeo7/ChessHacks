class MoveVocab:
    def __init__(self):
        self.move_to_idx = {}
        self.idx_to_move = {}
        # Add a default "unknown" move at index 0
        self.move_to_idx["<unk>"] = 0
        self.idx_to_move[0] = "<unk>"
    
    def build_from_positions(self, positions):
        """Build vocabulary from all positions"""
        moves = set()
        for fen, move in positions:
            moves.add(move)
        
        # Start indexing from 1, as 0 is reserved for "<unk>"
        for i, move in enumerate(sorted(moves), 1):
            self.move_to_idx[move] = i
            self.idx_to_move[i] = move
    
    def encode(self, move):
        """Convert move to index"""
        # Return 0 ("<unk>") if move is not found
        return self.move_to_idx.get(move, 0)
    
    def decode(self, idx):
        """Convert index back to move"""
        return self.idx_to_move.get(idx, None)