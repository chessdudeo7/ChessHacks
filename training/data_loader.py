import chess.pgn
import os

def load_positions_from_pgn_folder(folder_path, limit_games=None):
    positions = []
    game_count = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pgn"):
            continue  # ignore non-pgn files

        full_path = os.path.join(folder_path, filename)
        print(f"Loading {full_path}")

        with open(full_path, "r", encoding="utf-8", errors="ignore") as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break

                board = game.board()

                # extract FEN → chosen move
                for move in game.mainline_moves():
                    positions.append((board.fen(), move.uci()))
                    board.push(move)

                game_count += 1
                if limit_games and game_count >= limit_games:
                    return positions

    return positions
