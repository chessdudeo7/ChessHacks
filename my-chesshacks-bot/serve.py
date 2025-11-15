from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess
import os

from src.utils import chess_manager
from src import main

app = FastAPI()


@app.post("/")
async def root():
    return JSONResponse(content={"running": True})


@app.post("/move")
async def get_move(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    if "pgn" not in data or "timeleft" not in data:
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    pgn = data["pgn"]
    timeleft = data["timeleft"]  # in milliseconds

    chess_manager.set_context(pgn, timeleft)
    print("pgn", pgn)

    try:
        start_time = time.perf_counter()
        move, move_probs, logs = chess_manager.get_model_move()
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Bot raised an exception",
                "logs": None,
                "exception": str(e),
            },
            status_code=500,
        )

    # Ensure move is a chess.Move object
    if isinstance(move, tuple):
        move_obj = move[0]  # first element is the actual move
    else:
        move_obj = move

    if not isinstance(move_obj, chess.Move):
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Invalid move returned by bot",
                "logs": logs,
            },
            status_code=500,
        )

    # Confirm move_probs is a dict
    if not isinstance(move_probs, dict):
        move_probs_dict = None
    else:
        # Translate move_probs to Dict[str, float] safely
        move_probs_dict = {}
        for m, prob in move_probs.items():
            if isinstance(m, chess.Move) and isinstance(prob, float):
                move_probs_dict[m.uci()] = prob

    return JSONResponse(
        content={
            "move": move_obj.uci(),
            "error": None,
            "time_taken": time_taken,
            "move_probs": move_probs_dict,
            "logs": logs,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)