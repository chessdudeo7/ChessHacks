import modal
from pathlib import Path
import torch
import shutil
import os

# ----------------------
# Paths
# ----------------------
CLOUD_PGN_DIR = "/data/pgn"
CLOUD_MODEL_DIR = "/data/model"
CLOUD_MODEL_PATH = f"{CLOUD_MODEL_DIR}/trained_model.pt"

# ----------------------
# Modal App & Image
# ----------------------
app = modal.App("chess-trainer")

# Create an image
image = (
    modal.Image.debian_slim()
    .pip_install("python-chess", "torch", "transformers")
    .add_local_dir("training", remote_path="/app/training")  # mount your training folder
)

# ----------------------
# Volumes
# ----------------------
pgn_volume = modal.Volume.from_name("chess-pgn", create_if_missing=True)
model_volume = modal.Volume.from_name("chess-model", create_if_missing=True)

# ----------------------
# Training Function
# ----------------------
@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={
        CLOUD_PGN_DIR: pgn_volume,
        CLOUD_MODEL_DIR: model_volume,
    },
    env={"PGN_FOLDER": CLOUD_PGN_DIR},  # tells train.py where the PGNs are
)
def cloud_train():
    import sys
    sys.path.insert(0, "/app/training")  # make training folder importable

    import train
    import data_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading PGN files...")
    positions = data_loader.load_positions_from_pgn_folder(os.environ["PGN_FOLDER"], limit_games=None)
    print(f"Loaded {len(positions)} positions.")

    print("Training model...")
    # train_model no longer uses checkpointing
    train.train_model(
        positions,
        save_path=CLOUD_MODEL_PATH,
        device=device,
    )
    print(f"Model saved at {CLOUD_MODEL_PATH}")

# ----------------------
# Pull Model Back
# ----------------------
def pull_model_back():
    LOCAL_MODEL_FOLDER = Path("trained_model")
    LOCAL_MODEL_FOLDER.mkdir(exist_ok=True)
    with model_volume.mount() as mount_path:
        cloud_model = Path(mount_path) / "trained_model.pt"
        local_path = LOCAL_MODEL_FOLDER / "trained_model.pt"

        if cloud_model.exists():
            print("Copying trained model locally:", local_path)
            shutil.copy(cloud_model, local_path)
        else:
            print("❌ No trained model found in the cloud volume.")

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    print("Submitting cloud training job to Modal...")
    handle = cloud_train.remote()
    handle.get()

    print("Training job finished. Pulling model back...")
    pull_model_back()
    print("Done!")