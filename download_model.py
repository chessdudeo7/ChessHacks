import modal
import os
from datetime import datetime

stub = modal.App("chess-training-job")
vol = modal.Volume.from_name("chess-model-vol")

@stub.local_entrypoint()
def main():
    dest_folder = "./downloaded_model"
    os.makedirs(dest_folder, exist_ok=True)

    files = ["/model_data/chess_model.pt", "/model_data/move_vocab.pt"]

    for f in files:
        local_path = os.path.join(dest_folder, os.path.basename(f))
        vol.download_file(f, local_path)
        print(f"Downloaded {f} -> {local_path}")

        # Optionally print last modified info (Modal v1.2.2 only)
        try:
            info = vol.get_file_info(f)
            if info:
                print(f"  Last modified: {datetime.fromtimestamp(info.modified)}")
        except Exception:
            pass  # Some versions may not support get_file_info locally

    print(f"All files saved to {dest_folder}")