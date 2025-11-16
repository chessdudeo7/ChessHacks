import os
import torch
from torch.utils.data import DataLoader, random_split
from data_loader import load_positions_from_pgn_folder
from move_vocab import MoveVocab
# --- CHANGE ---
# We now import the new ChessDataset class
from chess_dataset import ChessDataset 
from model import ChessNet

# -----------------------------
# Config
# -----------------------------
PGN_FOLDER = os.environ.get("PGN_FOLDER", "pgn")
BATCH_SIZE = 128
EPOCHS = 5
VALID_SPLIT = 0.1
LR = 1e-4
# --- NEW ---
# Define paths for loading/saving
MODEL_PATH = "chess_model.pt"
VOCAB_PATH = "move_vocab.pt"

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Load positions
# -----------------------------
print("Loading PGN position data...")
positions = load_positions_from_pgn_folder(PGN_FOLDER, limit_games=None)
print(f"Loaded {len(positions)} positions.")

# -----------------------------
# Build vocab and model
# -----------------------------

# --- NEW: Check for existing model and vocab ---
if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
    print("Found existing model and vocab. Loading them...")
    
    # Load the vocab
    vocab = torch.load(VOCAB_PATH, map_location=device, weights_only=False)
    output_size = len(vocab.idx_to_move)
    print(f"Vocab loaded with {output_size} moves.")

    # Initialize model and load its weights
    model = ChessNet(output_size=output_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model weights loaded successfully.")

else:
    print("No existing model/vocab found. Building from scratch...")
    
    # Build vocab from PGNs
    vocab = MoveVocab()
    vocab.build_from_positions(positions)
    output_size = len(vocab.idx_to_move)
    print(f"New vocab built with {output_size} moves.")

    # Initialize a new model
    model = ChessNet(output_size=output_size).to(device)
    print("Initialized new model.")

# --- END NEW ---

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

# -----------------------------
# Prepare dataset
# -----------------------------
dataset = ChessDataset(positions, vocab) # This now uses the new ChessDataset
val_size = int(len(dataset) * VALID_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train positions: {len(train_dataset)}, Validation positions: {len(val_dataset)}")

# -----------------------------
# Training loop
# -----------------------------
print("Starting training...")
for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    train_loss = 0.0
    for X_spatial, X_scalar, y in train_loader:
        # Move all tensors to the device
        X_spatial, X_scalar, y = X_spatial.to(device), X_scalar.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X_spatial, X_scalar) 
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_spatial.size(0)

    avg_train_loss = train_loss / len(train_dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X_spatial, X_scalar, y in val_loader:
            X_spatial, X_scalar, y = X_spatial.to(device), X_scalar.to(device), y.to(device)
            
            preds = model(X_spatial, X_scalar)
            loss = loss_fn(preds, y)
            val_loss += loss.item() * X_spatial.size(0)
            predicted = preds.argmax(dim=1)
            correct += (predicted == y).sum().item()

    avg_val_loss = val_loss / len(val_dataset)
    val_accuracy = correct / len(val_dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Accuracy: {val_accuracy:.4f}")

# --- UPDATE THIS FUNCTION (if you use it) ---
# (Note: The duplicated `train_model` function below also needs these same changes
# to use the new (X_spatial, X_scalar) input format)
def train_model(positions, save_path, device):
    print("WARNING: The 'train_model' function is deprecated or needs to be updated.")
    
    # --- NEW: Logic applied to this function as well ---
    vocab_path = save_path.replace(".pt", "_vocab.pt")

    if os.path.exists(save_path) and os.path.exists(vocab_path):
        print(f"Loading existing model from {save_path}")
        vocab = torch.load(vocab_path, map_location=device, weights_only=False)
        output_size = len(vocab.idx_to_move)
        model = ChessNet(output_size=output_size).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("Building new model and vocab...")
        # Build vocab
        vocab = MoveVocab()
        vocab.build_from_positions(positions)
        # Build model
        output_size = len(vocab.idx_to_move)
        model = ChessNet(output_size=output_size).to(device)
    # --- END NEW ---

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Dataset
    dataset = ChessDataset(positions, vocab) # <-- Updated
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Training loop
    for epoch in range(5):
        model.train()
        for X_spatial, X_scalar, y in train_loader: # <-- Updated
            X_spatial, X_scalar, y = X_spatial.to(device), X_scalar.to(device), y.to(device) # <-- Updated
            optimizer.zero_grad()
            loss = loss_fn(model(X_spatial, X_scalar), y) # <-- Updated
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for X_spatial, X_scalar, y in val_loader: # <-- Updated
                X_spatial, X_scalar, y = X_spatial.to(device), X_scalar.to(device), y.to(device) # <-- Updated
                preds = model(X_spatial, X_scalar) # <-- Updated
                correct += (preds.argmax(dim=1) == y).sum().item()
        print(f"Epoch {epoch+1} | Val Accuracy: {correct / len(val_dataset):.4f}")

    # Save final model and vocab
    torch.save(model.state_dict(), save_path)
    torch.save(vocab, vocab_path) # <-- Updated to use specific path
    print(f"Model saved to {save_path}")

# -----------------------------
# Save final model
# -----------------------------
torch.save(model.state_dict(), MODEL_PATH) # Use defined path
torch.save(vocab, VOCAB_PATH) # Use defined path
print("Training complete! Model and vocab saved.")