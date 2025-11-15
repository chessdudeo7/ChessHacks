import os
import torch
from torch.utils.data import DataLoader, random_split
from data_loader import load_positions_from_pgn_folder
from move_vocab import MoveVocab
from chess_dataset import ChessDataset
from model import ChessNet

# -----------------------------
# Config
# -----------------------------
PGN_FOLDER = os.environ.get("PGN_FOLDER", "pgn")
BATCH_SIZE = 128
EPOCHS = 5
VALID_SPLIT = 0.1
LR = 1e-3

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
vocab = MoveVocab()
vocab.build_from_positions(positions)
input_size = 64 * 12 + 5
output_size = len(vocab.idx_to_move)

model = ChessNet(input_size=input_size, output_size=output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

# -----------------------------
# Prepare dataset
# -----------------------------
dataset = ChessDataset(positions, vocab)
val_size = int(len(dataset) * VALID_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train positions: {len(train_dataset)}, Validation positions: {len(val_dataset)}")

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)

    avg_train_loss = train_loss / len(train_dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            val_loss += loss.item() * X.size(0)
            predicted = preds.argmax(dim=1)
            correct += (predicted == y).sum().item()

    avg_val_loss = val_loss / len(val_dataset)
    val_accuracy = correct / len(val_dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Accuracy: {val_accuracy:.4f}")

def train_model(positions, save_path, device):
    # Build vocab
    vocab = MoveVocab()
    vocab.build_from_positions(positions)

    # Build model
    input_size = 64 * 12 + 5
    output_size = len(vocab.idx_to_move)
    model = ChessNet(input_size=input_size, output_size=output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Dataset
    dataset = ChessDataset(positions, vocab)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Training loop
    for epoch in range(5):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                correct += (preds.argmax(dim=1) == y).sum().item()
        print(f"Epoch {epoch+1} | Val Accuracy: {correct / len(val_dataset):.4f}")

    # Save final model and vocab
    torch.save(model.state_dict(), save_path)
    torch.save(vocab, save_path.replace(".pt", "_vocab.pt"))
    print(f"Model saved to {save_path}")

# -----------------------------
# Save final model
# -----------------------------
torch.save(model.state_dict(), "chess_model.pt")
torch.save(vocab, "move_vocab.pt")
print("Training complete!")