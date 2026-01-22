import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from convlstm_model import ConvLSTM
from data_utils_seq import load_seq_data
from tqdm import tqdm

# --- CONFIG ---
MODEL_SAVE_PATH = "best_convlstm.pth"
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 4  # Lower batch size because ConvLSTM is memory hungry!
SEQUENCE_LENGTH = 3  # 3 Time steps


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


def train():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        DEVICE = torch.device("cpu")

    print(f"Training ConvLSTM on: {DEVICE}")

    # Load Sequence Data
    train_loader, val_loader, input_channels = load_seq_data(
        BATCH_SIZE, SEQUENCE_LENGTH
    )

    # Initialize ConvLSTM (3 layers of 32 hidden channels)
    model = ConvLSTM(
        in_channels=input_channels, out_channels=1, hidden_dims=[32, 32, 32]
    ).to(DEVICE)

    criterion = DiceLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, (inputs, labels) in enumerate(pbar):
            # inputs shape: (Batch, Time, Channels, Height, Width)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if torch.isnan(inputs).any():
                continue

            optimizer.zero_grad()
            outputs = model(inputs)  # Returns (Batch, 1, H, W)

            loss = criterion(outputs, labels)
            loss.backward()

            # Clip Gradients (Essential for LSTM stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        avg_val = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(">>> Model Saved!")


if __name__ == "__main__":
    train()
