import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- THIS WAS MISSING
import torch.optim as optim
from data_utils import load_split_data
from tqdm import tqdm
from unet_model import UNet

# CONFIG
MODEL_SAVE_PATH = "best_fire_unet.pth"
SAMPLE_OUTPUT_PATH = "simulation_input/fire_prediction_sample.npy"
VISUALIZATION_PATH = "simulation_input/prediction_visual.png"
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3  # High LR to kickstart training

# --- MEMORY OPTIMIZATION ---
BATCH_SIZE = 8  # Fits in VRAM
ACCUMULATION_STEPS = 4  # Effective Batch Size = 32


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Activate logits to get probabilities (0 to 1)
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Calculate Dice Coefficient
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        # Return 1 - Dice (because we want to minimize loss, maximizing Dice)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # We use BCEWithLogits underneath for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # pt is the probability of being correct

        # Focal Loss Formula: -alpha * (1-pt)^gamma * log(pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_and_save():
    # Check for CUDA
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        DEVICE = torch.device("cpu")

    print(f"Training on device: {DEVICE}")
    os.makedirs("simulation_input", exist_ok=True)

    # Load Data
    train_loader, val_loader, input_channels = load_split_data(batch_size=BATCH_SIZE)

    model = UNet(in_channels=input_channels, out_channels=1).to(DEVICE)

    # Use Focal Loss to handle the class imbalance intelligently
    # criterion = FocalLoss(alpha=0.75, gamma=2).to(DEVICE)
    criterion = DiceLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print(
        f"Starting training with Batch Size {BATCH_SIZE} x {ACCUMULATION_STEPS} Steps = Effective {BATCH_SIZE * ACCUMULATION_STEPS}"
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Initialize gradients

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch")

        for i, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if torch.isnan(inputs).any():
                continue

            # Forward Pass
            outputs = model(inputs)

            # --- SANITY CHECK (First batch of first epoch only) ---
            if epoch == 0 and i == 0:
                print("\n--- SANITY CHECK ---")
                print(
                    f"Input Range:   {inputs.min().item():.3f} to {inputs.max().item():.3f}"
                )
                print(
                    f"Label Range:   {labels.min().item()} to {labels.max().item()} (Sum: {labels.sum().item()})"
                )
                probs = torch.sigmoid(outputs)
                print(
                    f"Init Predicts: {probs.min().item():.4f} to {probs.max().item():.4f} (Mean: {probs.mean().item():.4f})"
                )
                print("--------------------")
            # ------------------------------------------------------

            loss = criterion(outputs, labels)

            # Normalize loss by accumulation steps
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            # Step Optimizer only every 'ACCUMULATION_STEPS'
            if (i + 1) % ACCUMULATION_STEPS == 0:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

            # Record the "real" loss (multiply back for display)
            current_loss = loss.item() * ACCUMULATION_STEPS
            train_loss += current_loss
            pbar.set_postfix(loss=f"{current_loss:.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")

        if np.isnan(avg_val_loss):
            print("Error: Validation loss became NaN. Stopping.")
            break

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(">>> Model Saved!")

    # --- VISUALIZATION ---
    print("\nGenerating sample prediction...")
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    except:
        print("Using current weights.")

    model.eval()
    sample_inputs, sample_labels = next(iter(val_loader))
    sample_inputs = sample_inputs.to(DEVICE)

    with torch.no_grad():
        logits = model(sample_inputs)
        probs = torch.sigmoid(logits)

    prediction_np = probs[0, 0].float().cpu().numpy()
    ground_truth_np = sample_labels[0, 0].float().cpu().numpy()

    np.save(SAMPLE_OUTPUT_PATH, prediction_np)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(ground_truth_np, cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(prediction_np, cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.savefig(VISUALIZATION_PATH)
    print("Done.")


if __name__ == "__main__":
    train_and_save()
