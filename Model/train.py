import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import from our updated package
from src.dataset import load_split_data
from src.models import UNet
from src.utils import DiceLoss, calculate_accuracy
from tqdm import tqdm

# --- CONFIG ---
MODEL_SAVE_PATH = "best_fire_unet.pth"
SAMPLE_OUTPUT_PATH = "simulation_input/fire_prediction_sample.npy"
VISUALIZATION_PATH = "simulation_input/prediction_visual.png"

# Hyperparameters
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4  # Effective Batch Size = 32


def train_and_save():
    # Check for CUDA
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    print(f"Training on device: {DEVICE}")
    os.makedirs("simulation_input", exist_ok=True)

    # Load Data
    try:
        train_loader, val_loader, input_channels = load_split_data(
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    model = UNet(n_channels=input_channels, n_classes=1).to(DEVICE)

    # Loss and Optimizer
    criterion = DiceLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print(
        f"Starting training with Batch Size {BATCH_SIZE} x {ACCUMULATION_STEPS} Steps = Effective {BATCH_SIZE * ACCUMULATION_STEPS}"
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc_accum = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch")

        batch_count = 0
        for i, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if torch.isnan(inputs).any():
                continue

            # Forward Pass
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Normalize loss by accumulation steps
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            # Calculate Batch Accuracy
            acc = calculate_accuracy(outputs, labels)
            train_acc_accum += acc
            batch_count += 1

            # Step Optimizer only every 'ACCUMULATION_STEPS'
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Record the "real" loss (multiply back for display)
            current_loss = loss.item() * ACCUMULATION_STEPS
            train_loss += current_loss

            pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{acc:.4f}")

        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        avg_train_acc = train_acc_accum / batch_count if batch_count > 0 else 0

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_acc_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                acc = calculate_accuracy(outputs, labels)
                val_acc_accum += acc
                val_batches += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc_accum / val_batches if val_batches > 0 else 0

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc:.4f}")

        if np.isnan(avg_val_loss):
            print("Error: Validation loss became NaN. Stopping.")
            break

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(">>> Model Saved (New Best Validation Loss)!")

    # --- VISUALIZATION ---
    print("\nGenerating sample prediction...")
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded weights from {MODEL_SAVE_PATH}")
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
    print(f"Visualization saved to {VISUALIZATION_PATH}")


if __name__ == "__main__":
    train_and_save()
