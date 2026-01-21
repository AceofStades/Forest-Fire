import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import load_split_data
from tqdm import tqdm
from unet_model import UNet

# CONFIG
MODEL_SAVE_PATH = "best_fire_unet.pth"
SAMPLE_OUTPUT_PATH = "simulation_input/fire_prediction_sample.npy"
VISUALIZATION_PATH = "simulation_input/prediction_visual.png"
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
BATCH_SIZE = 32  # Increased from 4 to 32 for Speed


def train_and_save():
    # Check for CUDA
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Optimized for fixed input size
    else:
        DEVICE = torch.device("cpu")

    print(f"Training on device: {DEVICE}")
    os.makedirs("simulation_input", exist_ok=True)

    # Load Data (Now loads into RAM)
    train_loader, val_loader, input_channels = load_split_data(batch_size=BATCH_SIZE)

    model = UNet(in_channels=input_channels, out_channels=1).to(DEVICE)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch")

        for inputs, labels in pbar:
            # Non_blocking allows asynchronous transfer
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

            # Mixed Precision Forward Pass
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Mixed Precision Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(">>> Model Saved!")

    # --- VISUALIZATION (Sample Prediction) ---
    print("\nGenerating sample prediction...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    sample_inputs, sample_labels = next(iter(val_loader))
    sample_inputs = sample_inputs.to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            sample_logits = model(sample_inputs)
            sample_probs = torch.sigmoid(sample_logits)

    # Take first item in batch
    prediction_np = sample_probs[0, 0].float().cpu().numpy()
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
