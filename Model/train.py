import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import FireDataset, load_split_data
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_model import UNet

MODEL_SAVE_PATH = "best_fire_unet.pth"
SAMPLE_OUTPUT_PATH = "simulation_input/fire_prediction_sample.npy"
VISUALIZATION_PATH = "simulation_input/prediction_visual.png"
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
BATCH_SIZE = 4

os.makedirs("simulation_input", exist_ok=True)


def train_and_save():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        print("⚠️ CUDA not found. Training on CPU will be slow.")
        DEVICE = torch.device("cpu")
    print(f"Training on device: {DEVICE}")

    print("Loading and splitting data...")
    train_loader, val_loader, input_channels = load_split_data(batch_size=BATCH_SIZE)
    print(f"Input Channels: {input_channels}")

    model = UNet(in_channels=input_channels, out_channels=1).to(DEVICE)

    pos_weight = torch.tensor([50.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [TRAIN]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> Best model saved to {MODEL_SAVE_PATH}")

    print("\nGenerating sample prediction for Cellular Automata...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    sample_inputs, sample_labels = next(iter(val_loader))
    sample_inputs = sample_inputs.to(DEVICE)

    with torch.no_grad():
        sample_logits = model(sample_inputs)
        sample_probs = torch.sigmoid(sample_logits)

    prediction_np = sample_probs[0, 0].cpu().numpy()
    ground_truth_np = sample_labels[0, 0].cpu().numpy()

    np.save(SAMPLE_OUTPUT_PATH, prediction_np)
    print(
        f"Sample prediction saved to {SAMPLE_OUTPUT_PATH} (Shape: {prediction_np.shape})"
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth (Actual Fire)")
    plt.imshow(ground_truth_np, cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("U-Net Prediction (Probability)")
    plt.imshow(prediction_np, cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.savefig(VISUALIZATION_PATH)
    print(f"Visualization saved to {VISUALIZATION_PATH}")
    print("Training and generation complete!")


if __name__ == "__main__":
    train_and_save()
