import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import load_seq_data, load_split_data
from src.models import ConvLSTMFireNet, UNet
from src.utils import CombinedLoss, compute_best_threshold_metrics
from tqdm import tqdm

# --- CONFIG ---
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
ACCUMULATION_STEPS = 8


def train(model_type="unet"):
    global ACCUMULATION_STEPS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Checkouts dir
    os.makedirs("checkouts", exist_ok=True)
    model_save_path = f"checkouts/best_{model_type}.pth"

    # Load Data
    print(f"Loading data for {model_type}...")
    if model_type == "unet":
        train_loader, val_loader, in_channels = load_split_data(
            batch_size=BATCH_SIZE,
            weighted_sampling=True,
            fire_oversample_ratio=5.0,
            include_fire_input=True,  # CRITICAL: UNet needs to see current fire!
        )
        model = UNet(n_channels=in_channels, n_classes=1)
        model = model.to(device)
    elif model_type == "convlstm":
        # ConvLSTM requires extremely low batch size to prevent OOM
        conv_batch_size = 2
        conv_accum_steps = BATCH_SIZE * ACCUMULATION_STEPS // conv_batch_size

        train_loader, val_loader, in_channels = load_seq_data(
            batch_size=conv_batch_size,
            seq_len=4,
            weighted_sampling=True,
            fire_oversample_ratio=50.0,
            include_fire_input=True,
        )
        model = ConvLSTMFireNet(
            in_channels=in_channels, n_classes=1, hidden_dims=[64, 64]
        ).to(device)
        ACCUMULATION_STEPS = conv_accum_steps  # Update global for the loop
    else:
        raise ValueError("Invalid model type. Choose 'unet' or 'convlstm'.")

    print(f"Model {model_type} initialized with {in_channels} input channels.")

    # Using CombinedLoss (Focal + Dice) to handle extreme imbalance stably
    criterion = CombinedLoss(alpha=0.5, focal_alpha=0.95, focal_gamma=2.0).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    best_val_f1 = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [TRAIN]", unit="batch"
        )

        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            if torch.isnan(inputs).any():
                continue

            with torch.amp.autocast("cuda"):
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix(loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_batches = 0

        thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        tp_counts = {t: 0 for t in thresholds}
        fp_counts = {t: 0 for t in thresholds}
        fn_counts = {t: 0 for t in thresholds}
        tn_counts = {t: 0 for t in thresholds}

        print(f"Evaluating {model_type}...")
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = (
                    inputs.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )

                with torch.amp.autocast("cuda"):
                    logits = model(inputs)
                    v_loss = criterion(logits, labels)
                    val_loss += v_loss.item()

                probs = torch.sigmoid(logits).detach()

                for t in thresholds:
                    pred_bin = (probs > t).float()
                    tp_counts[t] += (pred_bin * labels).sum().item()
                    fp_counts[t] += (pred_bin * (1 - labels)).sum().item()
                    fn_counts[t] += ((1 - pred_bin) * labels).sum().item()
                    tn_counts[t] += ((1 - pred_bin) * (1 - labels)).sum().item()

                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)

        best_f1 = -1.0
        best_acc = 0.0
        for t in thresholds:
            tp = tp_counts[t]
            fp = fp_counts[t]
            fn = fn_counts[t]
            tn = tn_counts[t]

            if tp + fp + fn == 0:
                f1 = 1.0
            else:
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

            acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc

        avg_val_f1 = best_f1
        avg_val_acc = best_acc

        scheduler.step()

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss / len(train_loader):.4f}")
        print(
            f"  Val Loss:   {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.6f} | Val F1 (Best Thr): {avg_val_f1:.6f}"
        )

        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), model_save_path)
            print(
                f">>> Model Saved to {model_save_path} (New Best Validation F1: {best_val_f1:.6f})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="unet", choices=["unet", "convlstm"]
    )
    args = parser.parse_args()
    train(args.model)
