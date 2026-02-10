import os

import numpy as np
import torch
import torch.nn as nn

# Import from our clean package
from src.dataset import get_dataloaders
from src.models import UNet
from src.utils import compute_metrics, dice_loss
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# --- CONFIG ---
SEQ_LEN = 3
LEAD_TIME = 8  # 24 hours
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4

# HIGH WEIGHT to find sparse fires
POS_WEIGHT = 50.0

SAVE_DIR = "weights"
SAVE_PATH = os.path.join(SAVE_DIR, "best_unet.pth")


def main():
    print(f"\n--- Initializing Data ---")
    train_loader, val_loader, in_channels = get_dataloaders(
        BATCH_SIZE, SEQ_LEN, LEAD_TIME
    )
    print(f"Model Input Channels: {in_channels}")

    model = UNet(n_channels=in_channels, n_classes=1).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Scheduler to reduce LR when learning stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).cuda())
    scaler = GradScaler("cuda")

    best_iou = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\nStarting Training...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        batches_processed = 0

        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{EPOCHS} [Train]")

        for x, y in loop:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()

            with autocast("cuda"):
                pred = model(x)
                # Combined Loss: BCE (High Weight) + Dice
                loss = criterion_bce(pred, y) + dice_loss(pred, y)

            # --- STABILITY CHECKS ---
            if torch.isnan(loss) or torch.isinf(loss):
                loop.set_postfix(status="NaN Loss - Skipping")
                continue

            # Scaled Backward Pass
            scaler.scale(loss).backward()

            # --- CRITICAL FIX: GRADIENT CLIPPING ---
            # Prevents the scaler from crashing on massive gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Metrics
            with torch.no_grad():
                _, _, acc = compute_metrics(pred, y)
                train_acc += acc

            train_loss += loss.item()
            batches_processed += 1
            loop.set_postfix(loss=loss.item(), acc=acc)

        # --- VAL ---
        model.eval()
        val_iou = 0
        val_rec = 0
        val_acc = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with autocast("cuda"):
                    pred = model(x)

                iou, rec, acc = compute_metrics(pred, y)
                val_iou += iou
                val_rec += rec
                val_acc += acc

        if batches_processed > 0:
            avg_train_loss = train_loss / batches_processed
            avg_train_acc = train_acc / batches_processed
        else:
            avg_train_loss = 0
            avg_train_acc = 0

        avg_val_iou = val_iou / len(val_loader)
        avg_val_rec = val_rec / len(val_loader)

        scheduler.step(avg_val_iou)

        print(
            f"Summary: Loss {avg_train_loss:.4f} | Train Acc {avg_train_acc:.4f} || Val IoU {avg_val_iou:.4f} | Val Rec {avg_val_rec:.4f}"
        )

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"*** NEW RECORD! Model Saved ({best_iou:.4f}) ***")

        print("-" * 60)


if __name__ == "__main__":
    main()
