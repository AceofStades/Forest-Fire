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

# --- CONFIGURATION ---
SEQ_LEN = 3
LEAD_TIME = 8  # 24 hours
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 5.0

# Updated path
SAVE_DIR = "weights"
SAVE_PATH = os.path.join(SAVE_DIR, "best_unet.pth")


def main():
    # 1. Setup Data
    # This handles Spatial Split + Time Stacking automatically
    train_loader, val_loader, in_channels = get_dataloaders(
        BATCH_SIZE, SEQ_LEN, LEAD_TIME
    )
    print(f"Model Input Channels: {in_channels}")

    # 2. Setup Model
    model = UNet(n_channels=in_channels, n_classes=1).cuda()

    # 3. Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, verbose=True
    )
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).cuda())
    scaler = GradScaler("cuda")

    best_iou = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\nStarting Training... Saving to {SAVE_PATH}")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{EPOCHS} [Train]")

        for x, y in loop:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()

            with autocast("cuda"):
                pred = model(x)
                loss = criterion_bce(pred, y) + dice_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # --- VAL ---
        model.eval()
        val_iou = 0
        val_rec = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with autocast("cuda"):
                    pred = model(x)

                iou, rec = compute_metrics(pred, y)
                val_iou += iou
                val_rec += rec

        avg_loss = train_loss / len(train_loader)
        avg_iou = val_iou / len(val_loader)
        avg_rec = val_rec / len(val_loader)

        scheduler.step(avg_iou)

        print(
            f"Summary: Loss {avg_loss:.4f} | Val IoU {avg_iou:.4f} | Val Rec {avg_rec:.4f}"
        )

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"*** NEW RECORD! Model Saved ({best_iou:.4f}) ***")

        print("-" * 50)


if __name__ == "__main__":
    main()
