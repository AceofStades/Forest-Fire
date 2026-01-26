import numpy as np
import torch
import torch.nn as nn
from data_utils_hybrid import load_hybrid_data
from hybrid_model import HybridConvLSTMUNet
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision
from tqdm import tqdm

# --- Configuration ---
LEAD_TIME = 8
BATCH_SIZE = 32  # Increased from 4 -> 32 (Since we used AMP)
EPOCHS = 50
LR = 1e-4
POS_WEIGHT = 20.0


def dice_loss(pred, target):
    smooth = 1e-6
    p = torch.sigmoid(pred)
    intersection = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2.0 * intersection + smooth) / (union + smooth)).mean()


def compute_metrics(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float()

    # Create mask for frames that actually HAVE fire
    has_fire = target.sum(dim=(1, 2, 3)) > 0

    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))

    iou_per_image = tp / (tp + fp + fn + 1e-6)

    # Separate metrics
    mean_iou = iou_per_image.mean().item()

    # Only calculate Recall for images that actually had fire (avoid /0 or misleading 1.0s)
    if has_fire.sum() > 0:
        rec_per_image = tp[has_fire] / (tp[has_fire] + fn[has_fire] + 1e-6)
        mean_rec = rec_per_image.mean().item()
    else:
        mean_rec = 0.0

    return mean_iou, mean_rec


# --- Setup ---
train_loader, val_loader, in_dims = load_hybrid_data(
    BATCH_SIZE, seq_len=3, lead_time=LEAD_TIME
)

model = HybridConvLSTMUNet(in_dims).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# POS_WEIGHT needs to be consistent with AMP (handled automatically usually, but good to check)
bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).cuda())
scaler = GradScaler()  # Initialize Mixed Precision Scaler

print(f"\nStarting Fast Hybrid Training | Batch: {BATCH_SIZE} | AMP: Enabled")
print("=" * 70)

best_iou = 0.0

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    train_loss = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

    for x, y in train_pbar:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        optimizer.zero_grad()

        # Mixed Precision Context
        with autocast():
            out = model(x)
            loss = bce_criterion(out, y) + dice_loss(out, y)

        # Scale Loss & Step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # --- Validation Phase ---
    model.eval()
    val_loss, val_iou, val_rec = 0, 0, 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")

    with torch.no_grad():
        for x, y in val_pbar:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            with autocast():
                out = model(x)
                loss = bce_criterion(out, y) + dice_loss(out, y)

            iou, rec = compute_metrics(out, y)

            val_loss += loss.item()
            val_iou += iou
            val_rec += rec

            val_pbar.set_postfix(
                {
                    "IoU": f"{iou:.4f}",
                    "Rec (Active)": f"{rec:.4f}",
                }
            )

    # --- Epoch Summary ---
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    avg_val_rec = val_rec / len(val_loader)

    print(
        f"\nSummary - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    )
    print(
        f"Metrics - Val IoU: {avg_val_iou:.4f} | Val Recall (Active Fires): {avg_val_rec:.4f}\n"
    )
    print("-" * 70)

    if avg_val_iou > best_iou:
        best_iou = avg_val_iou
        torch.save(model.state_dict(), "best_hybrid_model.pth")
