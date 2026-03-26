"""
NDWS Training Script

Trains a UNet on the Next Day Wildfire Spread dataset.
Uses Combined Focal+Dice loss for class imbalance handling.
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parse_ndwd import get_ndws_dataloaders, INPUT_FEATURES
from src.models import UNet


# === Loss Functions ===

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.95, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Binary cross entropy per pixel
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        
        # Focal weight: down-weight easy examples
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting for positive class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for structural similarity."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss."""
    
    def __init__(self, focal_weight: float = 0.5, alpha: float = 0.95, gamma: float = 2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + (1 - self.focal_weight) * dice_loss


# === Metrics ===

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Compute classification metrics for fire prediction."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Flatten
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        # True positives, false positives, false negatives
        tp = ((preds_flat == 1) & (targets_flat == 1)).sum().float()
        fp = ((preds_flat == 1) & (targets_flat == 0)).sum().float()
        fn = ((preds_flat == 0) & (targets_flat == 1)).sum().float()
        tn = ((preds_flat == 0) & (targets_flat == 0)).sum().float()
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "tp": tp.item(),
            "fp": fp.item(),
            "fn": fn.item(),
        }


def find_best_threshold(logits: torch.Tensor, targets: torch.Tensor):
    """Find optimal threshold by sweeping [0.1, 0.9]."""
    best_f1 = 0.0
    best_thresh = 0.5
    
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        metrics = compute_metrics(logits, targets, threshold=thresh)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = thresh
    
    return best_thresh, best_f1


# === Training Loop ===

def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for i, (X, Y) in enumerate(pbar):
        X = X.to(device)
        Y = Y.to(device)
        
        logits = model(X)
        loss = criterion(logits, Y) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Store for metrics (subsample to save memory)
        if i % 10 == 0:
            all_logits.append(logits.detach().cpu())
            all_targets.append(Y.detach().cpu())
        
        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
    
    # Final optimizer step if needed
    if len(loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Compute epoch metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / len(loader)
    
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    
    for X, Y in tqdm(loader, desc="Validating", leave=False):
        X = X.to(device)
        Y = Y.to(device)
        
        logits = model(X)
        loss = criterion(logits, Y)
        
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_targets.append(Y.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Find best threshold
    best_thresh, best_f1 = find_best_threshold(all_logits, all_targets)
    
    # Compute metrics at best threshold
    metrics = compute_metrics(all_logits, all_targets, threshold=best_thresh)
    metrics["loss"] = total_loss / len(loader)
    metrics["best_threshold"] = best_thresh
    
    return metrics


# === Main ===

def main():
    parser = argparse.ArgumentParser(description="Train UNet on NDWS dataset")
    parser.add_argument("--data-dir", type=str, default="Model/dataset/ndws",
                        help="Path to NDWS TFRecord directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--save-dir", type=str, default="Model/weights",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"NDWS Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data
    print(f"\nLoading NDWS dataset from {args.data_dir}...")
    train_loader, val_loader, test_loader, n_channels = get_ndws_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    print(f"Input channels: {n_channels}")
    print(f"Batch size: {args.batch_size}")
    
    # Model
    model = UNet(n_channels=n_channels, n_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(focal_weight=0.5, alpha=0.95, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )
    
    # Resume if specified
    start_epoch = 0
    best_f1 = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_f1 = checkpoint.get("best_f1", 0.0)
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    patience_counter = 0
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            accumulation_steps=args.accumulation_steps
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics["f1"])
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Logging
        print(f"  Train Loss: {train_metrics['loss']:.4f} | F1: {train_metrics['f1']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | F1: {val_metrics['f1']:.4f} "
              f"(thresh={val_metrics['best_threshold']:.1f})")
        print(f"  Val   P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            
            checkpoint_path = os.path.join(args.save_dir, "ndws_unet_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "best_threshold": val_metrics["best_threshold"],
                "val_metrics": val_metrics,
            }, checkpoint_path)
            print(f"  ✓ New best model saved! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
        
        print()
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    
    # Load best model
    best_checkpoint = torch.load(
        os.path.join(args.save_dir, "ndws_unet_best.pth"),
        map_location=device
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Threshold: {test_metrics['best_threshold']:.1f}")
    
    print(f"\n✓ Training complete! Best model saved to: {args.save_dir}/ndws_unet_best.pth")


if __name__ == "__main__":
    main()
