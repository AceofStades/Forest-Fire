import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()


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


def compute_metrics(pred_logits, target, threshold=0.3):
    """
    Using threshold 0.3 to catch weaker signals.
    Returns: IoU, Recall, Accuracy, Max_Confidence
    """
    probs = torch.sigmoid(pred_logits)
    pred_bin = (probs > threshold).float()

    # Max Confidence (Debug metric: Is the model even trying?)
    max_conf = probs.max().item()

    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))

    # IoU
    iou = (tp / (tp + fp + fn + 1e-6)).mean().item()

    # Active Recall
    has_fire = target.sum(dim=(1, 2, 3)) > 0
    if has_fire.sum() > 0:
        rec = (tp[has_fire] / (tp[has_fire] + fn[has_fire] + 1e-6)).mean().item()
    else:
        rec = 0.0

    return iou, rec, max_conf


def calculate_accuracy(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == targets).float().sum()
    total = torch.numel(targets)
    return (correct / total).item()
