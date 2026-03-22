import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
#  Loss functions
# ============================================================================


def _safe_div(numerator, denominator, zero_division=0.0):
    if denominator > 0:
        return numerator / denominator
    return float(zero_division)


def compute_binary_metrics_from_counts(tp, fp, fn, tn, zero_division=0.0):
    """Compute binary segmentation metrics from confusion-matrix counts."""
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    tn = float(tn)

    precision = _safe_div(tp, tp + fp, zero_division=zero_division)
    recall = _safe_div(tp, tp + fn, zero_division=zero_division)
    f1 = _safe_div(
        2.0 * precision * recall, precision + recall, zero_division=zero_division
    )
    iou = _safe_div(tp, tp + fp + fn, zero_division=zero_division)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn, zero_division=zero_division)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.99, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)

        # We must penalize missing the rare fire class MORE than false positives.
        # Targets=1 gets alpha. Targets=0 gets (1-alpha).
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class CombinedLoss(nn.Module):
    """alpha * FocalLoss + (1 - alpha) * DiceLoss"""

    def __init__(self, alpha=0.2, focal_alpha=0.99, focal_gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        # Fire is incredibly rare. Dice loss provides a much better structural gradient
        # so we weight it heavily (80% Dice, 20% Focal).
        return self.alpha * self.focal(logits, targets) + (1 - self.alpha) * self.dice(
            logits, targets
        )


class BCEDiceLoss(nn.Module):
    """pos-weighted BCE + DiceLoss — good for heavily imbalanced fire data."""

    def __init__(self, pos_weight=20.0, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        # Move pos_weight to correct device lazily
        if self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        return self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class TverskyLoss(nn.Module):
    """Tversky index loss — generalisation of Dice with separate FP/FN weights.

    alpha < beta penalises false-negatives more, boosting recall for the
    rare fire class.  Default alpha=0.3, beta=0.7 is a common starting point
    for highly imbalanced segmentation.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        tgt = targets.view(-1)
        tp = (probs * tgt).sum()
        fp = (probs * (1 - tgt)).sum()
        fn = ((1 - probs) * tgt).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1 - tversky


class BCETverskyLoss(nn.Module):
    """pos-weighted BCE + Tversky — built for extreme class imbalance.

    Combines BCE with large pos_weight (loud per-pixel fire gradient) and
    Tversky with beta > alpha (penalise missed fire regions more than false
    alarms).
    """

    def __init__(
        self, pos_weight=500.0, tversky_weight=0.5, tversky_alpha=0.3, tversky_beta=0.7
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.tw = tversky_weight

    def forward(self, logits, targets):
        if self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        return self.bce(logits, targets) + self.tw * self.tversky(logits, targets)


# ============================================================================
#  Metric helpers (kept for backward compatibility with evaluate.py)
# ============================================================================


def compute_metrics(pred_logits, target, threshold=0.3):
    """
    Using threshold 0.3 to catch weaker signals.
    Returns: IoU, Recall, Max_Confidence
    """
    probs = torch.sigmoid(pred_logits)
    pred_bin = (probs > threshold).float()

    max_conf = probs.max().item()

    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))

    iou = (tp / (tp + fp + fn + 1e-6)).mean().item()

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


def compute_all_metrics(logits, targets, threshold=0.5):
    """Full metric suite for the sweep. Returns dict with all scores."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pred_bin = (probs > threshold).float()

        tp = (pred_bin * targets).sum().item()
        fp = (pred_bin * (1 - targets)).sum().item()
        fn = ((1 - pred_bin) * targets).sum().item()
        tn = ((1 - pred_bin) * (1 - targets)).sum().item()

        metrics = compute_binary_metrics_from_counts(
            tp, fp, fn, tn, zero_division=0.0
        )
        max_prob = probs.max().item()

    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "iou": metrics["iou"],
        "max_prob": max_prob,
        "threshold": threshold,
    }


def compute_best_threshold_metrics(logits, targets, thresholds=None):
    """Try several thresholds; return metrics at the one with highest F1.

    For datasets with extreme class imbalance (fire = 0.002%),
    threshold=0.5 often gives F1=0. This finds the best operating point.
    """
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    best_f1 = -1.0
    best = {}
    for t in thresholds:
        m = compute_all_metrics(logits, targets, threshold=t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best = m
    return best if best else compute_all_metrics(logits, targets, threshold=0.5)
