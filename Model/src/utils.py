import torch


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


def compute_metrics(pred, target, threshold=0.5):
    """
    Returns: IoU, Recall, Accuracy
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()

    # --- Intersection over Union (IoU) ---
    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))

    iou = (tp / (tp + fp + fn + 1e-6)).mean().item()

    # --- Recall (Sensitivity) ---
    # Only calculate for images that actually contain fire to avoid /0
    has_fire = target.sum(dim=(1, 2, 3)) > 0
    if has_fire.sum() > 0:
        rec = (tp[has_fire] / (tp[has_fire] + fn[has_fire] + 1e-6)).mean().item()
    else:
        rec = 0.0

    # --- Pixel Accuracy ---
    # (TP + TN) / Total
    correct_pixels = (pred_bin == target).float().sum(dim=(1, 2, 3))
    total_pixels = target.size(2) * target.size(3)  # H * W
    acc = (correct_pixels / total_pixels).mean().item()

    return iou, rec, acc
