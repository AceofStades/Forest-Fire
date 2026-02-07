import torch
import torch.nn as nn


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


def compute_metrics(pred, target, threshold=0.5):
    """
    Computes IoU and Recall (Sensitivity).
    'Active Recall' ignores images with no fire.
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()

    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))

    # Intersection over Union
    iou = (tp / (tp + fp + fn + 1e-6)).mean().item()

    # Active Recall (Only for frames that actually have fire)
    has_fire = target.sum(dim=(1, 2, 3)) > 0
    if has_fire.sum() > 0:
        recall = (tp[has_fire] / (tp[has_fire] + fn[has_fire] + 1e-6)).mean().item()
    else:
        recall = 0.0

    return iou, recall
