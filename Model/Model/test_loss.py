import sys; sys.path.append('.')
import torch
from src.utils import CombinedLoss
loss = CombinedLoss(alpha=0.2, focal_alpha=0.99, focal_gamma=2.0)
logits = torch.ones(1, 1, 10, 10) * -5.0 # Predict zero
targets = torch.zeros(1, 1, 10, 10) # True zero
print('Loss when predicting zeros perfectly:', loss(logits, targets).item())

targets[0, 0, 0, 0] = 1.0 # 1 positive
print('Loss when predicting zeros but 1 positive exists (Miss):', loss(logits, targets).item())

logits[0, 0, 0, 0] = 5.0 # Predict 1 positive perfectly
print('Loss when predicting 1 positive perfectly (Hit):', loss(logits, targets).item())
