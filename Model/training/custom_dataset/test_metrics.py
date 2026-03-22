import sys
import os
sys.path.append(os.path.abspath('Model'))
from src.utils import compute_binary_metrics_from_counts

tp = 0
fp = 0
fn = 1224
tn = 2048000

m = compute_binary_metrics_from_counts(tp, fp, fn, tn)
print(m)
