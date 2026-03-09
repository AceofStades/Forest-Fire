import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import src.dataset


def main():
    train_loader, val_loader, in_channels = src.dataset.load_split_data(batch_size=8)

    total_fire_pixels = 0
    total_pixels = 0
    for inputs, targets in val_loader:
        total_fire_pixels += targets.sum().item()
        total_pixels += targets.numel()

    print(f"Total fire pixels in val: {total_fire_pixels}")
    print(f"Total pixels in val: {total_pixels}")


if __name__ == "__main__":
    main()
