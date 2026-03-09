import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import src.dataset


def main():
    train_loader, val_loader, in_channels = src.dataset.load_split_data(batch_size=8)

    total_fire_pixels_train = 0
    total_pixels_train = 0
    for inputs, targets in train_loader:
        total_fire_pixels_train += targets.sum().item()
        total_pixels_train += targets.numel()

    print(f"Total fire pixels in train: {total_fire_pixels_train}")
    print(f"Total pixels in train: {total_pixels_train}")


if __name__ == "__main__":
    main()
