import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

# Update this path if needed
INPUT_NC_PATH = "dataset/final_feature_stack_MASTER.nc"
CACHE_PATH = "checkouts/stats_cache.pkl"


class FireDataset(Dataset):
    def __init__(self, data, targets, indices, seq_len, lead_time, augment=False):
        self.data = data
        self.targets = targets
        self.indices = indices
        self.seq_len = seq_len
        self.lead_time = lead_time
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_idx = self.indices[idx]

        # 1. Get Sequence [t - seq_len + 1 ... t]
        # Shape: (Seq, C, H, W)
        x_seq = self.data[t_idx - self.seq_len + 1 : t_idx + 1]

        # 2. Flatten Time into Channels for UNet
        # Shape: (Seq * C, H, W)
        x_flat = np.concatenate(list(x_seq), axis=0)

        # 3. Get Target
        y_data = self.targets[t_idx + self.lead_time]

        # 4. Convert to Tensor
        x_tensor = torch.from_numpy(x_flat).float()
        y_tensor = torch.from_numpy(y_data).float().unsqueeze(0)

        # 5. Augmentation
        if self.augment:
            if random.random() > 0.5:  # H-Flip
                x_tensor = torch.flip(x_tensor, [-1])
                y_tensor = torch.flip(y_tensor, [-1])
            if random.random() > 0.5:  # V-Flip
                x_tensor = torch.flip(x_tensor, [-2])
                y_tensor = torch.flip(y_tensor, [-2])
            k = random.randint(0, 3)  # Rotate
            if k > 0:
                x_tensor = torch.rot90(x_tensor, k, [-2, -1])
                y_tensor = torch.rot90(y_tensor, k, [-2, -1])

        # 6. Pad to multiple of 16
        _, h, w = x_tensor.shape
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        if ph > 0 or pw > 0:
            x_tensor = F.pad(x_tensor, (0, pw, 0, ph))
            y_tensor = F.pad(y_tensor, (0, pw, 0, ph))

        return x_tensor, y_tensor


def get_dataloaders(batch_size=32, seq_len=3, lead_time=1):
    print(f"Loading {INPUT_NC_PATH}...")
    with xr.open_dataset(INPUT_NC_PATH, engine="h5netcdf") as ds:
        # We use ALL variables (including fire history) for input
        features = list(ds.data_vars)
        print(f"Features: {features}")

        data_np = ds.to_array(dim="channel").values
        data_np = np.transpose(data_np, (1, 0, 2, 3)).astype(np.float32)
        target_np = ds["MODIS_FIRE_T1"].values.astype(np.float32)

    # Stats / Normalization
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            stats = pickle.load(f)
    else:
        stats = {
            "mean": np.nanmean(data_np, axis=(0, 2, 3), keepdims=True),
            "std": np.nanstd(data_np, axis=(0, 2, 3), keepdims=True),
        }
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(stats, f)

    data_np = (data_np - stats["mean"]) / (stats["std"] + 1e-6)

    # SPATIAL SPLIT (North Train / South Val)
    _, _, H, W = data_np.shape
    split_row = int(H * 0.80)

    print(f"Spatial Split at Row {split_row} (Top 80% Train, Bottom 20% Val)")

    train_data = data_np[:, :, :split_row, :]
    train_tgt = target_np[:, :split_row, :]

    val_data = data_np[:, :, split_row:, :]
    val_tgt = target_np[:, split_row:, :]

    # Indices
    indices = np.arange(seq_len - 1, data_np.shape[0] - lead_time)

    train_ds = FireDataset(
        train_data, train_tgt, indices, seq_len, lead_time, augment=True
    )
    val_ds = FireDataset(val_data, val_tgt, indices, seq_len, lead_time, augment=False)

    # Calc Input Channels: (Features * SeqLen)
    in_channels = len(features) * seq_len

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_dl, val_dl, in_channels
