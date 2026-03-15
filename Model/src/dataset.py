import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_NC_PATH = "dataset/final_feature_stack_DYNAMIC_interpolated.nc"
CACHE_PATH = "stats_cache.pkl"


def _resolve_path(nc_path):
    """Try to find the dataset file."""
    if os.path.exists(nc_path):
        return nc_path
    alt = os.path.join("Model", nc_path)
    if os.path.exists(alt):
        return alt
    return nc_path  # let it fail with a clear path


def _load_ds(nc_path=None, include_fire_input=False):
    """Load dataset into RAM. Returns (ds_loaded, feature_vars, total_steps).
    If include_fire_input=True, MODIS_FIRE_T1 at time T is included as a feature.
    Also dynamically calculates a Burn_Scar feature which is 1 if the pixel has ever been on fire.
    Engineers Water_Mask from LULC and calculates Slope from DEM."""
    path = _resolve_path(nc_path or INPUT_NC_PATH)
    print(f"--- Loading dataset into RAM ({path}) ---")
    with xr.open_dataset(path, engine="h5netcdf", chunks=None) as ds:
        ds_loaded = ds.load()

        # Calculate Burn_Scar
        print("--- Calculating Burn_Scar (cumulative fire history) ---")
        burn_scar = ds_loaded["MODIS_FIRE_T1"].cumsum(dim="valid_time")
        burn_scar = xr.where(burn_scar > 0, 1.0, 0.0).astype(np.float32)
        ds_loaded["Burn_Scar"] = burn_scar

        # Engineer Water Mask from LULC (Assuming class 0 or specific class is water)
        # Bhuvan LULC: typically Water bodies are a specific class.
        # For generalization, if LULC <= 0 (or specific value), it's non-burnable.
        # Let's assume class 0 is water/null.
        print("--- Engineering Water_Mask from LULC ---")
        lulc_data = ds_loaded["LULC"].values
        # 1.0 means burnable, 0.0 means water/barren
        water_mask = (lulc_data > 0).astype(np.float32)
        ds_loaded["Water_Mask"] = (("latitude", "longitude"), water_mask)

        # Engineer Urban Mask from GHS_BUILT
        print("--- Engineering Urban_Mask from GHS_BUILT ---")
        ghs_data = ds_loaded["GHS_BUILT"].values
        urban_mask = (ghs_data > 0).astype(np.float32)
        ds_loaded["Urban_Mask"] = (("latitude", "longitude"), urban_mask)

        # Calculate Slope from DEM
        print("--- Calculating Topographical Slope from DEM ---")
        dem_data = ds_loaded["DEM"].values
        # Compute spatial gradient (dy, dx)
        slope_y, slope_x = np.gradient(dem_data)
        ds_loaded["Slope_Y"] = (("latitude", "longitude"), slope_y.astype(np.float32))
        ds_loaded["Slope_X"] = (("latitude", "longitude"), slope_x.astype(np.float32))

        # We will drop the raw LULC, GHS, and DEM to avoid feeding raw unscaled categorical/absolute data
        vars_to_drop = ["LULC", "GHS_BUILT", "DEM"]
        ds_loaded = ds_loaded.drop_vars([v for v in vars_to_drop if v in ds_loaded])

        if include_fire_input:
            feature_vars = list(ds_loaded.data_vars)
        else:
            feature_vars = [v for v in ds_loaded.data_vars if v != "MODIS_FIRE_T1"]

        total_steps = ds_loaded.sizes["valid_time"]

    fire_frames = int((ds_loaded["MODIS_FIRE_T1"].values.sum(axis=(1, 2)) > 0).sum())
    print(
        f"--- Loaded: {total_steps} steps, {len(feature_vars)} features, "
        f"{fire_frames} fire frames ({100 * fire_frames / total_steps:.1f}%) ---"
    )
    if include_fire_input:
        print("--- MODIS_FIRE_T1 included as input feature (autoregressive mode) ---")
    return ds_loaded, feature_vars, total_steps


def _compute_global_stats(ds_loaded, feature_vars, cache_path):
    if os.path.exists(cache_path):
        print(f"--- Loading cached stats from {cache_path} ---")
        try:
            with open(cache_path, "rb") as f:
                stats = pickle.load(f)
            if isinstance(stats, dict) and "min" in stats and "max" in stats:
                if len(stats["min"]) == len(feature_vars):
                    return stats
                print("Cache channel count mismatch. Recomputing...")
        except Exception as e:
            print(f"Warning: cache load failed ({e}). Recomputing...")

    print("--- Computing Robust Stats (2nd/98th Percentiles) ---")
    num_channels = len(feature_vars)
    mins = np.zeros(num_channels)
    maxs = np.zeros(num_channels)
    means = np.zeros(num_channels)
    stds = np.zeros(num_channels)

    for i, var in enumerate(tqdm(feature_vars, desc="Analyzing Channels")):
        data = ds_loaded[var].values

        # Calculate mean and std for standard scaling
        means[i] = np.nanmean(data)
        stds[i] = np.nanstd(data) + 1e-6

        if var in ["MODIS_FIRE_T1", "Burn_Scar", "Water_Mask", "Urban_Mask"]:
            # These are binary/sparse, 98th percentile is often 0. Bypass scaling.
            mins[i] = 0.0
            maxs[i] = 1.0
        else:
            mins[i] = np.nanpercentile(data, 2)
            maxs[i] = np.nanpercentile(data, 98)
            if maxs[i] == mins[i]:
                maxs[i] = np.nanmax(data)

    stats = {
        "min": mins.astype(np.float32),
        "max": maxs.astype(np.float32),
        "mean": means.astype(np.float32),
        "std": stds.astype(np.float32),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(stats, f)
    return stats


class FireDataset(Dataset):
    """Single-frame dataset for UNet / AttentionUNet."""

    def __init__(self, loaded_ds, indices, stats, feature_vars):
        self.ds = loaded_ds
        self.indices = indices
        self.stats = stats
        self.feature_vars = feature_vars

        # Find indices for specific channels to apply custom scaling
        self.u10_idx = feature_vars.index("u10") if "u10" in feature_vars else -1
        self.v10_idx = feature_vars.index("v10") if "v10" in feature_vars else -1

        # Binary layers skip min-max scaling completely
        self.binary_indices = [
            i
            for i, v in enumerate(feature_vars)
            if v in ["MODIS_FIRE_T1", "Burn_Scar", "Water_Mask", "Urban_Mask"]
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_idx = self.indices[idx]

        X_data = (
            self.ds[self.feature_vars]
            .isel(valid_time=t_idx)
            .to_array(dim="channel")
            .values
        )

        # TARGET: We want to predict ONLY the newly ignited pixels (the Delta).
        current_fire = self.ds["MODIS_FIRE_T1"].isel(valid_time=t_idx).values
        next_fire = self.ds["MODIS_FIRE_T1"].isel(valid_time=t_idx + 1).values
        Y_data = np.clip(next_fire - current_fire, 0, 1)

        # Normalization
        min_v = self.stats["min"][:, None, None]
        max_v = self.stats["max"][:, None, None]
        mean_v = self.stats["mean"][:, None, None]
        std_v = self.stats["std"][:, None, None]

        X_norm = np.zeros_like(X_data)

        for c in range(X_data.shape[0]):
            if c in self.binary_indices:
                # Bypass scaling for binary masks
                X_norm[c] = np.clip(X_data[c], 0, 1)
            elif c == self.u10_idx or c == self.v10_idx:
                # Standard Scaling for Wind Vectors to preserve directional signs (-/+)
                X_norm[c] = (X_data[c] - mean_v[c]) / std_v[c]
            else:
                # Min-Max Scaling for everything else (Temperature, Slope, etc.)
                data_c = np.clip(X_data[c], min_v[c], max_v[c])
                denom = max_v[c] - min_v[c] + np.float32(1e-6)
                X_norm[c] = (data_c - min_v[c]) / denom

        X_tensor = torch.from_numpy(X_norm).float()
        Y_tensor = torch.from_numpy(Y_data).float().unsqueeze(0)

        _, h, w = X_tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        if pad_h > 0 or pad_w > 0:
            X_tensor = F.pad(X_tensor, (0, pad_w, 0, pad_h))
            Y_tensor = F.pad(Y_tensor, (0, pad_w, 0, pad_h))

        return X_tensor, Y_tensor


class FireSeqDataset(Dataset):
    """Sequence dataset for ConvLSTM / Hybrid models. Returns (T, C, H, W)."""

    def __init__(self, loaded_ds, indices, stats, feature_vars, seq_len=4):
        self.ds = loaded_ds
        self.indices = indices
        self.stats = stats
        self.feature_vars = feature_vars
        self.seq_len = seq_len

        self.u10_idx = feature_vars.index("u10") if "u10" in feature_vars else -1
        self.v10_idx = feature_vars.index("v10") if "v10" in feature_vars else -1
        self.binary_indices = [
            i
            for i, v in enumerate(feature_vars)
            if v in ["MODIS_FIRE_T1", "Burn_Scar", "Water_Mask", "Urban_Mask"]
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        target_t_idx = self.indices[idx]
        start_t = target_t_idx - self.seq_len + 1
        time_indices = [max(0, t) for t in range(start_t, target_t_idx + 1)]

        min_v = self.stats["min"][:, None, None]
        max_v = self.stats["max"][:, None, None]
        mean_v = self.stats["mean"][:, None, None]
        std_v = self.stats["std"][:, None, None]

        frames = []
        for t in time_indices:
            data = (
                self.ds[self.feature_vars]
                .isel(valid_time=t)
                .to_array(dim="channel")
                .values
            )

            norm_frame = np.zeros_like(data)
            for c in range(data.shape[0]):
                if c in self.binary_indices:
                    norm_frame[c] = np.clip(data[c], 0, 1)
                elif c == self.u10_idx or c == self.v10_idx:
                    norm_frame[c] = (data[c] - mean_v[c]) / std_v[c]
                else:
                    data_c = np.clip(data[c], min_v[c], max_v[c])
                    denom = max_v[c] - min_v[c] + np.float32(1e-6)
                    norm_frame[c] = (data_c - min_v[c]) / denom

            frames.append(norm_frame)

        X_seq = np.stack(frames, axis=0)  # (T, C, H, W)

        # TARGET: We want to predict ONLY the newly ignited pixels (the Delta).
        current_fire = self.ds["MODIS_FIRE_T1"].isel(valid_time=target_t_idx).values
        next_fire = self.ds["MODIS_FIRE_T1"].isel(valid_time=target_t_idx + 1).values
        Y_data = np.clip(next_fire - current_fire, 0, 1)

        X_tensor = torch.from_numpy(X_seq).float()
        Y_tensor = torch.from_numpy(Y_data).float().unsqueeze(0)

        _, _, h, w = X_tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            X_tensor = F.pad(X_tensor, (0, pad_w, 0, pad_h))
            Y_tensor = F.pad(Y_tensor, (0, pad_w, 0, pad_h))

        return X_tensor, Y_tensor


def _compute_sample_weights(ds_loaded, indices, fire_oversample_ratio=10.0):
    """Per-sample weights for WeightedRandomSampler.

    CRITICAL FIX: We ONLY care about frames where the fire actually expanded (Delta > 0).
    Due to 24-hour persistence, 23/24 frames have 0 spread.
    Frames with active expansion get a massive weight (fire_oversample_ratio).
    Frames with static fire get 0.0 weight (ignored).
    Frames with no fire get a very small weight to keep the network grounded.
    """
    fire_data = ds_loaded["MODIS_FIRE_T1"].values

    weights = []
    n_expansion = 0
    n_static = 0
    n_empty = 0

    for idx in indices:
        current_fire = fire_data[idx]
        next_fire = fire_data[idx + 1]

        # Calculate new spread
        delta = np.clip(next_fire - current_fire, 0, 1)
        spread_pixels = delta.sum()

        if spread_pixels > 0:
            # Active expansion: Focus heavily on this!
            weights.append(fire_oversample_ratio)
            n_expansion += 1
        elif current_fire.sum() > 0:
            # Fire exists, but didn't grow (static frame due to 24h persistence)
            # IGNORE IT. If we train on this, the model learns to predict 0 spread.
            weights.append(0.0)
            n_static += 1
        else:
            # No fire anywhere. Keep a few to prevent false positives.
            weights.append(1.0)
            n_empty += 1

    print(
        f"--- Sampler Weights: {n_expansion} Expansion (Focused), "
        f"{n_static} Static (Ignored), {n_empty} Empty (Baseline) ---"
    )
    return torch.DoubleTensor(weights)


# ---- Public loaders ----


def load_split_data(
    batch_size=32,
    nc_path=None,
    weighted_sampling=False,
    fire_oversample_ratio=10.0,
    include_fire_input=False,
):
    """Single-frame loader for UNet / AttentionUNet.

    Args:
        weighted_sampling: oversample fire-containing target frames.
        fire_oversample_ratio: weight multiplier for fire frames in sampler.
        include_fire_input: include MODIS_FIRE_T1 at time T as an input channel.
    """
    ds_loaded, feature_vars, total_steps = _load_ds(
        nc_path, include_fire_input=include_fire_input
    )
    cache = (
        CACHE_PATH
        if nc_path is None
        else CACHE_PATH.replace(".pkl", f"_{hash(nc_path) % 10000}.pkl")
    )
    if include_fire_input:
        cache = cache.replace(".pkl", "_fi.pkl")
    stats = _compute_global_stats(ds_loaded, feature_vars, cache)

    indices = np.arange(total_steps - 1)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=False)

    train_ds = FireDataset(ds_loaded, train_idx, stats, feature_vars)
    val_ds = FireDataset(ds_loaded, val_idx, stats, feature_vars)

    nw = min(8, os.cpu_count() or 4)

    if weighted_sampling:
        train_weights = _compute_sample_weights(
            ds_loaded, train_idx, fire_oversample_ratio
        )
        sampler = WeightedRandomSampler(
            train_weights, num_samples=len(train_weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=nw,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    return train_loader, val_loader, len(feature_vars)


def load_seq_data(
    batch_size=4,
    seq_len=4,
    nc_path=None,
    weighted_sampling=False,
    fire_oversample_ratio=10.0,
    include_fire_input=False,
):
    """Sequence loader for ConvLSTM / Hybrid. Returns (B, T, C, H, W) batches."""
    ds_loaded, feature_vars, total_steps = _load_ds(
        nc_path, include_fire_input=include_fire_input
    )
    cache = (
        CACHE_PATH
        if nc_path is None
        else CACHE_PATH.replace(".pkl", f"_{hash(nc_path) % 10000}.pkl")
    )
    if include_fire_input:
        cache = cache.replace(".pkl", "_fi.pkl")
    stats = _compute_global_stats(ds_loaded, feature_vars, cache)

    indices = np.arange(seq_len, total_steps - 1)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=False)

    train_ds = FireSeqDataset(ds_loaded, train_idx, stats, feature_vars, seq_len)
    val_ds = FireSeqDataset(ds_loaded, val_idx, stats, feature_vars, seq_len)

    nw = min(8, os.cpu_count() or 4)

    if weighted_sampling:
        train_weights = _compute_sample_weights(
            ds_loaded, train_idx, fire_oversample_ratio
        )
        sampler = WeightedRandomSampler(
            train_weights, num_samples=len(train_weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=nw,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )
    return train_loader, val_loader, len(feature_vars)
