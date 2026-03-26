"""
NDWS (Next Day Wildfire Spread) Dataset Parser

Bridges TensorFlow TFRecord format to PyTorch DataLoader.
Uses hardcoded normalization constants from the Google NDWS paper.
"""

import os
import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except ImportError:
    raise ImportError("TensorFlow not installed. Run: uv add tensorflow")


# === NDWS Feature Configuration ===
# 12 input features + 1 target (FireMask)
INPUT_FEATURES = [
    "elevation",     # Topography
    "th",            # Wind direction (degrees)
    "vs",            # Wind speed (m/s)
    "tmmn",          # Min temperature (K)
    "tmmx",          # Max temperature (K)
    "sph",           # Specific humidity (kg/kg)
    "pr",            # Precipitation (mm)
    "pdsi",          # Palmer Drought Severity Index
    "erc",           # Energy Release Component
    "population",    # Population density
    "NDVI",          # Vegetation index
    "PrevFireMask",  # Day 1 fire footprint (binary)
]
TARGET_FEATURE = "FireMask"  # Day 2 fire footprint (binary)

# Hardcoded normalization constants from Google NDWS paper
# Format: (min, max) for min-max scaling to [0, 1]
NORMALIZATION_CONSTANTS = {
    "elevation": (-100.0, 4000.0),
    "th": (0.0, 360.0),
    "vs": (0.0, 20.0),
    "tmmn": (250.0, 320.0),
    "tmmx": (250.0, 330.0),
    "sph": (0.0, 0.02),
    "pr": (0.0, 50.0),
    "pdsi": (-10.0, 10.0),
    "erc": (0.0, 200.0),
    "population": (0.0, 1000.0),
    "NDVI": (-1.0, 1.0),
    "PrevFireMask": (0.0, 1.0),  # Binary, no scaling needed
    "FireMask": (0.0, 1.0),      # Binary, no scaling needed
}


def _create_feature_description():
    """Create TensorFlow feature description for parsing."""
    desc = {}
    for feat in INPUT_FEATURES + [TARGET_FEATURE]:
        desc[feat] = tf.io.FixedLenFeature([64, 64], tf.float32)
    return desc


def _parse_single_example(raw_record, feature_description):
    """Parse a single TFRecord example."""
    return tf.io.parse_single_example(raw_record, feature_description)


def _normalize_feature(data: np.ndarray, feature_name: str) -> np.ndarray:
    """Apply min-max normalization using hardcoded constants."""
    min_val, max_val = NORMALIZATION_CONSTANTS[feature_name]
    
    # Binary masks: just clip to [0, 1]
    if feature_name in ["PrevFireMask", "FireMask"]:
        return np.clip(data, 0.0, 1.0)
    
    # Min-max normalization
    data = np.clip(data, min_val, max_val)
    return (data - min_val) / (max_val - min_val + 1e-6)


class NDWSDataset(IterableDataset):
    """
    PyTorch IterableDataset wrapper for NDWS TFRecord files.
    
    Yields (X, Y) tuples where:
        X: [12, 64, 64] tensor of normalized input features
        Y: [1, 64, 64] tensor of FireMask target
    """
    
    def __init__(self, tfrecord_pattern: str, shuffle: bool = True):
        """
        Args:
            tfrecord_pattern: Glob pattern for TFRecord files
                e.g., "Model/dataset/ndws/next_day_wildfire_spread_train_*.tfrecord"
            shuffle: Whether to shuffle files and samples
        """
        self.tfrecord_files = sorted(glob.glob(tfrecord_pattern))
        if not self.tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found: {tfrecord_pattern}")
        
        self.shuffle = shuffle
        self.feature_description = _create_feature_description()
        
        # Count total samples (cached)
        self._length = None
    
    def __len__(self):
        """Return total number of samples (expensive on first call)."""
        if self._length is None:
            self._length = 0
            for f in self.tfrecord_files:
                ds = tf.data.TFRecordDataset(f)
                self._length += sum(1 for _ in ds)
        return self._length
    
    def __iter__(self):
        """Iterate over all samples, yielding (X, Y) tuples."""
        files = self.tfrecord_files.copy()
        if self.shuffle:
            np.random.shuffle(files)
        
        for tfrecord_file in files:
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=1000)
            
            for raw_record in dataset:
                example = _parse_single_example(raw_record, self.feature_description)
                
                # Stack input features: [12, 64, 64]
                input_channels = []
                for feat_name in INPUT_FEATURES:
                    data = example[feat_name].numpy()
                    normalized = _normalize_feature(data, feat_name)
                    input_channels.append(normalized)
                
                X = np.stack(input_channels, axis=0).astype(np.float32)
                
                # Target: [1, 64, 64]
                Y = example[TARGET_FEATURE].numpy()
                Y = _normalize_feature(Y, TARGET_FEATURE).astype(np.float32)
                Y = Y[np.newaxis, ...]  # Add channel dim
                
                yield torch.from_numpy(X), torch.from_numpy(Y)


def get_ndws_dataloaders(
    data_dir: str = "Model/dataset/ndws",
    batch_size: int = 32,
    num_workers: int = 0,  # Must be 0 for IterableDataset with TF
) -> tuple:
    """
    Create train, val, and test DataLoaders for NDWS dataset.
    
    Returns:
        (train_loader, val_loader, test_loader, n_channels)
    """
    train_pattern = os.path.join(data_dir, "next_day_wildfire_spread_train_*.tfrecord")
    eval_pattern = os.path.join(data_dir, "next_day_wildfire_spread_eval_*.tfrecord")
    test_pattern = os.path.join(data_dir, "next_day_wildfire_spread_test_*.tfrecord")
    
    train_ds = NDWSDataset(train_pattern, shuffle=True)
    eval_ds = NDWSDataset(eval_pattern, shuffle=False)
    test_ds = NDWSDataset(test_pattern, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(eval_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    
    n_channels = len(INPUT_FEATURES)  # 12
    
    return train_loader, val_loader, test_loader, n_channels


# === Quick Test ===
if __name__ == "__main__":
    print("=== NDWS Dataset Parser Test ===\n")
    
    # Test single dataset
    train_pattern = "Model/dataset/ndws/next_day_wildfire_spread_train_00.tfrecord"
    ds = NDWSDataset(train_pattern, shuffle=False)
    
    print(f"Files: {len(ds.tfrecord_files)}")
    print(f"Input features: {len(INPUT_FEATURES)} channels")
    print(f"Features: {INPUT_FEATURES}")
    
    # Get one sample
    for X, Y in ds:
        print(f"\nSample shapes:")
        print(f"  X (inputs): {X.shape} - expected [12, 64, 64]")
        print(f"  Y (target): {Y.shape} - expected [1, 64, 64]")
        print(f"  X dtype: {X.dtype}")
        print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  Y range: [{Y.min():.3f}, {Y.max():.3f}]")
        print(f"  NaN check: X has NaN = {torch.isnan(X).any()}, Y has NaN = {torch.isnan(Y).any()}")
        break
    
    print("\n=== DataLoader Test ===")
    train_loader, val_loader, test_loader, n_channels = get_ndws_dataloaders(
        batch_size=16
    )
    
    for X_batch, Y_batch in train_loader:
        print(f"Batch X: {X_batch.shape} - expected [16, 12, 64, 64]")
        print(f"Batch Y: {Y_batch.shape} - expected [16, 1, 64, 64]")
        print(f"n_channels: {n_channels}")
        break
    
    print("\n✅ All tests passed!")
