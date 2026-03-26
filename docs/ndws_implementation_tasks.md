# NDWS Implementation Task List

**Goal:** Train a UNet on Google NDWS dataset and compare against Custom Hybrid model.

**Status Legend:** ✅ Done | 🔄 In Progress | ⏳ Pending | 🔴 Blocked

---

## Phase 1: Dataset Acquisition

| Status | ID | Task | Details |
|--------|----|----- |---------|
| ✅ | 1a | Find NDWS dataset source | Dataset source located |
| ✅ | 1b | Download TFRecord files | 15 train + 2 eval + 2 test files (5.8GB) → `Model/dataset/ndws/` |
| ✅ | 1c | Verify dataset integrity | 14,979 train / 1,877 eval / 1,689 test samples. All 64×64 with 12 inputs + FireMask ✓ |

---

## Phase 2: TFRecord → PyTorch Bridge

| Status | ID | Task | Details |
|--------|----|----- |---------|
| ✅ | 2a | Install TensorFlow (GPU) | TensorFlow 2.20.0 with GPU support via `uv add tensorflow` |
| ✅ | 2b | Define feature spec | 12 inputs + FireMask dict created |
| ✅ | 2c | Parse single record | `_parse_single_example()` verified [64,64] shapes |
| ✅ | 2d | Create IterableDataset | `NDWSDataset` class wraps TF in PyTorch |
| ✅ | 2e | Add normalization | Hardcoded Google paper min/max constants applied |
| ✅ | 2f | Test DataLoader | ✓ Batches [B,12,64,64], normalized [0,1], no NaNs |

**The 12 Input Channels:**
```
Topography:  elevation
Weather:     th (wind dir), vs (wind speed), tmmn, tmmx, sph, pr  
Drought:     pdsi, erc, NDVI
Human:       population
Fire:        PrevFireMask (Day 1 fire footprint)
```

---

## Phase 3: Training Script (`train_ndws.py`)

| Status | ID | Task | Details |
|--------|----|----- |---------|
| ⏳ | 3a | Confirm UNet architecture | Verify n_channels=12, output=1, works with 64×64 |
| ⏳ | 3b | Create script skeleton | argparse, device, epoch loop, checkpoint saving |
| ⏳ | 3c | Add Combined Loss | Focal (α=0.95, γ=2.0) + Dice for imbalance |
| ⏳ | 3d | Add evaluation metrics | Accuracy, Precision, Recall, F1 at thresholds |
| ⏳ | 3e | Add training logging | Loss/metrics/LR per epoch; TensorBoard/wandb |
| ⏳ | 3f | Add early stopping | Stop if val F1 stagnates, save best checkpoint |

---

## Phase 4: Training Execution

| Status | ID | Task | Details |
|--------|----|----- |---------|
| ⏳ | 4a | Run smoke test | 1 epoch to verify pipeline works |
| ⏳ | 4b | Run full training | 50+ epochs, expect F1 ~0.30-0.40 |
| ⏳ | 4c | Analyze results | Plot loss curves, F1 at thresholds |
| ⏳ | 4d | Save best weights | Export to `Model/weights/ndws_unet_best.pth` |

---

## Phase 5: Comparative Analysis & Documentation

| Status | ID | Task | Details |
|--------|----|----- |---------|
| ⏳ | 5a | Quantitative comparison | Table: Custom vs NDWS (F1, Precision, Recall, time) |
| ⏳ | 5b | Architectural comparison | Black-box ML (rigid) vs Hybrid ML+CA (flexible) |
| ⏳ | 5c | Deployability comparison | US-specific (pdsi/erc) vs global (ERA5/DEM) |
| ⏳ | 5d | Update research paper | Add NDWS section to research_paper_draft.md |

---

## Comparison Methodology Summary

### Pillar 1: Quantitative (F1 Score)
| Model | Expected F1 | Reasoning |
|-------|-------------|-----------|
| Custom (Fuel Model) | ~0.94 | Static topographical prediction is deterministic |
| NDWS (Spread Model) | ~0.30-0.40 | Fluid dynamics from Day 1→Day 2 sparse snapshots is brutal |

### Pillar 2: Flexibility
- **NDWS:** Black-box. Day 1 → Day 2. No "what if wind changes at 2PM?"
- **Custom:** ML fuel map + CA physics. User adjusts wind → instant recalculation

### Pillar 3: Global Deployability  
- **NDWS:** Requires US-specific `pdsi`, `erc` indices
- **Custom:** Uses globally available ERA5 + DEM data

---

## Key Insight

> *"Predicting fluid dynamics from sparse satellite Day 1→Day 2 snapshots is mathematically brutal. The NDWS model will struggle (F1 ~0.30-0.40), while the Custom Hybrid ML+CA approach sidesteps this by using ML for static fuel mapping (F1 ~0.94) and delegating temporal spread to explicit physics."*

---

**Last Updated:** 2026-03-26  
**Current Task:** 1c - Verify dataset integrity
