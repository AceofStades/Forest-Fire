#!/usr/bin/env python3
"""
Overnight Hyperparameter Sweep v3 -- Forest Fire Prediction Models.

Key improvements over v2 (addresses all-zero F1 from extreme class imbalance):
  - Weighted sampling: fire-containing frames oversampled 10x (~50% of batches)
  - Tversky loss: penalises missed fire regions harder than false alarms
  - Extreme pos_weight: 500-3000 to counteract 0.002% fire-pixel ratio
  - Multi-threshold F1: evaluates thresholds 0.05-0.5, reports best
  - Autoregressive option: MODIS_FIRE_T1 at T used as input feature
  - Aggressive early kill: if max_prob < 0.02 after 5 epochs, config is dead
  - Comprehensive grid search via itertools.product (36 configs)
  - Fixed: ReduceLROnPlateau (no verbose), torch.amp deprecations

Run from Model/ directory:
    uv run python sweep.py

Clear old results first:
    rm -f checkouts/*_report.json checkouts/*.pth
"""

import csv
import gc
import itertools
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.dataset import load_split_data, load_seq_data
from src.models import AttentionUNet, ConvLSTMFireNet, HybridFireNet
from src.utils import (
    BCEDiceLoss, BCETverskyLoss, CombinedLoss, DiceLoss, FocalLoss, TverskyLoss,
)

# ============================================================================
#  Global settings
# ============================================================================

CHECKOUTS_DIR = "checkouts"
SUMMARY_CSV = os.path.join(CHECKOUTS_DIR, "sweep_summary.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

QUICK_SCAN_EPOCHS = 5
MIN_MAX_PROB = 0.02

EVAL_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# ============================================================================
#  Hyperparameter grids  (total ~36 configs)
# ============================================================================

PARAM_GRID = {
    "AttentionUNet": {
        "lr":                 [3e-4, 1e-4],
        "pos_weight":         [500, 1000, 3000],
        "loss":               ["BCETverskyLoss", "BCEDiceLoss"],
        "include_fire_input": [True, False],
        "base_filters":       [32],
        "scheduler":          ["cosine"],
        "optimizer":          ["adamw"],
        "batch_size":         [8],
        "accum_steps":        [4],
        "epochs":             [30],
        "patience":           [8],
        "dropout":            [0.3],
        "weighted_sampling":  [True],
        "weight_decay":       [1e-4],
    },
    "ConvLSTMFireNet": {
        "lr":                 [3e-4],
        "pos_weight":         [500, 1000],
        "loss":               ["BCETverskyLoss", "BCEDiceLoss"],
        "seq_len":            [4, 6],
        "hidden_dims":        [[64, 64]],
        "scheduler":          ["cosine"],
        "optimizer":          ["adamw"],
        "batch_size":         [4],
        "accum_steps":        [4],
        "epochs":             [25],
        "patience":           [7],
        "dropout":            [0.3],
        "include_fire_input": [False],
        "weighted_sampling":  [True],
        "weight_decay":       [1e-4],
    },
    "HybridFireNet": {
        "lr":                 [3e-4],
        "pos_weight":         [500, 1000],
        "loss":               ["BCETverskyLoss", "BCEDiceLoss"],
        "seq_len":            [4],
        "base_filters":       [32],
        "scheduler":          ["cosine"],
        "optimizer":          ["adamw"],
        "batch_size":         [4],
        "accum_steps":        [4],
        "epochs":             [25],
        "patience":           [7],
        "dropout":            [0.3],
        "include_fire_input": [False],
        "weighted_sampling":  [True],
        "weight_decay":       [1e-4],
    },
}


def generate_configs():
    """Build all sweep configs from PARAM_GRID via itertools.product."""
    all_configs = []
    seen = set()

    for model_name, grid in PARAM_GRID.items():
        keys = sorted(grid.keys())
        vals = [grid[k] for k in keys]

        for combo in itertools.product(*vals):
            cfg = dict(zip(keys, combo))
            cfg["model"] = model_name

            tag = {"AttentionUNet": "attn", "ConvLSTMFireNet": "lstm",
                   "HybridFireNet": "hybrid"}[model_name]
            loss_short = "bcetvk" if "Tversky" in cfg["loss"] else "bcedice"
            parts = [tag, f"pw{int(cfg['pos_weight'])}", loss_short,
                     f"lr{cfg['lr']:.0e}"]
            if cfg.get("include_fire_input"):
                parts.append("fi")
            if cfg.get("seq_len"):
                parts.append(f"s{cfg['seq_len']}")

            name = "_".join(parts)
            if name in seen:
                i = 2
                while f"{name}_{i}" in seen:
                    i += 1
                name = f"{name}_{i}"
            seen.add(name)
            cfg["name"] = name
            all_configs.append(cfg)

    all_configs.sort(key=lambda c: (
        -c.get("pos_weight", 0),
        0 if "Tversky" in c.get("loss", "") else 1,
        0 if c.get("include_fire_input") else 1,
        {"AttentionUNet": 0, "ConvLSTMFireNet": 1, "HybridFireNet": 2}.get(c["model"], 3),
    ))
    return all_configs


# ============================================================================
#  Builder helpers
# ============================================================================

def build_model(cfg, in_channels):
    m = cfg["model"]
    if m == "AttentionUNet":
        return AttentionUNet(
            n_channels=in_channels,
            base_filters=cfg.get("base_filters", 64),
            dropout=cfg.get("dropout", 0.3),
        )
    elif m == "ConvLSTMFireNet":
        hd = cfg.get("hidden_dims", [64, 64])
        if isinstance(hd, str):
            hd = json.loads(hd)
        return ConvLSTMFireNet(
            in_channels=in_channels, hidden_dims=list(hd),
            dropout=cfg.get("dropout", 0.3),
        )
    elif m == "HybridFireNet":
        return HybridFireNet(
            in_channels=in_channels,
            base_filters=cfg.get("base_filters", 32),
            dropout=cfg.get("dropout", 0.3),
        )
    raise ValueError(f"Unknown model: {m}")


def build_loss(cfg):
    n = cfg["loss"]
    pw = cfg.get("pos_weight", 500.0)
    if n == "BCETverskyLoss":
        return BCETverskyLoss(pos_weight=pw)
    elif n == "BCEDiceLoss":
        return BCEDiceLoss(pos_weight=pw)
    elif n == "CombinedLoss":
        return CombinedLoss()
    elif n == "DiceLoss":
        return DiceLoss()
    elif n == "FocalLoss":
        return FocalLoss()
    elif n == "TverskyLoss":
        return TverskyLoss()
    raise ValueError(f"Unknown loss: {n}")


def build_optimizer(cfg, params):
    o = cfg.get("optimizer", "adamw")
    lr, wd = cfg["lr"], cfg.get("weight_decay", 1e-4)
    if o == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif o == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif o == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                               weight_decay=wd, nesterov=True)
    raise ValueError(f"Unknown optimizer: {o}")


def build_scheduler(cfg, optimizer, steps_per_epoch):
    s = cfg.get("scheduler", "cosine")
    epochs = cfg["epochs"]
    if s == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["lr"],
            total_steps=epochs * steps_per_epoch, pct_start=0.1,
        ), "batch"
    elif s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=cfg["lr"] * 0.01,
        ), "epoch"
    elif s == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4,
        ), "plateau"
    raise ValueError(f"Unknown scheduler: {s}")


def is_temporal(cfg):
    return cfg["model"] in ("ConvLSTMFireNet", "HybridFireNet")


def report_path(cfg):
    return os.path.join(CHECKOUTS_DIR, f"{cfg['name']}_report.json")


def ckpt_path(cfg):
    return os.path.join(CHECKOUTS_DIR, f"{cfg['name']}.pth")


def header(text):
    w = 76
    print("\n" + "=" * w, flush=True)
    for line in text.split("\n"):
        print(f"  {line}", flush=True)
    print("=" * w, flush=True)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================================
#  Core training function for one config
# ============================================================================

def train_one(cfg):
    name = cfg["name"]
    header(
        f"CONFIG: {name}\n"
        f"model={cfg['model']}  loss={cfg['loss']}  pos_weight={cfg.get('pos_weight', '-')}\n"
        f"lr={cfg['lr']}  opt={cfg.get('optimizer', 'adamw')}  "
        f"sched={cfg.get('scheduler', 'cosine')}\n"
        f"weighted={cfg.get('weighted_sampling', False)}  "
        f"fire_input={cfg.get('include_fire_input', False)}"
        + (f"  seq={cfg.get('seq_len', '-')}" if is_temporal(cfg) else "")
    )
    t0 = time.time()

    ws = cfg.get("weighted_sampling", False)
    fi = cfg.get("include_fire_input", False)

    if is_temporal(cfg):
        train_loader, val_loader, in_ch = load_seq_data(
            batch_size=cfg["batch_size"], seq_len=cfg.get("seq_len", 4),
            weighted_sampling=ws, include_fire_input=fi,
        )
    else:
        train_loader, val_loader, in_ch = load_split_data(
            batch_size=cfg["batch_size"],
            weighted_sampling=ws, include_fire_input=fi,
        )

    model = build_model(cfg, in_ch).to(DEVICE)
    criterion = build_loss(cfg).to(DEVICE)
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler, sched_mode = build_scheduler(cfg, optimizer, len(train_loader))
    scaler = GradScaler("cuda", enabled=AMP_ENABLED)

    accum = cfg["accum_steps"]
    patience = cfg["patience"]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Params: {n_params:,}  |  Train: {len(train_loader)} batches  "
        f"Val: {len(val_loader)} batches")

    best_f1 = -1.0
    best_metrics = {}
    no_improve = 0
    global_max_prob = 0.0
    epoch = 0

    for epoch in range(1, cfg["epochs"] + 1):

        # ---- Train --------------------------------------------------------
        model.train()
        train_loss = 0.0
        n_train = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"  Train {epoch}/{cfg['epochs']}",
                    unit="b", leave=False, ncols=100)
        for i, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if torch.isnan(inputs).any():
                continue

            with autocast("cuda", enabled=AMP_ENABLED):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accum

            scaler.scale(loss).backward()

            if (i + 1) % accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if sched_mode == "batch":
                    scheduler.step()

            raw_loss = loss.item() * accum
            train_loss += raw_loss
            n_train += 1
            pbar.set_postfix(loss=f"{raw_loss:.4f}")

        avg_train_loss = train_loss / max(n_train, 1)

        # ---- Validate (multi-threshold) -----------------------------------
        model.eval()
        val_loss = 0.0
        val_n = 0
        epoch_max_prob = 0.0
        thr_acc = {t: [0.0, 0.0, 0.0, 0.0] for t in EVAL_THRESHOLDS}

        pbar_v = tqdm(val_loader, desc=f"  Val   {epoch}/{cfg['epochs']}",
                      unit="b", leave=False, ncols=100)
        with torch.no_grad():
            for inputs, labels in pbar_v:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                if torch.isnan(inputs).any():
                    continue

                with autocast("cuda", enabled=AMP_ENABLED):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_n += 1

                probs = torch.sigmoid(outputs.float())
                batch_max = probs.max().item()
                if batch_max > epoch_max_prob:
                    epoch_max_prob = batch_max

                for t in EVAL_THRESHOLDS:
                    pred = (probs > t).float()
                    thr_acc[t][0] += (pred * labels).sum().item()
                    thr_acc[t][1] += (pred * (1 - labels)).sum().item()
                    thr_acc[t][2] += ((1 - pred) * labels).sum().item()
                    thr_acc[t][3] += ((1 - pred) * (1 - labels)).sum().item()

                tp3, fp3, fn3 = thr_acc[0.3][0], thr_acc[0.3][1], thr_acc[0.3][2]
                rf1 = 2 * tp3 / (2 * tp3 + fp3 + fn3 + 1e-8)
                pbar_v.set_postfix(F1=f"{rf1:.4f}", maxP=f"{epoch_max_prob:.4f}")

        if val_n == 0:
            log(f"  Epoch {epoch}: no valid val batches")
            continue

        avg_val_loss = val_loss / val_n
        global_max_prob = max(global_max_prob, epoch_max_prob)

        best_t_f1, best_t = -1.0, 0.5
        for t in EVAL_THRESHOLDS:
            tp, fp, fn, tn = thr_acc[t]
            pr = tp / (tp + fp + 1e-8)
            rc = tp / (tp + fn + 1e-8)
            ft = 2 * pr * rc / (pr + rc + 1e-8)
            if ft > best_t_f1:
                best_t_f1 = ft
                best_t = t

        tp, fp, fn, tn = thr_acc[best_t]
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        metrics = {"accuracy": acc, "precision": prec, "recall": rec,
                   "f1": f1, "iou": iou, "threshold": best_t,
                   "max_prob": epoch_max_prob}

        if sched_mode == "epoch":
            scheduler.step()
        elif sched_mode == "plateau":
            scheduler.step(f1)

        improved = f1 > best_f1
        if improved:
            best_f1 = f1
            best_metrics = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                **{k: round(v, 6) if isinstance(v, float) else v
                   for k, v in metrics.items()},
            }
            torch.save(model.state_dict(), ckpt_path(cfg))
            no_improve = 0
        else:
            no_improve += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        star = "  * NEW BEST" if improved else ""
        print(
            f"\n  Epoch [{epoch:>3}/{cfg['epochs']}]  lr={cur_lr:.2e}  "
            f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f}\n"
            f"  F1={f1:.4f}  IoU={iou:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
            f"maxP={epoch_max_prob:.4f}  thr={best_t}\n"
            f"  best_F1={best_f1:.4f}  stale={no_improve}/{patience}{star}",
            flush=True,
        )

        if epoch >= QUICK_SCAN_EPOCHS and global_max_prob < MIN_MAX_PROB:
            log(f"  EARLY KILL: max_prob={global_max_prob:.4f} < {MIN_MAX_PROB} "
                f"after {epoch} epochs -- model not learning fire at all")
            break

        if no_improve >= patience:
            log(f"  Early stop: no F1 improvement for {patience} epochs")
            break

    elapsed = time.time() - t0

    def _ser(v):
        if isinstance(v, list):
            return str(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    early_killed = (epoch >= QUICK_SCAN_EPOCHS and global_max_prob < MIN_MAX_PROB)
    report = {
        "name": name,
        "config": {k: _ser(v) for k, v in cfg.items()},
        "best_metrics": best_metrics,
        "global_max_prob": round(global_max_prob, 6),
        "total_epochs_trained": epoch,
        "early_killed": early_killed,
        "elapsed_seconds": round(elapsed, 1),
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat(),
    }
    with open(report_path(cfg), "w") as f:
        json.dump(report, f, indent=2)

    header(
        f"DONE: {name}\n"
        f"best_F1={best_f1:.4f}  maxP={global_max_prob:.4f}  "
        f"({elapsed / 60:.1f} min, {epoch} epochs)"
    )

    del model, optimizer, scheduler, scaler, criterion, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    return report


# ============================================================================
#  Main sweep loop
# ============================================================================

def main():
    os.makedirs(CHECKOUTS_DIR, exist_ok=True)
    configs = generate_configs()

    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log(f"GPU   : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    log(f"Total configs: {len(configs)}")
    log(f"Quick-scan: {QUICK_SCAN_EPOCHS} epochs, min maxP: {MIN_MAX_PROB}")

    for m, cnt in Counter(c["model"] for c in configs).items():
        log(f"  {m}: {cnt} configs")

    all_reports = []
    n_skip = n_kill = n_ok = n_fail = 0

    for idx, cfg in enumerate(configs):
        rp = report_path(cfg)
        if os.path.exists(rp):
            log(f"[{idx + 1}/{len(configs)}] SKIP (done): {cfg['name']}")
            try:
                with open(rp) as f:
                    all_reports.append(json.load(f))
            except Exception:
                pass
            n_skip += 1
            continue

        log(f"\n[{idx + 1}/{len(configs)}] Starting: {cfg['name']}")
        try:
            report = train_one(cfg)
            all_reports.append(report)
            if report.get("early_killed"):
                n_kill += 1
            else:
                n_ok += 1
        except Exception:
            log(f"FAILED: {cfg['name']}")
            traceback.print_exc()
            all_reports.append({
                "name": cfg["name"], "best_metrics": {}, "error": True,
            })
            n_fail += 1
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Summary -----------------------------------------------------------
    fields = ["rank", "name", "model", "loss", "pos_weight", "lr",
              "fire_input", "f1", "iou", "precision", "recall",
              "threshold", "max_prob", "val_loss", "epoch",
              "early_killed", "elapsed_min"]

    rows = []
    for r in all_reports:
        bm = r.get("best_metrics", {})
        c = r.get("config", r)
        rows.append({
            "name":          r.get("name", "?"),
            "model":         c.get("model", "?"),
            "loss":          c.get("loss", "?"),
            "pos_weight":    c.get("pos_weight", "?"),
            "lr":            c.get("lr", "?"),
            "fire_input":    c.get("include_fire_input", "?"),
            "f1":            bm.get("f1", -1),
            "iou":           bm.get("iou", -1),
            "precision":     bm.get("precision", -1),
            "recall":        bm.get("recall", -1),
            "threshold":     bm.get("threshold", -1),
            "max_prob":      r.get("global_max_prob", bm.get("max_prob", -1)),
            "val_loss":      bm.get("val_loss", -1),
            "epoch":         bm.get("epoch", -1),
            "early_killed":  r.get("early_killed", False),
            "elapsed_min":   round(r.get("elapsed_seconds", 0) / 60, 1),
        })

    def _sort_key(r):
        f = r["f1"] if isinstance(r["f1"], (int, float)) and r["f1"] >= 0 else -1
        mp = r["max_prob"] if isinstance(r["max_prob"], (int, float)) else 0
        return (-f, -mp)

    rows.sort(key=_sort_key)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    header("SWEEP COMPLETE -- Final Rankings")
    print(f"  {'#':<3} {'Name':<42} {'F1':>7} {'IoU':>7} {'maxP':>7} {'thr':>5}",
          flush=True)
    print("  " + "-" * 72, flush=True)
    for r in rows[:30]:
        if r["f1"] < 0:
            print(f"  {'-':<3} {r['name']:<42}   FAILED/ERROR", flush=True)
        else:
            k = " X" if r.get("early_killed") else ""
            mp = r.get("max_prob", 0)
            if not isinstance(mp, (int, float)):
                mp = 0
            thr = r.get("threshold", 0)
            if not isinstance(thr, (int, float)):
                thr = 0
            print(f"  {r['rank']:<3} {r['name']:<42} "
                  f"{r['f1']:>7.4f} {r['iou']:>7.4f} {mp:>7.4f} "
                  f"{thr:>5.2f}{k}", flush=True)

    log(f"\n  Trained: {n_ok}  Killed: {n_kill}  Skipped: {n_skip}  Failed: {n_fail}")
    log(f"  CSV: {os.path.abspath(SUMMARY_CSV)}")

    valid = [r for r in rows if isinstance(r["f1"], (int, float)) and r["f1"] > 0]
    if valid:
        b = valid[0]
        log(f"  BEST: {b['name']}  F1={b['f1']:.4f}  IoU={b['iou']:.4f}")
    else:
        log("  WARNING: No config achieved F1 > 0.")
        log("  Consider: even higher pos_weight, patch-based training, "
            "or oversampling fire pixels directly.")


if __name__ == "__main__":
    main()
