#!/usr/bin/env python3
"""
Fine-tune MotionCLIP for PerMo healthy-vs-flawed anomaly detection with truly unseen-action testing.
What this script does:
  1. Reads a metadata CSV with columns:
       motion_path, action_label, condition_label, is_anomaly
  1b. Holds out entire action/content classes from training/validation when requested.
  1c. Optionally holds out entire style/condition classes from training/validation too.
  2. Creates prompts per action:
       normal class  : "healthy {action}"
       anomaly class : "flawed {action}"
  3. Freezes the CLIP text encoder.
  4. Freezes most of MotionCLIP motion encoder and trains only the last N transformer layers.
  5. Trains with CLIP-style supervised contrastive loss between motion embeddings
     and the matching text prompt embedding for each sample.
  6. Runs inference on seen-content, unseen-content, and optional unseen-style test sets using healthy/flawed prompt similarity.
  7. Saves metrics, predictions, split CSVs, embeddings, training curves, and the best checkpoint.
Example:
  python finetune_permo_flawed_motionclip.py \
    --csv_path /scratch/mgirishnair/Thesis/PerMo_metadata.csv \
    --repo_root /scratch/mgirishnair/Thesis/MotionCLIP \
    --checkpoint /scratch/mgirishnair/Thesis/MotionCLIP/checkpoints/babel60.pth.tar \
    --output_dir /scratch/mgirishnair/Thesis/permo_flawed_runs/run1 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-5 \
    --trainable_layers 2
Notes:
  - This script tries to be compatible with common MotionCLIP repo layouts, but you may need
    to set --model_module if your Encoder_TRANSFORMER import path differs.
  - If your .npz motion key is not auto-detected, pass --motion_key <key>.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
# -----------------------------
# General utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
def save_training_curves(epoch_records: List[Dict[str, Any]], output_dir: str | Path) -> None:
    """Save CSV/NPZ history and PNG plots for train/validation curves."""
    if not epoch_records:
        return
    output_dir = Path(output_dir)
    hist_df = pd.DataFrame(epoch_records)
    hist_df.to_csv(output_dir / "epoch_metrics.csv", index=False)
    np.savez(
        output_dir / "training_history.npz",
        **{col: hist_df[col].to_numpy() for col in hist_df.columns if pd.api.types.is_numeric_dtype(hist_df[col])},
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib, skipping training curve plots: {exc}")
        return
    def _plot(columns: Sequence[str], filename: str, ylabel: str) -> None:
        available = [c for c in columns if c in hist_df.columns]
        if not available:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in available:
            ax.plot(hist_df["epoch"], hist_df[c], marker="o", label=c)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
    _plot(["train_loss", "val_loss"], "loss_curves.png", "loss")
    _plot(["val_auroc", "val_auprc", "val_f1", "val_balanced_accuracy"], "validation_metrics.png", "metric")
def normalize_action_text(text: str) -> str:
    text = str(text).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())
def normalize_actor_id(text: str) -> str:
    """Normalize actor IDs such as A01/a01 to A01."""
    text = str(text).strip().upper()
    return text
def extract_actor_id_from_path(path: str | Path) -> str:
    """Extract actor ID from filenames like Armaching_Hop_A01_001.npz."""
    name = Path(str(path)).stem
    match = re.search(r"(?:^|_)(A\d{2})(?:_|$)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Could not extract actor ID from {path!r}. Expected filename pattern like '*_A01_001.npz'. "
            "Pass --actor_col if the CSV already has an actor column."
        )
    return normalize_actor_id(match.group(1))
def load_unseen_actor_ids_from_file(path: str | Path) -> List[str]:
    path = Path(path)
    actors: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            actors.append(normalize_actor_id(line.split(",")[0]))
    return actors
def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out
def find_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Tries to extract a model state dict from common checkpoint formats.
    """
    if isinstance(obj, dict):
        # Direct state dict: all or most values are tensors
        tensor_values = sum(torch.is_tensor(v) for v in obj.values())
        if tensor_values > 0 and tensor_values >= max(1, len(obj) // 2):
            return obj
        for key in ["state_dict", "model_state_dict", "model", "encoder", "motion_encoder", "net"]:
            if key in obj:
                maybe = obj[key]
                if isinstance(maybe, dict):
                    return find_state_dict(maybe)
    raise ValueError(
        "Could not find a state_dict in the checkpoint. "
        "Inspect the checkpoint keys and adapt find_state_dict()."
    )
# -----------------------------
# Dataset and splitting
# -----------------------------
class PerMoMotionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        action_col: str,
        label_col: str,
        motion_key: str = "auto",
        expected_shape: Tuple[int, int, int] = (60, 25, 6),
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.path_col = path_col
        self.action_col = action_col
        self.label_col = label_col
        self.motion_key = motion_key
        self.expected_shape = expected_shape
    def __len__(self) -> int:
        return len(self.df)
    def _pick_npz_array(self, data: np.lib.npyio.NpzFile, path: str) -> np.ndarray:
        if self.motion_key != "auto":
            if self.motion_key not in data.files:
                raise KeyError(f"motion_key={self.motion_key!r} not found in {path}. Keys: {data.files}")
            return data[self.motion_key]
        preferred = [
            "motion", "motions", "x", "X", "data", "arr_0", "poses", "pose",
            "rot6d", "features", "joints", "input"
        ]
        for key in preferred:
            if key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.number):
                    return arr
        numeric_keys = [k for k in data.files if np.issubdtype(data[k].dtype, np.number)]
        if not numeric_keys:
            raise KeyError(f"No numeric arrays found in {path}. Keys: {data.files}")
        return data[numeric_keys[0]]
    def _standardize_motion_shape(self, arr: np.ndarray, path: str) -> np.ndarray:
        """
        Returns motion as [T, J, F].
        Expected default is [60, 25, 6].
        """
        arr = np.asarray(arr, dtype=np.float32)
        # Remove trivial dimensions, e.g. [1, 60, 25, 6]
        while arr.ndim > 3 and 1 in arr.shape:
            arr = np.squeeze(arr, axis=arr.shape.index(1))
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D motion array [T,J,F], got shape {arr.shape} in {path}")
        T, J, Feat = self.expected_shape
        # Already [T, J, F]
        if arr.shape == (T, J, Feat):
            return arr
        # Common alternatives
        if arr.shape == (J, Feat, T):
            return np.transpose(arr, (2, 0, 1))
        if arr.shape == (Feat, T, J):
            return np.transpose(arr, (1, 2, 0))
        if arr.shape == (T, Feat, J):
            return np.transpose(arr, (0, 2, 1))
        if arr.shape == (J, T, Feat):
            return np.transpose(arr, (1, 0, 2))
        # Last-resort heuristic: identify axes by expected sizes
        shape = list(arr.shape)
        try:
            t_axis = shape.index(T)
            j_axis = shape.index(J)
            f_axis = shape.index(Feat)
            return np.transpose(arr, (t_axis, j_axis, f_axis))
        except ValueError as e:
            raise ValueError(
                f"Cannot convert motion shape {arr.shape} to expected {(T, J, Feat)} for {path}. "
                "Pass --num_frames/--njoints/--nfeats or adapt _standardize_motion_shape()."
            ) from e
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        with np.load(path, allow_pickle=False) as data:
            arr = self._pick_npz_array(data, path)
        arr = self._standardize_motion_shape(arr, path)
        return {
            "motion": torch.from_numpy(arr),  # [T, J, F]
            "action": normalize_action_text(row[self.action_col]),
            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),  # 0 healthy, 1 flawed
            "path": path,
            "row_index": int(row.get("original_index", idx)),
        }
def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "motion": torch.stack([b["motion"] for b in batch], dim=0),  # [B,T,J,F]
        "action": [b["action"] for b in batch],
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
        "row_index": [b["row_index"] for b in batch],
    }
class BalancedBinaryBatchSampler(Sampler[List[int]]):
    """Yield batches with an approximately 50/50 split between labels 0 and 1.
    Minority-class samples are oversampled with replacement when needed.
    This is used only for training; validation/test loaders remain deterministic.
    """
    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 2:
            raise ValueError("BalancedBinaryBatchSampler requires batch_size >= 2.")
        labels_np = np.asarray(labels).astype(int)
        unique = set(labels_np.tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"BalancedBinaryBatchSampler expects binary labels 0/1, got {sorted(unique)}")
        self.indices_by_class = {
            0: np.where(labels_np == 0)[0],
            1: np.where(labels_np == 1)[0],
        }
        if len(self.indices_by_class[0]) == 0 or len(self.indices_by_class[1]) == 0:
            raise ValueError(
                "BalancedBinaryBatchSampler needs at least one healthy sample and one flawed sample in the training split."
            )
        self.batch_size = int(batch_size)
        self.n0 = self.batch_size // 2
        self.n1 = self.batch_size - self.n0
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        # Number of batches needed so that the larger class is seen roughly once per epoch.
        self.num_batches = int(max(
            math.ceil(len(self.indices_by_class[0]) / self.n0),
            math.ceil(len(self.indices_by_class[1]) / self.n1),
        ))
    def __len__(self) -> int:
        return self.num_batches
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
    def _sample_class_indices(self, cls: int, n_total: int, rng: np.random.Generator) -> np.ndarray:
        pool = self.indices_by_class[cls]
        if n_total <= len(pool):
            return rng.permutation(pool)[:n_total]
        # Use all samples once, then oversample the remainder with replacement.
        full = rng.permutation(pool)
        extra = rng.choice(pool, size=n_total - len(pool), replace=True)
        return np.concatenate([full, extra])
    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        labels0 = self._sample_class_indices(0, self.num_batches * self.n0, rng)
        labels1 = self._sample_class_indices(1, self.num_batches * self.n1, rng)
        for batch_idx in range(self.num_batches):
            b0 = labels0[batch_idx * self.n0:(batch_idx + 1) * self.n0]
            b1 = labels1[batch_idx * self.n1:(batch_idx + 1) * self.n1]
            batch = np.concatenate([b0, b1])
            rng.shuffle(batch)
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch.astype(int).tolist()
def stratified_group_split(
    df: pd.DataFrame,
    stratify_cols: Sequence[str],
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratifies within groups, e.g. by action + condition.
    val_fraction is taken from the non-test part.
    """
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []
    for _, group in df.groupby(list(stratify_cols), dropna=False):
        idxs = group.index.to_numpy()
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_fraction))
        if n >= 5 and test_fraction > 0:
            n_test = max(1, n_test)
        n_test = min(n_test, max(0, n - 2))
        test_idx = idxs[:n_test]
        rem_idx = idxs[n_test:]
        n_rem = len(rem_idx)
        n_val = int(round(n_rem * val_fraction))
        if n_rem >= 5 and val_fraction > 0:
            n_val = max(1, n_val)
        n_val = min(n_val, max(0, n_rem - 1))
        val_idx = rem_idx[:n_val]
        train_idx = rem_idx[n_val:]
        train_parts.append(df.loc[train_idx])
        val_parts.append(df.loc[val_idx])
        test_parts.append(df.loc[test_idx])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return train_df, val_df, test_df
def balance_binary_split(
    df: pd.DataFrame,
    label_col: str,
    seed: int,
    split_name: str = "test",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Downsample the majority label so the split has equal normal/anomaly counts.
    Label convention:
      0 = healthy/normal
      1 = flawed/anomaly
    The sampling is deterministic for a fixed seed.
    """
    counts_before = df[label_col].astype(int).value_counts().to_dict()
    n_normal = int((df[label_col].astype(int) == 0).sum())
    n_anomaly = int((df[label_col].astype(int) == 1).sum())
    if n_normal == 0 or n_anomaly == 0:
        raise ValueError(
            f"Cannot balance {split_name}: needs both labels, got "
            f"normal={n_normal}, anomaly={n_anomaly}."
        )
    n_keep = min(n_normal, n_anomaly)
    normal_df = df[df[label_col].astype(int) == 0].sample(
        n=n_keep,
        replace=False,
        random_state=seed,
    )
    anomaly_df = df[df[label_col].astype(int) == 1].sample(
        n=n_keep,
        replace=False,
        random_state=seed + 1,
    )
    balanced_df = pd.concat([normal_df, anomaly_df], axis=0).sample(
        frac=1.0,
        random_state=seed + 2,
    ).reset_index(drop=True)
    counts_after = balanced_df[label_col].astype(int).value_counts().to_dict()
    info = {
        "split_name": split_name,
        "balanced": True,
        "seed": int(seed),
        "n_before": int(len(df)),
        "n_after": int(len(balanced_df)),
        "counts_before": {int(k): int(v) for k, v in counts_before.items()},
        "counts_after": {int(k): int(v) for k, v in counts_after.items()},
        "n_kept_per_label": int(n_keep),
        "n_dropped": int(len(df) - len(balanced_df)),
    }
    return balanced_df, info
def load_unseen_actions_from_file(path: str | Path) -> List[str]:
    """Read unseen action names from a txt/csv-like file.
    Accepts one action per line. Empty lines and lines starting with # are ignored.
    If a line contains a comma, only the first field is used.
    """
    path = Path(path)
    actions: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            actions.append(normalize_action_text(line.split(",")[0]))
    return actions
def load_unseen_values_from_file(path: str | Path) -> List[str]:
    """Read normalized unseen class/style names from a txt/csv-like file.
    Accepts one value per line. Empty lines and lines starting with # are ignored.
    If a line contains a comma, only the first field is used.
    """
    path = Path(path)
    values: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values.append(normalize_action_text(line.split(",")[0]))
    return values
def select_unseen_values(
    df: pd.DataFrame,
    norm_col: str,
    label_col: str,
    seed: int,
    requested_values: Optional[Sequence[str]] = None,
    requested_file: str = "",
    requested_count: int = 0,
    requested_fraction: float = 0.0,
    require_both_labels: bool = True,
    value_name: str = "value",
) -> List[str]:
    """Choose full classes/styles that will be excluded from train/val.
    If no explicit values, file, count, or positive fraction is provided, returns an empty list.
    """
    if norm_col not in df.columns:
        raise ValueError(f"Column {norm_col!r} not found while selecting unseen {value_name}s.")
    all_values = sorted(df[norm_col].dropna().map(normalize_action_text).unique().tolist())
    requested: List[str] = []
    if requested_values:
        requested.extend([normalize_action_text(v) for v in requested_values if str(v).strip()])
    if requested_file:
        requested.extend(load_unseen_values_from_file(requested_file))
    if requested:
        requested = sorted(set(requested))
        missing = sorted(set(requested) - set(all_values))
        if missing:
            raise ValueError(
                f"Requested unseen {value_name}s were not found in the CSV after normalization: "
                f"{missing}. Available examples: {all_values[:30]}"
            )
        return requested
    if requested_count <= 0 and requested_fraction <= 0:
        return []
    eligible = all_values
    if require_both_labels:
        eligible = []
        for value, g in df.groupby(norm_col):
            labs = set(g[label_col].astype(int).tolist())
            if {0, 1}.issubset(labs):
                eligible.append(value)
        eligible = sorted(eligible)
    if not eligible:
        raise ValueError(f"No eligible unseen {value_name}s found. Try allowing single-label unseen {value_name}s.")
    if requested_count > 0:
        n_unseen = int(requested_count)
    else:
        n_unseen = int(round(len(eligible) * requested_fraction))
        n_unseen = max(1, n_unseen)
    n_unseen = min(n_unseen, max(1, len(eligible) - 1))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.asarray(eligible, dtype=object), size=n_unseen, replace=False).tolist())
def select_unseen_actions(
    df: pd.DataFrame,
    action_norm_col: str,
    label_col: str,
    seed: int,
    unseen_actions: Optional[Sequence[str]] = None,
    unseen_action_file: str = "",
    unseen_action_count: int = 0,
    unseen_action_fraction: float = 0.20,
    require_both_labels: bool = True,
) -> List[str]:
    """Choose full action classes that will be excluded from train/val.
    Priority:
      1. --unseen_actions / --unseen_actions_file
      2. --unseen_action_count
      3. --unseen_action_fraction
    """
    all_actions = sorted(df[action_norm_col].dropna().map(normalize_action_text).unique().tolist())
    requested: List[str] = []
    if unseen_actions:
        requested.extend([normalize_action_text(a) for a in unseen_actions if str(a).strip()])
    if unseen_action_file:
        requested.extend(load_unseen_actions_from_file(unseen_action_file))
    if requested:
        requested = sorted(set(requested))
        missing = sorted(set(requested) - set(all_actions))
        if missing:
            raise ValueError(
                "Requested unseen actions were not found in the CSV after normalization: "
                f"{missing}. Available examples: {all_actions[:30]}"
            )
        return requested
    eligible = all_actions
    if require_both_labels:
        eligible = []
        for action, g in df.groupby(action_norm_col):
            labs = set(g[label_col].astype(int).tolist())
            if {0, 1}.issubset(labs):
                eligible.append(action)
        eligible = sorted(eligible)
    if not eligible:
        raise ValueError("No eligible unseen actions found. Try disabling --require_unseen_both_labels.")
    if unseen_action_count > 0:
        n_unseen = int(unseen_action_count)
    else:
        if unseen_action_fraction <= 0:
            raise ValueError(
                "No unseen actions specified. Provide --unseen_actions, --unseen_actions_file, "
                "--unseen_action_count, or set --unseen_action_fraction > 0."
            )
        n_unseen = int(round(len(eligible) * unseen_action_fraction))
        n_unseen = max(1, n_unseen)
    n_unseen = min(n_unseen, max(1, len(eligible) - 1))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(np.asarray(eligible, dtype=object), size=n_unseen, replace=False).tolist())
    return chosen
def select_unseen_actors(
    df: pd.DataFrame,
    actor_norm_col: str,
    label_col: str,
    seed: int,
    unseen_actors: Optional[Sequence[str]] = None,
    unseen_actor_file: str = "",
    unseen_actor_count: int = 0,
    unseen_actor_fraction: float = 0.0,
    require_both_labels: bool = True,
) -> List[str]:
    """Choose full actor IDs that will be excluded from train/val."""
    if actor_norm_col not in df.columns:
        raise ValueError(f"Column {actor_norm_col!r} not found while selecting unseen actors.")
    all_actors = sorted(df[actor_norm_col].dropna().map(normalize_actor_id).unique().tolist())
    requested: List[str] = []
    if unseen_actors:
        requested.extend([normalize_actor_id(a) for a in unseen_actors if str(a).strip()])
    if unseen_actor_file:
        requested.extend(load_unseen_actor_ids_from_file(unseen_actor_file))
    if requested:
        requested = sorted(set(requested))
        missing = sorted(set(requested) - set(all_actors))
        if missing:
            raise ValueError(
                f"Requested unseen actors were not found after normalization: {missing}. "
                f"Available actors: {all_actors}"
            )
        return requested
    if unseen_actor_count <= 0 and unseen_actor_fraction <= 0:
        return []
    eligible = all_actors
    if require_both_labels:
        eligible = []
        for actor, g in df.groupby(actor_norm_col):
            labs = set(g[label_col].astype(int).tolist())
            if {0, 1}.issubset(labs):
                eligible.append(actor)
        eligible = sorted(eligible)
    if not eligible:
        raise ValueError("No eligible unseen actors found. Try --allow_unseen_actor_single_label.")
    if unseen_actor_count > 0:
        n_unseen = int(unseen_actor_count)
    else:
        n_unseen = int(round(len(eligible) * unseen_actor_fraction))
        n_unseen = max(1, n_unseen)
    n_unseen = min(n_unseen, max(1, len(eligible) - 1))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.asarray(eligible, dtype=object), size=n_unseen, replace=False).tolist())
def safe_balance_binary_split(
    df: pd.DataFrame,
    label_col: str,
    seed: int,
    split_name: str,
    allow_single_label: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Balance a split only when it contains both labels."""
    n_normal = int((df[label_col].astype(int) == 0).sum()) if len(df) else 0
    n_anomaly = int((df[label_col].astype(int) == 1).sum()) if len(df) else 0
    if len(df) == 0:
        return df.reset_index(drop=True), {
            "split_name": split_name,
            "balanced": False,
            "reason": "empty split",
            "n_samples": 0,
            "n_normal": 0,
            "n_anomaly": 0,
        }
    if n_normal == 0 or n_anomaly == 0:
        if allow_single_label:
            return df.reset_index(drop=True), {
                "split_name": split_name,
                "balanced": False,
                "reason": "single-label split; AUROC/AUPRC are not meaningful",
                "n_samples": int(len(df)),
                "n_normal": n_normal,
                "n_anomaly": n_anomaly,
            }
        raise ValueError(
            f"Cannot balance {split_name}: needs both labels, got normal={n_normal}, anomaly={n_anomaly}."
        )
    return balance_binary_split(df, label_col=label_col, seed=seed, split_name=split_name)
def metrics_without_ranking_for_single_label_style(eval_out: EvalOutput, threshold: float, split_name: str) -> Dict[str, Any]:
    """For unseen-style tests, do not report AUROC/AUPRC because Healthy is not held out as an unseen style."""
    base = classification_metrics_at_threshold(eval_out.y_true, eval_out.score, threshold)
    base.update({
        "auroc": None,
        "auprc": None,
        "ranking_metrics_reported": False,
        "ranking_metrics_reason": (
            "Unseen style/condition is evaluated as a held-out flawed condition. Healthy is the normal class and is not treated as an unseen style, "
            "so this split can be single-label and AUROC/AUPRC are not meaningful."
        ),
        "split_name": split_name,
        "n_samples": int(len(eval_out.y_true)),
        "n_normal": int((eval_out.y_true == 0).sum()),
        "n_anomaly": int((eval_out.y_true == 1).sum()),
        "score_mean": float(np.mean(eval_out.score)) if len(eval_out.score) else float("nan"),
        "score_std": float(np.std(eval_out.score)) if len(eval_out.score) else float("nan"),
        "threshold_source": "validation_threshold",
    })
    return base
def fallback_validation_split_if_needed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create a usable validation split from the seen train pool if grouped splitting made val empty.
    This function only operates on the already-seen pool returned by the split function,
    so it cannot introduce leakage from held-out action/style/actor groups.
    It is applied when the grouped validation split is empty or lacks one of the two labels,
    because validation is used for threshold selection and best-checkpoint selection.
    """
    def _label_counts(frame: pd.DataFrame) -> Dict[int, int]:
        if len(frame) == 0:
            return {}
        return {int(k): int(v) for k, v in frame[label_col].astype(int).value_counts().to_dict().items()}
    original_info = {
        "applied": False,
        "reason": "validation split already contains both labels",
        "train_before": int(len(train_df)),
        "val_before": int(len(val_df)),
        "train_label_counts_before": _label_counts(train_df),
        "val_label_counts_before": _label_counts(val_df),
    }
    if len(val_df) > 0 and set(val_df[label_col].astype(int).unique()).issuperset({0, 1}):
        original_info.update({
            "train_after": int(len(train_df)),
            "val_after": int(len(val_df)),
            "train_label_counts_after": _label_counts(train_df),
            "val_label_counts_after": _label_counts(val_df),
        })
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), original_info
    if val_fraction <= 0:
        raise ValueError(
            "Validation split is empty or single-label, and --val_fraction <= 0. "
            "Set --val_fraction to a positive value, e.g. 0.10."
        )
    pool_df = pd.concat([train_df, val_df], axis=0).sample(frac=1.0, random_state=seed + 901).reset_index(drop=True)
    labels = pool_df[label_col].astype(int)
    n_normal = int((labels == 0).sum())
    n_anomaly = int((labels == 1).sum())
    if n_normal < 2 or n_anomaly < 2:
        raise ValueError(
            "Cannot create fallback validation split with both labels while leaving train with both labels. "
            f"Seen pool label counts: normal={n_normal}, anomaly={n_anomaly}."
        )
    target_val = max(2, int(round(len(pool_df) * val_fraction)))
    target_per_label = max(1, target_val // 2)
    # Leave at least one sample of each label in train.
    n_each = min(target_per_label, n_normal - 1, n_anomaly - 1)
    normal_idx = pool_df[labels == 0].sample(n=n_each, replace=False, random_state=seed + 902).index
    anomaly_idx = pool_df[labels == 1].sample(n=n_each, replace=False, random_state=seed + 903).index
    val_idx = normal_idx.union(anomaly_idx)
    new_val_df = pool_df.loc[val_idx].sample(frac=1.0, random_state=seed + 904).reset_index(drop=True)
    new_train_df = pool_df.drop(index=val_idx).sample(frac=1.0, random_state=seed + 905).reset_index(drop=True)
    info = dict(original_info)
    info.update({
        "applied": True,
        "reason": "grouped validation split was empty or did not contain both labels; fallback validation was sampled from seen train/val pool",
        "n_per_label_moved_to_val": int(n_each),
        "train_after": int(len(new_train_df)),
        "val_after": int(len(new_val_df)),
        "train_label_counts_after": _label_counts(new_train_df),
        "val_label_counts_after": _label_counts(new_val_df),
        "leakage_note": "fallback uses only the seen pool after action/style/actor holdout; held-out samples are not moved into validation",
    })
    print(f"[INFO] Applied validation fallback: {info}")
    return new_train_df, new_val_df, info
def true_unseen_action_style_actor_split(
    df: pd.DataFrame,
    action_norm_col: str,
    style_norm_col: str,
    actor_norm_col: str,
    condition_col: str,
    label_col: str,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    unseen_actions: Optional[Sequence[str]],
    unseen_action_file: str,
    unseen_action_count: int,
    unseen_action_fraction: float,
    require_unseen_both_labels: bool,
    unseen_styles: Optional[Sequence[str]],
    unseen_styles_file: str,
    unseen_style_count: int,
    unseen_style_fraction: float,
    require_unseen_style_both_labels: bool,
    unseen_actors: Optional[Sequence[str]],
    unseen_actors_file: str,
    unseen_actor_count: int,
    unseen_actor_fraction: float,
    require_unseen_actor_both_labels: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], List[str], Dict[str, Any]]:
    """Split with truly held-out action/content, style/condition, and actor IDs.
    Train/val are created only from samples with seen action + seen style + seen actor.
    The test buckets are:
      - seen_test_df: seen action + seen style + seen actor
      - unseen_action_test_df: unseen action + seen style + seen actor
      - unseen_style_test_df: seen action + unseen style + seen actor
      - unseen_actor_test_df: seen action + seen style + unseen actor
      - unseen_all_test_df: unseen action + unseen style + unseen actor
      - other_unseen_combinations_df: all two-way unseen combinations
      - combined_test_df: all test buckets above combined
    """
    heldout_actions = select_unseen_actions(
        df=df,
        action_norm_col=action_norm_col,
        label_col=label_col,
        seed=seed,
        unseen_actions=unseen_actions,
        unseen_action_file=unseen_action_file,
        unseen_action_count=unseen_action_count,
        unseen_action_fraction=unseen_action_fraction,
        require_both_labels=require_unseen_both_labels,
    )
    heldout_styles = select_unseen_values(
        df=df,
        norm_col=style_norm_col,
        label_col=label_col,
        seed=seed + 17,
        requested_values=unseen_styles,
        requested_file=unseen_styles_file,
        requested_count=unseen_style_count,
        requested_fraction=unseen_style_fraction,
        require_both_labels=require_unseen_style_both_labels,
        value_name="style",
    )
    heldout_actors = select_unseen_actors(
        df=df,
        actor_norm_col=actor_norm_col,
        label_col=label_col,
        seed=seed + 31,
        unseen_actors=unseen_actors,
        unseen_actor_file=unseen_actors_file,
        unseen_actor_count=unseen_actor_count,
        unseen_actor_fraction=unseen_actor_fraction,
        require_both_labels=require_unseen_actor_both_labels,
    )
    unseen_action_set = set(heldout_actions)
    unseen_style_set = set(heldout_styles)
    unseen_actor_set = set(heldout_actors)
    action_unseen = df[action_norm_col].isin(unseen_action_set)
    style_unseen = df[style_norm_col].isin(unseen_style_set)
    actor_unseen = df[actor_norm_col].isin(unseen_actor_set)
    seen_mask = (~action_unseen) & (~style_unseen) & (~actor_unseen)
    unseen_action_only_mask = action_unseen & (~style_unseen) & (~actor_unseen)
    unseen_style_only_mask = (~action_unseen) & style_unseen & (~actor_unseen)
    unseen_actor_only_mask = (~action_unseen) & (~style_unseen) & actor_unseen
    unseen_all_mask = action_unseen & style_unseen & actor_unseen
    other_unseen_combinations_mask = (
        (action_unseen | style_unseen | actor_unseen)
        & ~(unseen_action_only_mask | unseen_style_only_mask | unseen_actor_only_mask | unseen_all_mask)
    )
    seen_df = df[seen_mask].copy()
    unseen_action_test_df = df[unseen_action_only_mask].copy()
    unseen_style_test_df = df[unseen_style_only_mask].copy()
    unseen_actor_test_df = df[unseen_actor_only_mask].copy()
    unseen_all_test_df = df[unseen_all_mask].copy()
    other_unseen_combinations_df = df[other_unseen_combinations_mask].copy()
    if len(seen_df) == 0:
        raise ValueError("No seen data remains for training after action/style/actor holdout.")
    if not set(seen_df[label_col].astype(int).unique()).issuperset({0, 1}):
        raise ValueError("Seen training pool must contain both labels 0 and 1 after action/style/actor holdout.")
    if heldout_actions and len(unseen_action_test_df) == 0:
        print("[WARN] No isolated unseen-action rows remain. They may only exist in combined unseen buckets.")
    if heldout_styles and len(unseen_style_test_df) == 0:
        print("[WARN] No isolated unseen-style rows remain. They may only exist in combined unseen buckets.")
    if heldout_actors and len(unseen_actor_test_df) == 0:
        print("[WARN] No isolated unseen-actor rows remain. They may only exist in combined unseen buckets.")
    if heldout_actions and heldout_styles and heldout_actors and len(unseen_all_test_df) == 0:
        print("[WARN] No rows exist where action, style, and actor are all unseen at the same time.")
    stratify_cols = [action_norm_col, actor_norm_col, label_col]
    if condition_col in seen_df.columns:
        stratify_cols.insert(1, condition_col)
    train_df, val_df, seen_test_df = stratified_group_split(
        seen_df,
        stratify_cols=stratify_cols,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_df, val_df, validation_fallback_info = fallback_validation_split_if_needed(
        train_df=train_df,
        val_df=val_df,
        label_col=label_col,
        val_fraction=val_fraction,
        seed=seed,
    )
    test_parts = [seen_test_df]
    for part in [unseen_action_test_df, unseen_style_test_df, unseen_actor_test_df, unseen_all_test_df, other_unseen_combinations_df]:
        if len(part) > 0:
            test_parts.append(part)
    combined_test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
    return (
        train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        val_df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True),
        seen_test_df.sample(frac=1.0, random_state=seed + 5).reset_index(drop=True),
        unseen_action_test_df.sample(frac=1.0, random_state=seed + 6).reset_index(drop=True),
        unseen_style_test_df.sample(frac=1.0, random_state=seed + 7).reset_index(drop=True),
        unseen_actor_test_df.sample(frac=1.0, random_state=seed + 8).reset_index(drop=True),
        unseen_all_test_df.sample(frac=1.0, random_state=seed + 9).reset_index(drop=True),
        other_unseen_combinations_df.sample(frac=1.0, random_state=seed + 10).reset_index(drop=True),
        combined_test_df,
        heldout_actions,
        heldout_styles,
        heldout_actors,
        validation_fallback_info,
    )
# -----------------------------
# MotionCLIP loading/freezing
# -----------------------------
def build_motionclip_encoder(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Build and load MotionCLIP exactly like finetune_unsupervised_updated.py.
    The encoder architecture is hardcoded to the paper MotionCLIP setup:
    rot6d, [60, 25, 6], latent_dim=512, 8 transformer layers.
    """
    encoder = Encoder_TRANSFORMER(
        modeltype="motionclip",
        njoints=25,
        nfeats=6,
        num_frames=60,
        num_classes=1,
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=[math.pi, 0.0, 0.0],
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    encoder_state = {}
    for k, v in ckpt.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys: {unexpected}")
    if missing:
        print("Warning: missing encoder keys:", missing)
    encoder = encoder.to(device)
    encoder.train()
    return encoder
def freeze_encoder_except_last_layers(encoder: nn.Module, num_trainable_blocks: int = 2) -> Dict[str, Any]:
    """
    Freeze everything except the last N transformer blocks and final norm,
    matching the logic in finetune_unsupervised_updated.py.
    """
    for p in encoder.parameters():
        p.requires_grad = False
    unfroze_any = False
    unfrozen_layer_indices: List[int] = []
    if hasattr(encoder, "seqTransEncoder"):
        seq_encoder = encoder.seqTransEncoder
        if hasattr(seq_encoder, "layers"):
            layers = seq_encoder.layers
            n = min(num_trainable_blocks, len(layers))
            start = len(layers) - n
            for i, layer in enumerate(layers[start:], start=start):
                for p in layer.parameters():
                    p.requires_grad = True
                unfrozen_layer_indices.append(i)
            unfroze_any = True
        if hasattr(seq_encoder, "norm") and seq_encoder.norm is not None:
            for p in seq_encoder.norm.parameters():
                p.requires_grad = True
    if not unfroze_any:
        print("Warning: could not find encoder.seqTransEncoder.layers; encoder may remain frozen.")
    trainable_names = [name for name, p in encoder.named_parameters() if p.requires_grad]
    return {
        "num_trainable_blocks_requested": int(num_trainable_blocks),
        "unfrozen_layer_indices": unfrozen_layer_indices,
        "num_trainable_params": int(sum(p.numel() for p in encoder.parameters() if p.requires_grad)),
        "num_total_params": int(sum(p.numel() for p in encoder.parameters())),
        "trainable_param_names_first_100": trainable_names[:100],
    }
def encode_motion_auto(model: nn.Module, motion: torch.Tensor) -> torch.Tensor:
    """
    MotionCLIP-specific forward pass.
    Input from dataset is [B, T, J, F].
    MotionCLIP encoder expects batch['x'] as [B, J, F, T].
    """
    motion = motion.float()
    x = motion.permute(0, 2, 3, 1).contiguous()  # [B, 25, 6, 60]
    B, T = motion.shape[0], motion.shape[1]
    lengths = torch.full((B,), T, dtype=torch.long, device=motion.device)
    mask = torch.arange(T, device=motion.device).unsqueeze(0) < lengths.unsqueeze(1)
    batch = {
        "x": x,
        "y": torch.zeros(B, dtype=torch.long, device=motion.device),
        "lengths": lengths,
        "mask": mask,
    }
    out = model(batch)
    if not isinstance(out, dict) or "mu" not in out:
        raise RuntimeError("Expected MotionCLIP encoder output dict with key 'mu'.")
    return out["mu"]  # [B, 512]
# -----------------------------
# Text encoder and prompts
# -----------------------------
class FrozenCLIPTextEncoder:
    def __init__(self, clip_model_name: str, device: torch.device):
        try:
            import clip  # OpenAI CLIP package
        except ImportError as e:
            raise ImportError(
                "Could not import `clip`. Install OpenAI CLIP in your environment, e.g.:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            ) from e
        self.clip = clip
        self.model, _ = clip.load(clip_model_name, device=device)
        self.model = self.model.float()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 256) -> torch.Tensor:
        feats = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            tokens = self.clip.tokenize(chunk, truncate=True).to(self.device)
            f = self.model.encode_text(tokens).float()
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)
def build_prompt_cache(
    actions: Sequence[str],
    text_encoder: FrozenCLIPTextEncoder,
    normal_prompt_template: str,
    anomaly_prompt_template: str,
    device: torch.device,
) -> Tuple[Dict[str, int], torch.Tensor, Dict[str, Dict[str, str]]]:
    actions = sorted({normalize_action_text(a) for a in actions})
    action_to_idx = {a: i for i, a in enumerate(actions)}
    prompt_info: Dict[str, Dict[str, str]] = {}
    texts = []
    for action in actions:
        normal_prompt = normal_prompt_template.format(action=action)
        anomaly_prompt = anomaly_prompt_template.format(action=action)
        prompt_info[action] = {
            "normal": normal_prompt,
            "anomaly": anomaly_prompt,
        }
        texts.extend([normal_prompt, anomaly_prompt])
    text_feats = text_encoder.encode(texts)  # [2*A, D]
    text_feats = text_feats.reshape(len(actions), 2, -1).to(device)
    return action_to_idx, text_feats, prompt_info
# -----------------------------
# Metrics
# -----------------------------
def binary_auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Fallback AUROC using rank statistics.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average tied ranks
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for k, count in enumerate(counts):
            if count > 1:
                tied = inverse == k
                ranks[tied] = ranks[tied].mean()
    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)
def average_precision_fallback(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    order = np.argsort(-scores)
    y = y_true[order]
    total_pos = y.sum()
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / total_pos)
def classification_metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, criterion: str = "f1") -> Tuple[float, Dict[str, Any]]:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 0.0, {}
    # Candidate thresholds: unique scores plus ends
    candidates = np.unique(scores)
    if len(candidates) > 1000:
        candidates = np.quantile(scores, np.linspace(0, 1, 1000))
    best_thr = float(candidates[0])
    best_metrics = None
    best_value = -float("inf")
    for thr in candidates:
        m = classification_metrics_at_threshold(y_true, scores, float(thr))
        value = m.get(criterion, m["f1"])
        if value > best_value:
            best_value = value
            best_thr = float(thr)
            best_metrics = m
    assert best_metrics is not None
    return best_thr, best_metrics
def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    threshold_criterion: str = "f1",
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
        auprc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        auroc = binary_auc_rank(y_true, scores)
        auprc = average_precision_fallback(y_true, scores)
    if threshold is None:
        threshold, threshold_metrics = find_best_threshold(y_true, scores, threshold_criterion)
        threshold_source = f"best_{threshold_criterion}_on_this_split"
    else:
        threshold_metrics = classification_metrics_at_threshold(y_true, scores, threshold)
        threshold_source = "provided"
    return {
        "auroc": auroc,
        "auprc": auprc,
        "n_samples": int(len(y_true)),
        "n_normal": int((y_true == 0).sum()),
        "n_anomaly": int((y_true == 1).sum()),
        "score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
        "score_std": float(np.std(scores)) if len(scores) else float("nan"),
        "threshold_source": threshold_source,
        **threshold_metrics,
    }
# -----------------------------
# Train/evaluate
# -----------------------------
@dataclass
class EvalOutput:
    loss: float
    y_true: np.ndarray
    score: np.ndarray
    prob_anomaly: np.ndarray
    s_healthy: np.ndarray
    s_flawed: np.ndarray
    embeddings: np.ndarray
    paths: List[str]
    actions: List[str]
    row_indices: List[int]
def make_class_weights(labels: Sequence[int], device: torch.device, mode: str) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    labels_np = np.asarray(labels).astype(int)
    counts = np.bincount(labels_np, minlength=2).astype(float)
    weights = counts.sum() / np.maximum(1.0, 2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)
def logits_from_motion_and_prompts(
    motion_encoder: nn.Module,
    motion: torch.Tensor,
    actions: Sequence[str],
    action_to_idx: Dict[str, int],
    text_feats: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z = encode_motion_auto(motion_encoder, motion)
    z = F.normalize(z.float(), dim=-1)
    idx = torch.tensor([action_to_idx[normalize_action_text(a)] for a in actions], dtype=torch.long, device=motion.device)
    prompts = text_feats[idx]  # [B,2,D]
    if z.shape[-1] != prompts.shape[-1]:
        raise RuntimeError(
            f"Motion embedding dim ({z.shape[-1]}) does not match text embedding dim ({prompts.shape[-1]}). "
            "Check --latent_dim and CLIP model."
        )
    logits = torch.bmm(prompts, z.unsqueeze(-1)).squeeze(-1) / temperature  # [B,2]
    return logits, z
def target_text_embeddings_and_group_ids_for_batch(
    actions: Sequence[str],
    labels: torch.Tensor,
    action_to_idx: Dict[str, int],
    text_feats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return matching text embeddings and class ids for healthy/flawed-per-action groups."""
    idx = torch.tensor(
        [action_to_idx[normalize_action_text(a)] for a in actions],
        dtype=torch.long,
        device=labels.device,
    )
    labels = labels.long()
    target_text = text_feats[idx, labels]  # [B, D]
    group_ids = idx * 2 + labels           # same id = same action and same healthy/flawed label
    return target_text, group_ids
def symmetric_motion_text_contrastive_loss(
    motion_z: torch.Tensor,
    text_z: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Bidirectional supervised contrastive loss between motions and matching text prompts.
    Samples with the same action and healthy/flawed label are treated as positives.
    This avoids treating duplicate prompts within a batch as false negatives.
    """
    motion_z = F.normalize(motion_z.float(), dim=-1)
    text_z = F.normalize(text_z.float(), dim=-1)
    logits = motion_z @ text_z.t() / temperature  # [B, B]
    positive_mask = group_ids[:, None].eq(group_ids[None, :]).float()
    log_prob_m2t = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_m2t = -(positive_mask * log_prob_m2t).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1.0)
    log_prob_t2m = logits.t() - torch.logsumexp(logits.t(), dim=1, keepdim=True)
    loss_t2m = -(positive_mask.t() * log_prob_t2m).sum(dim=1) / positive_mask.t().sum(dim=1).clamp_min(1.0)
    return 0.5 * (loss_m2t.mean() + loss_t2m.mean())
def train_one_epoch(
    motion_encoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    action_to_idx: Dict[str, int],
    text_feats: torch.Tensor,
    temperature: float,
    grad_clip: float,
    use_amp: bool,
) -> float:
    motion_encoder.train()
    total_loss = 0.0
    total_n = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            z = encode_motion_auto(motion_encoder, motion)
            target_text, group_ids = target_text_embeddings_and_group_ids_for_batch(batch["action"], labels, action_to_idx, text_feats)
            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in motion_encoder.parameters() if p.requires_grad],
                grad_clip,
            )
        scaler.step(optimizer)
        scaler.update()
        bs = labels.numel()
        total_loss += float(loss.detach().cpu()) * bs
        total_n += bs
    return total_loss / max(1, total_n)
@torch.no_grad()
def evaluate(
    motion_encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_to_idx: Dict[str, int],
    text_feats: torch.Tensor,
    temperature: float,
    compute_contrastive_loss: bool = True,
) -> EvalOutput:
    motion_encoder.eval()
    losses = []
    y_true = []
    scores = []
    probs = []
    s_h = []
    s_f = []
    embeddings = []
    paths = []
    actions = []
    row_indices = []
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits, z = logits_from_motion_and_prompts(
            motion_encoder, motion, batch["action"], action_to_idx, text_feats, temperature
        )
        if compute_contrastive_loss:
            target_text, group_ids = target_text_embeddings_and_group_ids_for_batch(batch["action"], labels, action_to_idx, text_feats)
            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
            losses.append(float(loss.detach().cpu()) * labels.numel())
        soft = torch.softmax(logits, dim=-1)
        # anomaly score: positive means closer to flawed than healthy
        score = logits[:, 1] - logits[:, 0]
        embeddings.append(z.detach().cpu().numpy().astype(np.float32))
        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        scores.extend(score.cpu().numpy().astype(float).tolist())
        probs.extend(soft[:, 1].cpu().numpy().astype(float).tolist())
        s_h.extend(logits[:, 0].cpu().numpy().astype(float).tolist())
        s_f.extend(logits[:, 1].cpu().numpy().astype(float).tolist())
        paths.extend(batch["path"])
        actions.extend(batch["action"])
        row_indices.extend(batch["row_index"])
    total_n = max(1, len(y_true))
    avg_loss = sum(losses) / total_n if losses else float("nan")
    return EvalOutput(
        loss=avg_loss,
        y_true=np.asarray(y_true, dtype=int),
        score=np.asarray(scores, dtype=float),
        prob_anomaly=np.asarray(probs, dtype=float),
        s_healthy=np.asarray(s_h, dtype=float),
        s_flawed=np.asarray(s_f, dtype=float),
        embeddings=np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 512), dtype=np.float32),
        paths=paths,
        actions=actions,
        row_indices=row_indices,
    )
def save_predictions(eval_out: EvalOutput, path: str | Path, threshold: float) -> None:
    pred = (eval_out.score >= threshold).astype(int)
    out_df = pd.DataFrame({
        "row_index": eval_out.row_indices,
        "motion_path": eval_out.paths,
        "action": eval_out.actions,
        "y_true_is_anomaly": eval_out.y_true,
        "anomaly_score_flawed_minus_healthy": eval_out.score,
        "prob_anomaly": eval_out.prob_anomaly,
        "logit_healthy": eval_out.s_healthy,
        "logit_flawed": eval_out.s_flawed,
        "pred_is_anomaly": pred,
    })
    out_df.to_csv(path, index=False)
def save_embeddings(eval_out: EvalOutput, npz_path: str | Path, metadata_csv_path: str | Path, threshold: float) -> None:
    """Save normalized MotionCLIP embeddings and metadata for plotting/debugging.
    NPZ keys:
      embeddings: [N, D] normalized motion embeddings used for prompt scoring
      y_true: [N] 0=healthy, 1=flawed/anomaly
      score: [N] flawed_minus_healthy anomaly score
      prob_anomaly: [N] softmax probability for flawed/anomaly prompt
      logit_healthy/logit_flawed: [N] prompt logits
      row_index: [N] original CSV row index
      pred_is_anomaly: [N] prediction using the selected threshold
    """
    pred = (eval_out.score >= threshold).astype(int)
    np.savez_compressed(
        npz_path,
        embeddings=eval_out.embeddings.astype(np.float32),
        y_true=eval_out.y_true.astype(np.int64),
        score=eval_out.score.astype(np.float32),
        prob_anomaly=eval_out.prob_anomaly.astype(np.float32),
        logit_healthy=eval_out.s_healthy.astype(np.float32),
        logit_flawed=eval_out.s_flawed.astype(np.float32),
        row_index=np.asarray(eval_out.row_indices, dtype=np.int64),
        pred_is_anomaly=pred.astype(np.int64),
    )
    meta_df = pd.DataFrame({
        "embedding_index": np.arange(len(eval_out.row_indices), dtype=int),
        "row_index": eval_out.row_indices,
        "motion_path": eval_out.paths,
        "action": eval_out.actions,
        "y_true_is_anomaly": eval_out.y_true,
        "anomaly_score_flawed_minus_healthy": eval_out.score,
        "prob_anomaly": eval_out.prob_anomaly,
        "logit_healthy": eval_out.s_healthy,
        "logit_flawed": eval_out.s_flawed,
        "pred_is_anomaly": pred,
    })
    meta_df.to_csv(metadata_csv_path, index=False)
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MotionCLIP with healthy/flawed action prompts on PerMo.")
    # Data
    parser.add_argument("--csv_path", required=True, help="Path to PerMo metadata CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory where outputs are saved.")
    parser.add_argument("--path_col", default="motion_path")
    parser.add_argument("--action_col", default="action_label")
    parser.add_argument("--condition_col", default="condition_label")
    parser.add_argument("--actor_col", default="", help="Optional CSV column containing actor IDs. If omitted, actor is parsed from filename, e.g. *_A01_001.npz.")
    parser.add_argument("--label_col", default="is_anomaly")
    parser.add_argument("--motion_key", default="auto", help="NPZ key. Use 'auto' to infer.")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--njoints", type=int, default=25)
    parser.add_argument("--nfeats", type=int, default=6)
    # Split
    parser.add_argument("--test_fraction", type=float, default=0.20, help="Seen-pool test fraction. Held-out classes/actors are entirely test-only.")
    parser.add_argument("--val_fraction", type=float, default=0.10, help="Fraction of non-test seen-pool data used for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance_test_sets",
        action="store_true",
        default=True,
        help="Downsample majority label in rankable test sets so normal/anomaly counts are equal.",
    )
    parser.add_argument(
        "--no_balance_test_sets",
        action="store_false",
        dest="balance_test_sets",
        help="Keep natural label ratios in test sets.",
    )
    parser.add_argument("--unseen_actions", "--unseen_contents", nargs="*", default=[], help="Action/content names to hold out completely from training/validation.")
    parser.add_argument("--unseen_actions_file", "--unseen_contents_file", default="", help="Optional text file with one unseen action/content name per line.")
    parser.add_argument("--unseen_action_count", type=int, default=0, help="Randomly hold out this many action classes if --unseen_actions is not given.")
    parser.add_argument("--unseen_action_fraction", type=float, default=0.20, help="Randomly hold out this fraction of action classes if no explicit unseen actions/count are given.")
    parser.add_argument("--require_unseen_both_labels", action="store_true", default=True, help="Only randomly select unseen actions that contain both healthy and anomaly labels.")
    parser.add_argument("--allow_unseen_single_label", action="store_false", dest="require_unseen_both_labels", help="Allow unseen actions with only one label when randomly selecting unseen actions.")
    parser.add_argument("--unseen_styles", nargs="*", default=[], help="Style/condition names to hold out completely from training/validation. Do not include Healthy as an unseen style.")
    parser.add_argument("--unseen_styles_file", default="", help="Optional text file with one unseen style/condition name per line.")
    parser.add_argument("--unseen_style_count", type=int, default=0, help="Randomly hold out this many style/condition classes if --unseen_styles is not given.")
    parser.add_argument("--unseen_style_fraction", type=float, default=0.0, help="Randomly hold out this fraction of style/condition classes. Default 0 means no random style holdout.")
    parser.add_argument("--require_unseen_style_both_labels", action="store_true", default=True, help="Only randomly select unseen styles that contain both healthy and anomaly labels.")
    parser.add_argument("--allow_unseen_style_single_label", action="store_false", dest="require_unseen_style_both_labels", help="Allow unseen styles with only one label when randomly selecting unseen styles.")
    parser.add_argument("--unseen_actors", nargs="*", default=[], help="Actor IDs to hold out completely from training/validation, e.g. A01 A02.")
    parser.add_argument("--unseen_actors_file", default="", help="Optional text file with one unseen actor ID per line.")
    parser.add_argument("--unseen_actor_count", type=int, default=0, help="Randomly hold out this many actors if --unseen_actors is not given.")
    parser.add_argument("--unseen_actor_fraction", type=float, default=0.0, help="Randomly hold out this fraction of actors. Default 0 means no random actor holdout.")
    parser.add_argument("--require_unseen_actor_both_labels", action="store_true", default=True, help="Only randomly select unseen actors that contain both healthy and anomaly labels.")
    parser.add_argument("--allow_unseen_actor_single_label", action="store_false", dest="require_unseen_actor_both_labels", help="Allow unseen actors with only one label when randomly selecting unseen actors.")
    # MotionCLIP model
    parser.add_argument("--project_root", default="", help="Parent directory containing the MotionCLIP folder.")
    parser.add_argument("--checkpoint", required=True, help="Pretrained MotionCLIP checkpoint.")
    parser.add_argument("--trainable_layers", type=int, default=2)
    # Text prompts
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--normal_prompt_template", default="healthy {action}")
    parser.add_argument("--anomaly_prompt_template", default="flawed {action}")
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--class_weight", choices=["auto", "none"], default="auto", help="Kept for compatibility; contrastive training does not use class weights.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision.")
    parser.add_argument("--threshold_criterion", default="f1", choices=["f1", "balanced_accuracy", "accuracy"])
    args = parser.parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_json(vars(args), output_dir / "args.json")
    # Load metadata
    df = pd.read_csv(args.csv_path)
    df = df.copy()
    df["original_index"] = np.arange(len(df))
    required_cols = [args.path_col, args.action_col, args.label_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")
    # Normalize labels/actions/styles/actors
    df[args.label_col] = df[args.label_col].astype(int)
    if not set(df[args.label_col].unique()).issubset({0, 1}):
        raise ValueError(f"{args.label_col} must contain only 0/1 labels.")
    df["_action_norm"] = df[args.action_col].map(normalize_action_text)
    style_holdout_requested = bool(args.unseen_styles or args.unseen_styles_file or args.unseen_style_count > 0 or args.unseen_style_fraction > 0)
    if args.condition_col not in df.columns:
        if style_holdout_requested:
            raise ValueError(
                f"Style holdout was requested, but condition/style column {args.condition_col!r} is missing. Found: {list(df.columns)}"
            )
        df["_style_norm"] = "__no_condition_column__"
    else:
        df["_style_norm"] = df[args.condition_col].map(normalize_action_text)
    if args.actor_col:
        if args.actor_col not in df.columns:
            raise ValueError(f"--actor_col {args.actor_col!r} was provided but is missing from CSV. Found: {list(df.columns)}")
        df["_actor_norm"] = df[args.actor_col].map(normalize_actor_id)
    else:
        df["_actor_norm"] = df[args.path_col].map(extract_actor_id_from_path)
    # Basic file existence check
    missing_paths = [p for p in df[args.path_col].head(20).tolist() if not Path(str(p)).exists()]
    if missing_paths:
        print("[WARN] Some example motion files do not exist from this machine.")
        print("       This is okay only if you are testing the script outside the data machine.")
        print(f"       First missing example: {missing_paths[0]}")
    (
        train_df,
        val_df,
        seen_test_df,
        unseen_action_test_df,
        unseen_style_test_df,
        unseen_actor_test_df,
        unseen_all_test_df,
        other_unseen_combinations_df,
        test_df,
        heldout_unseen_actions,
        heldout_unseen_styles,
        heldout_unseen_actors,
        validation_fallback_info,
    ) = true_unseen_action_style_actor_split(
        df=df,
        action_norm_col="_action_norm",
        style_norm_col="_style_norm",
        actor_norm_col="_actor_norm",
        condition_col=args.condition_col,
        label_col=args.label_col,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        unseen_actions=args.unseen_actions,
        unseen_action_file=args.unseen_actions_file,
        unseen_action_count=args.unseen_action_count,
        unseen_action_fraction=args.unseen_action_fraction,
        require_unseen_both_labels=args.require_unseen_both_labels,
        unseen_styles=args.unseen_styles,
        unseen_styles_file=args.unseen_styles_file,
        unseen_style_count=args.unseen_style_count,
        unseen_style_fraction=args.unseen_style_fraction,
        require_unseen_style_both_labels=args.require_unseen_style_both_labels,
        unseen_actors=args.unseen_actors,
        unseen_actors_file=args.unseen_actors_file,
        unseen_actor_count=args.unseen_actor_count,
        unseen_actor_fraction=args.unseen_actor_fraction,
        require_unseen_actor_both_labels=args.require_unseen_actor_both_labels,
    )
    # Hard leakage checks: no held-out action/style/actor may appear in train/val.
    train_actions = set(train_df["_action_norm"].unique().tolist())
    val_actions = set(val_df["_action_norm"].unique().tolist())
    train_styles = set(train_df["_style_norm"].unique().tolist())
    val_styles = set(val_df["_style_norm"].unique().tolist())
    train_actors = set(train_df["_actor_norm"].unique().tolist())
    val_actors = set(val_df["_actor_norm"].unique().tolist())
    action_leakage = sorted((train_actions | val_actions) & set(heldout_unseen_actions))
    style_leakage = sorted((train_styles | val_styles) & set(heldout_unseen_styles))
    actor_leakage = sorted((train_actors | val_actors) & set(heldout_unseen_actors))
    if action_leakage:
        raise RuntimeError(f"Unseen action leakage detected in train/val: {action_leakage}")
    if style_leakage:
        raise RuntimeError(f"Unseen style leakage detected in train/val: {style_leakage}")
    if actor_leakage:
        raise RuntimeError(f"Unseen actor leakage detected in train/val: {actor_leakage}")
    test_balance_info = {
        "enabled": bool(args.balance_test_sets),
        "note": "If enabled, rankable splits with both labels are downsampled with a fixed seed so normal/anomaly counts are equal. Unseen-style ranking metrics are not reported.",
    }
    if args.balance_test_sets:
        seen_test_df, seen_balance_info = safe_balance_binary_split(
            seen_test_df, args.label_col, args.seed + 100, "seen_action_seen_style_seen_actor_test"
        )
        unseen_action_test_df, unseen_action_balance_info = safe_balance_binary_split(
            unseen_action_test_df, args.label_col, args.seed + 200, "unseen_action_only_test", allow_single_label=True
        )
        unseen_actor_test_df, unseen_actor_balance_info = safe_balance_binary_split(
            unseen_actor_test_df, args.label_col, args.seed + 300, "unseen_actor_only_test", allow_single_label=True
        )
        unseen_all_test_df, unseen_all_balance_info = safe_balance_binary_split(
            unseen_all_test_df, args.label_col, args.seed + 350, "unseen_action_style_actor_test", allow_single_label=True
        )
        other_unseen_combinations_df, other_balance_info = safe_balance_binary_split(
            other_unseen_combinations_df, args.label_col, args.seed + 375, "other_unseen_combinations_test", allow_single_label=True
        )
        # Do not balance unseen-style as a binary split: held-out flawed styles can be anomaly-only by design.
        unseen_style_balance_info = {
            "split_name": "unseen_style_only_test",
            "balanced": False,
            "reason": "Unseen styles exclude Healthy, so this split may contain anomaly samples only; AUROC/AUPRC are not reported.",
            "n_samples": int(len(unseen_style_test_df)),
            "n_normal": int((unseen_style_test_df[args.label_col].astype(int) == 0).sum()) if len(unseen_style_test_df) else 0,
            "n_anomaly": int((unseen_style_test_df[args.label_col].astype(int) == 1).sum()) if len(unseen_style_test_df) else 0,
        }
        test_parts = [seen_test_df]
        for part in [unseen_action_test_df, unseen_style_test_df, unseen_actor_test_df, unseen_all_test_df, other_unseen_combinations_df]:
            if len(part) > 0:
                test_parts.append(part)
        test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=args.seed + 400).reset_index(drop=True)
        test_df, combined_balance_info = safe_balance_binary_split(
            test_df, args.label_col, args.seed + 500, "combined_test", allow_single_label=True
        )
        test_balance_info.update({
            "seen_action_seen_style_seen_actor_test": seen_balance_info,
            "unseen_action_only_test": unseen_action_balance_info,
            "unseen_style_only_test": unseen_style_balance_info,
            "unseen_actor_only_test": unseen_actor_balance_info,
            "unseen_action_style_actor_test": unseen_all_balance_info,
            "other_unseen_combinations_test": other_balance_info,
            "combined_test": combined_balance_info,
        })
    # Save split CSVs
    train_df.to_csv(output_dir / "split_train.csv", index=False)
    val_df.to_csv(output_dir / "split_val.csv", index=False)
    seen_test_df.to_csv(output_dir / "split_test_seen_action_seen_style_seen_actor.csv", index=False)
    unseen_action_test_df.to_csv(output_dir / "split_test_unseen_actions.csv", index=False)
    unseen_style_test_df.to_csv(output_dir / "split_test_unseen_styles.csv", index=False)
    unseen_actor_test_df.to_csv(output_dir / "split_test_unseen_actors.csv", index=False)
    unseen_all_test_df.to_csv(output_dir / "split_test_unseen_action_style_actor.csv", index=False)
    other_unseen_combinations_df.to_csv(output_dir / "split_test_other_unseen_combinations.csv", index=False)
    test_df.to_csv(output_dir / "split_test_combined.csv", index=False)
    test_df.to_csv(output_dir / "split_test.csv", index=False)
    split_summary = {
        "split_type": "true_unseen_action_style_actor",
        "total": int(len(df)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "seen_action_seen_style_seen_actor_test": int(len(seen_test_df)),
        "unseen_action_only_test": int(len(unseen_action_test_df)),
        "unseen_style_only_test": int(len(unseen_style_test_df)),
        "unseen_actor_only_test": int(len(unseen_actor_test_df)),
        "unseen_action_style_actor_test": int(len(unseen_all_test_df)),
        "other_unseen_combinations_test": int(len(other_unseen_combinations_df)),
        "combined_test": int(len(test_df)),
        "train_label_counts": train_df[args.label_col].value_counts().to_dict(),
        "val_label_counts": val_df[args.label_col].value_counts().to_dict(),
        "seen_test_label_counts": seen_test_df[args.label_col].value_counts().to_dict(),
        "unseen_action_test_label_counts": unseen_action_test_df[args.label_col].value_counts().to_dict(),
        "unseen_style_test_label_counts": unseen_style_test_df[args.label_col].value_counts().to_dict(),
        "unseen_actor_test_label_counts": unseen_actor_test_df[args.label_col].value_counts().to_dict(),
        "unseen_action_style_actor_test_label_counts": unseen_all_test_df[args.label_col].value_counts().to_dict(),
        "other_unseen_combinations_label_counts": other_unseen_combinations_df[args.label_col].value_counts().to_dict(),
        "combined_test_label_counts": test_df[args.label_col].value_counts().to_dict(),
        "test_balance_info": test_balance_info,
        "all_actions": sorted(df["_action_norm"].unique().tolist()),
        "all_styles": sorted(df["_style_norm"].unique().tolist()),
        "all_actors": sorted(df["_actor_norm"].unique().tolist()),
        "seen_train_val_actions": sorted((train_actions | val_actions)),
        "seen_train_val_styles": sorted((train_styles | val_styles)),
        "seen_train_val_actors": sorted((train_actors | val_actors)),
        "heldout_unseen_actions": heldout_unseen_actions,
        "heldout_unseen_styles": heldout_unseen_styles,
        "heldout_unseen_actors": heldout_unseen_actors,
        "n_heldout_unseen_actions": int(len(heldout_unseen_actions)),
        "n_heldout_unseen_styles": int(len(heldout_unseen_styles)),
        "n_heldout_unseen_actors": int(len(heldout_unseen_actors)),
        "unseen_action_leakage_check_passed": True,
        "unseen_style_leakage_check_passed": True,
        "unseen_actor_leakage_check_passed": True,
        "unseen_style_ranking_metrics_reported": False,
        "validation_fallback_info": validation_fallback_info,
    }
    save_json(split_summary, output_dir / "split_summary.json")
    print("[INFO] Split summary:", split_summary)
    expected_shape = (args.num_frames, args.njoints, args.nfeats)
    train_ds = PerMoMotionDataset(train_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
    val_ds = PerMoMotionDataset(val_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
    test_ds = PerMoMotionDataset(test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
    seen_test_ds = PerMoMotionDataset(seen_test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
    unseen_action_test_ds = PerMoMotionDataset(unseen_action_test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
    unseen_style_test_ds = PerMoMotionDataset(unseen_style_test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
    unseen_actor_test_ds = PerMoMotionDataset(unseen_actor_test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
    unseen_all_test_ds = PerMoMotionDataset(unseen_all_test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
    other_unseen_combinations_ds = PerMoMotionDataset(other_unseen_combinations_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
    train_labels = train_df[args.label_col].astype(int).to_numpy()
    train_batch_sampler = BalancedBinaryBatchSampler(labels=train_labels, batch_size=args.batch_size, seed=args.seed, drop_last=False)
    sampler_info = {
        "type": "BalancedBinaryBatchSampler",
        "purpose": "class-aware healthy/flawed training batches",
        "batch_size": int(args.batch_size),
        "healthy_per_batch": int(train_batch_sampler.n0),
        "flawed_per_batch": int(train_batch_sampler.n1),
        "num_batches_per_epoch": int(len(train_batch_sampler)),
        "train_label_counts": {"healthy_0": int((train_labels == 0).sum()), "flawed_1": int((train_labels == 1).sum())},
    }
    save_json(sampler_info, output_dir / "train_sampler_info.json")
    print("[INFO] Train sampler info:", sampler_info)
    def make_loader(ds: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
    train_eval_loader = make_loader(train_ds)
    val_loader = make_loader(val_ds)
    test_loader = make_loader(test_ds)
    seen_test_loader = make_loader(seen_test_ds)
    unseen_action_test_loader = make_loader(unseen_action_test_ds)
    unseen_style_test_loader = make_loader(unseen_style_test_ds)
    unseen_actor_test_loader = make_loader(unseen_actor_test_ds)
    unseen_all_test_loader = make_loader(unseen_all_test_ds)
    other_unseen_combinations_loader = make_loader(other_unseen_combinations_ds)
    # Model
    if args.project_root:
        sys.path.insert(0, str(Path(args.project_root).resolve()))
    global Encoder_TRANSFORMER
    from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER
    motion_encoder = build_motionclip_encoder(args.checkpoint, device)
    save_json({"loaded": True, "checkpoint_path": args.checkpoint, "loader": "MotionCLIP encoder loader from finetune_unsupervised_updated.py"}, output_dir / "checkpoint_load_info.json")
    unfreeze_info = freeze_encoder_except_last_layers(motion_encoder, num_trainable_blocks=args.trainable_layers)
    save_json(unfreeze_info, output_dir / "unfreeze_info.json")
    print("[INFO] Unfreeze info:", unfreeze_info)
    # Text features
    print("[INFO] Loading frozen CLIP text encoder...")
    text_encoder = FrozenCLIPTextEncoder(args.clip_model, device)
    all_actions = df[args.action_col].map(normalize_action_text).tolist()
    action_to_idx, text_feats, prompt_info = build_prompt_cache(all_actions, text_encoder, args.normal_prompt_template, args.anomaly_prompt_template, device)
    save_json(prompt_info, output_dir / "prompts.json")
    torch.save({"action_to_idx": action_to_idx, "text_feats": text_feats.detach().cpu(), "prompt_info": prompt_info}, output_dir / "text_prompt_cache.pt")
    # Loss/optim
    class_weights = make_class_weights(train_df[args.label_col].tolist(), device, args.class_weight)
    if class_weights is not None:
        print("[INFO] Class weights [healthy, flawed]:", class_weights.detach().cpu().tolist())
        print("[INFO] Contrastive training uses motion-text positives grouped by action and healthy/flawed label; class weights are saved but not applied to the loss.")
    optimizer = torch.optim.AdamW([p for p in motion_encoder.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    best_val_auroc = -float("inf")
    best_epoch = -1
    best_threshold = 0.0
    epoch_records = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_batch_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            motion_encoder=motion_encoder,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            action_to_idx=action_to_idx,
            text_feats=text_feats,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
        )
        val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, text_feats, args.temperature, True)
        val_metrics = compute_binary_metrics(val_out.y_true, val_out.score, threshold=None, threshold_criterion=args.threshold_criterion)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_out.loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_threshold": val_metrics["threshold"],
            "seconds": time.time() - t0,
        }
        epoch_records.append(record)
        save_training_curves(epoch_records, output_dir)
        print(f"[EPOCH {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_out.loss:.4f} val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f} val_f1={val_metrics['f1']:.4f} thr={val_metrics['threshold']:.4f}")
        current_auroc = val_metrics["auroc"]
        if np.isfinite(current_auroc) and current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_epoch = epoch
            best_threshold = float(val_metrics["threshold"])
            torch.save({
                "epoch": epoch,
                "motion_encoder_state_dict": motion_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "action_to_idx": action_to_idx,
                "prompt_info": prompt_info,
                "text_feats": text_feats.detach().cpu(),
                "best_val_metrics": val_metrics,
                "unfreeze_info": unfreeze_info,
            }, ckpt_dir / "best_model.pt")
            save_predictions(val_out, output_dir / "val_predictions_best.csv", threshold=best_threshold)
    # Load best model for final test inference
    best_ckpt_path = ckpt_dir / "best_model.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        motion_encoder.load_state_dict(best_ckpt["motion_encoder_state_dict"], strict=True)
        best_threshold = float(best_ckpt["best_val_metrics"]["threshold"])
    else:
        print("[WARN] No best checkpoint saved. Testing final epoch model.")
        best_threshold = 0.0
    # Final eval
    train_out = evaluate(motion_encoder, train_eval_loader, device, action_to_idx, text_feats, args.temperature, True)
    val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, text_feats, args.temperature, True)
    test_out = evaluate(motion_encoder, test_loader, device, action_to_idx, text_feats, args.temperature, True)
    seen_test_out = evaluate(motion_encoder, seen_test_loader, device, action_to_idx, text_feats, args.temperature, True)
    unseen_action_test_out = evaluate(motion_encoder, unseen_action_test_loader, device, action_to_idx, text_feats, args.temperature, True)
    unseen_style_test_out = evaluate(motion_encoder, unseen_style_test_loader, device, action_to_idx, text_feats, args.temperature, True)
    unseen_actor_test_out = evaluate(motion_encoder, unseen_actor_test_loader, device, action_to_idx, text_feats, args.temperature, True)
    unseen_all_test_out = evaluate(motion_encoder, unseen_all_test_loader, device, action_to_idx, text_feats, args.temperature, True)
    other_unseen_combinations_out = evaluate(motion_encoder, other_unseen_combinations_loader, device, action_to_idx, text_feats, args.temperature, True)
    train_metrics = compute_binary_metrics(train_out.y_true, train_out.score, threshold=best_threshold)
    val_metrics_final = compute_binary_metrics(val_out.y_true, val_out.score, threshold=best_threshold)
    test_metrics = compute_binary_metrics(test_out.y_true, test_out.score, threshold=best_threshold)
    seen_test_metrics = compute_binary_metrics(seen_test_out.y_true, seen_test_out.score, threshold=best_threshold)
    unseen_action_test_metrics = compute_binary_metrics(unseen_action_test_out.y_true, unseen_action_test_out.score, threshold=best_threshold)
    unseen_actor_test_metrics = compute_binary_metrics(unseen_actor_test_out.y_true, unseen_actor_test_out.score, threshold=best_threshold)
    unseen_all_test_metrics = compute_binary_metrics(unseen_all_test_out.y_true, unseen_all_test_out.score, threshold=best_threshold)
    other_unseen_combinations_metrics = compute_binary_metrics(other_unseen_combinations_out.y_true, other_unseen_combinations_out.score, threshold=best_threshold)
    unseen_style_test_metrics = metrics_without_ranking_for_single_label_style(unseen_style_test_out, best_threshold, "unseen_style_only_test")
    for m, out in [
        (train_metrics, train_out),
        (val_metrics_final, val_out),
        (test_metrics, test_out),
        (seen_test_metrics, seen_test_out),
        (unseen_action_test_metrics, unseen_action_test_out),
        (unseen_style_test_metrics, unseen_style_test_out),
        (unseen_actor_test_metrics, unseen_actor_test_out),
        (unseen_all_test_metrics, unseen_all_test_out),
        (other_unseen_combinations_metrics, other_unseen_combinations_out),
    ]:
        m["loss"] = out.loss
    # Save predictions
    save_predictions(train_out, output_dir / "train_predictions.csv", threshold=best_threshold)
    save_predictions(val_out, output_dir / "val_predictions.csv", threshold=best_threshold)
    save_predictions(test_out, output_dir / "test_predictions_combined.csv", threshold=best_threshold)
    save_predictions(seen_test_out, output_dir / "test_predictions_seen_action_seen_style_seen_actor.csv", threshold=best_threshold)
    save_predictions(unseen_action_test_out, output_dir / "test_predictions_unseen_actions.csv", threshold=best_threshold)
    save_predictions(unseen_style_test_out, output_dir / "test_predictions_unseen_styles.csv", threshold=best_threshold)
    save_predictions(unseen_actor_test_out, output_dir / "test_predictions_unseen_actors.csv", threshold=best_threshold)
    save_predictions(unseen_all_test_out, output_dir / "test_predictions_unseen_action_style_actor.csv", threshold=best_threshold)
    save_predictions(other_unseen_combinations_out, output_dir / "test_predictions_other_unseen_combinations.csv", threshold=best_threshold)
    save_predictions(test_out, output_dir / "test_predictions.csv", threshold=best_threshold)
    # Save embeddings
    save_embeddings(train_out, output_dir / "train_embeddings.npz", output_dir / "train_embeddings_metadata.csv", threshold=best_threshold)
    save_embeddings(val_out, output_dir / "val_embeddings.npz", output_dir / "val_embeddings_metadata.csv", threshold=best_threshold)
    save_embeddings(test_out, output_dir / "test_embeddings_combined.npz", output_dir / "test_embeddings_combined_metadata.csv", threshold=best_threshold)
    save_embeddings(seen_test_out, output_dir / "test_embeddings_seen_action_seen_style_seen_actor.npz", output_dir / "test_embeddings_seen_action_seen_style_seen_actor_metadata.csv", threshold=best_threshold)
    save_embeddings(unseen_action_test_out, output_dir / "test_embeddings_unseen_actions.npz", output_dir / "test_embeddings_unseen_actions_metadata.csv", threshold=best_threshold)
    save_embeddings(unseen_style_test_out, output_dir / "test_embeddings_unseen_styles.npz", output_dir / "test_embeddings_unseen_styles_metadata.csv", threshold=best_threshold)
    save_embeddings(unseen_actor_test_out, output_dir / "test_embeddings_unseen_actors.npz", output_dir / "test_embeddings_unseen_actors_metadata.csv", threshold=best_threshold)
    save_embeddings(unseen_all_test_out, output_dir / "test_embeddings_unseen_action_style_actor.npz", output_dir / "test_embeddings_unseen_action_style_actor_metadata.csv", threshold=best_threshold)
    save_embeddings(other_unseen_combinations_out, output_dir / "test_embeddings_other_unseen_combinations.npz", output_dir / "test_embeddings_other_unseen_combinations_metadata.csv", threshold=best_threshold)
    save_embeddings(test_out, output_dir / "test_embeddings.npz", output_dir / "test_embeddings_metadata.csv", threshold=best_threshold)
    final_summary = {
        "best_epoch": best_epoch,
        "best_val_auroc_during_training": best_val_auroc,
        "threshold_selected_on_validation": best_threshold,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics_final,
        "test_metrics": test_metrics,
        "seen_action_seen_style_seen_actor_test_metrics": seen_test_metrics,
        "unseen_action_test_metrics": unseen_action_test_metrics,
        "unseen_style_test_metrics": unseen_style_test_metrics,
        "unseen_actor_test_metrics": unseen_actor_test_metrics,
        "unseen_action_style_actor_test_metrics": unseen_all_test_metrics,
        "other_unseen_combinations_test_metrics": other_unseen_combinations_metrics,
        "split_summary": split_summary,
        "prompt_templates": {"normal": args.normal_prompt_template, "anomaly": args.anomaly_prompt_template},
        "output_files": {
            "best_checkpoint": str(best_ckpt_path),
            "epoch_metrics": str(output_dir / "epoch_metrics.csv"),
            "test_predictions_combined": str(output_dir / "test_predictions_combined.csv"),
            "test_predictions_seen_action_seen_style_seen_actor": str(output_dir / "test_predictions_seen_action_seen_style_seen_actor.csv"),
            "test_predictions_unseen_actions": str(output_dir / "test_predictions_unseen_actions.csv"),
            "test_predictions_unseen_styles": str(output_dir / "test_predictions_unseen_styles.csv"),
            "test_predictions_unseen_actors": str(output_dir / "test_predictions_unseen_actors.csv"),
            "test_predictions_unseen_action_style_actor": str(output_dir / "test_predictions_unseen_action_style_actor.csv"),
            "test_predictions_other_unseen_combinations": str(output_dir / "test_predictions_other_unseen_combinations.csv"),
            "train_embeddings": str(output_dir / "train_embeddings.npz"),
            "val_embeddings": str(output_dir / "val_embeddings.npz"),
            "test_embeddings_combined": str(output_dir / "test_embeddings_combined.npz"),
            "test_embeddings_seen_action_seen_style_seen_actor": str(output_dir / "test_embeddings_seen_action_seen_style_seen_actor.npz"),
            "test_embeddings_unseen_actions": str(output_dir / "test_embeddings_unseen_actions.npz"),
            "test_embeddings_unseen_styles": str(output_dir / "test_embeddings_unseen_styles.npz"),
            "test_embeddings_unseen_actors": str(output_dir / "test_embeddings_unseen_actors.npz"),
            "test_embeddings_unseen_action_style_actor": str(output_dir / "test_embeddings_unseen_action_style_actor.npz"),
            "test_embeddings_other_unseen_combinations": str(output_dir / "test_embeddings_other_unseen_combinations.npz"),
            "metrics": str(output_dir / "metrics.json"),
            "training_history_npz": str(output_dir / "training_history.npz"),
            "loss_curves": str(output_dir / "loss_curves.png"),
            "validation_metrics_plot": str(output_dir / "validation_metrics.png"),
        },
    }
    save_json(final_summary, output_dir / "metrics.json")
    print("[DONE] Final combined test metrics:")
    print(json.dumps(test_metrics, indent=2, sort_keys=True))
    print("[DONE] Seen action/style/actor test metrics:")
    print(json.dumps(seen_test_metrics, indent=2, sort_keys=True))
    print("[DONE] Truly unseen-action-only test metrics:")
    print(json.dumps(unseen_action_test_metrics, indent=2, sort_keys=True))
    print("[DONE] Truly unseen-style-only test metrics; AUROC/AUPRC intentionally not reported:")
    print(json.dumps(unseen_style_test_metrics, indent=2, sort_keys=True))
    print("[DONE] Truly unseen-actor-only test metrics:")
    print(json.dumps(unseen_actor_test_metrics, indent=2, sort_keys=True))
    print("[DONE] Hard test where action, style, and actor are all unseen:")
    print(json.dumps(unseen_all_test_metrics, indent=2, sort_keys=True))
    print(f"[DONE] Outputs saved to: {output_dir}")
if __name__ == "__main__":
    main()
#
##!/usr/bin/env python3
#"""
#Fine-tune MotionCLIP for PerMo healthy-vs-flawed anomaly detection with truly unseen-action testing.
#What this script does:
#  1. Reads a metadata CSV with columns:
#       motion_path, action_label, condition_label, is_anomaly
#  1b. Holds out entire action/content classes from training/validation when requested.
#  1c. Optionally holds out entire style/condition classes from training/validation too.
#  2. Creates prompts per action:
#       normal class  : "healthy {action}"
#       anomaly class : "flawed {action}"
#  3. Freezes the CLIP text encoder.
#  4. Freezes most of MotionCLIP motion encoder and trains only the last N transformer layers.
#  5. Trains with CLIP-style supervised contrastive loss between motion embeddings
#     and the matching text prompt embedding for each sample.
#  6. Runs inference on seen-content, unseen-content, and optional unseen-style test sets using healthy/flawed prompt similarity.
#  7. Saves metrics, predictions, split CSVs, embeddings, training curves, and the best checkpoint.
#Example:
#  python finetune_permo_flawed_motionclip.py \
#    --csv_path /scratch/mgirishnair/Thesis/PerMo_metadata.csv \
#    --repo_root /scratch/mgirishnair/Thesis/MotionCLIP \
#    --checkpoint /scratch/mgirishnair/Thesis/MotionCLIP/checkpoints/babel60.pth.tar \
#    --output_dir /scratch/mgirishnair/Thesis/permo_flawed_runs/run1 \
#    --epochs 20 \
#    --batch_size 32 \
#    --lr 1e-5 \
#    --trainable_layers 2
#Notes:
#  - This script tries to be compatible with common MotionCLIP repo layouts, but you may need
#    to set --model_module if your Encoder_TRANSFORMER import path differs.
#  - If your .npz motion key is not auto-detected, pass --motion_key <key>.
#"""
#from __future__ import annotations
#import argparse
#import csv
#import json
#import math
#import os
#import re
#import random
#import sys
#import time
#from dataclasses import asdict, dataclass
#from pathlib import Path
#from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
#import numpy as np
#import pandas as pd
#import torch
#import torch.nn.functional as F
#from torch import nn
#from torch.utils.data import DataLoader, Dataset, Sampler
## -----------------------------
## General utilities
## -----------------------------
#def set_seed(seed: int) -> None:
#    random.seed(seed)
#    np.random.seed(seed)
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#def ensure_dir(path: str | Path) -> Path:
#    p = Path(path)
#    p.mkdir(parents=True, exist_ok=True)
#    return p
#def save_json(obj: Any, path: str | Path) -> None:
#    with open(path, "w", encoding="utf-8") as f:
#        json.dump(obj, f, indent=2, sort_keys=True)
#def save_training_curves(epoch_records: List[Dict[str, Any]], output_dir: str | Path) -> None:
#    """Save CSV/NPZ history and PNG plots for train/validation curves."""
#    if not epoch_records:
#        return
#    output_dir = Path(output_dir)
#    hist_df = pd.DataFrame(epoch_records)
#    hist_df.to_csv(output_dir / "epoch_metrics.csv", index=False)
#    np.savez(
#        output_dir / "training_history.npz",
#        **{col: hist_df[col].to_numpy() for col in hist_df.columns if pd.api.types.is_numeric_dtype(hist_df[col])},
#    )
#    try:
#        import matplotlib
#        matplotlib.use("Agg")
#        import matplotlib.pyplot as plt
#    except Exception as exc:
#        print(f"[WARN] Could not import matplotlib, skipping training curve plots: {exc}")
#        return
#    def _plot(columns: Sequence[str], filename: str, ylabel: str) -> None:
#        available = [c for c in columns if c in hist_df.columns]
#        if not available:
#            return
#        fig, ax = plt.subplots(figsize=(8, 5))
#        for c in available:
#            ax.plot(hist_df["epoch"], hist_df[c], marker="o", label=c)
#        ax.set_xlabel("epoch")
#        ax.set_ylabel(ylabel)
#        ax.grid(True, alpha=0.3)
#        ax.legend()
#        fig.tight_layout()
#        fig.savefig(output_dir / filename, dpi=160)
#        plt.close(fig)
#    _plot(["train_loss", "val_loss"], "loss_curves.png", "loss")
#    _plot(["val_auroc", "val_auprc", "val_f1", "val_balanced_accuracy"], "validation_metrics.png", "metric")
#def normalize_action_text(text: str) -> str:
#    text = str(text).strip().lower().replace("_", " ").replace("-", " ")
#    return " ".join(text.split())
#def normalize_actor_id(text: str) -> str:
#    """Normalize actor IDs such as A01/a01 to A01."""
#    text = str(text).strip().upper()
#    return text
#def extract_actor_id_from_path(path: str | Path) -> str:
#    """Extract actor ID from filenames like Armaching_Hop_A01_001.npz."""
#    name = Path(str(path)).stem
#    match = re.search(r"(?:^|_)(A\d{2})(?:_|$)", name, flags=re.IGNORECASE)
#    if not match:
#        raise ValueError(
#            f"Could not extract actor ID from {path!r}. Expected filename pattern like '*_A01_001.npz'. "
#            "Pass --actor_col if the CSV already has an actor column."
#        )
#    return normalize_actor_id(match.group(1))
#def load_unseen_actor_ids_from_file(path: str | Path) -> List[str]:
#    path = Path(path)
#    actors: List[str] = []
#    with open(path, "r", encoding="utf-8") as f:
#        for raw in f:
#            line = raw.strip()
#            if not line or line.startswith("#"):
#                continue
#            actors.append(normalize_actor_id(line.split(",")[0]))
#    return actors
#def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#    out = {}
#    for k, v in state_dict.items():
#        if k.startswith("module."):
#            k = k[len("module."):]
#        out[k] = v
#    return out
#def find_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
#    """
#    Tries to extract a model state dict from common checkpoint formats.
#    """
#    if isinstance(obj, dict):
#        # Direct state dict: all or most values are tensors
#        tensor_values = sum(torch.is_tensor(v) for v in obj.values())
#        if tensor_values > 0 and tensor_values >= max(1, len(obj) // 2):
#            return obj
#        for key in ["state_dict", "model_state_dict", "model", "encoder", "motion_encoder", "net"]:
#            if key in obj:
#                maybe = obj[key]
#                if isinstance(maybe, dict):
#                    return find_state_dict(maybe)
#    raise ValueError(
#        "Could not find a state_dict in the checkpoint. "
#        "Inspect the checkpoint keys and adapt find_state_dict()."
#    )
## -----------------------------
## Dataset and splitting
## -----------------------------
#class PerMoMotionDataset(Dataset):
#    def __init__(
#        self,
#        df: pd.DataFrame,
#        path_col: str,
#        action_col: str,
#        label_col: str,
#        motion_key: str = "auto",
#        expected_shape: Tuple[int, int, int] = (60, 25, 6),
#    ) -> None:
#        self.df = df.reset_index(drop=True).copy()
#        self.path_col = path_col
#        self.action_col = action_col
#        self.label_col = label_col
#        self.motion_key = motion_key
#        self.expected_shape = expected_shape
#    def __len__(self) -> int:
#        return len(self.df)
#    def _pick_npz_array(self, data: np.lib.npyio.NpzFile, path: str) -> np.ndarray:
#        if self.motion_key != "auto":
#            if self.motion_key not in data.files:
#                raise KeyError(f"motion_key={self.motion_key!r} not found in {path}. Keys: {data.files}")
#            return data[self.motion_key]
#        preferred = [
#            "motion", "motions", "x", "X", "data", "arr_0", "poses", "pose",
#            "rot6d", "features", "joints", "input"
#        ]
#        for key in preferred:
#            if key in data.files:
#                arr = data[key]
#                if np.issubdtype(arr.dtype, np.number):
#                    return arr
#        numeric_keys = [k for k in data.files if np.issubdtype(data[k].dtype, np.number)]
#        if not numeric_keys:
#            raise KeyError(f"No numeric arrays found in {path}. Keys: {data.files}")
#        return data[numeric_keys[0]]
#    def _standardize_motion_shape(self, arr: np.ndarray, path: str) -> np.ndarray:
#        """
#        Returns motion as [T, J, F].
#        Expected default is [60, 25, 6].
#        """
#        arr = np.asarray(arr, dtype=np.float32)
#        # Remove trivial dimensions, e.g. [1, 60, 25, 6]
#        while arr.ndim > 3 and 1 in arr.shape:
#            arr = np.squeeze(arr, axis=arr.shape.index(1))
#        if arr.ndim != 3:
#            raise ValueError(f"Expected 3D motion array [T,J,F], got shape {arr.shape} in {path}")
#        T, J, Feat = self.expected_shape
#        # Already [T, J, F]
#        if arr.shape == (T, J, Feat):
#            return arr
#        # Common alternatives
#        if arr.shape == (J, Feat, T):
#            return np.transpose(arr, (2, 0, 1))
#        if arr.shape == (Feat, T, J):
#            return np.transpose(arr, (1, 2, 0))
#        if arr.shape == (T, Feat, J):
#            return np.transpose(arr, (0, 2, 1))
#        if arr.shape == (J, T, Feat):
#            return np.transpose(arr, (1, 0, 2))
#        # Last-resort heuristic: identify axes by expected sizes
#        shape = list(arr.shape)
#        try:
#            t_axis = shape.index(T)
#            j_axis = shape.index(J)
#            f_axis = shape.index(Feat)
#            return np.transpose(arr, (t_axis, j_axis, f_axis))
#        except ValueError as e:
#            raise ValueError(
#                f"Cannot convert motion shape {arr.shape} to expected {(T, J, Feat)} for {path}. "
#                "Pass --num_frames/--njoints/--nfeats or adapt _standardize_motion_shape()."
#            ) from e
#    def __getitem__(self, idx: int) -> Dict[str, Any]:
#        row = self.df.iloc[idx]
#        path = str(row[self.path_col])
#        with np.load(path, allow_pickle=False) as data:
#            arr = self._pick_npz_array(data, path)
#        arr = self._standardize_motion_shape(arr, path)
#        return {
#            "motion": torch.from_numpy(arr),  # [T, J, F]
#            "action": normalize_action_text(row[self.action_col]),
#            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),  # 0 healthy, 1 flawed
#            "path": path,
#            "row_index": int(row.get("original_index", idx)),
#        }
#def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#    return {
#        "motion": torch.stack([b["motion"] for b in batch], dim=0),  # [B,T,J,F]
#        "action": [b["action"] for b in batch],
#        "label": torch.stack([b["label"] for b in batch], dim=0),
#        "path": [b["path"] for b in batch],
#        "row_index": [b["row_index"] for b in batch],
#    }
#class BalancedBinaryBatchSampler(Sampler[List[int]]):
#    """Yield batches with an approximately 50/50 split between labels 0 and 1.
#    Minority-class samples are oversampled with replacement when needed.
#    This is used only for training; validation/test loaders remain deterministic.
#    """
#    def __init__(
#        self,
#        labels: Sequence[int],
#        batch_size: int,
#        seed: int = 42,
#        drop_last: bool = False,
#    ) -> None:
#        if batch_size < 2:
#            raise ValueError("BalancedBinaryBatchSampler requires batch_size >= 2.")
#        labels_np = np.asarray(labels).astype(int)
#        unique = set(labels_np.tolist())
#        if not unique.issubset({0, 1}):
#            raise ValueError(f"BalancedBinaryBatchSampler expects binary labels 0/1, got {sorted(unique)}")
#        self.indices_by_class = {
#            0: np.where(labels_np == 0)[0],
#            1: np.where(labels_np == 1)[0],
#        }
#        if len(self.indices_by_class[0]) == 0 or len(self.indices_by_class[1]) == 0:
#            raise ValueError(
#                "BalancedBinaryBatchSampler needs at least one healthy sample and one flawed sample in the training split."
#            )
#        self.batch_size = int(batch_size)
#        self.n0 = self.batch_size // 2
#        self.n1 = self.batch_size - self.n0
#        self.seed = int(seed)
#        self.drop_last = bool(drop_last)
#        self.epoch = 0
#        # Number of batches needed so that the larger class is seen roughly once per epoch.
#        self.num_batches = int(max(
#            math.ceil(len(self.indices_by_class[0]) / self.n0),
#            math.ceil(len(self.indices_by_class[1]) / self.n1),
#        ))
#    def __len__(self) -> int:
#        return self.num_batches
#    def set_epoch(self, epoch: int) -> None:
#        self.epoch = int(epoch)
#    def _sample_class_indices(self, cls: int, n_total: int, rng: np.random.Generator) -> np.ndarray:
#        pool = self.indices_by_class[cls]
#        if n_total <= len(pool):
#            return rng.permutation(pool)[:n_total]
#        # Use all samples once, then oversample the remainder with replacement.
#        full = rng.permutation(pool)
#        extra = rng.choice(pool, size=n_total - len(pool), replace=True)
#        return np.concatenate([full, extra])
#    def __iter__(self):
#        rng = np.random.default_rng(self.seed + self.epoch)
#        labels0 = self._sample_class_indices(0, self.num_batches * self.n0, rng)
#        labels1 = self._sample_class_indices(1, self.num_batches * self.n1, rng)
#        for batch_idx in range(self.num_batches):
#            b0 = labels0[batch_idx * self.n0:(batch_idx + 1) * self.n0]
#            b1 = labels1[batch_idx * self.n1:(batch_idx + 1) * self.n1]
#            batch = np.concatenate([b0, b1])
#            rng.shuffle(batch)
#            if self.drop_last and len(batch) < self.batch_size:
#                continue
#            yield batch.astype(int).tolist()
#def stratified_group_split(
#    df: pd.DataFrame,
#    stratify_cols: Sequence[str],
#    test_fraction: float,
#    val_fraction: float,
#    seed: int,
#) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#    """
#    Stratifies within groups, e.g. by action + condition.
#    val_fraction is taken from the non-test part.
#    """
#    rng = np.random.default_rng(seed)
#    train_parts = []
#    val_parts = []
#    test_parts = []
#    for _, group in df.groupby(list(stratify_cols), dropna=False):
#        idxs = group.index.to_numpy()
#        rng.shuffle(idxs)
#        n = len(idxs)
#        n_test = int(round(n * test_fraction))
#        if n >= 5 and test_fraction > 0:
#            n_test = max(1, n_test)
#        n_test = min(n_test, max(0, n - 2))
#        test_idx = idxs[:n_test]
#        rem_idx = idxs[n_test:]
#        n_rem = len(rem_idx)
#        n_val = int(round(n_rem * val_fraction))
#        if n_rem >= 5 and val_fraction > 0:
#            n_val = max(1, n_val)
#        n_val = min(n_val, max(0, n_rem - 1))
#        val_idx = rem_idx[:n_val]
#        train_idx = rem_idx[n_val:]
#        train_parts.append(df.loc[train_idx])
#        val_parts.append(df.loc[val_idx])
#        test_parts.append(df.loc[test_idx])
#    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
#    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
#    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
#    return train_df, val_df, test_df
#def balance_binary_split(
#    df: pd.DataFrame,
#    label_col: str,
#    seed: int,
#    split_name: str = "test",
#) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#    """Downsample the majority label so the split has equal normal/anomaly counts.
#    Label convention:
#      0 = healthy/normal
#      1 = flawed/anomaly
#    The sampling is deterministic for a fixed seed.
#    """
#    counts_before = df[label_col].astype(int).value_counts().to_dict()
#    n_normal = int((df[label_col].astype(int) == 0).sum())
#    n_anomaly = int((df[label_col].astype(int) == 1).sum())
#    if n_normal == 0 or n_anomaly == 0:
#        raise ValueError(
#            f"Cannot balance {split_name}: needs both labels, got "
#            f"normal={n_normal}, anomaly={n_anomaly}."
#        )
#    n_keep = min(n_normal, n_anomaly)
#    normal_df = df[df[label_col].astype(int) == 0].sample(
#        n=n_keep,
#        replace=False,
#        random_state=seed,
#    )
#    anomaly_df = df[df[label_col].astype(int) == 1].sample(
#        n=n_keep,
#        replace=False,
#        random_state=seed + 1,
#    )
#    balanced_df = pd.concat([normal_df, anomaly_df], axis=0).sample(
#        frac=1.0,
#        random_state=seed + 2,
#    ).reset_index(drop=True)
#    counts_after = balanced_df[label_col].astype(int).value_counts().to_dict()
#    info = {
#        "split_name": split_name,
#        "balanced": True,
#        "seed": int(seed),
#        "n_before": int(len(df)),
#        "n_after": int(len(balanced_df)),
#        "counts_before": {int(k): int(v) for k, v in counts_before.items()},
#        "counts_after": {int(k): int(v) for k, v in counts_after.items()},
#        "n_kept_per_label": int(n_keep),
#        "n_dropped": int(len(df) - len(balanced_df)),
#    }
#    return balanced_df, info
#def load_unseen_actions_from_file(path: str | Path) -> List[str]:
#    """Read unseen action names from a txt/csv-like file.
#    Accepts one action per line. Empty lines and lines starting with # are ignored.
#    If a line contains a comma, only the first field is used.
#    """
#    path = Path(path)
#    actions: List[str] = []
#    with open(path, "r", encoding="utf-8") as f:
#        for raw in f:
#            line = raw.strip()
#            if not line or line.startswith("#"):
#                continue
#            actions.append(normalize_action_text(line.split(",")[0]))
#    return actions
#def load_unseen_values_from_file(path: str | Path) -> List[str]:
#    """Read normalized unseen class/style names from a txt/csv-like file.
#    Accepts one value per line. Empty lines and lines starting with # are ignored.
#    If a line contains a comma, only the first field is used.
#    """
#    path = Path(path)
#    values: List[str] = []
#    with open(path, "r", encoding="utf-8") as f:
#        for raw in f:
#            line = raw.strip()
#            if not line or line.startswith("#"):
#                continue
#            values.append(normalize_action_text(line.split(",")[0]))
#    return values
#def select_unseen_values(
#    df: pd.DataFrame,
#    norm_col: str,
#    label_col: str,
#    seed: int,
#    requested_values: Optional[Sequence[str]] = None,
#    requested_file: str = "",
#    requested_count: int = 0,
#    requested_fraction: float = 0.0,
#    require_both_labels: bool = True,
#    value_name: str = "value",
#) -> List[str]:
#    """Choose full classes/styles that will be excluded from train/val.
#    If no explicit values, file, count, or positive fraction is provided, returns an empty list.
#    """
#    if norm_col not in df.columns:
#        raise ValueError(f"Column {norm_col!r} not found while selecting unseen {value_name}s.")
#    all_values = sorted(df[norm_col].dropna().map(normalize_action_text).unique().tolist())
#    requested: List[str] = []
#    if requested_values:
#        requested.extend([normalize_action_text(v) for v in requested_values if str(v).strip()])
#    if requested_file:
#        requested.extend(load_unseen_values_from_file(requested_file))
#    if requested:
#        requested = sorted(set(requested))
#        missing = sorted(set(requested) - set(all_values))
#        if missing:
#            raise ValueError(
#                f"Requested unseen {value_name}s were not found in the CSV after normalization: "
#                f"{missing}. Available examples: {all_values[:30]}"
#            )
#        return requested
#    if requested_count <= 0 and requested_fraction <= 0:
#        return []
#    eligible = all_values
#    if require_both_labels:
#        eligible = []
#        for value, g in df.groupby(norm_col):
#            labs = set(g[label_col].astype(int).tolist())
#            if {0, 1}.issubset(labs):
#                eligible.append(value)
#        eligible = sorted(eligible)
#    if not eligible:
#        raise ValueError(f"No eligible unseen {value_name}s found. Try allowing single-label unseen {value_name}s.")
#    if requested_count > 0:
#        n_unseen = int(requested_count)
#    else:
#        n_unseen = int(round(len(eligible) * requested_fraction))
#        n_unseen = max(1, n_unseen)
#    n_unseen = min(n_unseen, max(1, len(eligible) - 1))
#    rng = np.random.default_rng(seed)
#    return sorted(rng.choice(np.asarray(eligible, dtype=object), size=n_unseen, replace=False).tolist())
#def select_unseen_actions(
#    df: pd.DataFrame,
#    action_norm_col: str,
#    label_col: str,
#    seed: int,
#    unseen_actions: Optional[Sequence[str]] = None,
#    unseen_action_file: str = "",
#    unseen_action_count: int = 0,
#    unseen_action_fraction: float = 0.20,
#    require_both_labels: bool = True,
#) -> List[str]:
#    """Choose full action classes that will be excluded from train/val.
#    Priority:
#      1. --unseen_actions / --unseen_actions_file
#      2. --unseen_action_count
#      3. --unseen_action_fraction
#    """
#    all_actions = sorted(df[action_norm_col].dropna().map(normalize_action_text).unique().tolist())
#    requested: List[str] = []
#    if unseen_actions:
#        requested.extend([normalize_action_text(a) for a in unseen_actions if str(a).strip()])
#    if unseen_action_file:
#        requested.extend(load_unseen_actions_from_file(unseen_action_file))
#    if requested:
#        requested = sorted(set(requested))
#        missing = sorted(set(requested) - set(all_actions))
#        if missing:
#            raise ValueError(
#                "Requested unseen actions were not found in the CSV after normalization: "
#                f"{missing}. Available examples: {all_actions[:30]}"
#            )
#        return requested
#    eligible = all_actions
#    if require_both_labels:
#        eligible = []
#        for action, g in df.groupby(action_norm_col):
#            labs = set(g[label_col].astype(int).tolist())
#            if {0, 1}.issubset(labs):
#                eligible.append(action)
#        eligible = sorted(eligible)
#    if not eligible:
#        raise ValueError("No eligible unseen actions found. Try disabling --require_unseen_both_labels.")
#    if unseen_action_count > 0:
#        n_unseen = int(unseen_action_count)
#    else:
#        if unseen_action_fraction <= 0:
#            raise ValueError(
#                "No unseen actions specified. Provide --unseen_actions, --unseen_actions_file, "
#                "--unseen_action_count, or set --unseen_action_fraction > 0."
#            )
#        n_unseen = int(round(len(eligible) * unseen_action_fraction))
#        n_unseen = max(1, n_unseen)
#    n_unseen = min(n_unseen, max(1, len(eligible) - 1))
#    rng = np.random.default_rng(seed)
#    chosen = sorted(rng.choice(np.asarray(eligible, dtype=object), size=n_unseen, replace=False).tolist())
#    return chosen
#def select_unseen_actors(
#    df: pd.DataFrame,
#    actor_norm_col: str,
#    label_col: str,
#    seed: int,
#    unseen_actors: Optional[Sequence[str]] = None,
#    unseen_actor_file: str = "",
#    unseen_actor_count: int = 0,
#    unseen_actor_fraction: float = 0.0,
#    require_both_labels: bool = True,
#) -> List[str]:
#    """Choose full actor IDs that will be excluded from train/val."""
#    if actor_norm_col not in df.columns:
#        raise ValueError(f"Column {actor_norm_col!r} not found while selecting unseen actors.")
#    all_actors = sorted(df[actor_norm_col].dropna().map(normalize_actor_id).unique().tolist())
#    requested: List[str] = []
#    if unseen_actors:
#        requested.extend([normalize_actor_id(a) for a in unseen_actors if str(a).strip()])
#    if unseen_actor_file:
#        requested.extend(load_unseen_actor_ids_from_file(unseen_actor_file))
#    if requested:
#        requested = sorted(set(requested))
#        missing = sorted(set(requested) - set(all_actors))
#        if missing:
#            raise ValueError(
#                f"Requested unseen actors were not found after normalization: {missing}. "
#                f"Available actors: {all_actors}"
#            )
#        return requested
#    if unseen_actor_count <= 0 and unseen_actor_fraction <= 0:
#        return []
#    eligible = all_actors
#    if require_both_labels:
#        eligible = []
#        for actor, g in df.groupby(actor_norm_col):
#            labs = set(g[label_col].astype(int).tolist())
#            if {0, 1}.issubset(labs):
#                eligible.append(actor)
#        eligible = sorted(eligible)
#    if not eligible:
#        raise ValueError("No eligible unseen actors found. Try --allow_unseen_actor_single_label.")
#    if unseen_actor_count > 0:
#        n_unseen = int(unseen_actor_count)
#    else:
#        n_unseen = int(round(len(eligible) * unseen_actor_fraction))
#        n_unseen = max(1, n_unseen)
#    n_unseen = min(n_unseen, max(1, len(eligible) - 1))
#    rng = np.random.default_rng(seed)
#    return sorted(rng.choice(np.asarray(eligible, dtype=object), size=n_unseen, replace=False).tolist())
#def safe_balance_binary_split(
#    df: pd.DataFrame,
#    label_col: str,
#    seed: int,
#    split_name: str,
#    allow_single_label: bool = False,
#) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#    """Balance a split only when it contains both labels."""
#    n_normal = int((df[label_col].astype(int) == 0).sum()) if len(df) else 0
#    n_anomaly = int((df[label_col].astype(int) == 1).sum()) if len(df) else 0
#    if len(df) == 0:
#        return df.reset_index(drop=True), {
#            "split_name": split_name,
#            "balanced": False,
#            "reason": "empty split",
#            "n_samples": 0,
#            "n_normal": 0,
#            "n_anomaly": 0,
#        }
#    if n_normal == 0 or n_anomaly == 0:
#        if allow_single_label:
#            return df.reset_index(drop=True), {
#                "split_name": split_name,
#                "balanced": False,
#                "reason": "single-label split; AUROC/AUPRC are not meaningful",
#                "n_samples": int(len(df)),
#                "n_normal": n_normal,
#                "n_anomaly": n_anomaly,
#            }
#        raise ValueError(
#            f"Cannot balance {split_name}: needs both labels, got normal={n_normal}, anomaly={n_anomaly}."
#        )
#    return balance_binary_split(df, label_col=label_col, seed=seed, split_name=split_name)
#def metrics_without_ranking_for_single_label_style(eval_out: EvalOutput, threshold: float, split_name: str) -> Dict[str, Any]:
#    """For unseen-style tests, do not report AUROC/AUPRC because Healthy is not held out as an unseen style."""
#    base = classification_metrics_at_threshold(eval_out.y_true, eval_out.score, threshold)
#    base.update({
#        "auroc": None,
#        "auprc": None,
#        "ranking_metrics_reported": False,
#        "ranking_metrics_reason": (
#            "Unseen style/condition is evaluated as a held-out flawed condition. Healthy is the normal class and is not treated as an unseen style, "
#            "so this split can be single-label and AUROC/AUPRC are not meaningful."
#        ),
#        "split_name": split_name,
#        "n_samples": int(len(eval_out.y_true)),
#        "n_normal": int((eval_out.y_true == 0).sum()),
#        "n_anomaly": int((eval_out.y_true == 1).sum()),
#        "score_mean": float(np.mean(eval_out.score)) if len(eval_out.score) else float("nan"),
#        "score_std": float(np.std(eval_out.score)) if len(eval_out.score) else float("nan"),
#        "threshold_source": "validation_threshold",
#    })
#    return base
#def true_unseen_action_style_actor_split(
#    df: pd.DataFrame,
#    action_norm_col: str,
#    style_norm_col: str,
#    actor_norm_col: str,
#    condition_col: str,
#    label_col: str,
#    test_fraction: float,
#    val_fraction: float,
#    seed: int,
#    unseen_actions: Optional[Sequence[str]],
#    unseen_action_file: str,
#    unseen_action_count: int,
#    unseen_action_fraction: float,
#    require_unseen_both_labels: bool,
#    unseen_styles: Optional[Sequence[str]],
#    unseen_styles_file: str,
#    unseen_style_count: int,
#    unseen_style_fraction: float,
#    require_unseen_style_both_labels: bool,
#    unseen_actors: Optional[Sequence[str]],
#    unseen_actors_file: str,
#    unseen_actor_count: int,
#    unseen_actor_fraction: float,
#    require_unseen_actor_both_labels: bool,
#) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
#    """Split with truly held-out action/content, style/condition, and actor IDs.
#    Train/val are created only from samples with seen action + seen style + seen actor.
#    The test buckets are:
#      - seen_test_df: seen action + seen style + seen actor
#      - unseen_action_test_df: unseen action + seen style + seen actor
#      - unseen_style_test_df: seen action + unseen style + seen actor
#      - unseen_actor_test_df: seen action + seen style + unseen actor
#      - unseen_all_test_df: unseen action + unseen style + unseen actor
#      - other_unseen_combinations_df: all two-way unseen combinations
#      - combined_test_df: all test buckets above combined
#    """
#    heldout_actions = select_unseen_actions(
#        df=df,
#        action_norm_col=action_norm_col,
#        label_col=label_col,
#        seed=seed,
#        unseen_actions=unseen_actions,
#        unseen_action_file=unseen_action_file,
#        unseen_action_count=unseen_action_count,
#        unseen_action_fraction=unseen_action_fraction,
#        require_both_labels=require_unseen_both_labels,
#    )
#    heldout_styles = select_unseen_values(
#        df=df,
#        norm_col=style_norm_col,
#        label_col=label_col,
#        seed=seed + 17,
#        requested_values=unseen_styles,
#        requested_file=unseen_styles_file,
#        requested_count=unseen_style_count,
#        requested_fraction=unseen_style_fraction,
#        require_both_labels=require_unseen_style_both_labels,
#        value_name="style",
#    )
#    heldout_actors = select_unseen_actors(
#        df=df,
#        actor_norm_col=actor_norm_col,
#        label_col=label_col,
#        seed=seed + 31,
#        unseen_actors=unseen_actors,
#        unseen_actor_file=unseen_actors_file,
#        unseen_actor_count=unseen_actor_count,
#        unseen_actor_fraction=unseen_actor_fraction,
#        require_both_labels=require_unseen_actor_both_labels,
#    )
#    unseen_action_set = set(heldout_actions)
#    unseen_style_set = set(heldout_styles)
#    unseen_actor_set = set(heldout_actors)
#    action_unseen = df[action_norm_col].isin(unseen_action_set)
#    style_unseen = df[style_norm_col].isin(unseen_style_set)
#    actor_unseen = df[actor_norm_col].isin(unseen_actor_set)
#    seen_mask = (~action_unseen) & (~style_unseen) & (~actor_unseen)
#    unseen_action_only_mask = action_unseen & (~style_unseen) & (~actor_unseen)
#    unseen_style_only_mask = (~action_unseen) & style_unseen & (~actor_unseen)
#    unseen_actor_only_mask = (~action_unseen) & (~style_unseen) & actor_unseen
#    unseen_all_mask = action_unseen & style_unseen & actor_unseen
#    other_unseen_combinations_mask = (
#        (action_unseen | style_unseen | actor_unseen)
#        & ~(unseen_action_only_mask | unseen_style_only_mask | unseen_actor_only_mask | unseen_all_mask)
#    )
#    seen_df = df[seen_mask].copy()
#    unseen_action_test_df = df[unseen_action_only_mask].copy()
#    unseen_style_test_df = df[unseen_style_only_mask].copy()
#    unseen_actor_test_df = df[unseen_actor_only_mask].copy()
#    unseen_all_test_df = df[unseen_all_mask].copy()
#    other_unseen_combinations_df = df[other_unseen_combinations_mask].copy()
#    if len(seen_df) == 0:
#        raise ValueError("No seen data remains for training after action/style/actor holdout.")
#    if not set(seen_df[label_col].astype(int).unique()).issuperset({0, 1}):
#        raise ValueError("Seen training pool must contain both labels 0 and 1 after action/style/actor holdout.")
#    if heldout_actions and len(unseen_action_test_df) == 0:
#        print("[WARN] No isolated unseen-action rows remain. They may only exist in combined unseen buckets.")
#    if heldout_styles and len(unseen_style_test_df) == 0:
#        print("[WARN] No isolated unseen-style rows remain. They may only exist in combined unseen buckets.")
#    if heldout_actors and len(unseen_actor_test_df) == 0:
#        print("[WARN] No isolated unseen-actor rows remain. They may only exist in combined unseen buckets.")
#    if heldout_actions and heldout_styles and heldout_actors and len(unseen_all_test_df) == 0:
#        print("[WARN] No rows exist where action, style, and actor are all unseen at the same time.")
#    stratify_cols = [action_norm_col, actor_norm_col, label_col]
#    if condition_col in seen_df.columns:
#        stratify_cols.insert(1, condition_col)
#    train_df, val_df, seen_test_df = stratified_group_split(
#        seen_df,
#        stratify_cols=stratify_cols,
#        test_fraction=test_fraction,
#        val_fraction=val_fraction,
#        seed=seed,
#    )
#    test_parts = [seen_test_df]
#    for part in [unseen_action_test_df, unseen_style_test_df, unseen_actor_test_df, unseen_all_test_df, other_unseen_combinations_df]:
#        if len(part) > 0:
#            test_parts.append(part)
#    combined_test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
#    return (
#        train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
#        val_df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True),
#        seen_test_df.sample(frac=1.0, random_state=seed + 5).reset_index(drop=True),
#        unseen_action_test_df.sample(frac=1.0, random_state=seed + 6).reset_index(drop=True),
#        unseen_style_test_df.sample(frac=1.0, random_state=seed + 7).reset_index(drop=True),
#        unseen_actor_test_df.sample(frac=1.0, random_state=seed + 8).reset_index(drop=True),
#        unseen_all_test_df.sample(frac=1.0, random_state=seed + 9).reset_index(drop=True),
#        other_unseen_combinations_df.sample(frac=1.0, random_state=seed + 10).reset_index(drop=True),
#        combined_test_df,
#        heldout_actions,
#        heldout_styles,
#        heldout_actors,
#    )
## -----------------------------
## MotionCLIP loading/freezing
## -----------------------------
#def build_motionclip_encoder(checkpoint_path: str, device: torch.device) -> nn.Module:
#    """
#    Build and load MotionCLIP exactly like finetune_unsupervised_updated.py.
#    The encoder architecture is hardcoded to the paper MotionCLIP setup:
#    rot6d, [60, 25, 6], latent_dim=512, 8 transformer layers.
#    """
#    encoder = Encoder_TRANSFORMER(
#        modeltype="motionclip",
#        njoints=25,
#        nfeats=6,
#        num_frames=60,
#        num_classes=1,
#        translation=True,
#        pose_rep="rot6d",
#        glob=True,
#        glob_rot=[math.pi, 0.0, 0.0],
#        latent_dim=512,
#        ff_size=1024,
#        num_layers=8,
#        num_heads=4,
#        dropout=0.1,
#        ablation=None,
#        activation="gelu",
#    )
#    ckpt = torch.load(checkpoint_path, map_location="cpu")
#    if "state_dict" in ckpt:
#        ckpt = ckpt["state_dict"]
#    encoder_state = {}
#    for k, v in ckpt.items():
#        if k.startswith("encoder."):
#            encoder_state[k[len("encoder."):]] = v
#    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
#    if unexpected:
#        raise RuntimeError(f"Unexpected encoder keys: {unexpected}")
#    if missing:
#        print("Warning: missing encoder keys:", missing)
#    encoder = encoder.to(device)
#    encoder.train()
#    return encoder
#def freeze_encoder_except_last_layers(encoder: nn.Module, num_trainable_blocks: int = 2) -> Dict[str, Any]:
#    """
#    Freeze everything except the last N transformer blocks and final norm,
#    matching the logic in finetune_unsupervised_updated.py.
#    """
#    for p in encoder.parameters():
#        p.requires_grad = False
#    unfroze_any = False
#    unfrozen_layer_indices: List[int] = []
#    if hasattr(encoder, "seqTransEncoder"):
#        seq_encoder = encoder.seqTransEncoder
#        if hasattr(seq_encoder, "layers"):
#            layers = seq_encoder.layers
#            n = min(num_trainable_blocks, len(layers))
#            start = len(layers) - n
#            for i, layer in enumerate(layers[start:], start=start):
#                for p in layer.parameters():
#                    p.requires_grad = True
#                unfrozen_layer_indices.append(i)
#            unfroze_any = True
#        if hasattr(seq_encoder, "norm") and seq_encoder.norm is not None:
#            for p in seq_encoder.norm.parameters():
#                p.requires_grad = True
#    if not unfroze_any:
#        print("Warning: could not find encoder.seqTransEncoder.layers; encoder may remain frozen.")
#    trainable_names = [name for name, p in encoder.named_parameters() if p.requires_grad]
#    return {
#        "num_trainable_blocks_requested": int(num_trainable_blocks),
#        "unfrozen_layer_indices": unfrozen_layer_indices,
#        "num_trainable_params": int(sum(p.numel() for p in encoder.parameters() if p.requires_grad)),
#        "num_total_params": int(sum(p.numel() for p in encoder.parameters())),
#        "trainable_param_names_first_100": trainable_names[:100],
#    }
#def encode_motion_auto(model: nn.Module, motion: torch.Tensor) -> torch.Tensor:
#    """
#    MotionCLIP-specific forward pass.
#    Input from dataset is [B, T, J, F].
#    MotionCLIP encoder expects batch['x'] as [B, J, F, T].
#    """
#    motion = motion.float()
#    x = motion.permute(0, 2, 3, 1).contiguous()  # [B, 25, 6, 60]
#    B, T = motion.shape[0], motion.shape[1]
#    lengths = torch.full((B,), T, dtype=torch.long, device=motion.device)
#    mask = torch.arange(T, device=motion.device).unsqueeze(0) < lengths.unsqueeze(1)
#    batch = {
#        "x": x,
#        "y": torch.zeros(B, dtype=torch.long, device=motion.device),
#        "lengths": lengths,
#        "mask": mask,
#    }
#    out = model(batch)
#    if not isinstance(out, dict) or "mu" not in out:
#        raise RuntimeError("Expected MotionCLIP encoder output dict with key 'mu'.")
#    return out["mu"]  # [B, 512]
## -----------------------------
## Text encoder and prompts
## -----------------------------
#class FrozenCLIPTextEncoder:
#    def __init__(self, clip_model_name: str, device: torch.device):
#        try:
#            import clip  # OpenAI CLIP package
#        except ImportError as e:
#            raise ImportError(
#                "Could not import `clip`. Install OpenAI CLIP in your environment, e.g.:\n"
#                "  pip install git+https://github.com/openai/CLIP.git"
#            ) from e
#        self.clip = clip
#        self.model, _ = clip.load(clip_model_name, device=device)
#        self.model = self.model.float()
#        self.model.eval()
#        for p in self.model.parameters():
#            p.requires_grad = False
#        self.device = device
#    @torch.no_grad()
#    def encode(self, texts: Sequence[str], batch_size: int = 256) -> torch.Tensor:
#        feats = []
#        for start in range(0, len(texts), batch_size):
#            chunk = list(texts[start:start + batch_size])
#            tokens = self.clip.tokenize(chunk, truncate=True).to(self.device)
#            f = self.model.encode_text(tokens).float()
#            f = F.normalize(f, dim=-1)
#            feats.append(f.cpu())
#        return torch.cat(feats, dim=0)
#def build_prompt_cache(
#    actions: Sequence[str],
#    text_encoder: FrozenCLIPTextEncoder,
#    normal_prompt_template: str,
#    anomaly_prompt_template: str,
#    device: torch.device,
#) -> Tuple[Dict[str, int], torch.Tensor, Dict[str, Dict[str, str]]]:
#    actions = sorted({normalize_action_text(a) for a in actions})
#    action_to_idx = {a: i for i, a in enumerate(actions)}
#    prompt_info: Dict[str, Dict[str, str]] = {}
#    texts = []
#    for action in actions:
#        normal_prompt = normal_prompt_template.format(action=action)
#        anomaly_prompt = anomaly_prompt_template.format(action=action)
#        prompt_info[action] = {
#            "normal": normal_prompt,
#            "anomaly": anomaly_prompt,
#        }
#        texts.extend([normal_prompt, anomaly_prompt])
#    text_feats = text_encoder.encode(texts)  # [2*A, D]
#    text_feats = text_feats.reshape(len(actions), 2, -1).to(device)
#    return action_to_idx, text_feats, prompt_info
## -----------------------------
## Metrics
## -----------------------------
#def binary_auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
#    """
#    Fallback AUROC using rank statistics.
#    """
#    y_true = np.asarray(y_true).astype(int)
#    scores = np.asarray(scores).astype(float)
#    pos = y_true == 1
#    neg = y_true == 0
#    n_pos = pos.sum()
#    n_neg = neg.sum()
#    if n_pos == 0 or n_neg == 0:
#        return float("nan")
#    order = np.argsort(scores)
#    ranks = np.empty_like(order, dtype=float)
#    ranks[order] = np.arange(1, len(scores) + 1)
#    # Average tied ranks
#    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
#    if np.any(counts > 1):
#        for k, count in enumerate(counts):
#            if count > 1:
#                tied = inverse == k
#                ranks[tied] = ranks[tied].mean()
#    rank_sum_pos = ranks[pos].sum()
#    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
#    return float(auc)
#def average_precision_fallback(y_true: np.ndarray, scores: np.ndarray) -> float:
#    y_true = np.asarray(y_true).astype(int)
#    scores = np.asarray(scores).astype(float)
#    order = np.argsort(-scores)
#    y = y_true[order]
#    total_pos = y.sum()
#    if total_pos == 0:
#        return float("nan")
#    tp = np.cumsum(y)
#    precision = tp / (np.arange(len(y)) + 1)
#    return float((precision * y).sum() / total_pos)
#def classification_metrics_at_threshold(
#    y_true: np.ndarray,
#    scores: np.ndarray,
#    threshold: float,
#) -> Dict[str, Any]:
#    y_true = np.asarray(y_true).astype(int)
#    pred = (np.asarray(scores) >= threshold).astype(int)
#    tp = int(((pred == 1) & (y_true == 1)).sum())
#    tn = int(((pred == 0) & (y_true == 0)).sum())
#    fp = int(((pred == 1) & (y_true == 0)).sum())
#    fn = int(((pred == 0) & (y_true == 1)).sum())
#    accuracy = (tp + tn) / max(1, len(y_true))
#    precision = tp / max(1, tp + fp)
#    recall = tp / max(1, tp + fn)
#    specificity = tn / max(1, tn + fp)
#    f1 = 2 * precision * recall / max(1e-12, precision + recall)
#    balanced_accuracy = 0.5 * (recall + specificity)
#    return {
#        "threshold": float(threshold),
#        "accuracy": float(accuracy),
#        "balanced_accuracy": float(balanced_accuracy),
#        "precision": float(precision),
#        "recall": float(recall),
#        "specificity": float(specificity),
#        "f1": float(f1),
#        "tp": tp,
#        "tn": tn,
#        "fp": fp,
#        "fn": fn,
#    }
#def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, criterion: str = "f1") -> Tuple[float, Dict[str, Any]]:
#    scores = np.asarray(scores, dtype=float)
#    if len(scores) == 0:
#        return 0.0, {}
#    # Candidate thresholds: unique scores plus ends
#    candidates = np.unique(scores)
#    if len(candidates) > 1000:
#        candidates = np.quantile(scores, np.linspace(0, 1, 1000))
#    best_thr = float(candidates[0])
#    best_metrics = None
#    best_value = -float("inf")
#    for thr in candidates:
#        m = classification_metrics_at_threshold(y_true, scores, float(thr))
#        value = m.get(criterion, m["f1"])
#        if value > best_value:
#            best_value = value
#            best_thr = float(thr)
#            best_metrics = m
#    assert best_metrics is not None
#    return best_thr, best_metrics
#def compute_binary_metrics(
#    y_true: np.ndarray,
#    scores: np.ndarray,
#    threshold: Optional[float] = None,
#    threshold_criterion: str = "f1",
#) -> Dict[str, Any]:
#    y_true = np.asarray(y_true).astype(int)
#    scores = np.asarray(scores).astype(float)
#    try:
#        from sklearn.metrics import roc_auc_score, average_precision_score
#        auroc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
#        auprc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
#    except Exception:
#        auroc = binary_auc_rank(y_true, scores)
#        auprc = average_precision_fallback(y_true, scores)
#    if threshold is None:
#        threshold, threshold_metrics = find_best_threshold(y_true, scores, threshold_criterion)
#        threshold_source = f"best_{threshold_criterion}_on_this_split"
#    else:
#        threshold_metrics = classification_metrics_at_threshold(y_true, scores, threshold)
#        threshold_source = "provided"
#    return {
#        "auroc": auroc,
#        "auprc": auprc,
#        "n_samples": int(len(y_true)),
#        "n_normal": int((y_true == 0).sum()),
#        "n_anomaly": int((y_true == 1).sum()),
#        "score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
#        "score_std": float(np.std(scores)) if len(scores) else float("nan"),
#        "threshold_source": threshold_source,
#        **threshold_metrics,
#    }
## -----------------------------
## Train/evaluate
## -----------------------------
#@dataclass
#class EvalOutput:
#    loss: float
#    y_true: np.ndarray
#    score: np.ndarray
#    prob_anomaly: np.ndarray
#    s_healthy: np.ndarray
#    s_flawed: np.ndarray
#    embeddings: np.ndarray
#    paths: List[str]
#    actions: List[str]
#    row_indices: List[int]
#def make_class_weights(labels: Sequence[int], device: torch.device, mode: str) -> Optional[torch.Tensor]:
#    if mode == "none":
#        return None
#    labels_np = np.asarray(labels).astype(int)
#    counts = np.bincount(labels_np, minlength=2).astype(float)
#    weights = counts.sum() / np.maximum(1.0, 2.0 * counts)
#    return torch.tensor(weights, dtype=torch.float32, device=device)
#def logits_from_motion_and_prompts(
#    motion_encoder: nn.Module,
#    motion: torch.Tensor,
#    actions: Sequence[str],
#    action_to_idx: Dict[str, int],
#    text_feats: torch.Tensor,
#    temperature: float,
#) -> Tuple[torch.Tensor, torch.Tensor]:
#    z = encode_motion_auto(motion_encoder, motion)
#    z = F.normalize(z.float(), dim=-1)
#    idx = torch.tensor([action_to_idx[normalize_action_text(a)] for a in actions], dtype=torch.long, device=motion.device)
#    prompts = text_feats[idx]  # [B,2,D]
#    if z.shape[-1] != prompts.shape[-1]:
#        raise RuntimeError(
#            f"Motion embedding dim ({z.shape[-1]}) does not match text embedding dim ({prompts.shape[-1]}). "
#            "Check --latent_dim and CLIP model."
#        )
#    logits = torch.bmm(prompts, z.unsqueeze(-1)).squeeze(-1) / temperature  # [B,2]
#    return logits, z
#def target_text_embeddings_and_group_ids_for_batch(
#    actions: Sequence[str],
#    labels: torch.Tensor,
#    action_to_idx: Dict[str, int],
#    text_feats: torch.Tensor,
#) -> Tuple[torch.Tensor, torch.Tensor]:
#    """Return matching text embeddings and class ids for healthy/flawed-per-action groups."""
#    idx = torch.tensor(
#        [action_to_idx[normalize_action_text(a)] for a in actions],
#        dtype=torch.long,
#        device=labels.device,
#    )
#    labels = labels.long()
#    target_text = text_feats[idx, labels]  # [B, D]
#    group_ids = idx * 2 + labels           # same id = same action and same healthy/flawed label
#    return target_text, group_ids
#def symmetric_motion_text_contrastive_loss(
#    motion_z: torch.Tensor,
#    text_z: torch.Tensor,
#    group_ids: torch.Tensor,
#    temperature: float,
#) -> torch.Tensor:
#    """Bidirectional supervised contrastive loss between motions and matching text prompts.
#    Samples with the same action and healthy/flawed label are treated as positives.
#    This avoids treating duplicate prompts within a batch as false negatives.
#    """
#    motion_z = F.normalize(motion_z.float(), dim=-1)
#    text_z = F.normalize(text_z.float(), dim=-1)
#    logits = motion_z @ text_z.t() / temperature  # [B, B]
#    positive_mask = group_ids[:, None].eq(group_ids[None, :]).float()
#    log_prob_m2t = logits - torch.logsumexp(logits, dim=1, keepdim=True)
#    loss_m2t = -(positive_mask * log_prob_m2t).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1.0)
#    log_prob_t2m = logits.t() - torch.logsumexp(logits.t(), dim=1, keepdim=True)
#    loss_t2m = -(positive_mask.t() * log_prob_t2m).sum(dim=1) / positive_mask.t().sum(dim=1).clamp_min(1.0)
#    return 0.5 * (loss_m2t.mean() + loss_t2m.mean())
#def train_one_epoch(
#    motion_encoder: nn.Module,
#    loader: DataLoader,
#    optimizer: torch.optim.Optimizer,
#    device: torch.device,
#    action_to_idx: Dict[str, int],
#    text_feats: torch.Tensor,
#    temperature: float,
#    grad_clip: float,
#    use_amp: bool,
#) -> float:
#    motion_encoder.train()
#    total_loss = 0.0
#    total_n = 0
#    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#    for batch in loader:
#        motion = batch["motion"].to(device, non_blocking=True)
#        labels = batch["label"].to(device, non_blocking=True)
#        optimizer.zero_grad(set_to_none=True)
#        with torch.cuda.amp.autocast(enabled=use_amp):
#            z = encode_motion_auto(motion_encoder, motion)
#            target_text, group_ids = target_text_embeddings_and_group_ids_for_batch(batch["action"], labels, action_to_idx, text_feats)
#            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
#        scaler.scale(loss).backward()
#        if grad_clip > 0:
#            scaler.unscale_(optimizer)
#            torch.nn.utils.clip_grad_norm_(
#                [p for p in motion_encoder.parameters() if p.requires_grad],
#                grad_clip,
#            )
#        scaler.step(optimizer)
#        scaler.update()
#        bs = labels.numel()
#        total_loss += float(loss.detach().cpu()) * bs
#        total_n += bs
#    return total_loss / max(1, total_n)
#@torch.no_grad()
#def evaluate(
#    motion_encoder: nn.Module,
#    loader: DataLoader,
#    device: torch.device,
#    action_to_idx: Dict[str, int],
#    text_feats: torch.Tensor,
#    temperature: float,
#    compute_contrastive_loss: bool = True,
#) -> EvalOutput:
#    motion_encoder.eval()
#    losses = []
#    y_true = []
#    scores = []
#    probs = []
#    s_h = []
#    s_f = []
#    embeddings = []
#    paths = []
#    actions = []
#    row_indices = []
#    for batch in loader:
#        motion = batch["motion"].to(device, non_blocking=True)
#        labels = batch["label"].to(device, non_blocking=True)
#        logits, z = logits_from_motion_and_prompts(
#            motion_encoder, motion, batch["action"], action_to_idx, text_feats, temperature
#        )
#        if compute_contrastive_loss:
#            target_text, group_ids = target_text_embeddings_and_group_ids_for_batch(batch["action"], labels, action_to_idx, text_feats)
#            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
#            losses.append(float(loss.detach().cpu()) * labels.numel())
#        soft = torch.softmax(logits, dim=-1)
#        # anomaly score: positive means closer to flawed than healthy
#        score = logits[:, 1] - logits[:, 0]
#        embeddings.append(z.detach().cpu().numpy().astype(np.float32))
#        y_true.extend(labels.cpu().numpy().astype(int).tolist())
#        scores.extend(score.cpu().numpy().astype(float).tolist())
#        probs.extend(soft[:, 1].cpu().numpy().astype(float).tolist())
#        s_h.extend(logits[:, 0].cpu().numpy().astype(float).tolist())
#        s_f.extend(logits[:, 1].cpu().numpy().astype(float).tolist())
#        paths.extend(batch["path"])
#        actions.extend(batch["action"])
#        row_indices.extend(batch["row_index"])
#    total_n = max(1, len(y_true))
#    avg_loss = sum(losses) / total_n if losses else float("nan")
#    return EvalOutput(
#        loss=avg_loss,
#        y_true=np.asarray(y_true, dtype=int),
#        score=np.asarray(scores, dtype=float),
#        prob_anomaly=np.asarray(probs, dtype=float),
#        s_healthy=np.asarray(s_h, dtype=float),
#        s_flawed=np.asarray(s_f, dtype=float),
#        embeddings=np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 512), dtype=np.float32),
#        paths=paths,
#        actions=actions,
#        row_indices=row_indices,
#    )
#def save_predictions(eval_out: EvalOutput, path: str | Path, threshold: float) -> None:
#    pred = (eval_out.score >= threshold).astype(int)
#    out_df = pd.DataFrame({
#        "row_index": eval_out.row_indices,
#        "motion_path": eval_out.paths,
#        "action": eval_out.actions,
#        "y_true_is_anomaly": eval_out.y_true,
#        "anomaly_score_flawed_minus_healthy": eval_out.score,
#        "prob_anomaly": eval_out.prob_anomaly,
#        "logit_healthy": eval_out.s_healthy,
#        "logit_flawed": eval_out.s_flawed,
#        "pred_is_anomaly": pred,
#    })
#    out_df.to_csv(path, index=False)
#def save_embeddings(eval_out: EvalOutput, npz_path: str | Path, metadata_csv_path: str | Path, threshold: float) -> None:
#    """Save normalized MotionCLIP embeddings and metadata for plotting/debugging.
#    NPZ keys:
#      embeddings: [N, D] normalized motion embeddings used for prompt scoring
#      y_true: [N] 0=healthy, 1=flawed/anomaly
#      score: [N] flawed_minus_healthy anomaly score
#      prob_anomaly: [N] softmax probability for flawed/anomaly prompt
#      logit_healthy/logit_flawed: [N] prompt logits
#      row_index: [N] original CSV row index
#      pred_is_anomaly: [N] prediction using the selected threshold
#    """
#    pred = (eval_out.score >= threshold).astype(int)
#    np.savez_compressed(
#        npz_path,
#        embeddings=eval_out.embeddings.astype(np.float32),
#        y_true=eval_out.y_true.astype(np.int64),
#        score=eval_out.score.astype(np.float32),
#        prob_anomaly=eval_out.prob_anomaly.astype(np.float32),
#        logit_healthy=eval_out.s_healthy.astype(np.float32),
#        logit_flawed=eval_out.s_flawed.astype(np.float32),
#        row_index=np.asarray(eval_out.row_indices, dtype=np.int64),
#        pred_is_anomaly=pred.astype(np.int64),
#    )
#    meta_df = pd.DataFrame({
#        "embedding_index": np.arange(len(eval_out.row_indices), dtype=int),
#        "row_index": eval_out.row_indices,
#        "motion_path": eval_out.paths,
#        "action": eval_out.actions,
#        "y_true_is_anomaly": eval_out.y_true,
#        "anomaly_score_flawed_minus_healthy": eval_out.score,
#        "prob_anomaly": eval_out.prob_anomaly,
#        "logit_healthy": eval_out.s_healthy,
#        "logit_flawed": eval_out.s_flawed,
#        "pred_is_anomaly": pred,
#    })
#    meta_df.to_csv(metadata_csv_path, index=False)
#def main() -> None:
#    parser = argparse.ArgumentParser(description="Fine-tune MotionCLIP with healthy/flawed action prompts on PerMo.")
#    # Data
#    parser.add_argument("--csv_path", required=True, help="Path to PerMo metadata CSV.")
#    parser.add_argument("--output_dir", required=True, help="Directory where outputs are saved.")
#    parser.add_argument("--path_col", default="motion_path")
#    parser.add_argument("--action_col", default="action_label")
#    parser.add_argument("--condition_col", default="condition_label")
#    parser.add_argument("--actor_col", default="", help="Optional CSV column containing actor IDs. If omitted, actor is parsed from filename, e.g. *_A01_001.npz.")
#    parser.add_argument("--label_col", default="is_anomaly")
#    parser.add_argument("--motion_key", default="auto", help="NPZ key. Use 'auto' to infer.")
#    parser.add_argument("--num_frames", type=int, default=60)
#    parser.add_argument("--njoints", type=int, default=25)
#    parser.add_argument("--nfeats", type=int, default=6)
#    # Split
#    parser.add_argument("--test_fraction", type=float, default=0.20, help="Seen-pool test fraction. Held-out classes/actors are entirely test-only.")
#    parser.add_argument("--val_fraction", type=float, default=0.10, help="Fraction of non-test seen-pool data used for validation.")
#    parser.add_argument("--seed", type=int, default=42)
#    parser.add_argument(
#        "--balance_test_sets",
#        action="store_true",
#        default=True,
#        help="Downsample majority label in rankable test sets so normal/anomaly counts are equal.",
#    )
#    parser.add_argument(
#        "--no_balance_test_sets",
#        action="store_false",
#        dest="balance_test_sets",
#        help="Keep natural label ratios in test sets.",
#    )
#    parser.add_argument("--unseen_actions", "--unseen_contents", nargs="*", default=[], help="Action/content names to hold out completely from training/validation.")
#    parser.add_argument("--unseen_actions_file", "--unseen_contents_file", default="", help="Optional text file with one unseen action/content name per line.")
#    parser.add_argument("--unseen_action_count", type=int, default=0, help="Randomly hold out this many action classes if --unseen_actions is not given.")
#    parser.add_argument("--unseen_action_fraction", type=float, default=0.20, help="Randomly hold out this fraction of action classes if no explicit unseen actions/count are given.")
#    parser.add_argument("--require_unseen_both_labels", action="store_true", default=True, help="Only randomly select unseen actions that contain both healthy and anomaly labels.")
#    parser.add_argument("--allow_unseen_single_label", action="store_false", dest="require_unseen_both_labels", help="Allow unseen actions with only one label when randomly selecting unseen actions.")
#    parser.add_argument("--unseen_styles", nargs="*", default=[], help="Style/condition names to hold out completely from training/validation. Do not include Healthy as an unseen style.")
#    parser.add_argument("--unseen_styles_file", default="", help="Optional text file with one unseen style/condition name per line.")
#    parser.add_argument("--unseen_style_count", type=int, default=0, help="Randomly hold out this many style/condition classes if --unseen_styles is not given.")
#    parser.add_argument("--unseen_style_fraction", type=float, default=0.0, help="Randomly hold out this fraction of style/condition classes. Default 0 means no random style holdout.")
#    parser.add_argument("--require_unseen_style_both_labels", action="store_true", default=True, help="Only randomly select unseen styles that contain both healthy and anomaly labels.")
#    parser.add_argument("--allow_unseen_style_single_label", action="store_false", dest="require_unseen_style_both_labels", help="Allow unseen styles with only one label when randomly selecting unseen styles.")
#    parser.add_argument("--unseen_actors", nargs="*", default=[], help="Actor IDs to hold out completely from training/validation, e.g. A01 A02.")
#    parser.add_argument("--unseen_actors_file", default="", help="Optional text file with one unseen actor ID per line.")
#    parser.add_argument("--unseen_actor_count", type=int, default=0, help="Randomly hold out this many actors if --unseen_actors is not given.")
#    parser.add_argument("--unseen_actor_fraction", type=float, default=0.0, help="Randomly hold out this fraction of actors. Default 0 means no random actor holdout.")
#    parser.add_argument("--require_unseen_actor_both_labels", action="store_true", default=True, help="Only randomly select unseen actors that contain both healthy and anomaly labels.")
#    parser.add_argument("--allow_unseen_actor_single_label", action="store_false", dest="require_unseen_actor_both_labels", help="Allow unseen actors with only one label when randomly selecting unseen actors.")
#    # MotionCLIP model
#    parser.add_argument("--project_root", default="", help="Parent directory containing the MotionCLIP folder.")
#    parser.add_argument("--checkpoint", required=True, help="Pretrained MotionCLIP checkpoint.")
#    parser.add_argument("--trainable_layers", type=int, default=2)
#    # Text prompts
#    parser.add_argument("--clip_model", default="ViT-B/32")
#    parser.add_argument("--normal_prompt_template", default="healthy {action}")
#    parser.add_argument("--anomaly_prompt_template", default="flawed {action}")
#    # Training
#    parser.add_argument("--epochs", type=int, default=20)
#    parser.add_argument("--batch_size", type=int, default=32)
#    parser.add_argument("--num_workers", type=int, default=4)
#    parser.add_argument("--lr", type=float, default=1e-5)
#    parser.add_argument("--weight_decay", type=float, default=1e-2)
#    parser.add_argument("--temperature", type=float, default=0.07)
#    parser.add_argument("--class_weight", choices=["auto", "none"], default="auto", help="Kept for compatibility; contrastive training does not use class weights.")
#    parser.add_argument("--grad_clip", type=float, default=1.0)
#    parser.add_argument("--amp", action="store_true", help="Use mixed precision.")
#    parser.add_argument("--threshold_criterion", default="f1", choices=["f1", "balanced_accuracy", "accuracy"])
#    args = parser.parse_args()
#    set_seed(args.seed)
#    output_dir = ensure_dir(args.output_dir)
#    ckpt_dir = ensure_dir(output_dir / "checkpoints")
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    save_json(vars(args), output_dir / "args.json")
#    # Load metadata
#    df = pd.read_csv(args.csv_path)
#    df = df.copy()
#    df["original_index"] = np.arange(len(df))
#    required_cols = [args.path_col, args.action_col, args.label_col]
#    missing = [c for c in required_cols if c not in df.columns]
#    if missing:
#        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")
#    # Normalize labels/actions/styles/actors
#    df[args.label_col] = df[args.label_col].astype(int)
#    if not set(df[args.label_col].unique()).issubset({0, 1}):
#        raise ValueError(f"{args.label_col} must contain only 0/1 labels.")
#    df["_action_norm"] = df[args.action_col].map(normalize_action_text)
#    style_holdout_requested = bool(args.unseen_styles or args.unseen_styles_file or args.unseen_style_count > 0 or args.unseen_style_fraction > 0)
#    if args.condition_col not in df.columns:
#        if style_holdout_requested:
#            raise ValueError(
#                f"Style holdout was requested, but condition/style column {args.condition_col!r} is missing. Found: {list(df.columns)}"
#            )
#        df["_style_norm"] = "__no_condition_column__"
#    else:
#        df["_style_norm"] = df[args.condition_col].map(normalize_action_text)
#    if args.actor_col:
#        if args.actor_col not in df.columns:
#            raise ValueError(f"--actor_col {args.actor_col!r} was provided but is missing from CSV. Found: {list(df.columns)}")
#        df["_actor_norm"] = df[args.actor_col].map(normalize_actor_id)
#    else:
#        df["_actor_norm"] = df[args.path_col].map(extract_actor_id_from_path)
#    # Basic file existence check
#    missing_paths = [p for p in df[args.path_col].head(20).tolist() if not Path(str(p)).exists()]
#    if missing_paths:
#        print("[WARN] Some example motion files do not exist from this machine.")
#        print("       This is okay only if you are testing the script outside the data machine.")
#        print(f"       First missing example: {missing_paths[0]}")
#    (
#        train_df,
#        val_df,
#        seen_test_df,
#        unseen_action_test_df,
#        unseen_style_test_df,
#        unseen_actor_test_df,
#        unseen_all_test_df,
#        other_unseen_combinations_df,
#        test_df,
#        heldout_unseen_actions,
#        heldout_unseen_styles,
#        heldout_unseen_actors,
#    ) = true_unseen_action_style_actor_split(
#        df=df,
#        action_norm_col="_action_norm",
#        style_norm_col="_style_norm",
#        actor_norm_col="_actor_norm",
#        condition_col=args.condition_col,
#        label_col=args.label_col,
#        test_fraction=args.test_fraction,
#        val_fraction=args.val_fraction,
#        seed=args.seed,
#        unseen_actions=args.unseen_actions,
#        unseen_action_file=args.unseen_actions_file,
#        unseen_action_count=args.unseen_action_count,
#        unseen_action_fraction=args.unseen_action_fraction,
#        require_unseen_both_labels=args.require_unseen_both_labels,
#        unseen_styles=args.unseen_styles,
#        unseen_styles_file=args.unseen_styles_file,
#        unseen_style_count=args.unseen_style_count,
#        unseen_style_fraction=args.unseen_style_fraction,
#        require_unseen_style_both_labels=args.require_unseen_style_both_labels,
#        unseen_actors=args.unseen_actors,
#        unseen_actors_file=args.unseen_actors_file,
#        unseen_actor_count=args.unseen_actor_count,
#        unseen_actor_fraction=args.unseen_actor_fraction,
#        require_unseen_actor_both_labels=args.require_unseen_actor_both_labels,
#    )
#    # Hard leakage checks: no held-out action/style/actor may appear in train/val.
#    train_actions = set(train_df["_action_norm"].unique().tolist())
#    val_actions = set(val_df["_action_norm"].unique().tolist())
#    train_styles = set(train_df["_style_norm"].unique().tolist())
#    val_styles = set(val_df["_style_norm"].unique().tolist())
#    train_actors = set(train_df["_actor_norm"].unique().tolist())
#    val_actors = set(val_df["_actor_norm"].unique().tolist())
#    action_leakage = sorted((train_actions | val_actions) & set(heldout_unseen_actions))
#    style_leakage = sorted((train_styles | val_styles) & set(heldout_unseen_styles))
#    actor_leakage = sorted((train_actors | val_actors) & set(heldout_unseen_actors))
#    if action_leakage:
#        raise RuntimeError(f"Unseen action leakage detected in train/val: {action_leakage}")
#    if style_leakage:
#        raise RuntimeError(f"Unseen style leakage detected in train/val: {style_leakage}")
#    if actor_leakage:
#        raise RuntimeError(f"Unseen actor leakage detected in train/val: {actor_leakage}")
#    test_balance_info = {
#        "enabled": bool(args.balance_test_sets),
#        "note": "If enabled, rankable splits with both labels are downsampled with a fixed seed so normal/anomaly counts are equal. Unseen-style ranking metrics are not reported.",
#    }
#    if args.balance_test_sets:
#        seen_test_df, seen_balance_info = safe_balance_binary_split(
#            seen_test_df, args.label_col, args.seed + 100, "seen_action_seen_style_seen_actor_test"
#        )
#        unseen_action_test_df, unseen_action_balance_info = safe_balance_binary_split(
#            unseen_action_test_df, args.label_col, args.seed + 200, "unseen_action_only_test", allow_single_label=True
#        )
#        unseen_actor_test_df, unseen_actor_balance_info = safe_balance_binary_split(
#            unseen_actor_test_df, args.label_col, args.seed + 300, "unseen_actor_only_test", allow_single_label=True
#        )
#        unseen_all_test_df, unseen_all_balance_info = safe_balance_binary_split(
#            unseen_all_test_df, args.label_col, args.seed + 350, "unseen_action_style_actor_test", allow_single_label=True
#        )
#        other_unseen_combinations_df, other_balance_info = safe_balance_binary_split(
#            other_unseen_combinations_df, args.label_col, args.seed + 375, "other_unseen_combinations_test", allow_single_label=True
#        )
#        # Do not balance unseen-style as a binary split: held-out flawed styles can be anomaly-only by design.
#        unseen_style_balance_info = {
#            "split_name": "unseen_style_only_test",
#            "balanced": False,
#            "reason": "Unseen styles exclude Healthy, so this split may contain anomaly samples only; AUROC/AUPRC are not reported.",
#            "n_samples": int(len(unseen_style_test_df)),
#            "n_normal": int((unseen_style_test_df[args.label_col].astype(int) == 0).sum()) if len(unseen_style_test_df) else 0,
#            "n_anomaly": int((unseen_style_test_df[args.label_col].astype(int) == 1).sum()) if len(unseen_style_test_df) else 0,
#        }
#        test_parts = [seen_test_df]
#        for part in [unseen_action_test_df, unseen_style_test_df, unseen_actor_test_df, unseen_all_test_df, other_unseen_combinations_df]:
#            if len(part) > 0:
#                test_parts.append(part)
#        test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=args.seed + 400).reset_index(drop=True)
#        test_df, combined_balance_info = safe_balance_binary_split(
#            test_df, args.label_col, args.seed + 500, "combined_test", allow_single_label=True
#        )
#        test_balance_info.update({
#            "seen_action_seen_style_seen_actor_test": seen_balance_info,
#            "unseen_action_only_test": unseen_action_balance_info,
#            "unseen_style_only_test": unseen_style_balance_info,
#            "unseen_actor_only_test": unseen_actor_balance_info,
#            "unseen_action_style_actor_test": unseen_all_balance_info,
#            "other_unseen_combinations_test": other_balance_info,
#            "combined_test": combined_balance_info,
#        })
#    # Save split CSVs
#    train_df.to_csv(output_dir / "split_train.csv", index=False)
#    val_df.to_csv(output_dir / "split_val.csv", index=False)
#    seen_test_df.to_csv(output_dir / "split_test_seen_action_seen_style_seen_actor.csv", index=False)
#    unseen_action_test_df.to_csv(output_dir / "split_test_unseen_actions.csv", index=False)
#    unseen_style_test_df.to_csv(output_dir / "split_test_unseen_styles.csv", index=False)
#    unseen_actor_test_df.to_csv(output_dir / "split_test_unseen_actors.csv", index=False)
#    unseen_all_test_df.to_csv(output_dir / "split_test_unseen_action_style_actor.csv", index=False)
#    other_unseen_combinations_df.to_csv(output_dir / "split_test_other_unseen_combinations.csv", index=False)
#    test_df.to_csv(output_dir / "split_test_combined.csv", index=False)
#    test_df.to_csv(output_dir / "split_test.csv", index=False)
#    split_summary = {
#        "split_type": "true_unseen_action_style_actor",
#        "total": int(len(df)),
#        "train": int(len(train_df)),
#        "val": int(len(val_df)),
#        "seen_action_seen_style_seen_actor_test": int(len(seen_test_df)),
#        "unseen_action_only_test": int(len(unseen_action_test_df)),
#        "unseen_style_only_test": int(len(unseen_style_test_df)),
#        "unseen_actor_only_test": int(len(unseen_actor_test_df)),
#        "unseen_action_style_actor_test": int(len(unseen_all_test_df)),
#        "other_unseen_combinations_test": int(len(other_unseen_combinations_df)),
#        "combined_test": int(len(test_df)),
#        "train_label_counts": train_df[args.label_col].value_counts().to_dict(),
#        "val_label_counts": val_df[args.label_col].value_counts().to_dict(),
#        "seen_test_label_counts": seen_test_df[args.label_col].value_counts().to_dict(),
#        "unseen_action_test_label_counts": unseen_action_test_df[args.label_col].value_counts().to_dict(),
#        "unseen_style_test_label_counts": unseen_style_test_df[args.label_col].value_counts().to_dict(),
#        "unseen_actor_test_label_counts": unseen_actor_test_df[args.label_col].value_counts().to_dict(),
#        "unseen_action_style_actor_test_label_counts": unseen_all_test_df[args.label_col].value_counts().to_dict(),
#        "other_unseen_combinations_label_counts": other_unseen_combinations_df[args.label_col].value_counts().to_dict(),
#        "combined_test_label_counts": test_df[args.label_col].value_counts().to_dict(),
#        "test_balance_info": test_balance_info,
#        "all_actions": sorted(df["_action_norm"].unique().tolist()),
#        "all_styles": sorted(df["_style_norm"].unique().tolist()),
#        "all_actors": sorted(df["_actor_norm"].unique().tolist()),
#        "seen_train_val_actions": sorted((train_actions | val_actions)),
#        "seen_train_val_styles": sorted((train_styles | val_styles)),
#        "seen_train_val_actors": sorted((train_actors | val_actors)),
#        "heldout_unseen_actions": heldout_unseen_actions,
#        "heldout_unseen_styles": heldout_unseen_styles,
#        "heldout_unseen_actors": heldout_unseen_actors,
#        "n_heldout_unseen_actions": int(len(heldout_unseen_actions)),
#        "n_heldout_unseen_styles": int(len(heldout_unseen_styles)),
#        "n_heldout_unseen_actors": int(len(heldout_unseen_actors)),
#        "unseen_action_leakage_check_passed": True,
#        "unseen_style_leakage_check_passed": True,
#        "unseen_actor_leakage_check_passed": True,
#        "unseen_style_ranking_metrics_reported": False,
#    }
#    save_json(split_summary, output_dir / "split_summary.json")
#    print("[INFO] Split summary:", split_summary)
#    expected_shape = (args.num_frames, args.njoints, args.nfeats)
#    train_ds = PerMoMotionDataset(train_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
#    val_ds = PerMoMotionDataset(val_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
#    test_ds = PerMoMotionDataset(test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
#    seen_test_ds = PerMoMotionDataset(seen_test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
#    unseen_action_test_ds = PerMoMotionDataset(unseen_action_test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
#    unseen_style_test_ds = PerMoMotionDataset(unseen_style_test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
#    unseen_actor_test_ds = PerMoMotionDataset(unseen_actor_test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
#    unseen_all_test_ds = PerMoMotionDataset(unseen_all_test_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
#    other_unseen_combinations_ds = PerMoMotionDataset(other_unseen_combinations_df, args.path_col, args.action_col, args.label_col, args.motion_key, expected_shape)
#    train_labels = train_df[args.label_col].astype(int).to_numpy()
#    train_batch_sampler = BalancedBinaryBatchSampler(labels=train_labels, batch_size=args.batch_size, seed=args.seed, drop_last=False)
#    sampler_info = {
#        "type": "BalancedBinaryBatchSampler",
#        "purpose": "class-aware healthy/flawed training batches",
#        "batch_size": int(args.batch_size),
#        "healthy_per_batch": int(train_batch_sampler.n0),
#        "flawed_per_batch": int(train_batch_sampler.n1),
#        "num_batches_per_epoch": int(len(train_batch_sampler)),
#        "train_label_counts": {"healthy_0": int((train_labels == 0).sum()), "flawed_1": int((train_labels == 1).sum())},
#    }
#    save_json(sampler_info, output_dir / "train_sampler_info.json")
#    print("[INFO] Train sampler info:", sampler_info)
#    def make_loader(ds: Dataset, shuffle: bool = False) -> DataLoader:
#        return DataLoader(
#            ds,
#            batch_size=args.batch_size,
#            shuffle=shuffle,
#            num_workers=args.num_workers,
#            pin_memory=torch.cuda.is_available(),
#            collate_fn=collate_batch,
#        )
#    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
#    train_eval_loader = make_loader(train_ds)
#    val_loader = make_loader(val_ds)
#    test_loader = make_loader(test_ds)
#    seen_test_loader = make_loader(seen_test_ds)
#    unseen_action_test_loader = make_loader(unseen_action_test_ds)
#    unseen_style_test_loader = make_loader(unseen_style_test_ds)
#    unseen_actor_test_loader = make_loader(unseen_actor_test_ds)
#    unseen_all_test_loader = make_loader(unseen_all_test_ds)
#    other_unseen_combinations_loader = make_loader(other_unseen_combinations_ds)
#    # Model
#    if args.project_root:
#        sys.path.insert(0, str(Path(args.project_root).resolve()))
#    global Encoder_TRANSFORMER
#    from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER
#    motion_encoder = build_motionclip_encoder(args.checkpoint, device)
#    save_json({"loaded": True, "checkpoint_path": args.checkpoint, "loader": "MotionCLIP encoder loader from finetune_unsupervised_updated.py"}, output_dir / "checkpoint_load_info.json")
#    unfreeze_info = freeze_encoder_except_last_layers(motion_encoder, num_trainable_blocks=args.trainable_layers)
#    save_json(unfreeze_info, output_dir / "unfreeze_info.json")
#    print("[INFO] Unfreeze info:", unfreeze_info)
#    # Text features
#    print("[INFO] Loading frozen CLIP text encoder...")
#    text_encoder = FrozenCLIPTextEncoder(args.clip_model, device)
#    all_actions = df[args.action_col].map(normalize_action_text).tolist()
#    action_to_idx, text_feats, prompt_info = build_prompt_cache(all_actions, text_encoder, args.normal_prompt_template, args.anomaly_prompt_template, device)
#    save_json(prompt_info, output_dir / "prompts.json")
#    torch.save({"action_to_idx": action_to_idx, "text_feats": text_feats.detach().cpu(), "prompt_info": prompt_info}, output_dir / "text_prompt_cache.pt")
#    # Loss/optim
#    class_weights = make_class_weights(train_df[args.label_col].tolist(), device, args.class_weight)
#    if class_weights is not None:
#        print("[INFO] Class weights [healthy, flawed]:", class_weights.detach().cpu().tolist())
#        print("[INFO] Contrastive training uses motion-text positives grouped by action and healthy/flawed label; class weights are saved but not applied to the loss.")
#    optimizer = torch.optim.AdamW([p for p in motion_encoder.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
#    best_val_auroc = -float("inf")
#    best_epoch = -1
#    best_threshold = 0.0
#    epoch_records = []
#    for epoch in range(1, args.epochs + 1):
#        t0 = time.time()
#        train_batch_sampler.set_epoch(epoch)
#        train_loss = train_one_epoch(
#            motion_encoder=motion_encoder,
#            loader=train_loader,
#            optimizer=optimizer,
#            device=device,
#            action_to_idx=action_to_idx,
#            text_feats=text_feats,
#            temperature=args.temperature,
#            grad_clip=args.grad_clip,
#            use_amp=args.amp,
#        )
#        val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, text_feats, args.temperature, True)
#        val_metrics = compute_binary_metrics(val_out.y_true, val_out.score, threshold=None, threshold_criterion=args.threshold_criterion)
#        record = {
#            "epoch": epoch,
#            "train_loss": train_loss,
#            "val_loss": val_out.loss,
#            "val_auroc": val_metrics["auroc"],
#            "val_auprc": val_metrics["auprc"],
#            "val_f1": val_metrics["f1"],
#            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
#            "val_threshold": val_metrics["threshold"],
#            "seconds": time.time() - t0,
#        }
#        epoch_records.append(record)
#        save_training_curves(epoch_records, output_dir)
#        print(f"[EPOCH {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_out.loss:.4f} val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f} val_f1={val_metrics['f1']:.4f} thr={val_metrics['threshold']:.4f}")
#        current_auroc = val_metrics["auroc"]
#        if np.isfinite(current_auroc) and current_auroc > best_val_auroc:
#            best_val_auroc = current_auroc
#            best_epoch = epoch
#            best_threshold = float(val_metrics["threshold"])
#            torch.save({
#                "epoch": epoch,
#                "motion_encoder_state_dict": motion_encoder.state_dict(),
#                "optimizer_state_dict": optimizer.state_dict(),
#                "args": vars(args),
#                "action_to_idx": action_to_idx,
#                "prompt_info": prompt_info,
#                "text_feats": text_feats.detach().cpu(),
#                "best_val_metrics": val_metrics,
#                "unfreeze_info": unfreeze_info,
#            }, ckpt_dir / "best_model.pt")
#            save_predictions(val_out, output_dir / "val_predictions_best.csv", threshold=best_threshold)
#    # Load best model for final test inference
#    best_ckpt_path = ckpt_dir / "best_model.pt"
#    if best_ckpt_path.exists():
#        best_ckpt = torch.load(best_ckpt_path, map_location=device)
#        motion_encoder.load_state_dict(best_ckpt["motion_encoder_state_dict"], strict=True)
#        best_threshold = float(best_ckpt["best_val_metrics"]["threshold"])
#    else:
#        print("[WARN] No best checkpoint saved. Testing final epoch model.")
#        best_threshold = 0.0
#    # Final eval
#    train_out = evaluate(motion_encoder, train_eval_loader, device, action_to_idx, text_feats, args.temperature, True)
#    val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, text_feats, args.temperature, True)
#    test_out = evaluate(motion_encoder, test_loader, device, action_to_idx, text_feats, args.temperature, True)
#    seen_test_out = evaluate(motion_encoder, seen_test_loader, device, action_to_idx, text_feats, args.temperature, True)
#    unseen_action_test_out = evaluate(motion_encoder, unseen_action_test_loader, device, action_to_idx, text_feats, args.temperature, True)
#    unseen_style_test_out = evaluate(motion_encoder, unseen_style_test_loader, device, action_to_idx, text_feats, args.temperature, True)
#    unseen_actor_test_out = evaluate(motion_encoder, unseen_actor_test_loader, device, action_to_idx, text_feats, args.temperature, True)
#    unseen_all_test_out = evaluate(motion_encoder, unseen_all_test_loader, device, action_to_idx, text_feats, args.temperature, True)
#    other_unseen_combinations_out = evaluate(motion_encoder, other_unseen_combinations_loader, device, action_to_idx, text_feats, args.temperature, True)
#    train_metrics = compute_binary_metrics(train_out.y_true, train_out.score, threshold=best_threshold)
#    val_metrics_final = compute_binary_metrics(val_out.y_true, val_out.score, threshold=best_threshold)
#    test_metrics = compute_binary_metrics(test_out.y_true, test_out.score, threshold=best_threshold)
#    seen_test_metrics = compute_binary_metrics(seen_test_out.y_true, seen_test_out.score, threshold=best_threshold)
#    unseen_action_test_metrics = compute_binary_metrics(unseen_action_test_out.y_true, unseen_action_test_out.score, threshold=best_threshold)
#    unseen_actor_test_metrics = compute_binary_metrics(unseen_actor_test_out.y_true, unseen_actor_test_out.score, threshold=best_threshold)
#    unseen_all_test_metrics = compute_binary_metrics(unseen_all_test_out.y_true, unseen_all_test_out.score, threshold=best_threshold)
#    other_unseen_combinations_metrics = compute_binary_metrics(other_unseen_combinations_out.y_true, other_unseen_combinations_out.score, threshold=best_threshold)
#    unseen_style_test_metrics = metrics_without_ranking_for_single_label_style(unseen_style_test_out, best_threshold, "unseen_style_only_test")
#    for m, out in [
#        (train_metrics, train_out),
#        (val_metrics_final, val_out),
#        (test_metrics, test_out),
#        (seen_test_metrics, seen_test_out),
#        (unseen_action_test_metrics, unseen_action_test_out),
#        (unseen_style_test_metrics, unseen_style_test_out),
#        (unseen_actor_test_metrics, unseen_actor_test_out),
#        (unseen_all_test_metrics, unseen_all_test_out),
#        (other_unseen_combinations_metrics, other_unseen_combinations_out),
#    ]:
#        m["loss"] = out.loss
#    # Save predictions
#    save_predictions(train_out, output_dir / "train_predictions.csv", threshold=best_threshold)
#    save_predictions(val_out, output_dir / "val_predictions.csv", threshold=best_threshold)
#    save_predictions(test_out, output_dir / "test_predictions_combined.csv", threshold=best_threshold)
#    save_predictions(seen_test_out, output_dir / "test_predictions_seen_action_seen_style_seen_actor.csv", threshold=best_threshold)
#    save_predictions(unseen_action_test_out, output_dir / "test_predictions_unseen_actions.csv", threshold=best_threshold)
#    save_predictions(unseen_style_test_out, output_dir / "test_predictions_unseen_styles.csv", threshold=best_threshold)
#    save_predictions(unseen_actor_test_out, output_dir / "test_predictions_unseen_actors.csv", threshold=best_threshold)
#    save_predictions(unseen_all_test_out, output_dir / "test_predictions_unseen_action_style_actor.csv", threshold=best_threshold)
#    save_predictions(other_unseen_combinations_out, output_dir / "test_predictions_other_unseen_combinations.csv", threshold=best_threshold)
#    save_predictions(test_out, output_dir / "test_predictions.csv", threshold=best_threshold)
#    # Save embeddings
#    save_embeddings(train_out, output_dir / "train_embeddings.npz", output_dir / "train_embeddings_metadata.csv", threshold=best_threshold)
#    save_embeddings(val_out, output_dir / "val_embeddings.npz", output_dir / "val_embeddings_metadata.csv", threshold=best_threshold)
#    save_embeddings(test_out, output_dir / "test_embeddings_combined.npz", output_dir / "test_embeddings_combined_metadata.csv", threshold=best_threshold)
#    save_embeddings(seen_test_out, output_dir / "test_embeddings_seen_action_seen_style_seen_actor.npz", output_dir / "test_embeddings_seen_action_seen_style_seen_actor_metadata.csv", threshold=best_threshold)
#    save_embeddings(unseen_action_test_out, output_dir / "test_embeddings_unseen_actions.npz", output_dir / "test_embeddings_unseen_actions_metadata.csv", threshold=best_threshold)
#    save_embeddings(unseen_style_test_out, output_dir / "test_embeddings_unseen_styles.npz", output_dir / "test_embeddings_unseen_styles_metadata.csv", threshold=best_threshold)
#    save_embeddings(unseen_actor_test_out, output_dir / "test_embeddings_unseen_actors.npz", output_dir / "test_embeddings_unseen_actors_metadata.csv", threshold=best_threshold)
#    save_embeddings(unseen_all_test_out, output_dir / "test_embeddings_unseen_action_style_actor.npz", output_dir / "test_embeddings_unseen_action_style_actor_metadata.csv", threshold=best_threshold)
#    save_embeddings(other_unseen_combinations_out, output_dir / "test_embeddings_other_unseen_combinations.npz", output_dir / "test_embeddings_other_unseen_combinations_metadata.csv", threshold=best_threshold)
#    save_embeddings(test_out, output_dir / "test_embeddings.npz", output_dir / "test_embeddings_metadata.csv", threshold=best_threshold)
#    final_summary = {
#        "best_epoch": best_epoch,
#        "best_val_auroc_during_training": best_val_auroc,
#        "threshold_selected_on_validation": best_threshold,
#        "train_metrics": train_metrics,
#        "val_metrics": val_metrics_final,
#        "test_metrics": test_metrics,
#        "seen_action_seen_style_seen_actor_test_metrics": seen_test_metrics,
#        "unseen_action_test_metrics": unseen_action_test_metrics,
#        "unseen_style_test_metrics": unseen_style_test_metrics,
#        "unseen_actor_test_metrics": unseen_actor_test_metrics,
#        "unseen_action_style_actor_test_metrics": unseen_all_test_metrics,
#        "other_unseen_combinations_test_metrics": other_unseen_combinations_metrics,
#        "split_summary": split_summary,
#        "prompt_templates": {"normal": args.normal_prompt_template, "anomaly": args.anomaly_prompt_template},
#        "output_files": {
#            "best_checkpoint": str(best_ckpt_path),
#            "epoch_metrics": str(output_dir / "epoch_metrics.csv"),
#            "test_predictions_combined": str(output_dir / "test_predictions_combined.csv"),
#            "test_predictions_seen_action_seen_style_seen_actor": str(output_dir / "test_predictions_seen_action_seen_style_seen_actor.csv"),
#            "test_predictions_unseen_actions": str(output_dir / "test_predictions_unseen_actions.csv"),
#            "test_predictions_unseen_styles": str(output_dir / "test_predictions_unseen_styles.csv"),
#            "test_predictions_unseen_actors": str(output_dir / "test_predictions_unseen_actors.csv"),
#            "test_predictions_unseen_action_style_actor": str(output_dir / "test_predictions_unseen_action_style_actor.csv"),
#            "test_predictions_other_unseen_combinations": str(output_dir / "test_predictions_other_unseen_combinations.csv"),
#            "train_embeddings": str(output_dir / "train_embeddings.npz"),
#            "val_embeddings": str(output_dir / "val_embeddings.npz"),
#            "test_embeddings_combined": str(output_dir / "test_embeddings_combined.npz"),
#            "test_embeddings_seen_action_seen_style_seen_actor": str(output_dir / "test_embeddings_seen_action_seen_style_seen_actor.npz"),
#            "test_embeddings_unseen_actions": str(output_dir / "test_embeddings_unseen_actions.npz"),
#            "test_embeddings_unseen_styles": str(output_dir / "test_embeddings_unseen_styles.npz"),
#            "test_embeddings_unseen_actors": str(output_dir / "test_embeddings_unseen_actors.npz"),
#            "test_embeddings_unseen_action_style_actor": str(output_dir / "test_embeddings_unseen_action_style_actor.npz"),
#            "test_embeddings_other_unseen_combinations": str(output_dir / "test_embeddings_other_unseen_combinations.npz"),
#            "metrics": str(output_dir / "metrics.json"),
#            "training_history_npz": str(output_dir / "training_history.npz"),
#            "loss_curves": str(output_dir / "loss_curves.png"),
#            "validation_metrics_plot": str(output_dir / "validation_metrics.png"),
#        },
#    }
#    save_json(final_summary, output_dir / "metrics.json")
#    print("[DONE] Final combined test metrics:")
#    print(json.dumps(test_metrics, indent=2, sort_keys=True))
#    print("[DONE] Seen action/style/actor test metrics:")
#    print(json.dumps(seen_test_metrics, indent=2, sort_keys=True))
#    print("[DONE] Truly unseen-action-only test metrics:")
#    print(json.dumps(unseen_action_test_metrics, indent=2, sort_keys=True))
#    print("[DONE] Truly unseen-style-only test metrics; AUROC/AUPRC intentionally not reported:")
#    print(json.dumps(unseen_style_test_metrics, indent=2, sort_keys=True))
#    print("[DONE] Truly unseen-actor-only test metrics:")
#    print(json.dumps(unseen_actor_test_metrics, indent=2, sort_keys=True))
#    print("[DONE] Hard test where action, style, and actor are all unseen:")
#    print(json.dumps(unseen_all_test_metrics, indent=2, sort_keys=True))
#    print(f"[DONE] Outputs saved to: {output_dir}")
#if __name__ == "__main__":
#    main()
