#!/usr/bin/env python3
"""
Two-stage AA-CLIP adaptation of pretrained MotionCLIP for PerMo Condition anomaly detection.
Stage 1 freezes MotionCLIP and the original CLIP backbone, then trains shallow CLIP text
residual adapters plus the text projection to produce disentangled normal/anomaly anchors.
Stage 2 freezes those anchors and all original MotionCLIP parameters, then trains shallow
MotionCLIP residual adapters and multi-level projectors. The script performs splitting,
training, validation-loss checkpoint selection, validation-threshold selection, strict test inference, optional prompted unseen-style
inference, per-style/action/actor metrics, and embedding export.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
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
def torch_load_compat(path: str | Path, map_location: Any = "cpu") -> Any:
    """Load trusted local checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
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
    _plot(["train_auroc", "val_auroc"], "auroc_curves.png", "AUROC")
    _plot(["val_auroc", "val_auprc", "val_f1", "val_balanced_accuracy"], "validation_metrics.png", "metric")
def normalize_action_text(text: str) -> str:
    text = str(text).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())
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
        condition_col: str,
        label_col: str,
        motion_key: str = "auto",
        expected_shape: Tuple[int, int, int] = (60, 25, 6),
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.path_col = path_col
        self.action_col = action_col
        self.condition_col = condition_col
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
            "condition": normalize_action_text(row[self.condition_col]),
            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),  # 0 healthy, 1 anomaly/non-healthy
            "path": path,
            "row_index": int(row.get("original_index", idx)),
        }
def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "motion": torch.stack([b["motion"] for b in batch], dim=0),  # [B,T,J,F]
        "action": [b["action"] for b in batch],
        "condition": [b["condition"] for b in batch],
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
        "row_index": [b["row_index"] for b in batch],
    }
class BalancedBinaryBatchSampler(Sampler[List[int]]):
    """Yield batches with an approximately 50/50 split between labels 0 and 1.
    Minority-class samples are oversampled with replacement when needed.
    This is used only for training; validation/test loaders remain deterministic.
    Two seeds are intentionally separated:
      - seed controls the selected training index multiset, including any oversampled duplicates.
      - order_seed controls only the order/batch arrangement of that fixed index multiset.
    Therefore, changing order_seed changes the order in which the same selected training
    examples are seen, without changing the train/val/test split or adapter initialization.
    """
    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        seed: int = 42,
        drop_last: bool = False,
        order_seed: Optional[int] = None,
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
        self.order_seed = int(seed if order_seed is None else order_seed)
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
        sample_rng = np.random.default_rng(self.seed + self.epoch)
        order_rng = np.random.default_rng(self.order_seed + self.epoch)
        # Fixed by self.seed: this controls which original rows, including any
        # oversampled duplicates, will be used in this epoch.
        labels0 = self._sample_class_indices(0, self.num_batches * self.n0, sample_rng)
        labels1 = self._sample_class_indices(1, self.num_batches * self.n1, sample_rng)
        # Fixed by self.order_seed: this changes only the order/batch arrangement
        # of the already selected indices above.
        labels0 = labels0[order_rng.permutation(len(labels0))]
        labels1 = labels1[order_rng.permutation(len(labels1))]
        for batch_idx in range(self.num_batches):
            b0 = labels0[batch_idx * self.n0:(batch_idx + 1) * self.n0]
            b1 = labels1[batch_idx * self.n1:(batch_idx + 1) * self.n1]
            batch = np.concatenate([b0, b1])
            order_rng.shuffle(batch)
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
      1 = non-healthy/anomaly
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
def parse_list_arg(values: Optional[Sequence[str]]) -> List[str]:
    """Accepts repeated args and/or comma-separated strings."""
    out: List[str] = []
    if not values:
        return out
    for v in values:
        for part in str(v).split(','):
            part = part.strip()
            if part:
                out.append(normalize_action_text(part))
    return sorted(set(out))
def _quota_counts(keys: Sequence[str], total: int) -> Dict[str, int]:
    """Nearly equal integer quotas that sum to total."""
    keys = sorted(set(map(str, keys)))
    if not keys:
        return {}
    base = total // len(keys)
    rem = total % len(keys)
    return {k: base + (1 if i < rem else 0) for i, k in enumerate(keys)}
def balanced_marginal_sample(
    df: pd.DataFrame,
    n_total: int,
    style_col: str,
    action_col: str,
    actor_col: str,
    seed: int,
    split_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Sample n_total rows while keeping style/action/actor marginals as even as possible.
    For PerMo Condition with 6 anomaly styles, 10 actions and 5 actors, n_total=200 gives:
      style quota: 33/34 each
      action quota: 20 each
      actor quota: 40 each
    The exact joint distribution cannot always be perfectly equal because 200 is not divisible by
    style*action*actor combinations, so this uses deterministic greedy marginal balancing.
    """
    if len(df) < n_total:
        raise ValueError(f"Cannot sample {n_total} rows for {split_name}; only {len(df)} rows available.")
    rng = np.random.default_rng(seed)
    work = df.copy()
    work[style_col] = work[style_col].map(normalize_action_text)
    work[action_col] = work[action_col].map(normalize_action_text)
    work[actor_col] = work[actor_col].astype(str).map(normalize_action_text)
    style_quota = _quota_counts(work[style_col].unique().tolist(), n_total)
    action_quota = _quota_counts(work[action_col].unique().tolist(), n_total)
    actor_quota = _quota_counts(work[actor_col].unique().tolist(), n_total)
    remaining = work.sample(frac=1.0, random_state=seed).copy()
    selected_indices: List[int] = []
    selected_style = {k: 0 for k in style_quota}
    selected_action = {k: 0 for k in action_quota}
    selected_actor = {k: 0 for k in actor_quota}
    for _ in range(n_total):
        best_i = None
        best_score = None
        # Randomly inspect all remaining rows; this is small for PerMo, so simple is okay.
        for idx, row in remaining.iterrows():
            st, ac, ar = row[style_col], row[action_col], row[actor_col]
            # Positive terms prioritize under-filled marginals. Negative terms discourage overfill.
            score = (
                3.0 * (style_quota[st] - selected_style[st]) +
                2.0 * (action_quota[ac] - selected_action[ac]) +
                2.0 * (actor_quota[ar] - selected_actor[ar])
            )
            # Prefer not taking more than available quota, but allow it if necessary.
            over_penalty = 0.0
            over_penalty += max(0, selected_style[st] + 1 - style_quota[st]) * 10.0
            over_penalty += max(0, selected_action[ac] + 1 - action_quota[ac]) * 5.0
            over_penalty += max(0, selected_actor[ar] + 1 - actor_quota[ar]) * 5.0
            jitter = float(rng.normal(0, 1e-6))
            score = score - over_penalty + jitter
            if best_score is None or score > best_score:
                best_score = score
                best_i = idx
        assert best_i is not None
        row = remaining.loc[best_i]
        selected_indices.append(best_i)
        selected_style[row[style_col]] += 1
        selected_action[row[action_col]] += 1
        selected_actor[row[actor_col]] += 1
        remaining = remaining.drop(index=best_i)
    sampled = work.loc[selected_indices].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    info = {
        "split_name": split_name,
        "n_requested": int(n_total),
        "n_sampled": int(len(sampled)),
        "style_counts": sampled[style_col].value_counts().sort_index().to_dict(),
        "action_counts": sampled[action_col].value_counts().sort_index().to_dict(),
        "actor_counts": sampled[actor_col].value_counts().sort_index().to_dict(),
        "style_quota": style_quota,
        "action_quota": action_quota,
        "actor_quota": actor_quota,
    }
    return sampled, info
def build_balanced_condition_subset(
    df: pd.DataFrame,
    action_col: str,
    condition_col: str,
    actor_col: str,
    label_col: str,
    healthy_condition: str,
    normal_target: int,
    anomaly_target: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create the 200 healthy + 200 anomaly Condition subset used by all experiments."""
    healthy_condition = normalize_action_text(healthy_condition)
    work = df.copy()
    work[action_col] = work[action_col].map(normalize_action_text)
    work[condition_col] = work[condition_col].map(normalize_action_text)
    work[actor_col] = work[actor_col].astype(str).map(normalize_action_text)
    healthy_pool = work[work[condition_col] == healthy_condition].copy()
    anomaly_pool = work[work[condition_col] != healthy_condition].copy()
    if len(healthy_pool) < normal_target:
        raise ValueError(f"Need {normal_target} healthy samples, found only {len(healthy_pool)}.")
    if len(anomaly_pool) < anomaly_target:
        raise ValueError(f"Need {anomaly_target} anomaly samples, found only {len(anomaly_pool)}.")
    # Usually exactly 200. If more, downsample while balancing actions/actors. Style is constant healthy.
    if len(healthy_pool) == normal_target:
        healthy_sample = healthy_pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        healthy_info = {
            "n_sampled": int(len(healthy_sample)),
            "action_counts": healthy_sample[action_col].value_counts().sort_index().to_dict(),
            "actor_counts": healthy_sample[actor_col].value_counts().sort_index().to_dict(),
        }
    else:
        healthy_tmp = healthy_pool.copy()
        healthy_tmp["_healthy_style_for_sampling"] = healthy_condition
        healthy_sample, healthy_info = balanced_marginal_sample(
            healthy_tmp,
            n_total=normal_target,
            style_col="_healthy_style_for_sampling",
            action_col=action_col,
            actor_col=actor_col,
            seed=seed,
            split_name="healthy_normal_subset",
        )
        healthy_sample = healthy_sample.drop(columns=["_healthy_style_for_sampling"], errors="ignore")
    anomaly_sample, anomaly_info = balanced_marginal_sample(
        anomaly_pool,
        n_total=anomaly_target,
        style_col=condition_col,
        action_col=action_col,
        actor_col=actor_col,
        seed=seed + 10,
        split_name="balanced_anomaly_subset",
    )
    healthy_sample[label_col] = 0
    anomaly_sample[label_col] = 1
    subset = pd.concat([healthy_sample, anomaly_sample], axis=0).sample(frac=1.0, random_state=seed + 20).reset_index(drop=True)
    info = {
        "normal_target": int(normal_target),
        "anomaly_target": int(anomaly_target),
        "healthy_info": healthy_info,
        "anomaly_info": anomaly_info,
        "final_label_counts": subset[label_col].value_counts().sort_index().to_dict(),
        "final_condition_counts": subset[condition_col].value_counts().sort_index().to_dict(),
        "final_action_counts": subset[action_col].value_counts().sort_index().to_dict(),
        "final_actor_counts": subset[actor_col].value_counts().sort_index().to_dict(),
    }
    return subset, info
def _marginal_split_counts(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for c in cols:
        if c in df.columns:
            out[c] = {str(k): int(v) for k, v in df[c].value_counts().sort_index().to_dict().items()}
    return out
def _pick_marginally_balanced_rows(
    df: pd.DataFrame,
    n_total: int,
    balance_cols: Sequence[str],
    seed: int,
    split_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Pick n_total rows while keeping the requested marginal columns as even as possible.
    This does NOT require each exact (style, action, actor) combination to be splittable.
    Instead, each selected row counts simultaneously toward its style, action, and actor margins.
    That avoids the empty-validation issue caused by tiny groups.
    """
    if n_total <= 0:
        return df.iloc[0:0].copy().reset_index(drop=True), {
            "split_name": split_name,
            "n_requested": int(n_total),
            "n_selected": 0,
            "balance_columns": list(balance_cols),
            "counts": {},
        }
    if len(df) < n_total:
        raise ValueError(f"Cannot pick {n_total} rows for {split_name}; only {len(df)} rows available.")
    rng = np.random.default_rng(seed)
    work = df.copy()
    for c in balance_cols:
        work[c] = work[c].astype(str).map(normalize_action_text)
    # Uniform quotas over values that exist in this pool.
    quotas: Dict[str, Dict[str, int]] = {}
    selected_counts: Dict[str, Dict[str, int]] = {}
    for c in balance_cols:
        values = sorted(work[c].dropna().unique().tolist())
        quotas[c] = _quota_counts(values, n_total)
        selected_counts[c] = {v: 0 for v in values}
    remaining = work.sample(frac=1.0, random_state=seed).copy()
    selected_indices: List[int] = []
    # Earlier columns get slightly more weight. In this script that means:
    # label handled outside, then style, action, actor within label.
    col_weights = {c: float(len(balance_cols) - i) for i, c in enumerate(balance_cols)}
    for _ in range(n_total):
        best_idx = None
        best_score = None
        for idx, row in remaining.iterrows():
            score = 0.0
            over_penalty = 0.0
            for c in balance_cols:
                value = row[c]
                quota = quotas[c].get(value, 0)
                current = selected_counts[c].get(value, 0)
                score += col_weights[c] * (quota - current)
                over_penalty += col_weights[c] * max(0, current + 1 - quota) * 10.0
            score = score - over_penalty + float(rng.normal(0, 1e-6))
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None
        row = remaining.loc[best_idx]
        selected_indices.append(best_idx)
        for c in balance_cols:
            selected_counts[c][row[c]] += 1
        remaining = remaining.drop(index=best_idx)
    selected = work.loc[selected_indices].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    info = {
        "split_name": split_name,
        "n_requested": int(n_total),
        "n_selected": int(len(selected)),
        "balance_columns": list(balance_cols),
        "quotas": quotas,
        "counts": _marginal_split_counts(selected, balance_cols),
    }
    return selected, info
def balanced_label_marginal_train_val_test_split(
    df: pd.DataFrame,
    label_col: str,
    condition_col: str,
    action_col: str,
    actor_col: str,
    healthy_condition: str,
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split a balanced subset by label first, then balance style/action/actor margins.
    Important difference from the old splitter:
      old: split every tiny (label, style, action, actor) group separately
      new: split normal/anomaly separately, and within each label keep style/action/actor
           distributions approximately uniform.
    This keeps validation non-empty and keeps the test set label-balanced.
    """
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1).")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1).")
    work = df.copy()
    work[condition_col] = work[condition_col].astype(str).map(normalize_action_text)
    work[action_col] = work[action_col].astype(str).map(normalize_action_text)
    work[actor_col] = work[actor_col].astype(str).map(normalize_action_text)
    healthy_condition = normalize_action_text(healthy_condition)
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    split_infos: Dict[str, Any] = {}
    for label_value in sorted(work[label_col].astype(int).unique().tolist()):
        pool = work[work[label_col].astype(int) == int(label_value)].copy()
        n = len(pool)
        if n < 3:
            raise ValueError(f"Label {label_value} has only {n} rows after unseen holdout; cannot make train/val/test.")
        # Keep the existing script semantics: test is taken first, val is from the remaining pool.
        n_test = int(round(n * test_fraction))
        n_test = min(max(1 if test_fraction > 0 else 0, n_test), max(0, n - 2))
        n_remaining_after_test = n - n_test
        n_val = int(round(n_remaining_after_test * val_fraction))
        n_val = min(max(1 if val_fraction > 0 else 0, n_val), max(0, n_remaining_after_test - 1))
        # Healthy has only one style, so balancing condition is pointless for label 0.
        # Anomaly uses condition/style + action + actor.
        if int(label_value) == 0:
            balance_cols = [action_col, actor_col]
        else:
            balance_cols = [condition_col, action_col, actor_col]
        test_df, test_info = _pick_marginally_balanced_rows(
            pool, n_test, balance_cols, seed + 1000 + int(label_value), f"label_{label_value}_test"
        )
        remaining = pool.drop(index=test_df["original_index"].values, errors="ignore") if "original_index" in test_df.columns else pool.drop(index=test_df.index, errors="ignore")
        # Because selected/test_df is reset_index'ed, dropping by original_index only works if original_index equals index.
        # Use a stable helper key instead.
        if "_split_row_id" not in pool.columns:
            pass
        # Stable removal by original dataframe index saved before reset.
        selected_orig_indices = set(test_df.get("original_index", pd.Series([], dtype=int)).tolist())
        if selected_orig_indices:
            remaining = pool[~pool["original_index"].isin(selected_orig_indices)].copy()
        else:
            remaining = pool.drop(index=test_df.index, errors="ignore").copy()
        val_df, val_info = _pick_marginally_balanced_rows(
            remaining, n_val, balance_cols, seed + 2000 + int(label_value), f"label_{label_value}_val"
        )
        selected_val_orig_indices = set(val_df.get("original_index", pd.Series([], dtype=int)).tolist())
        if selected_val_orig_indices:
            train_df = remaining[~remaining["original_index"].isin(selected_val_orig_indices)].copy()
        else:
            train_df = remaining.drop(index=val_df.index, errors="ignore").copy()
        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)
        split_infos[str(label_value)] = {
            "n_pool": int(n),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "balance_cols": balance_cols,
            "test_info": test_info,
            "val_info": val_info,
            "train_counts": _marginal_split_counts(train_df, balance_cols),
        }
    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed + 10).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=seed + 11).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed + 12).reset_index(drop=True)
    info = {
        "splitter": "balanced_label_marginal_train_val_test_split",
        "test_fraction": float(test_fraction),
        "val_fraction_from_non_test_pool": float(val_fraction),
        "label_infos": split_infos,
        "train_label_counts": {int(k): int(v) for k, v in train_df[label_col].value_counts().sort_index().to_dict().items()},
        "val_label_counts": {int(k): int(v) for k, v in val_df[label_col].value_counts().sort_index().to_dict().items()},
        "test_label_counts": {int(k): int(v) for k, v in test_df[label_col].value_counts().sort_index().to_dict().items()},
    }
    return train_df, val_df, test_df, info
def split_balanced_condition_experiment(
    df: pd.DataFrame,
    action_col: str,
    condition_col: str,
    actor_col: str,
    label_col: str,
    healthy_condition: str,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    unseen_actions: Sequence[str],
    unseen_actors: Sequence[str],
    unseen_styles: Sequence[str],
    include_seen_healthy_in_unseen_style_test: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """One splitter for all three requested experiments.
    Experiment 1: no unseen args -> split the balanced 200+200 subset by label first,
      then keep style/action/actor marginals approximately uniform.
    Experiment 2: pass --unseen_actions and/or --unseen_actors -> those rows are test-only.
    Experiment 3: additionally pass --unseen_styles -> those anomaly styles are test-only.
    """
    work = df.copy()
    work[action_col] = work[action_col].map(normalize_action_text)
    work[condition_col] = work[condition_col].map(normalize_action_text)
    work[actor_col] = work[actor_col].astype(str).map(normalize_action_text)
    unseen_actions = set(parse_list_arg(unseen_actions))
    unseen_actors = set(parse_list_arg(unseen_actors))
    unseen_styles = set(parse_list_arg(unseen_styles))
    healthy_condition = normalize_action_text(healthy_condition)
    if healthy_condition in unseen_styles:
        raise ValueError("Do not pass healthy as an unseen style. Healthy is the normal class.")
    unknown_actions = sorted(unseen_actions - set(work[action_col].unique()))
    unknown_actors = sorted(unseen_actors - set(work[actor_col].unique()))
    unknown_styles = sorted(unseen_styles - set(work[condition_col].unique()))
    if unknown_actions or unknown_actors or unknown_styles:
        raise ValueError(
            f"Unknown unseen values. actions={unknown_actions}, actors={unknown_actors}, styles={unknown_styles}."
        )
    has_unseen = bool(unseen_actions or unseen_actors or unseen_styles)
    if not has_unseen:
        train_df, val_df, seen_test_df, marginal_split_info = balanced_label_marginal_train_val_test_split(
            work,
            label_col=label_col,
            condition_col=condition_col,
            action_col=action_col,
            actor_col=actor_col,
            healthy_condition=healthy_condition,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            seed=seed,
        )
        unseen_test_df = work.iloc[0:0].copy()
        combined_test_df = seen_test_df.copy()
        split_type = "balanced_random_seen_split_marginal"
    else:
        heldout_mask = pd.Series(False, index=work.index)
        if unseen_actions:
            heldout_mask |= work[action_col].isin(unseen_actions)
        if unseen_actors:
            heldout_mask |= work[actor_col].isin(unseen_actors)
        if unseen_styles:
            heldout_mask |= work[condition_col].isin(unseen_styles)
        unseen_test_df = work[heldout_mask].copy()
        seen_pool = work[~heldout_mask].copy()
        if len(unseen_test_df) == 0:
            raise ValueError("The unseen arguments selected zero test rows.")
        if len(seen_pool) == 0:
            raise ValueError("The unseen arguments removed all rows, so no training data remains.")
        if not set(seen_pool[label_col].astype(int).unique()).issuperset({0, 1}):
            raise ValueError("The remaining seen training pool must contain both healthy and anomaly samples.")
        train_df, val_df, seen_test_df, marginal_split_info = balanced_label_marginal_train_val_test_split(
            seen_pool,
            label_col=label_col,
            condition_col=condition_col,
            action_col=action_col,
            actor_col=actor_col,
            healthy_condition=healthy_condition,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            seed=seed,
        )
        # If only styles are unseen, unseen_test has anomalies only because healthy has no matching style.
        # Add held-out healthy rows from the seen test split so AUROC on this test is defined.
        if include_seen_healthy_in_unseen_style_test and unseen_styles and int((unseen_test_df[label_col] == 0).sum()) == 0:
            normal_support = seen_test_df[seen_test_df[label_col].astype(int) == 0].copy()
            if len(normal_support) > 0:
                unseen_test_df = pd.concat([normal_support, unseen_test_df], axis=0).sample(
                    frac=1.0, random_state=seed + 123
                ).reset_index(drop=True)
        combined_test_df = pd.concat([seen_test_df, unseen_test_df], axis=0).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
        split_type = "balanced_unseen_holdout_split_marginal"
    train_df = train_df.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed + 4).reset_index(drop=True)
    seen_test_df = seen_test_df.sample(frac=1.0, random_state=seed + 5).reset_index(drop=True)
    unseen_test_df = unseen_test_df.sample(frac=1.0, random_state=seed + 6).reset_index(drop=True)
    combined_test_df = combined_test_df.sample(frac=1.0, random_state=seed + 7).reset_index(drop=True)
    info = {
        "split_type": split_type,
        "unseen_actions": sorted(unseen_actions),
        "unseen_actors": sorted(unseen_actors),
        "unseen_styles": sorted(unseen_styles),
        "include_seen_healthy_in_unseen_style_test": bool(include_seen_healthy_in_unseen_style_test),
        "marginal_split_info": marginal_split_info,
    }
    return train_df, val_df, seen_test_df, unseen_test_df, combined_test_df, info
def _concat_unique_rows(dfs: Sequence[pd.DataFrame], seed: int) -> pd.DataFrame:
    """Concatenate dataframes and remove duplicate CSV rows using original_index when available."""
    non_empty = [d.copy() for d in dfs if d is not None and len(d) > 0]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, axis=0)
    if "original_index" in out.columns:
        out = out.drop_duplicates(subset=["original_index"], keep="first")
    else:
        out = out.drop_duplicates(keep="first")
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
def _balance_eval_binary(df: pd.DataFrame, label_col: str, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Downsample an evaluation bucket to equal normal/anomaly counts when both labels exist."""
    if len(df) == 0:
        return df.copy(), {"balanced": False, "reason": "empty", "n_before": 0, "n_after": 0}
    counts_before = {int(k): int(v) for k, v in df[label_col].astype(int).value_counts().sort_index().to_dict().items()}
    n0 = counts_before.get(0, 0)
    n1 = counts_before.get(1, 0)
    if n0 == 0 or n1 == 0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True), {
            "balanced": False,
            "reason": "needs both labels",
            "n_before": int(len(df)),
            "n_after": int(len(df)),
            "counts_before": counts_before,
            "counts_after": counts_before,
        }
    n = min(n0, n1)
    normal = df[df[label_col].astype(int) == 0].sample(n=n, replace=False, random_state=seed)
    anomaly = df[df[label_col].astype(int) == 1].sample(n=n, replace=False, random_state=seed + 1)
    out = pd.concat([normal, anomaly], axis=0).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    counts_after = {int(k): int(v) for k, v in out[label_col].astype(int).value_counts().sort_index().to_dict().items()}
    return out, {
        "balanced": True,
        "n_before": int(len(df)),
        "n_after": int(len(out)),
        "counts_before": counts_before,
        "counts_after": counts_after,
        "n_kept_per_label": int(n),
    }
def build_unseen_test_buckets(
    balanced_df: pd.DataFrame,
    seen_test_df: pd.DataFrame,
    action_col: str,
    condition_col: str,
    actor_col: str,
    label_col: str,
    healthy_condition: str,
    unseen_actions: Sequence[str],
    unseen_actors: Sequence[str],
    unseen_styles: Sequence[str],
    seed: int,
    balance_binary: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Build separate unseen evaluation buckets.
    The training split removes the union of requested unseen actions/actors/styles. This helper
    creates more interpretable test subsets from that held-out space:
      - unseen_action: action unseen, actor/style seen
      - unseen_actor: actor unseen, action/style seen
      - unseen_style: anomaly style unseen, action/actor seen; normal support is held-out healthy
        from the seen test split, preferably excluding unseen actions/actors.
      - unseen_action_actor, unseen_action_style, unseen_actor_style, unseen_action_actor_style
      - unseen_any_combined: OR bucket over all requested unseen dimensions, with healthy support
        for style-only anomalies.
    For style buckets, the normal class is always healthy. Healthy is a seen style, but the rows are
    held out from training/validation either through the action/actor holdout or from the seen test split.
    """
    work = balanced_df.copy()
    seen = seen_test_df.copy()
    for d in (work, seen):
        if len(d) == 0:
            continue
        d[action_col] = d[action_col].astype(str).map(normalize_action_text)
        d[condition_col] = d[condition_col].astype(str).map(normalize_action_text)
        d[actor_col] = d[actor_col].astype(str).map(normalize_action_text)
    Aset = set(parse_list_arg(unseen_actions))
    Rset = set(parse_list_arg(unseen_actors))
    Sset = set(parse_list_arg(unseen_styles))
    healthy_condition = normalize_action_text(healthy_condition)
    A = work[action_col].isin(Aset) if Aset else pd.Series(False, index=work.index)
    R = work[actor_col].isin(Rset) if Rset else pd.Series(False, index=work.index)
    S = work[condition_col].isin(Sset) if Sset else pd.Series(False, index=work.index)
    H = (work[condition_col] == healthy_condition) & (work[label_col].astype(int) == 0)
    Y1 = work[label_col].astype(int) == 1
    seen_A = seen[action_col].isin(Aset) if Aset and len(seen) else pd.Series(False, index=seen.index)
    seen_R = seen[actor_col].isin(Rset) if Rset and len(seen) else pd.Series(False, index=seen.index)
    seen_H = (seen[condition_col] == healthy_condition) & (seen[label_col].astype(int) == 0) if len(seen) else pd.Series(False, index=seen.index)
    buckets_raw: Dict[str, pd.DataFrame] = {}
    if Aset:
        buckets_raw["unseen_action"] = work[A & ~R & ~S].copy()
    if Rset:
        buckets_raw["unseen_actor"] = work[R & ~A & ~S].copy()
    if Sset:
        # Style-only: anomalies with unseen style, but seen action/actor if possible.
        style_anom = work[S & ~A & ~R & Y1].copy()
        normal_support = seen[seen_H & ~seen_A & ~seen_R].copy()
        if len(normal_support) == 0:
            normal_support = seen[seen_H].copy()
        buckets_raw["unseen_style"] = _concat_unique_rows([normal_support, style_anom], seed + 300)
    if Aset and Rset:
        buckets_raw["unseen_action_actor"] = work[A & R & ~S].copy()
    if Aset and Sset:
        action_style_anom = work[A & S & ~R & Y1].copy()
        # Healthy normal support has unseen action and healthy style; actor is seen if possible.
        action_style_normal = work[A & ~R & H].copy()
        if len(action_style_normal) == 0:
            action_style_normal = work[A & H].copy()
        buckets_raw["unseen_action_style"] = _concat_unique_rows([action_style_normal, action_style_anom], seed + 301)
    if Rset and Sset:
        actor_style_anom = work[R & S & ~A & Y1].copy()
        actor_style_normal = work[R & ~A & H].copy()
        if len(actor_style_normal) == 0:
            actor_style_normal = work[R & H].copy()
        buckets_raw["unseen_actor_style"] = _concat_unique_rows([actor_style_normal, actor_style_anom], seed + 302)
    if Aset and Rset and Sset:
        ars_anom = work[A & R & S & Y1].copy()
        # Healthy style is seen, but action+actor are unseen.
        ars_normal = work[A & R & H].copy()
        buckets_raw["unseen_action_actor_style"] = _concat_unique_rows([ars_normal, ars_anom], seed + 303)
    if Aset or Rset or Sset:
        # Any-combined: all held-out action/actor rows plus unseen-style anomalies.
        any_anom = work[((A | R | S) & Y1)].copy()
        any_normal_parts = []
        if Aset or Rset:
            any_normal_parts.append(work[(A | R) & H].copy())
        if Sset:
            # Adds held-out healthy support for style-only anomalies; prefer seen action/actor.
            style_normal = seen[seen_H & ~seen_A & ~seen_R].copy()
            if len(style_normal) == 0:
                style_normal = seen[seen_H].copy()
            any_normal_parts.append(style_normal)
        buckets_raw["unseen_any_combined"] = _concat_unique_rows(any_normal_parts + [any_anom], seed + 304)
    buckets: Dict[str, pd.DataFrame] = {}
    info: Dict[str, Any] = {}
    for name, raw in buckets_raw.items():
        raw = raw.copy()
        if len(raw) == 0:
            info[name] = {"n_raw": 0, "saved": False, "reason": "empty bucket"}
            continue
        if balance_binary:
            final, bal_info = _balance_eval_binary(raw, label_col, seed + 400 + len(info))
        else:
            final = raw.sample(frac=1.0, random_state=seed + 400 + len(info)).reset_index(drop=True)
            bal_info = {"balanced": False, "reason": "disabled"}
        buckets[name] = final
        info[name] = {
            "saved": True,
            "n_raw": int(len(raw)),
            "n_final": int(len(final)),
            "label_counts": {int(k): int(v) for k, v in final[label_col].astype(int).value_counts().sort_index().to_dict().items()},
            "condition_counts": final[condition_col].value_counts().sort_index().to_dict() if condition_col in final.columns else {},
            "action_counts": final[action_col].value_counts().sort_index().to_dict() if action_col in final.columns else {},
            "actor_counts": final[actor_col].value_counts().sort_index().to_dict() if actor_col in final.columns else {},
            "balance_info": bal_info,
        }
    return buckets, info
def true_unseen_action_split(
    df: pd.DataFrame,
    action_norm_col: str,
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Split so selected actions are never used in training or validation.
    Returns:
      train_df: seen actions only
      val_df: seen actions only
      seen_test_df: held-out samples from seen actions
      unseen_test_df: all samples from unseen actions
      combined_test_df: seen_test + unseen_test
      unseen_actions: normalized action names held out
    """
    unseen = select_unseen_actions(
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
    unseen_set = set(unseen)
    unseen_test_df = df[df[action_norm_col].isin(unseen_set)].copy()
    seen_df = df[~df[action_norm_col].isin(unseen_set)].copy()
    if len(seen_df) == 0:
        raise ValueError("All actions were selected as unseen; no seen-action data remains for training.")
    if len(unseen_test_df) == 0:
        raise ValueError("No unseen test rows were selected.")
    if not set(seen_df[label_col].astype(int).unique()).issuperset({0, 1}):
        raise ValueError("Seen training pool must contain both labels 0 and 1.")
    if condition_col in seen_df.columns:
        stratify_cols = [action_norm_col, condition_col, label_col]
    else:
        stratify_cols = [action_norm_col, label_col]
    train_df, val_df, seen_test_df = stratified_group_split(
        seen_df,
        stratify_cols=stratify_cols,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )
    combined_test_df = pd.concat([seen_test_df, unseen_test_df], axis=0).sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
    unseen_test_df = unseen_test_df.sample(frac=1.0, random_state=seed + 4).reset_index(drop=True)
    seen_test_df = seen_test_df.sample(frac=1.0, random_state=seed + 5).reset_index(drop=True)
    return train_df, val_df, seen_test_df, unseen_test_df, combined_test_df, unseen
# -----------------------------
# -----------------------------
# AA-MotionCLIP: frozen backbones + residual adapters
# -----------------------------
def count_parameters(module: nn.Module) -> Dict[str, int]:
    return {
        "total": int(sum(p.numel() for p in module.parameters())),
        "trainable": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
    }
class ResidualAdapter(nn.Module):
    """AA-CLIP-style residual adapter operating on the final feature dimension."""
    def __init__(self, dim: int, ratio: float = 0.1) -> None:
        super().__init__()
        self.ratio = float(ratio)
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.linear.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm(self.activation(self.linear(x)))
        return (1.0 - self.ratio) * x + self.ratio * residual
def _primary_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported transformer layer output type: {type(output)}")
def _replace_primary_tensor(output: Any, new_tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return new_tensor
    if isinstance(output, tuple):
        return (new_tensor, *output[1:])
    if isinstance(output, list):
        return [new_tensor, *output[1:]]
    raise TypeError(f"Unsupported transformer layer output type: {type(output)}")
def build_motionclip_encoder(
    checkpoint_path: str,
    device: torch.device,
    num_frames: int = 60,
    njoints: int = 25,
    nfeats: int = 6,
    latent_dim: int = 512,
    ff_size: int = 1024,
    num_layers: int = 8,
    num_heads: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    """Construct and load the pretrained MotionCLIP encoder."""
    encoder = Encoder_TRANSFORMER(
        modeltype="motionclip",
        njoints=njoints,
        nfeats=nfeats,
        num_frames=num_frames,
        num_classes=1,
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=[math.pi, 0.0, 0.0],
        latent_dim=latent_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        ablation=None,
        activation="gelu",
    )
    raw = torch_load_compat(checkpoint_path, map_location="cpu")
    state = find_state_dict(raw)
    state = strip_module_prefix(state)
    encoder_state: Dict[str, torch.Tensor] = {}
    if any(k.startswith("encoder.") for k in state):
        for k, v in state.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v
    else:
        encoder_state = state
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected MotionCLIP encoder keys: {unexpected[:30]}")
    if missing:
        print(f"[WARN] Missing MotionCLIP encoder keys ({len(missing)}): {missing[:30]}")
    encoder = encoder.to(device).float()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder
def motion_batch_dict(motion: torch.Tensor) -> Dict[str, torch.Tensor]:
    motion = motion.float()
    x = motion.permute(0, 2, 3, 1).contiguous()  # [B,J,F,T]
    batch_size, num_frames = motion.shape[0], motion.shape[1]
    lengths = torch.full((batch_size,), num_frames, dtype=torch.long, device=motion.device)
    mask = torch.arange(num_frames, device=motion.device).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "x": x,
        "y": torch.zeros(batch_size, dtype=torch.long, device=motion.device),
        "lengths": lengths,
        "mask": mask,
    }
def encode_frozen_motion(model: nn.Module, motion: torch.Tensor) -> torch.Tensor:
    out = model(motion_batch_dict(motion))
    if not isinstance(out, dict) or "mu" not in out:
        raise RuntimeError("Expected MotionCLIP encoder output dictionary containing 'mu'.")
    return F.normalize(out["mu"].float(), dim=-1)
class AATextEncoder(nn.Module):
    """OpenAI CLIP text encoder with trainable shallow residual adapters."""
    def __init__(
        self,
        clip_model_name: str,
        device: torch.device,
        adapter_layer_count: int = 3,
        adapter_ratio: float = 0.1,
        train_text_projection: bool = True,
    ) -> None:
        super().__init__()
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "OpenAI CLIP is required. Install it with:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            ) from exc
        self.clip_package = clip
        self.clip_model_name = clip_model_name
        model, _ = clip.load(clip_model_name, device=device)
        self.model = model.float()
        for p in self.model.parameters():
            p.requires_grad = False
        blocks = getattr(getattr(self.model, "transformer", None), "resblocks", None)
        if blocks is None:
            raise AttributeError("Could not find CLIP text transformer blocks at model.transformer.resblocks.")
        width = int(self.model.ln_final.weight.numel())
        n_layers = len(blocks)
        n_adapt = min(max(0, int(adapter_layer_count)), n_layers)
        self.adapter_indices = list(range(n_adapt))
        self.adapters = nn.ModuleDict({str(i): ResidualAdapter(width, adapter_ratio) for i in self.adapter_indices}).to(device)
        # The hook flag lets us export *true* original CLIP features with all
        # residual adapters bypassed. This is required for a valid before/after
        # comparison such as AA-CLIP Figures 2 and 3.
        self.adapters_enabled = True
        self._handles: List[Any] = []
        for i in self.adapter_indices:
            self._handles.append(blocks[i].register_forward_hook(self._make_hook(i)))
        self.train_text_projection = bool(train_text_projection)
        if self.train_text_projection:
            self.model.text_projection.requires_grad = True
        self.device = device
        self.model.eval()
    def _make_hook(self, index: int):
        def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
            if not self.adapters_enabled:
                return output
            x = _primary_tensor(output)
            x = self.adapters[str(index)](x)
            return _replace_primary_tensor(output, x)
        return hook
    def set_adapters_enabled(self, enabled: bool) -> None:
        """Enable adapted AA-CLIP text features or bypass adapters for original CLIP."""
        self.adapters_enabled = bool(enabled)
    def tokenize(self, texts: Sequence[str]) -> torch.Tensor:
        return self.clip_package.tokenize(list(texts), truncate=True)
    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_text(tokens.to(self.device)).float()
        return F.normalize(features, dim=-1)
    def trainable_parameters(self) -> List[nn.Parameter]:
        params = [p for p in self.adapters.parameters() if p.requires_grad]
        if self.train_text_projection and self.model.text_projection.requires_grad:
            params.append(self.model.text_projection)
        return params
    def adaptation_state_dict(self) -> Dict[str, Any]:
        return {
            "adapters": self.adapters.state_dict(),
            "text_projection": self.model.text_projection.detach().cpu(),
            "adapter_indices": self.adapter_indices,
            "clip_model_name": self.clip_model_name,
            "train_text_projection": self.train_text_projection,
        }
    def load_adaptation_state_dict(self, state: Dict[str, Any]) -> None:
        self.adapters.load_state_dict(state["adapters"], strict=True)
        if "text_projection" in state:
            with torch.no_grad():
                self.model.text_projection.copy_(state["text_projection"].to(self.model.text_projection.device))
    def freeze_adaptation(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the pretrained CLIP backbone deterministic; only adapters/projection learn.
        self.model.eval()
        self.adapters.train(mode)
        return self
class AAMotionEncoder(nn.Module):
    """Frozen MotionCLIP encoder with residual adapters and multi-level projectors."""
    def __init__(
        self,
        base_encoder: nn.Module,
        latent_dim: int = 512,
        adapter_layers: Sequence[int] = (0, 1, 2, 3, 4, 5),
        feature_layers: Sequence[int] = (1, 3, 5, 7),
        adapter_ratio: float = 0.1,
        pooling: str = "prefix_mean",
        feature_fusion_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        for p in self.base_encoder.parameters():
            p.requires_grad = False
        self.base_encoder.eval()
        seq_encoder = getattr(self.base_encoder, "seqTransEncoder", None)
        layers = getattr(seq_encoder, "layers", None)
        if layers is None:
            raise AttributeError("Could not find MotionCLIP transformer layers at encoder.seqTransEncoder.layers.")
        self.num_layers = len(layers)
        self.latent_dim = int(latent_dim)
        self.pooling = pooling
        self.feature_fusion_ratio = float(feature_fusion_ratio)
        if not 0.0 <= self.feature_fusion_ratio <= 1.0:
            raise ValueError("feature_fusion_ratio must be in [0, 1].")
        adapter_indices = sorted({int(i) for i in adapter_layers})
        feature_indices = sorted({int(i) for i in feature_layers})
        invalid = [i for i in adapter_indices + feature_indices if i < 0 or i >= self.num_layers]
        if invalid:
            raise ValueError(f"Invalid MotionCLIP layer indices {invalid}; model has {self.num_layers} layers.")
        self.adapter_indices = adapter_indices
        self.feature_indices = feature_indices
        self.adapters = nn.ModuleDict({str(i): ResidualAdapter(self.latent_dim, adapter_ratio) for i in adapter_indices})
        pooled_dim = self.latent_dim * 2 if pooling == "prefix_mean" else self.latent_dim
        self.projectors = nn.ModuleDict({
            str(i): nn.Linear(pooled_dim, self.latent_dim, bias=False)
            for i in feature_indices
        })
        for projector in self.projectors.values():
            nn.init.normal_(projector.weight, mean=0.0, std=1e-3)
        self.global_projector = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        nn.init.eye_(self.global_projector.weight)
        self._captured: Dict[int, torch.Tensor] = {}
        self._handles: List[Any] = []
        hooked = sorted(set(adapter_indices) | set(feature_indices))
        for i in hooked:
            self._handles.append(layers[i].register_forward_hook(self._make_hook(i)))
    def _make_hook(self, index: int):
        def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
            x = _primary_tensor(output)
            if index in self.adapter_indices:
                x = self.adapters[str(index)](x)
            if index in self.feature_indices:
                self._captured[index] = x
            return _replace_primary_tensor(output, x)
        return hook
    def _as_batch_first(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D transformer feature, got shape {tuple(x.shape)}")
        if x.shape[0] == batch_size:
            return x
        if x.shape[1] == batch_size:
            return x.transpose(0, 1)
        raise ValueError(
            f"Cannot infer batch dimension from transformer feature {tuple(x.shape)} and batch size {batch_size}."
        )
    def _pool(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = self._as_batch_first(x, batch_size)  # [B,S,D]
        prefix = x[:, 0]
        temporal = x[:, 1:].mean(dim=1) if x.shape[1] > 1 else prefix
        if self.pooling == "prefix":
            return prefix
        if self.pooling == "temporal_mean":
            return temporal
        if self.pooling == "prefix_mean":
            return torch.cat([prefix, temporal], dim=-1)
        raise ValueError(f"Unsupported pooling mode: {self.pooling}")
    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        self._captured = {}
        out = self.base_encoder(motion_batch_dict(motion))
        if not isinstance(out, dict) or "mu" not in out:
            raise RuntimeError("Expected MotionCLIP encoder output dictionary containing 'mu'.")
        mu = out["mu"].float()
        global_feature = self.global_projector(mu)
        feature_pieces: List[torch.Tensor] = []
        for i in self.feature_indices:
            if i not in self._captured:
                raise RuntimeError(f"Motion feature hook for layer {i} did not run.")
            pooled = self._pool(self._captured[i], motion.shape[0])
            feature_pieces.append(self.projectors[str(i)](pooled))
        if feature_pieces:
            multi_level = torch.stack(feature_pieces, dim=0).mean(dim=0)
            fused = (1.0 - self.feature_fusion_ratio) * global_feature + self.feature_fusion_ratio * multi_level
        else:
            fused = global_feature
        return F.normalize(fused, dim=-1)
    def adaptation_state_dict(self) -> Dict[str, Any]:
        return {
            "adapters": self.adapters.state_dict(),
            "projectors": self.projectors.state_dict(),
            "global_projector": self.global_projector.state_dict(),
            "adapter_indices": self.adapter_indices,
            "feature_indices": self.feature_indices,
            "pooling": self.pooling,
            "feature_fusion_ratio": self.feature_fusion_ratio,
            "latent_dim": self.latent_dim,
        }
    def load_adaptation_state_dict(self, state: Dict[str, Any]) -> None:
        self.adapters.load_state_dict(state["adapters"], strict=True)
        self.projectors.load_state_dict(state["projectors"], strict=True)
        self.global_projector.load_state_dict(state["global_projector"], strict=True)
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
    def train(self, mode: bool = True):
        super().train(mode)
        self.base_encoder.eval()
        self.adapters.train(mode)
        self.projectors.train(mode)
        self.global_projector.train(mode)
        return self
# -----------------------------
# Prompt ensembles and anchor caches
# -----------------------------
@dataclass
class AnchorCache:
    action_names: List[str]
    condition_names: List[str]
    action_to_idx: Dict[str, int]
    condition_to_idx: Dict[str, int]
    normal: torch.Tensor             # [A,D]
    anomaly: torch.Tensor            # [A,D]
    style: torch.Tensor              # [A,C,D]
    anomaly_styles: List[str]
class PromptTokenBank:
    def __init__(
        self,
        text_encoder: AATextEncoder,
        actions: Sequence[str],
        conditions: Sequence[str],
        healthy_condition: str,
        normal_templates: Sequence[str],
        anomaly_templates: Sequence[str],
    ) -> None:
        self.actions = sorted({normalize_action_text(a) for a in actions})
        self.healthy_condition = normalize_action_text(healthy_condition)
        condition_set = {normalize_action_text(c) for c in conditions}
        condition_set.add(self.healthy_condition)
        self.conditions = [self.healthy_condition] + sorted(c for c in condition_set if c != self.healthy_condition)
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.condition_to_idx = {c: i for i, c in enumerate(self.conditions)}
        self.normal_templates = list(normal_templates)
        self.anomaly_templates = list(anomaly_templates)
        if not self.normal_templates or not self.anomaly_templates:
            raise ValueError("At least one normal and one anomaly prompt template are required.")
        if len(self.normal_templates) != len(self.anomaly_templates):
            raise ValueError(
                "Normal and anomaly prompt template lists must have equal length so the prompt tensor is dense."
            )
        self.prompt_texts: Dict[str, Dict[str, List[str]]] = {}
        normal_tokens: List[torch.Tensor] = []
        style_tokens: List[torch.Tensor] = []
        for action in self.actions:
            self.prompt_texts[action] = {}
            normal_texts = [t.format(action=action, condition=self.healthy_condition) for t in self.normal_templates]
            self.prompt_texts[action][self.healthy_condition] = normal_texts
            normal_tokens.append(text_encoder.tokenize(normal_texts))
            per_condition = []
            for condition in self.conditions:
                if condition == self.healthy_condition:
                    texts = normal_texts
                    tokens = text_encoder.tokenize(texts)
                else:
                    texts = [t.format(action=action, condition=condition) for t in self.anomaly_templates]
                    tokens = text_encoder.tokenize(texts)
                self.prompt_texts[action][condition] = texts
                per_condition.append(tokens)
            style_tokens.append(torch.stack(per_condition, dim=0))
        self.normal_tokens = torch.stack(normal_tokens, dim=0)  # [A,Pn,77]
        self.style_tokens = torch.stack(style_tokens, dim=0)    # [A,C,P?,77]
        self.num_templates = int(self.normal_tokens.shape[1])
    def encode_per_template(
        self,
        text_encoder: AATextEncoder,
        device: torch.device,
        detach: bool = True,
    ) -> torch.Tensor:
        """Encode every rendered prompt without averaging templates.
        Returns:
            Tensor [num_actions, num_conditions, num_templates, embedding_dim].
        """
        A, C, P, L = self.style_tokens.shape
        flat_tokens = self.style_tokens.reshape(A * C * P, L).to(device)
        features = text_encoder.encode_tokens(flat_tokens).reshape(A, C, P, -1)
        features = F.normalize(features, dim=-1)
        return features.detach() if detach else features
    def encode(
        self,
        text_encoder: AATextEncoder,
        anomaly_styles: Sequence[str],
        device: torch.device,
        detach: bool = False,
        template_indices: Optional[Sequence[int]] = None,
    ) -> AnchorCache:
        anomaly_styles_n = sorted({normalize_action_text(s) for s in anomaly_styles})
        if self.healthy_condition in anomaly_styles_n:
            anomaly_styles_n.remove(self.healthy_condition)
        missing = sorted(set(anomaly_styles_n) - set(self.conditions))
        if missing:
            raise ValueError(f"Prompt bank is missing anomaly styles: {missing}")
        if template_indices is None:
            normal_token_tensor = self.normal_tokens
            style_token_tensor = self.style_tokens
        else:
            chosen = sorted({int(i) for i in template_indices})
            if not chosen:
                raise ValueError("template_indices must contain at least one template index.")
            invalid = [i for i in chosen if i < 0 or i >= self.num_templates]
            if invalid:
                raise ValueError(f"Invalid prompt template indices: {invalid}; valid range is 0..{self.num_templates - 1}.")
            index_tensor = torch.tensor(chosen, dtype=torch.long)
            normal_token_tensor = self.normal_tokens.index_select(1, index_tensor)
            style_token_tensor = self.style_tokens.index_select(2, index_tensor)
        A, P, L = normal_token_tensor.shape
        normal_tokens = normal_token_tensor.reshape(A * P, L).to(device)
        normal = text_encoder.encode_tokens(normal_tokens).reshape(A, P, -1).mean(dim=1)
        normal = F.normalize(normal, dim=-1)
        C = len(self.conditions)
        style_flat = style_token_tensor.reshape(A * C * P, L).to(device)
        style = text_encoder.encode_tokens(style_flat).reshape(A, C, P, -1).mean(dim=2)
        style = F.normalize(style, dim=-1)
        anomaly_indices = [self.condition_to_idx[s] for s in anomaly_styles_n]
        if not anomaly_indices:
            raise ValueError("No anomaly styles were supplied for anomaly-anchor construction.")
        anomaly = F.normalize(style[:, anomaly_indices].mean(dim=1), dim=-1)
        if detach:
            normal = normal.detach()
            anomaly = anomaly.detach()
            style = style.detach()
        return AnchorCache(
            action_names=self.actions,
            condition_names=self.conditions,
            action_to_idx=self.action_to_idx,
            condition_to_idx=self.condition_to_idx,
            normal=normal,
            anomaly=anomaly,
            style=style,
            anomaly_styles=anomaly_styles_n,
        )
def anchor_diagnostics(cache: AnchorCache) -> Dict[str, Any]:
    pair_cos = (cache.normal * cache.anomaly).sum(dim=-1)
    return {
        "normal_anomaly_cosine_mean": float(pair_cos.mean().detach().cpu()),
        "normal_anomaly_cosine_std": float(pair_cos.std(unbiased=False).detach().cpu()),
        "normal_anomaly_cosine_min": float(pair_cos.min().detach().cpu()),
        "normal_anomaly_cosine_max": float(pair_cos.max().detach().cpu()),
        "actions": {
            action: float(pair_cos[i].detach().cpu()) for i, action in enumerate(cache.action_names)
        },
        "anomaly_styles": cache.anomaly_styles,
    }
def save_anchor_cache(cache: AnchorCache, path: str | Path) -> None:
    torch.save({
        "action_names": cache.action_names,
        "condition_names": cache.condition_names,
        "action_to_idx": cache.action_to_idx,
        "condition_to_idx": cache.condition_to_idx,
        "normal": cache.normal.detach().cpu(),
        "anomaly": cache.anomaly.detach().cpu(),
        "style": cache.style.detach().cpu(),
        "anomaly_styles": cache.anomaly_styles,
    }, path)
def save_prompt_feature_export(
    prompt_bank: PromptTokenBank,
    text_encoder: AATextEncoder,
    output_path: str | Path,
    metadata_csv_path: str | Path,
    device: torch.device,
    encoder_state: str,
    train_actions: Sequence[str],
    train_conditions: Sequence[str],
    train_pairs: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    """Save per-template text features and metadata for AA-CLIP-style Figures 2 and 3.
    The tensor order is [action, condition, template, embedding]. The flattened
    feature matrix and metadata rows use exactly the same order. The export also
    contains a full cosine-similarity matrix for Figure 2 and seen/unseen labels
    for Figure 3.
    """
    with torch.no_grad():
        feature_tensor = prompt_bank.encode_per_template(text_encoder, device, detach=True).cpu()
    A, C, P, D = feature_tensor.shape
    flat_features = feature_tensor.reshape(A * C * P, D)
    cosine_similarity = flat_features @ flat_features.t()
    train_action_set = {normalize_action_text(v) for v in train_actions}
    train_condition_set = {normalize_action_text(v) for v in train_conditions}
    train_pair_set = {
        (normalize_action_text(action), normalize_action_text(condition))
        for action, condition in train_pairs
    }
    metadata: List[Dict[str, Any]] = []
    for action_idx, action in enumerate(prompt_bank.actions):
        action_seen = action in train_action_set
        for condition_idx, condition in enumerate(prompt_bank.conditions):
            condition_seen = condition in train_condition_set
            pair_seen = (action, condition) in train_pair_set
            is_normal = condition == prompt_bank.healthy_condition
            if action_seen and condition_seen:
                held_out_group = "seen"
            elif not action_seen and not condition_seen:
                held_out_group = "unseen_action_and_style"
            elif not action_seen:
                held_out_group = "unseen_action"
            else:
                held_out_group = "unseen_style"
            prompts = prompt_bank.prompt_texts[action][condition]
            for template_idx, prompt_text in enumerate(prompts):
                metadata.append({
                    "flat_index": len(metadata),
                    "action_index": action_idx,
                    "condition_index": condition_idx,
                    "template_index": template_idx,
                    "action": action,
                    "condition": condition,
                    "prompt_text": prompt_text,
                    "semantic_label": "normal" if is_normal else "anomaly",
                    "is_normal": bool(is_normal),
                    # 'class_seen' follows the AA-CLIP paper convention where the
                    # category/class is the object category; action is the closest
                    # equivalent category in this MotionCLIP adaptation.
                    "class_seen": bool(action_seen),
                    "action_seen": bool(action_seen),
                    "style_seen": bool(condition_seen),
                    "action_style_pair_seen": bool(pair_seen),
                    "held_out_group": held_out_group,
                    "encoder_state": encoder_state,
                })
    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = Path(metadata_csv_path)
    metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(metadata_csv_path, index=False)
    mean_features = F.normalize(feature_tensor.mean(dim=2), dim=-1)
    payload = {
        "encoder_state": encoder_state,
        "features": flat_features,
        "features_by_action_condition_template": feature_tensor,
        "mean_features_by_action_condition": mean_features,
        "cosine_similarity_matrix": cosine_similarity,
        "metadata": metadata,
        "metadata_csv": str(metadata_csv_path),
        "action_names": prompt_bank.actions,
        "condition_names": prompt_bank.conditions,
        "healthy_condition": prompt_bank.healthy_condition,
        "normal_templates": prompt_bank.normal_templates,
        "anomaly_templates": prompt_bank.anomaly_templates,
        "num_actions": A,
        "num_conditions": C,
        "num_templates": P,
        "embedding_dim": D,
        "train_actions": sorted(train_action_set),
        "train_conditions": sorted(train_condition_set),
        "format_version": 1,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "path": str(output_path),
        "metadata_csv": str(metadata_csv_path),
        "shape": [A, C, P, D],
        "n_flat_features": int(len(metadata)),
        "encoder_state": encoder_state,
    }
# -----------------------------
# Dataset wrapper including actor metadata
# -----------------------------
class AAPerMoMotionDataset(PerMoMotionDataset):
    def __init__(self, *args: Any, actor_col: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.actor_col = actor_col
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        row = self.df.iloc[idx]
        item["actor"] = normalize_action_text(str(row[self.actor_col]))
        return item
def aa_collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = collate_batch(batch)
    out["actor"] = [b["actor"] for b in batch]
    return out
# -----------------------------
# Metrics and persistence
# -----------------------------
def binary_auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for k, count in enumerate(counts):
        if count > 1:
            tied = inverse == k
            ranks[tied] = ranks[tied].mean()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
def average_precision_fallback(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    order = np.argsort(-scores)
    y = y_true[order]
    total_pos = int(y.sum())
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / total_pos)
def fpr_at_95_tpr(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) != 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        eligible = np.where(tpr >= 0.95)[0]
        return float(fpr[eligible[0]]) if len(eligible) else float("nan")
    except Exception:
        order = np.argsort(-scores)
        y = y_true[order]
        n_pos = max(1, int((y == 1).sum()))
        n_neg = max(1, int((y == 0).sum()))
        tp = np.cumsum(y == 1) / n_pos
        fp = np.cumsum(y == 0) / n_neg
        eligible = np.where(tp >= 0.95)[0]
        return float(fp[eligible[0]]) if len(eligible) else float("nan")
def classification_metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
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
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, criterion: str) -> Tuple[float, Dict[str, Any]]:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 0.5, classification_metrics_at_threshold(np.asarray([], dtype=int), scores, 0.5)
    candidates = np.unique(scores)
    if len(candidates) > 1000:
        candidates = np.quantile(scores, np.linspace(0.0, 1.0, 1000))
    best_value = -float("inf")
    best_threshold = float(candidates[0])
    best_metrics: Optional[Dict[str, Any]] = None
    for threshold in candidates:
        metrics = classification_metrics_at_threshold(y_true, scores, float(threshold))
        value = float(metrics[criterion])
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics
def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    threshold_criterion: str = "balanced_accuracy",
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(y_true) == 0:
        return {
            "auroc": float("nan"), "auprc": float("nan"), "fpr_at_95_tpr": float("nan"),
            "n_samples": 0, "n_normal": 0, "n_anomaly": 0,
            "score_mean": float("nan"), "score_std": float("nan"),
            **classification_metrics_at_threshold(y_true, scores, 0.5 if threshold is None else threshold),
        }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auroc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
        auprc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        auroc = binary_auc_rank(y_true, scores)
        auprc = average_precision_fallback(y_true, scores)
    if threshold is None:
        threshold, threshold_metrics = find_best_threshold(y_true, scores, threshold_criterion)
        source = f"best_{threshold_criterion}_on_validation"
    else:
        threshold_metrics = classification_metrics_at_threshold(y_true, scores, threshold)
        source = "provided_validation_threshold"
    return {
        "auroc": auroc,
        "auprc": auprc,
        "fpr_at_95_tpr": fpr_at_95_tpr(y_true, scores),
        "n_samples": int(len(y_true)),
        "n_normal": int((y_true == 0).sum()),
        "n_anomaly": int((y_true == 1).sum()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "threshold_source": source,
        **threshold_metrics,
    }
def save_stage_curves(records: List[Dict[str, Any]], stage_dir: Path) -> None:
    save_training_curves(records, stage_dir)
@dataclass
class EvalOutput:
    # loss is the full validation objective (classification + weighted contrastive).
    loss: float
    # classification_loss is used for checkpoint selection and Optuna because it
    # is comparable across trials with different contrastive-loss weights.
    classification_loss: float
    contrastive_loss: float
    y_true: np.ndarray
    score: np.ndarray
    margin: np.ndarray
    sim_normal: np.ndarray
    sim_anomaly: np.ndarray
    embeddings: np.ndarray
    paths: List[str]
    actions: List[str]
    conditions: List[str]
    actors: List[str]
    row_indices: List[int]
def _action_indices(actions: Sequence[str], cache: AnchorCache, device: torch.device) -> torch.Tensor:
    try:
        return torch.tensor(
            [cache.action_to_idx[normalize_action_text(a)] for a in actions],
            dtype=torch.long,
            device=device,
        )
    except KeyError as exc:
        raise KeyError(f"Action {exc} is missing from the prompt bank.") from exc
def _condition_indices(conditions: Sequence[str], cache: AnchorCache, device: torch.device) -> torch.Tensor:
    try:
        return torch.tensor(
            [cache.condition_to_idx[normalize_action_text(c)] for c in conditions],
            dtype=torch.long,
            device=device,
        )
    except KeyError as exc:
        raise KeyError(f"Condition {exc} is missing from the prompt bank.") from exc
def binary_logits(z: torch.Tensor, actions: Sequence[str], cache: AnchorCache, temperature: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_idx = _action_indices(actions, cache, z.device)
    normal = cache.normal.index_select(0, a_idx)
    anomaly = cache.anomaly.index_select(0, a_idx)
    sim_n = (z * normal).sum(dim=-1)
    sim_a = (z * anomaly).sum(dim=-1)
    logits = torch.stack([sim_n, sim_a], dim=-1) / float(temperature)
    return logits, sim_n, sim_a
def exact_style_targets(actions: Sequence[str], conditions: Sequence[str], cache: AnchorCache, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    a_idx = _action_indices(actions, cache, device)
    c_idx = _condition_indices(conditions, cache, device)
    targets = cache.style[a_idx, c_idx]
    group_ids = a_idx * len(cache.condition_names) + c_idx
    return targets, group_ids
def symmetric_supervised_contrastive_loss(
    motion_z: torch.Tensor,
    text_z: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    motion_z = F.normalize(motion_z.float(), dim=-1)
    text_z = F.normalize(text_z.float(), dim=-1)
    logits = motion_z @ text_z.t() / float(temperature)
    positive = group_ids[:, None].eq(group_ids[None, :]).float()
    log_m2t = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_m2t = -(positive * log_m2t).sum(dim=1) / positive.sum(dim=1).clamp_min(1.0)
    logits_t = logits.t()
    log_t2m = logits_t - torch.logsumexp(logits_t, dim=1, keepdim=True)
    loss_t2m = -(positive.t() * log_t2m).sum(dim=1) / positive.t().sum(dim=1).clamp_min(1.0)
    return 0.5 * (loss_m2t.mean() + loss_t2m.mean())
@torch.no_grad()
def evaluate_with_cache(
    encoder: nn.Module,
    loader: DataLoader,
    cache: AnchorCache,
    device: torch.device,
    temperature: float,
    contrastive_weight: float = 0.0,
    is_base_encoder: bool = False,
) -> EvalOutput:
    encoder.eval()
    all_y: List[int] = []
    all_score: List[float] = []
    all_margin: List[float] = []
    all_n: List[float] = []
    all_a: List[float] = []
    all_z: List[np.ndarray] = []
    paths: List[str] = []
    actions: List[str] = []
    conditions: List[str] = []
    actors: List[str] = []
    row_indices: List[int] = []
    total_loss = 0.0
    total_classification_loss = 0.0
    total_contrastive_loss = 0.0
    total_n = 0
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        z = encode_frozen_motion(encoder, motion) if is_base_encoder else encoder(motion)
        logits, sim_n, sim_a = binary_logits(z, batch["action"], cache, temperature)
        classification_loss = F.cross_entropy(logits, labels)
        contrastive_loss = torch.zeros((), device=device, dtype=classification_loss.dtype)
        if contrastive_weight > 0:
            targets, groups = exact_style_targets(batch["action"], batch["condition"], cache, device)
            contrastive_loss = symmetric_supervised_contrastive_loss(z, targets, groups, temperature)
        loss = classification_loss + float(contrastive_weight) * contrastive_loss
        probability = torch.softmax(logits, dim=-1)[:, 1]
        margin = sim_a - sim_n
        batch_size = labels.numel()
        total_loss += float(loss.detach().cpu()) * batch_size
        total_classification_loss += float(classification_loss.detach().cpu()) * batch_size
        total_contrastive_loss += float(contrastive_loss.detach().cpu()) * batch_size
        total_n += batch_size
        all_y.extend(labels.detach().cpu().numpy().astype(int).tolist())
        all_score.extend(probability.detach().cpu().numpy().astype(float).tolist())
        all_margin.extend(margin.detach().cpu().numpy().astype(float).tolist())
        all_n.extend(sim_n.detach().cpu().numpy().astype(float).tolist())
        all_a.extend(sim_a.detach().cpu().numpy().astype(float).tolist())
        all_z.append(z.detach().cpu().numpy().astype(np.float32))
        paths.extend(batch["path"])
        actions.extend(batch["action"])
        conditions.extend(batch["condition"])
        actors.extend(batch["actor"])
        row_indices.extend(batch["row_index"])
    return EvalOutput(
        loss=total_loss / max(1, total_n),
        classification_loss=total_classification_loss / max(1, total_n),
        contrastive_loss=total_contrastive_loss / max(1, total_n),
        y_true=np.asarray(all_y, dtype=int),
        score=np.asarray(all_score, dtype=float),
        margin=np.asarray(all_margin, dtype=float),
        sim_normal=np.asarray(all_n, dtype=float),
        sim_anomaly=np.asarray(all_a, dtype=float),
        embeddings=np.concatenate(all_z, axis=0) if all_z else np.empty((0, cache.normal.shape[-1]), dtype=np.float32),
        paths=paths,
        actions=actions,
        conditions=conditions,
        actors=actors,
        row_indices=row_indices,
    )
def eval_output_dataframe(output: EvalOutput, threshold: float) -> pd.DataFrame:
    return pd.DataFrame({
        "row_index": output.row_indices,
        "motion_path": output.paths,
        "action": output.actions,
        "condition": output.conditions,
        "actor": output.actors,
        "y_true_is_anomaly": output.y_true,
        "anomaly_probability": output.score,
        "anomaly_margin": output.margin,
        "cosine_normal_anchor": output.sim_normal,
        "cosine_anomaly_anchor": output.sim_anomaly,
        "pred_is_anomaly": (output.score >= threshold).astype(int),
    })
def save_eval_output(output: EvalOutput, prefix: Path, threshold: float) -> pd.DataFrame:
    df = eval_output_dataframe(output, threshold)
    df.to_csv(str(prefix) + "_predictions.csv", index=False)
    np.savez_compressed(
        str(prefix) + "_embeddings.npz",
        embeddings=output.embeddings,
        y_true=output.y_true,
        anomaly_probability=output.score,
        anomaly_margin=output.margin,
        cosine_normal_anchor=output.sim_normal,
        cosine_anomaly_anchor=output.sim_anomaly,
        row_index=np.asarray(output.row_indices, dtype=np.int64),
        pred_is_anomaly=(output.score >= threshold).astype(np.int64),
    )
    return df
def grouped_metrics_from_predictions(df: pd.DataFrame, group_col: str, threshold: float) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if group_col == "condition":
        healthy = df[df["y_true_is_anomaly"].astype(int) == 0]
        styles = sorted(df.loc[df["y_true_is_anomaly"].astype(int) == 1, "condition"].unique().tolist())
        for style in styles:
            subset = pd.concat([healthy, df[(df["condition"] == style) & (df["y_true_is_anomaly"].astype(int) == 1)]])
            metrics = compute_binary_metrics(subset["y_true_is_anomaly"].to_numpy(), subset["anomaly_probability"].to_numpy(), threshold)
            records.append({group_col: style, **metrics})
        return records
    for value, subset in df.groupby(group_col):
        metrics = compute_binary_metrics(subset["y_true_is_anomaly"].to_numpy(), subset["anomaly_probability"].to_numpy(), threshold)
        records.append({group_col: value, **metrics})
    return records
def save_breakdowns(df: pd.DataFrame, output_dir: Path, split_name: str, threshold: float) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for group in ["condition", "action", "actor"]:
        records = grouped_metrics_from_predictions(df, group, threshold)
        pd.DataFrame(records).to_csv(output_dir / f"{split_name}_metrics_by_{group}.csv", index=False)
        summary[group] = records
    return summary
# -----------------------------
# Stage 1 and Stage 2 training
# -----------------------------
def train_stage1_epoch(
    motion_encoder: nn.Module,
    text_encoder: AATextEncoder,
    prompt_bank: PromptTokenBank,
    train_anomaly_styles: Sequence[str],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    disentangle_weight: float,
    grad_clip: float,
    use_amp: bool,
    templates_per_batch: int,
) -> Dict[str, float]:
    motion_encoder.eval()
    text_encoder.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    totals = {"loss": 0.0, "classification": 0.0, "disentangle": 0.0, "n": 0.0}
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            motion_z = encode_frozen_motion(motion_encoder, motion)
        template_count = min(max(1, int(templates_per_batch)), prompt_bank.num_templates)
        if template_count == prompt_bank.num_templates:
            selected_templates = list(range(prompt_bank.num_templates))
        else:
            selected_templates = random.sample(range(prompt_bank.num_templates), k=template_count)
        with torch.cuda.amp.autocast(enabled=use_amp):
            cache = prompt_bank.encode(
                text_encoder, train_anomaly_styles, device, detach=False,
                template_indices=selected_templates,
            )
            logits, _, _ = binary_logits(motion_z, batch["action"], cache, temperature)
            classification = F.cross_entropy(logits, labels)
            disentangle = ((cache.normal * cache.anomaly).sum(dim=-1).abs() ** 2).mean()
            loss = classification + float(disentangle_weight) * disentangle
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(text_encoder.trainable_parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        n = labels.numel()
        totals["loss"] += float(loss.detach().cpu()) * n
        totals["classification"] += float(classification.detach().cpu()) * n
        totals["disentangle"] += float(disentangle.detach().cpu()) * n
        totals["n"] += n
    return {k: v / max(1.0, totals["n"]) for k, v in totals.items() if k != "n"}
def train_stage2_epoch(
    motion_model: AAMotionEncoder,
    cache: AnchorCache,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    contrastive_weight: float,
    grad_clip: float,
    use_amp: bool,
) -> Dict[str, float]:
    motion_model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    totals = {"loss": 0.0, "classification": 0.0, "contrastive": 0.0, "n": 0.0}
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            z = motion_model(motion)
            logits, _, _ = binary_logits(z, batch["action"], cache, temperature)
            classification = F.cross_entropy(logits, labels)
            target_text, groups = exact_style_targets(batch["action"], batch["condition"], cache, device)
            contrastive = symmetric_supervised_contrastive_loss(z, target_text, groups, temperature)
            loss = classification + float(contrastive_weight) * contrastive
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(motion_model.trainable_parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        n = labels.numel()
        totals["loss"] += float(loss.detach().cpu()) * n
        totals["classification"] += float(classification.detach().cpu()) * n
        totals["contrastive"] += float(contrastive.detach().cpu()) * n
        totals["n"] += n
    return {k: v / max(1.0, totals["n"]) for k, v in totals.items() if k != "n"}
def should_stop_without_improvement(epoch: int, best_epoch: int, patience: int) -> bool:
    return patience > 0 and best_epoch > 0 and epoch - best_epoch >= patience
# -----------------------------
# Main experiment
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage AA-CLIP adaptation of pretrained MotionCLIP for PerMo Condition anomaly detection."
    )
    # Data
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--path_col", default="motion_path")
    parser.add_argument("--action_col", default="action_label")
    parser.add_argument("--condition_col", default="condition_label")
    parser.add_argument("--actor_col", default="actor_label")
    parser.add_argument("--label_col", default="is_anomaly")
    parser.add_argument("--motion_key", default="auto")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--njoints", type=int, default=25)
    parser.add_argument("--nfeats", type=int, default=6)
    # Split
    parser.add_argument("--test_fraction", type=float, default=0.20)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--train_seed", type=int, default=None)
    parser.add_argument(
        "--order_seed", type=int, default=None,
        help=("Controls only the training batch order/batch arrangement. "
              "Changing this keeps the split, selected/oversampled train indices, "
              "and model initialization fixed. Defaults to train_seed."),
    )
    parser.add_argument("--normal_target", type=int, default=200)
    parser.add_argument("--anomaly_target", type=int, default=200)
    parser.add_argument("--unseen_actions", nargs="*", default=[])
    parser.add_argument("--unseen_actors", nargs="*", default=[])
    parser.add_argument("--unseen_styles", nargs="*", default=[])
    parser.add_argument("--no_seen_healthy_in_unseen_style_test", action="store_true")
    # Backbones
    parser.add_argument("--project_root", default="", help="Parent directory containing the MotionCLIP package folder.")
    parser.add_argument("--checkpoint", required=True, help="Pretrained MotionCLIP checkpoint.")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--motion_num_layers", type=int, default=8)
    parser.add_argument("--motion_num_heads", type=int, default=4)
    parser.add_argument("--motion_dropout", type=float, default=0.1)
    # Prompting
    parser.add_argument("--healthy_condition", default="healthy")
    parser.add_argument(
        "--normal_prompt_templates", nargs="+",
        default=["healthy {action}"],
    )
    parser.add_argument(
        "--anomaly_prompt_templates", nargs="+",
        default=["{condition} {action}"],
    )
    # AA adapters
    parser.add_argument("--adapter_ratio", type=float, default=0.1)
    parser.add_argument("--text_adapter_layer_count", type=int, default=3)
    parser.add_argument("--no_train_text_projection", action="store_true")
    parser.add_argument("--motion_adapter_layers", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--motion_feature_layers", nargs="*", type=int, default=[1, 3, 5, 7])
    parser.add_argument("--motion_pooling", choices=["prefix", "temporal_mean", "prefix_mean"], default="prefix_mean")
    parser.add_argument("--motion_feature_fusion_ratio", type=float, default=0.1)
    # Training
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage2_epochs", type=int, default=20)
    parser.add_argument("--stage1_lr", type=float, default=1e-5)
    parser.add_argument("--stage2_lr", type=float, default=5e-4)
    parser.add_argument("--stage1_weight_decay", type=float, default=1e-4)
    parser.add_argument("--stage2_weight_decay", type=float, default=1e-4)
    parser.add_argument("--stage1_disentangle_weight", type=float, default=0.1)
    parser.add_argument(
        "--stage1_templates_per_batch", type=int, default=1,
        help=("Number of prompt templates encoded with gradients in each Stage-1 batch. "
              "Using 1 cycles randomly across the full prompt ensemble and is recommended for 10 GB GPUs. "
              "Validation and final anchors still average all templates."),
    )
    parser.add_argument("--stage2_contrastive_weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--threshold_criterion", choices=["balanced_accuracy", "f1", "accuracy"], default="balanced_accuracy")
    parser.add_argument("--disable_prompted_unseen_style_eval", action="store_true")
    args = parser.parse_args()
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    args.train_seed = args.seed if args.train_seed is None else args.train_seed
    args.order_seed = args.train_seed if args.order_seed is None else args.order_seed
    set_seed(args.train_seed)
    output_dir = ensure_dir(args.output_dir)
    split_dir = ensure_dir(output_dir / "splits")
    stage1_dir = ensure_dir(output_dir / "stage1_text")
    figure_export_dir = ensure_dir(stage1_dir / "figure2_3_exports")
    stage2_dir = ensure_dir(output_dir / "stage2_motion")
    test_dir = ensure_dir(output_dir / "test")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_json(vars(args), output_dir / "args.json")
    # Metadata and leakage-safe splits
    df = pd.read_csv(args.csv_path).copy()
    df["original_index"] = np.arange(len(df))
    required = [args.path_col, args.action_col, args.condition_col, args.actor_col, args.label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns {missing}. Available columns: {list(df.columns)}")
    df[args.label_col] = df[args.label_col].astype(int)
    if not set(df[args.label_col].unique()).issubset({0, 1}):
        raise ValueError(f"{args.label_col} must contain binary labels 0/1.")
    df["_action_norm"] = df[args.action_col].map(normalize_action_text)
    df["_condition_norm"] = df[args.condition_col].map(normalize_action_text)
    df[args.actor_col] = df[args.actor_col].astype(str).map(normalize_action_text)
    args.healthy_condition = normalize_action_text(args.healthy_condition)
    balanced_df, balanced_info = build_balanced_condition_subset(
        df=df,
        action_col="_action_norm",
        condition_col="_condition_norm",
        actor_col=args.actor_col,
        label_col=args.label_col,
        healthy_condition=args.healthy_condition,
        normal_target=args.normal_target,
        anomaly_target=args.anomaly_target,
        seed=args.split_seed,
    )
    train_df, val_df, seen_test_df, unseen_union_df, combined_test_df, split_info = split_balanced_condition_experiment(
        df=balanced_df,
        action_col="_action_norm",
        condition_col="_condition_norm",
        actor_col=args.actor_col,
        label_col=args.label_col,
        healthy_condition=args.healthy_condition,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
        unseen_actions=args.unseen_actions,
        unseen_actors=args.unseen_actors,
        unseen_styles=args.unseen_styles,
        include_seen_healthy_in_unseen_style_test=not args.no_seen_healthy_in_unseen_style_test,
    )
    extra_buckets, extra_bucket_info = build_unseen_test_buckets(
        balanced_df=balanced_df,
        seen_test_df=seen_test_df,
        action_col="_action_norm",
        condition_col="_condition_norm",
        actor_col=args.actor_col,
        label_col=args.label_col,
        healthy_condition=args.healthy_condition,
        unseen_actions=args.unseen_actions,
        unseen_actors=args.unseen_actors,
        unseen_styles=args.unseen_styles,
        seed=args.split_seed,
        balance_binary=True,
    )
    requested_actions = set(parse_list_arg(args.unseen_actions))
    requested_actors = set(parse_list_arg(args.unseen_actors))
    requested_styles = set(parse_list_arg(args.unseen_styles))
    leakage = {
        "actions": sorted((set(train_df["_action_norm"]) | set(val_df["_action_norm"])) & requested_actions),
        "actors": sorted((set(train_df[args.actor_col]) | set(val_df[args.actor_col])) & requested_actors),
        "styles": sorted((set(train_df["_condition_norm"]) | set(val_df["_condition_norm"])) & requested_styles),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Unseen data leakage detected: {leakage}")
    split_frames = {
        "train": train_df,
        "val": val_df,
        "test_seen": seen_test_df,
        "test_unseen_union": unseen_union_df,
        "test_combined": combined_test_df,
        **{f"test_{name}": frame for name, frame in extra_buckets.items()},
    }
    for name, frame in split_frames.items():
        frame.to_csv(split_dir / f"{name}.csv", index=False)
    split_summary = {
        "split_type": split_info["split_type"],
        "seeds": {
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "train_seed": int(args.train_seed),
            "order_seed": int(args.order_seed),
        },
        "balanced_subset_info": balanced_info,
        "counts": {name: int(len(frame)) for name, frame in split_frames.items()},
        "label_counts": {
            name: {str(k): int(v) for k, v in frame[args.label_col].value_counts().sort_index().to_dict().items()}
            for name, frame in split_frames.items()
        },
        "train_styles": sorted(train_df["_condition_norm"].unique().tolist()),
        "train_actions": sorted(train_df["_action_norm"].unique().tolist()),
        "train_actors": sorted(train_df[args.actor_col].unique().tolist()),
        "unseen_actions": sorted(requested_actions),
        "unseen_actors": sorted(requested_actors),
        "unseen_styles": sorted(requested_styles),
        "extra_bucket_info": extra_bucket_info,
        "leakage_check": leakage,
        "leakage_check_passed": True,
    }
    save_json(split_summary, output_dir / "split_summary.json")
    expected_shape = (args.num_frames, args.njoints, args.nfeats)
    def make_dataset(frame: pd.DataFrame) -> AAPerMoMotionDataset:
        return AAPerMoMotionDataset(
            frame,
            args.path_col,
            args.action_col,
            args.condition_col,
            args.label_col,
            args.motion_key,
            expected_shape,
            actor_col=args.actor_col,
        )
    train_ds = make_dataset(train_df)
    train_sampler = BalancedBinaryBatchSampler(
        train_df[args.label_col].astype(int).to_numpy(),
        batch_size=args.batch_size,
        seed=args.train_seed,
        drop_last=False,
        order_seed=args.order_seed,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch,
    )
    train_eval_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch,
    )
    eval_loaders: Dict[str, DataLoader] = {
        "train": train_eval_loader,
        "val": DataLoader(make_dataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch),
        "test_seen": DataLoader(make_dataset(seen_test_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch),
        "test_unseen_union": DataLoader(make_dataset(unseen_union_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch),
        "test_combined": DataLoader(make_dataset(combined_test_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch),
    }
    for name, frame in extra_buckets.items():
        eval_loaders[f"test_{name}"] = DataLoader(
            make_dataset(frame), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch,
        )
    if args.project_root:
        sys.path.insert(0, str(Path(args.project_root).resolve()))
    global Encoder_TRANSFORMER
    from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER
    base_motion = build_motionclip_encoder(
        args.checkpoint, device, args.num_frames, args.njoints, args.nfeats,
        args.latent_dim, args.ff_size, args.motion_num_layers, args.motion_num_heads, args.motion_dropout,
    )
    text_encoder = AATextEncoder(
        args.clip_model, device, args.text_adapter_layer_count, args.adapter_ratio,
        train_text_projection=not args.no_train_text_projection,
    )
    all_actions = balanced_df["_action_norm"].unique().tolist()
    all_conditions = balanced_df["_condition_norm"].unique().tolist()
    prompt_bank = PromptTokenBank(
        text_encoder, all_actions, all_conditions, args.healthy_condition,
        args.normal_prompt_templates, args.anomaly_prompt_templates,
    )
    save_json(prompt_bank.prompt_texts, output_dir / "prompts.json")
    train_anomaly_styles = sorted(
        c for c in train_df["_condition_norm"].unique().tolist() if c != args.healthy_condition
    )
    if not train_anomaly_styles:
        raise ValueError("The training split contains no anomaly styles.")
    # Stage 1: anomaly-aware text anchors
    print("[INFO] Stage 1: training text residual adapters against frozen MotionCLIP features.")
    train_actions_for_export = sorted(train_df["_action_norm"].unique().tolist())
    train_conditions_for_export = sorted(train_df["_condition_norm"].unique().tolist())
    train_pairs_for_export = sorted({
        (str(action), str(condition))
        for action, condition in zip(train_df["_action_norm"], train_df["_condition_norm"])
    })
    # Export true original CLIP features. The adapters are explicitly bypassed;
    # merely exporting before training would not be sufficient because randomly
    # initialized residual adapters already alter the forward pass.
    text_encoder.set_adapters_enabled(False)
    with torch.no_grad():
        initial_cache = prompt_bank.encode(text_encoder, train_anomaly_styles, device, detach=True)
    save_anchor_cache(initial_cache, figure_export_dir / "original_clip_anchor_cache.pt")
    save_json(anchor_diagnostics(initial_cache), stage1_dir / "anchors_before_training.json")
    original_prompt_export = save_prompt_feature_export(
        prompt_bank=prompt_bank,
        text_encoder=text_encoder,
        output_path=figure_export_dir / "original_text_features_per_template.pt",
        metadata_csv_path=figure_export_dir / "original_text_features_metadata.csv",
        device=device,
        encoder_state="original_clip",
        train_actions=train_actions_for_export,
        train_conditions=train_conditions_for_export,
        train_pairs=train_pairs_for_export,
    )
    text_encoder.set_adapters_enabled(True)
    stage1_trainable_count = int(sum(p.numel() for p in text_encoder.trainable_parameters()))
    optimizer1 = torch.optim.AdamW(
        text_encoder.trainable_parameters(), lr=args.stage1_lr, weight_decay=args.stage1_weight_decay
    )
    best_stage1_loss = float("inf")
    best_stage1_auroc = float("nan")
    best_stage1_epoch = -1
    stage1_records: List[Dict[str, Any]] = []
    stage1_ckpt_path = checkpoint_dir / "stage1_text_adapters_best.pt"
    for epoch in range(1, args.stage1_epochs + 1):
        started = time.time()
        train_sampler.set_epoch(epoch)
        losses = train_stage1_epoch(
            base_motion, text_encoder, prompt_bank, train_anomaly_styles, train_loader,
            optimizer1, device, args.temperature, args.stage1_disentangle_weight,
            args.grad_clip, args.amp, args.stage1_templates_per_batch,
        )
        with torch.no_grad():
            cache = prompt_bank.encode(text_encoder, train_anomaly_styles, device, detach=True)
        train_output = evaluate_with_cache(base_motion, train_eval_loader, cache, device, args.temperature, is_base_encoder=True)
        val_output = evaluate_with_cache(base_motion, eval_loaders["val"], cache, device, args.temperature, is_base_encoder=True)
        train_metrics = compute_binary_metrics(train_output.y_true, train_output.score, None, args.threshold_criterion)
        val_metrics = compute_binary_metrics(val_output.y_true, val_output.score, None, args.threshold_criterion)
        record = {
            "epoch": epoch,
            "train_loss": losses["loss"],
            "train_classification_loss": losses["classification"],
            "train_disentangle_loss": losses["disentangle"],
            "val_loss": val_output.classification_loss,
            "val_total_loss": val_output.loss,
            "train_auroc": train_metrics["auroc"],
            "val_auroc": val_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "normal_anomaly_anchor_cosine": anchor_diagnostics(cache)["normal_anomaly_cosine_mean"],
            "seconds": time.time() - started,
        }
        stage1_records.append(record)
        save_stage_curves(stage1_records, stage1_dir)
        print(
            f"[STAGE1 {epoch:03d}] loss={losses['loss']:.4f} "
            f"dis={losses['disentangle']:.4f} val_loss={val_output.classification_loss:.6f} "
            f"val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f}"
        )
        current = val_output.classification_loss
        if np.isfinite(current) and current < best_stage1_loss:
            best_stage1_loss = float(current)
            best_stage1_auroc = float(val_metrics["auroc"])
            best_stage1_epoch = epoch
            torch.save({
                "epoch": epoch,
                "text_adaptation": text_encoder.adaptation_state_dict(),
                "best_val_loss": best_stage1_loss,
                "best_val_metrics": val_metrics,
                "train_anomaly_styles": train_anomaly_styles,
                "args": vars(args),
            }, stage1_ckpt_path)
        if should_stop_without_improvement(epoch, best_stage1_epoch, args.early_stopping_patience):
            print(f"[INFO] Stage 1 early stopping at epoch {epoch}.")
            break
    if not stage1_ckpt_path.exists():
        torch.save({
            "epoch": args.stage1_epochs,
            "text_adaptation": text_encoder.adaptation_state_dict(),
            "best_val_metrics": {},
            "train_anomaly_styles": train_anomaly_styles,
            "args": vars(args),
        }, stage1_ckpt_path)
    stage1_ckpt = torch_load_compat(stage1_ckpt_path, map_location=device)
    text_encoder.load_adaptation_state_dict(stage1_ckpt["text_adaptation"])
    text_encoder.freeze_adaptation()
    text_encoder.set_adapters_enabled(True)
    with torch.no_grad():
        strict_cache = prompt_bank.encode(text_encoder, train_anomaly_styles, device, detach=True)
    save_anchor_cache(strict_cache, stage1_dir / "strict_anchor_cache.pt")
    save_anchor_cache(strict_cache, figure_export_dir / "adapted_anchor_cache.pt")
    # Compatibility alias for scripts that expect this historical filename.
    save_anchor_cache(strict_cache, output_dir / "text_prompt_cache.pt")
    save_json(anchor_diagnostics(strict_cache), stage1_dir / "anchors_after_training.json")
    adapted_prompt_export = save_prompt_feature_export(
        prompt_bank=prompt_bank,
        text_encoder=text_encoder,
        output_path=figure_export_dir / "adapted_text_features_per_template.pt",
        metadata_csv_path=figure_export_dir / "adapted_text_features_metadata.csv",
        device=device,
        encoder_state="adapted_aaclip",
        train_actions=train_actions_for_export,
        train_conditions=train_conditions_for_export,
        train_pairs=train_pairs_for_export,
    )
    figure_export_manifest = {
        "purpose": "Inputs for AA-CLIP-style Figure 2 similarity heatmaps and Figure 3 t-SNE plots.",
        "original_features_are_true_original_clip": True,
        "original": original_prompt_export,
        "adapted": adapted_prompt_export,
        "original_anchor_cache": str(figure_export_dir / "original_clip_anchor_cache.pt"),
        "adapted_anchor_cache": str(figure_export_dir / "adapted_anchor_cache.pt"),
        "text_prompt_cache_alias": str(output_dir / "text_prompt_cache.pt"),
        "figure2_guidance": (
            "Use cosine_similarity_matrix with metadata ordering, or compute cosine similarity "
            "from mean_features_by_action_condition for a compact normal/anomaly heatmap."
        ),
        "figure3_guidance": (
            "Run t-SNE jointly on original and adapted per-template features using the same random seed. "
            "Use semantic_label for normal/anomaly and class_seen/action_seen or style_seen for seen/unseen."
        ),
    }
    save_json(figure_export_manifest, figure_export_dir / "manifest.json")
    # Stage 2: align frozen MotionCLIP through motion adapters
    print("[INFO] Stage 2: training MotionCLIP residual adapters/projectors against fixed text anchors.")
    motion_model = AAMotionEncoder(
        base_motion,
        latent_dim=args.latent_dim,
        adapter_layers=args.motion_adapter_layers,
        feature_layers=args.motion_feature_layers,
        adapter_ratio=args.adapter_ratio,
        pooling=args.motion_pooling,
        feature_fusion_ratio=args.motion_feature_fusion_ratio,
    ).to(device)
    optimizer2 = torch.optim.AdamW(
        motion_model.trainable_parameters(), lr=args.stage2_lr, weight_decay=args.stage2_weight_decay
    )
    best_stage2_loss = float("inf")
    best_stage2_auroc = float("nan")
    best_stage2_epoch = -1
    best_threshold = 0.5
    stage2_records: List[Dict[str, Any]] = []
    stage2_ckpt_path = checkpoint_dir / "stage2_motion_adapters_best.pt"
    for epoch in range(1, args.stage2_epochs + 1):
        started = time.time()
        train_sampler.set_epoch(10000 + epoch)
        losses = train_stage2_epoch(
            motion_model, strict_cache, train_loader, optimizer2, device,
            args.temperature, args.stage2_contrastive_weight, args.grad_clip, args.amp,
        )
        train_output = evaluate_with_cache(
            motion_model, train_eval_loader, strict_cache, device, args.temperature,
            args.stage2_contrastive_weight, is_base_encoder=False,
        )
        val_output = evaluate_with_cache(
            motion_model, eval_loaders["val"], strict_cache, device, args.temperature,
            args.stage2_contrastive_weight, is_base_encoder=False,
        )
        train_metrics = compute_binary_metrics(train_output.y_true, train_output.score, None, args.threshold_criterion)
        val_metrics = compute_binary_metrics(val_output.y_true, val_output.score, None, args.threshold_criterion)
        record = {
            "epoch": epoch,
            "train_loss": losses["loss"],
            "train_classification_loss": losses["classification"],
            "train_contrastive_loss": losses["contrastive"],
            "val_loss": val_output.classification_loss,
            "val_total_loss": val_output.loss,
            "val_contrastive_loss": val_output.contrastive_loss,
            "train_auroc": train_metrics["auroc"],
            "val_auroc": val_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_threshold": val_metrics["threshold"],
            "seconds": time.time() - started,
        }
        stage2_records.append(record)
        save_stage_curves(stage2_records, stage2_dir)
        print(
            f"[STAGE2 {epoch:03d}] loss={losses['loss']:.4f} "
            f"contrast={losses['contrastive']:.4f} val_loss={val_output.classification_loss:.6f} "
            f"val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f} "
            f"threshold={val_metrics['threshold']:.4f}"
        )
        current = val_output.classification_loss
        if np.isfinite(current) and current < best_stage2_loss:
            best_stage2_loss = float(current)
            best_stage2_auroc = float(val_metrics["auroc"])
            best_stage2_epoch = epoch
            best_threshold = float(val_metrics["threshold"])
            torch.save({
                "epoch": epoch,
                "motion_adaptation": motion_model.adaptation_state_dict(),
                "best_val_loss": best_stage2_loss,
                "best_val_metrics": val_metrics,
                "threshold": best_threshold,
                "strict_anchor_cache_path": str(stage1_dir / "strict_anchor_cache.pt"),
                "args": vars(args),
            }, stage2_ckpt_path)
        if should_stop_without_improvement(epoch, best_stage2_epoch, args.early_stopping_patience):
            print(f"[INFO] Stage 2 early stopping at epoch {epoch}.")
            break
    if not stage2_ckpt_path.exists():
        torch.save({
            "epoch": args.stage2_epochs,
            "motion_adaptation": motion_model.adaptation_state_dict(),
            "best_val_metrics": {},
            "threshold": best_threshold,
            "strict_anchor_cache_path": str(stage1_dir / "strict_anchor_cache.pt"),
            "args": vars(args),
        }, stage2_ckpt_path)
    stage2_ckpt = torch_load_compat(stage2_ckpt_path, map_location=device)
    motion_model.load_adaptation_state_dict(stage2_ckpt["motion_adaptation"])
    best_threshold = float(stage2_ckpt.get("threshold", 0.5))
    motion_model.eval()
    # Final strict inference on every saved split/bucket
    strict_metrics: Dict[str, Any] = {}
    strict_breakdowns: Dict[str, Any] = {}
    for name, loader in eval_loaders.items():
        output = evaluate_with_cache(
            motion_model, loader, strict_cache, device, args.temperature,
            args.stage2_contrastive_weight, is_base_encoder=False,
        )
        metrics = compute_binary_metrics(output.y_true, output.score, best_threshold, args.threshold_criterion)
        metrics["loss"] = output.classification_loss
        metrics["total_loss"] = output.loss
        metrics["contrastive_loss"] = output.contrastive_loss
        strict_metrics[name] = metrics
        prediction_df = save_eval_output(output, test_dir / f"strict_{name}", best_threshold)
        strict_breakdowns[name] = save_breakdowns(prediction_df, test_dir, f"strict_{name}", best_threshold)
    # Optional prompted unseen-style inference. The held-out style names are introduced only here.
    prompted_metrics: Dict[str, Any] = {}
    if requested_styles and not args.disable_prompted_unseen_style_eval:
        with torch.no_grad():
            prompted_cache = prompt_bank.encode(text_encoder, sorted(requested_styles), device, detach=True)
        save_anchor_cache(prompted_cache, stage1_dir / "prompted_unseen_style_anchor_cache.pt")
        save_json(anchor_diagnostics(prompted_cache), stage1_dir / "prompted_unseen_style_anchor_diagnostics.json")
        style_loader_names = [name for name in eval_loaders if "style" in name]
        for name in style_loader_names:
            output = evaluate_with_cache(
                motion_model, eval_loaders[name], prompted_cache, device, args.temperature,
                0.0, is_base_encoder=False,
            )
            metrics = compute_binary_metrics(output.y_true, output.score, best_threshold, args.threshold_criterion)
            metrics["loss"] = output.classification_loss
            metrics["total_loss"] = output.loss
            metrics["contrastive_loss"] = output.contrastive_loss
            prompted_metrics[name] = metrics
            save_eval_output(output, test_dir / f"prompted_{name}", best_threshold)
    model_info = {
        "base_motion_encoder": count_parameters(base_motion),
        "text_model_total": count_parameters(text_encoder),
        "text_adapter_trainable_during_stage1": stage1_trainable_count,
        "motion_adapter_model": count_parameters(motion_model),
        "motion_trainable": int(sum(p.numel() for p in motion_model.trainable_parameters())),
        "text_adapter_layers": text_encoder.adapter_indices,
        "motion_adapter_layers": motion_model.adapter_indices,
        "motion_feature_layers": motion_model.feature_indices,
        "frozen_backbone_policy": "All original CLIP and MotionCLIP backbone parameters remain frozen.",
    }
    save_json(model_info, output_dir / "model_info.json")
    final_summary = {
        "method": "AA-MotionCLIP two-stage residual-adapter adaptation",
        "stage1": {
            "best_epoch": best_stage1_epoch,
            "best_val_loss": best_stage1_loss,
            "best_val_auroc": best_stage1_auroc,
            "checkpoint": str(stage1_ckpt_path),
            "anchor_diagnostics": anchor_diagnostics(strict_cache),
        },
        "stage2": {
            "best_epoch": best_stage2_epoch,
            "best_val_loss": best_stage2_loss,
            "best_val_auroc": best_stage2_auroc,
            "validation_threshold": best_threshold,
            "checkpoint": str(stage2_ckpt_path),
        },
        "strict_metrics": strict_metrics,
        "prompted_unseen_style_metrics": prompted_metrics,
        "strict_breakdowns": strict_breakdowns,
        "split_summary": split_summary,
        "model_info": model_info,
        "figure2_3_exports": figure_export_manifest,
        "important_interpretation": {
            "strict_unseen_style": "The anomaly anchor contains only seen training anomaly styles.",
            "prompted_unseen_style": "Held-out style names are used only at inference to construct the anomaly anchor.",
        },
    }
    save_json(final_summary, output_dir / "metrics.json")
    print("[DONE] Strict combined-test metrics:")
    print(json.dumps(strict_metrics.get("test_combined", {}), indent=2, sort_keys=True))
    if prompted_metrics:
        print("[DONE] Prompted unseen-style AUROCs:")
        print(json.dumps({k: v.get("auroc") for k, v in prompted_metrics.items()}, indent=2, sort_keys=True))
    print(f"[DONE] All outputs saved to {output_dir}")
if __name__ == "__main__":
    main()
#
#
##!/usr/bin/env python3
#"""
#Two-stage AA-CLIP adaptation of pretrained MotionCLIP for PerMo Condition anomaly detection.
#Stage 1 freezes MotionCLIP and the original CLIP backbone, then trains shallow CLIP text
#residual adapters plus the text projection to produce disentangled normal/anomaly anchors.
#Stage 2 freezes those anchors and all original MotionCLIP parameters, then trains shallow
#MotionCLIP residual adapters and multi-level projectors. The script performs splitting,
#training, validation-loss checkpoint selection, validation-threshold selection, strict test inference, optional prompted unseen-style
#inference, per-style/action/actor metrics, and embedding export.
#"""
#from __future__ import annotations
#import argparse
#import csv
#import json
#import math
#import os
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
#def torch_load_compat(path: str | Path, map_location: Any = "cpu") -> Any:
#    """Load trusted local checkpoints across PyTorch versions."""
#    try:
#        return torch.load(path, map_location=map_location, weights_only=False)
#    except TypeError:
#        return torch.load(path, map_location=map_location)
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
#    _plot(["train_auroc", "val_auroc"], "auroc_curves.png", "AUROC")
#    _plot(["val_auroc", "val_auprc", "val_f1", "val_balanced_accuracy"], "validation_metrics.png", "metric")
#def normalize_action_text(text: str) -> str:
#    text = str(text).strip().lower().replace("_", " ").replace("-", " ")
#    return " ".join(text.split())
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
#        condition_col: str,
#        label_col: str,
#        motion_key: str = "auto",
#        expected_shape: Tuple[int, int, int] = (60, 25, 6),
#    ) -> None:
#        self.df = df.reset_index(drop=True).copy()
#        self.path_col = path_col
#        self.action_col = action_col
#        self.condition_col = condition_col
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
#            "condition": normalize_action_text(row[self.condition_col]),
#            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),  # 0 healthy, 1 anomaly/non-healthy
#            "path": path,
#            "row_index": int(row.get("original_index", idx)),
#        }
#def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#    return {
#        "motion": torch.stack([b["motion"] for b in batch], dim=0),  # [B,T,J,F]
#        "action": [b["action"] for b in batch],
#        "condition": [b["condition"] for b in batch],
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
#      1 = non-healthy/anomaly
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
#def parse_list_arg(values: Optional[Sequence[str]]) -> List[str]:
#    """Accepts repeated args and/or comma-separated strings."""
#    out: List[str] = []
#    if not values:
#        return out
#    for v in values:
#        for part in str(v).split(','):
#            part = part.strip()
#            if part:
#                out.append(normalize_action_text(part))
#    return sorted(set(out))
#def _quota_counts(keys: Sequence[str], total: int) -> Dict[str, int]:
#    """Nearly equal integer quotas that sum to total."""
#    keys = sorted(set(map(str, keys)))
#    if not keys:
#        return {}
#    base = total // len(keys)
#    rem = total % len(keys)
#    return {k: base + (1 if i < rem else 0) for i, k in enumerate(keys)}
#def balanced_marginal_sample(
#    df: pd.DataFrame,
#    n_total: int,
#    style_col: str,
#    action_col: str,
#    actor_col: str,
#    seed: int,
#    split_name: str,
#) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#    """Sample n_total rows while keeping style/action/actor marginals as even as possible.
#    For PerMo Condition with 6 anomaly styles, 10 actions and 5 actors, n_total=200 gives:
#      style quota: 33/34 each
#      action quota: 20 each
#      actor quota: 40 each
#    The exact joint distribution cannot always be perfectly equal because 200 is not divisible by
#    style*action*actor combinations, so this uses deterministic greedy marginal balancing.
#    """
#    if len(df) < n_total:
#        raise ValueError(f"Cannot sample {n_total} rows for {split_name}; only {len(df)} rows available.")
#    rng = np.random.default_rng(seed)
#    work = df.copy()
#    work[style_col] = work[style_col].map(normalize_action_text)
#    work[action_col] = work[action_col].map(normalize_action_text)
#    work[actor_col] = work[actor_col].astype(str).map(normalize_action_text)
#    style_quota = _quota_counts(work[style_col].unique().tolist(), n_total)
#    action_quota = _quota_counts(work[action_col].unique().tolist(), n_total)
#    actor_quota = _quota_counts(work[actor_col].unique().tolist(), n_total)
#    remaining = work.sample(frac=1.0, random_state=seed).copy()
#    selected_indices: List[int] = []
#    selected_style = {k: 0 for k in style_quota}
#    selected_action = {k: 0 for k in action_quota}
#    selected_actor = {k: 0 for k in actor_quota}
#    for _ in range(n_total):
#        best_i = None
#        best_score = None
#        # Randomly inspect all remaining rows; this is small for PerMo, so simple is okay.
#        for idx, row in remaining.iterrows():
#            st, ac, ar = row[style_col], row[action_col], row[actor_col]
#            # Positive terms prioritize under-filled marginals. Negative terms discourage overfill.
#            score = (
#                3.0 * (style_quota[st] - selected_style[st]) +
#                2.0 * (action_quota[ac] - selected_action[ac]) +
#                2.0 * (actor_quota[ar] - selected_actor[ar])
#            )
#            # Prefer not taking more than available quota, but allow it if necessary.
#            over_penalty = 0.0
#            over_penalty += max(0, selected_style[st] + 1 - style_quota[st]) * 10.0
#            over_penalty += max(0, selected_action[ac] + 1 - action_quota[ac]) * 5.0
#            over_penalty += max(0, selected_actor[ar] + 1 - actor_quota[ar]) * 5.0
#            jitter = float(rng.normal(0, 1e-6))
#            score = score - over_penalty + jitter
#            if best_score is None or score > best_score:
#                best_score = score
#                best_i = idx
#        assert best_i is not None
#        row = remaining.loc[best_i]
#        selected_indices.append(best_i)
#        selected_style[row[style_col]] += 1
#        selected_action[row[action_col]] += 1
#        selected_actor[row[actor_col]] += 1
#        remaining = remaining.drop(index=best_i)
#    sampled = work.loc[selected_indices].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
#    info = {
#        "split_name": split_name,
#        "n_requested": int(n_total),
#        "n_sampled": int(len(sampled)),
#        "style_counts": sampled[style_col].value_counts().sort_index().to_dict(),
#        "action_counts": sampled[action_col].value_counts().sort_index().to_dict(),
#        "actor_counts": sampled[actor_col].value_counts().sort_index().to_dict(),
#        "style_quota": style_quota,
#        "action_quota": action_quota,
#        "actor_quota": actor_quota,
#    }
#    return sampled, info
#def build_balanced_condition_subset(
#    df: pd.DataFrame,
#    action_col: str,
#    condition_col: str,
#    actor_col: str,
#    label_col: str,
#    healthy_condition: str,
#    normal_target: int,
#    anomaly_target: int,
#    seed: int,
#) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#    """Create the 200 healthy + 200 anomaly Condition subset used by all experiments."""
#    healthy_condition = normalize_action_text(healthy_condition)
#    work = df.copy()
#    work[action_col] = work[action_col].map(normalize_action_text)
#    work[condition_col] = work[condition_col].map(normalize_action_text)
#    work[actor_col] = work[actor_col].astype(str).map(normalize_action_text)
#    healthy_pool = work[work[condition_col] == healthy_condition].copy()
#    anomaly_pool = work[work[condition_col] != healthy_condition].copy()
#    if len(healthy_pool) < normal_target:
#        raise ValueError(f"Need {normal_target} healthy samples, found only {len(healthy_pool)}.")
#    if len(anomaly_pool) < anomaly_target:
#        raise ValueError(f"Need {anomaly_target} anomaly samples, found only {len(anomaly_pool)}.")
#    # Usually exactly 200. If more, downsample while balancing actions/actors. Style is constant healthy.
#    if len(healthy_pool) == normal_target:
#        healthy_sample = healthy_pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
#        healthy_info = {
#            "n_sampled": int(len(healthy_sample)),
#            "action_counts": healthy_sample[action_col].value_counts().sort_index().to_dict(),
#            "actor_counts": healthy_sample[actor_col].value_counts().sort_index().to_dict(),
#        }
#    else:
#        healthy_tmp = healthy_pool.copy()
#        healthy_tmp["_healthy_style_for_sampling"] = healthy_condition
#        healthy_sample, healthy_info = balanced_marginal_sample(
#            healthy_tmp,
#            n_total=normal_target,
#            style_col="_healthy_style_for_sampling",
#            action_col=action_col,
#            actor_col=actor_col,
#            seed=seed,
#            split_name="healthy_normal_subset",
#        )
#        healthy_sample = healthy_sample.drop(columns=["_healthy_style_for_sampling"], errors="ignore")
#    anomaly_sample, anomaly_info = balanced_marginal_sample(
#        anomaly_pool,
#        n_total=anomaly_target,
#        style_col=condition_col,
#        action_col=action_col,
#        actor_col=actor_col,
#        seed=seed + 10,
#        split_name="balanced_anomaly_subset",
#    )
#    healthy_sample[label_col] = 0
#    anomaly_sample[label_col] = 1
#    subset = pd.concat([healthy_sample, anomaly_sample], axis=0).sample(frac=1.0, random_state=seed + 20).reset_index(drop=True)
#    info = {
#        "normal_target": int(normal_target),
#        "anomaly_target": int(anomaly_target),
#        "healthy_info": healthy_info,
#        "anomaly_info": anomaly_info,
#        "final_label_counts": subset[label_col].value_counts().sort_index().to_dict(),
#        "final_condition_counts": subset[condition_col].value_counts().sort_index().to_dict(),
#        "final_action_counts": subset[action_col].value_counts().sort_index().to_dict(),
#        "final_actor_counts": subset[actor_col].value_counts().sort_index().to_dict(),
#    }
#    return subset, info
#def _marginal_split_counts(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Dict[str, int]]:
#    out: Dict[str, Dict[str, int]] = {}
#    for c in cols:
#        if c in df.columns:
#            out[c] = {str(k): int(v) for k, v in df[c].value_counts().sort_index().to_dict().items()}
#    return out
#def _pick_marginally_balanced_rows(
#    df: pd.DataFrame,
#    n_total: int,
#    balance_cols: Sequence[str],
#    seed: int,
#    split_name: str,
#) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#    """Pick n_total rows while keeping the requested marginal columns as even as possible.
#    This does NOT require each exact (style, action, actor) combination to be splittable.
#    Instead, each selected row counts simultaneously toward its style, action, and actor margins.
#    That avoids the empty-validation issue caused by tiny groups.
#    """
#    if n_total <= 0:
#        return df.iloc[0:0].copy().reset_index(drop=True), {
#            "split_name": split_name,
#            "n_requested": int(n_total),
#            "n_selected": 0,
#            "balance_columns": list(balance_cols),
#            "counts": {},
#        }
#    if len(df) < n_total:
#        raise ValueError(f"Cannot pick {n_total} rows for {split_name}; only {len(df)} rows available.")
#    rng = np.random.default_rng(seed)
#    work = df.copy()
#    for c in balance_cols:
#        work[c] = work[c].astype(str).map(normalize_action_text)
#    # Uniform quotas over values that exist in this pool.
#    quotas: Dict[str, Dict[str, int]] = {}
#    selected_counts: Dict[str, Dict[str, int]] = {}
#    for c in balance_cols:
#        values = sorted(work[c].dropna().unique().tolist())
#        quotas[c] = _quota_counts(values, n_total)
#        selected_counts[c] = {v: 0 for v in values}
#    remaining = work.sample(frac=1.0, random_state=seed).copy()
#    selected_indices: List[int] = []
#    # Earlier columns get slightly more weight. In this script that means:
#    # label handled outside, then style, action, actor within label.
#    col_weights = {c: float(len(balance_cols) - i) for i, c in enumerate(balance_cols)}
#    for _ in range(n_total):
#        best_idx = None
#        best_score = None
#        for idx, row in remaining.iterrows():
#            score = 0.0
#            over_penalty = 0.0
#            for c in balance_cols:
#                value = row[c]
#                quota = quotas[c].get(value, 0)
#                current = selected_counts[c].get(value, 0)
#                score += col_weights[c] * (quota - current)
#                over_penalty += col_weights[c] * max(0, current + 1 - quota) * 10.0
#            score = score - over_penalty + float(rng.normal(0, 1e-6))
#            if best_score is None or score > best_score:
#                best_score = score
#                best_idx = idx
#        assert best_idx is not None
#        row = remaining.loc[best_idx]
#        selected_indices.append(best_idx)
#        for c in balance_cols:
#            selected_counts[c][row[c]] += 1
#        remaining = remaining.drop(index=best_idx)
#    selected = work.loc[selected_indices].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
#    info = {
#        "split_name": split_name,
#        "n_requested": int(n_total),
#        "n_selected": int(len(selected)),
#        "balance_columns": list(balance_cols),
#        "quotas": quotas,
#        "counts": _marginal_split_counts(selected, balance_cols),
#    }
#    return selected, info
#def balanced_label_marginal_train_val_test_split(
#    df: pd.DataFrame,
#    label_col: str,
#    condition_col: str,
#    action_col: str,
#    actor_col: str,
#    healthy_condition: str,
#    test_fraction: float,
#    val_fraction: float,
#    seed: int,
#) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
#    """Split a balanced subset by label first, then balance style/action/actor margins.
#    Important difference from the old splitter:
#      old: split every tiny (label, style, action, actor) group separately
#      new: split normal/anomaly separately, and within each label keep style/action/actor
#           distributions approximately uniform.
#    This keeps validation non-empty and keeps the test set label-balanced.
#    """
#    if not 0 <= test_fraction < 1:
#        raise ValueError("test_fraction must be in [0, 1).")
#    if not 0 <= val_fraction < 1:
#        raise ValueError("val_fraction must be in [0, 1).")
#    work = df.copy()
#    work[condition_col] = work[condition_col].astype(str).map(normalize_action_text)
#    work[action_col] = work[action_col].astype(str).map(normalize_action_text)
#    work[actor_col] = work[actor_col].astype(str).map(normalize_action_text)
#    healthy_condition = normalize_action_text(healthy_condition)
#    train_parts: List[pd.DataFrame] = []
#    val_parts: List[pd.DataFrame] = []
#    test_parts: List[pd.DataFrame] = []
#    split_infos: Dict[str, Any] = {}
#    for label_value in sorted(work[label_col].astype(int).unique().tolist()):
#        pool = work[work[label_col].astype(int) == int(label_value)].copy()
#        n = len(pool)
#        if n < 3:
#            raise ValueError(f"Label {label_value} has only {n} rows after unseen holdout; cannot make train/val/test.")
#        # Keep the existing script semantics: test is taken first, val is from the remaining pool.
#        n_test = int(round(n * test_fraction))
#        n_test = min(max(1 if test_fraction > 0 else 0, n_test), max(0, n - 2))
#        n_remaining_after_test = n - n_test
#        n_val = int(round(n_remaining_after_test * val_fraction))
#        n_val = min(max(1 if val_fraction > 0 else 0, n_val), max(0, n_remaining_after_test - 1))
#        # Healthy has only one style, so balancing condition is pointless for label 0.
#        # Anomaly uses condition/style + action + actor.
#        if int(label_value) == 0:
#            balance_cols = [action_col, actor_col]
#        else:
#            balance_cols = [condition_col, action_col, actor_col]
#        test_df, test_info = _pick_marginally_balanced_rows(
#            pool, n_test, balance_cols, seed + 1000 + int(label_value), f"label_{label_value}_test"
#        )
#        remaining = pool.drop(index=test_df["original_index"].values, errors="ignore") if "original_index" in test_df.columns else pool.drop(index=test_df.index, errors="ignore")
#        # Because selected/test_df is reset_index'ed, dropping by original_index only works if original_index equals index.
#        # Use a stable helper key instead.
#        if "_split_row_id" not in pool.columns:
#            pass
#        # Stable removal by original dataframe index saved before reset.
#        selected_orig_indices = set(test_df.get("original_index", pd.Series([], dtype=int)).tolist())
#        if selected_orig_indices:
#            remaining = pool[~pool["original_index"].isin(selected_orig_indices)].copy()
#        else:
#            remaining = pool.drop(index=test_df.index, errors="ignore").copy()
#        val_df, val_info = _pick_marginally_balanced_rows(
#            remaining, n_val, balance_cols, seed + 2000 + int(label_value), f"label_{label_value}_val"
#        )
#        selected_val_orig_indices = set(val_df.get("original_index", pd.Series([], dtype=int)).tolist())
#        if selected_val_orig_indices:
#            train_df = remaining[~remaining["original_index"].isin(selected_val_orig_indices)].copy()
#        else:
#            train_df = remaining.drop(index=val_df.index, errors="ignore").copy()
#        train_parts.append(train_df)
#        val_parts.append(val_df)
#        test_parts.append(test_df)
#        split_infos[str(label_value)] = {
#            "n_pool": int(n),
#            "n_train": int(len(train_df)),
#            "n_val": int(len(val_df)),
#            "n_test": int(len(test_df)),
#            "balance_cols": balance_cols,
#            "test_info": test_info,
#            "val_info": val_info,
#            "train_counts": _marginal_split_counts(train_df, balance_cols),
#        }
#    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed + 10).reset_index(drop=True)
#    val_df = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=seed + 11).reset_index(drop=True)
#    test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed + 12).reset_index(drop=True)
#    info = {
#        "splitter": "balanced_label_marginal_train_val_test_split",
#        "test_fraction": float(test_fraction),
#        "val_fraction_from_non_test_pool": float(val_fraction),
#        "label_infos": split_infos,
#        "train_label_counts": {int(k): int(v) for k, v in train_df[label_col].value_counts().sort_index().to_dict().items()},
#        "val_label_counts": {int(k): int(v) for k, v in val_df[label_col].value_counts().sort_index().to_dict().items()},
#        "test_label_counts": {int(k): int(v) for k, v in test_df[label_col].value_counts().sort_index().to_dict().items()},
#    }
#    return train_df, val_df, test_df, info
#def split_balanced_condition_experiment(
#    df: pd.DataFrame,
#    action_col: str,
#    condition_col: str,
#    actor_col: str,
#    label_col: str,
#    healthy_condition: str,
#    test_fraction: float,
#    val_fraction: float,
#    seed: int,
#    unseen_actions: Sequence[str],
#    unseen_actors: Sequence[str],
#    unseen_styles: Sequence[str],
#    include_seen_healthy_in_unseen_style_test: bool = True,
#) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
#    """One splitter for all three requested experiments.
#    Experiment 1: no unseen args -> split the balanced 200+200 subset by label first,
#      then keep style/action/actor marginals approximately uniform.
#    Experiment 2: pass --unseen_actions and/or --unseen_actors -> those rows are test-only.
#    Experiment 3: additionally pass --unseen_styles -> those anomaly styles are test-only.
#    """
#    work = df.copy()
#    work[action_col] = work[action_col].map(normalize_action_text)
#    work[condition_col] = work[condition_col].map(normalize_action_text)
#    work[actor_col] = work[actor_col].astype(str).map(normalize_action_text)
#    unseen_actions = set(parse_list_arg(unseen_actions))
#    unseen_actors = set(parse_list_arg(unseen_actors))
#    unseen_styles = set(parse_list_arg(unseen_styles))
#    healthy_condition = normalize_action_text(healthy_condition)
#    if healthy_condition in unseen_styles:
#        raise ValueError("Do not pass healthy as an unseen style. Healthy is the normal class.")
#    unknown_actions = sorted(unseen_actions - set(work[action_col].unique()))
#    unknown_actors = sorted(unseen_actors - set(work[actor_col].unique()))
#    unknown_styles = sorted(unseen_styles - set(work[condition_col].unique()))
#    if unknown_actions or unknown_actors or unknown_styles:
#        raise ValueError(
#            f"Unknown unseen values. actions={unknown_actions}, actors={unknown_actors}, styles={unknown_styles}."
#        )
#    has_unseen = bool(unseen_actions or unseen_actors or unseen_styles)
#    if not has_unseen:
#        train_df, val_df, seen_test_df, marginal_split_info = balanced_label_marginal_train_val_test_split(
#            work,
#            label_col=label_col,
#            condition_col=condition_col,
#            action_col=action_col,
#            actor_col=actor_col,
#            healthy_condition=healthy_condition,
#            test_fraction=test_fraction,
#            val_fraction=val_fraction,
#            seed=seed,
#        )
#        unseen_test_df = work.iloc[0:0].copy()
#        combined_test_df = seen_test_df.copy()
#        split_type = "balanced_random_seen_split_marginal"
#    else:
#        heldout_mask = pd.Series(False, index=work.index)
#        if unseen_actions:
#            heldout_mask |= work[action_col].isin(unseen_actions)
#        if unseen_actors:
#            heldout_mask |= work[actor_col].isin(unseen_actors)
#        if unseen_styles:
#            heldout_mask |= work[condition_col].isin(unseen_styles)
#        unseen_test_df = work[heldout_mask].copy()
#        seen_pool = work[~heldout_mask].copy()
#        if len(unseen_test_df) == 0:
#            raise ValueError("The unseen arguments selected zero test rows.")
#        if len(seen_pool) == 0:
#            raise ValueError("The unseen arguments removed all rows, so no training data remains.")
#        if not set(seen_pool[label_col].astype(int).unique()).issuperset({0, 1}):
#            raise ValueError("The remaining seen training pool must contain both healthy and anomaly samples.")
#        train_df, val_df, seen_test_df, marginal_split_info = balanced_label_marginal_train_val_test_split(
#            seen_pool,
#            label_col=label_col,
#            condition_col=condition_col,
#            action_col=action_col,
#            actor_col=actor_col,
#            healthy_condition=healthy_condition,
#            test_fraction=test_fraction,
#            val_fraction=val_fraction,
#            seed=seed,
#        )
#        # If only styles are unseen, unseen_test has anomalies only because healthy has no matching style.
#        # Add held-out healthy rows from the seen test split so AUROC on this test is defined.
#        if include_seen_healthy_in_unseen_style_test and unseen_styles and int((unseen_test_df[label_col] == 0).sum()) == 0:
#            normal_support = seen_test_df[seen_test_df[label_col].astype(int) == 0].copy()
#            if len(normal_support) > 0:
#                unseen_test_df = pd.concat([normal_support, unseen_test_df], axis=0).sample(
#                    frac=1.0, random_state=seed + 123
#                ).reset_index(drop=True)
#        combined_test_df = pd.concat([seen_test_df, unseen_test_df], axis=0).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
#        split_type = "balanced_unseen_holdout_split_marginal"
#    train_df = train_df.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
#    val_df = val_df.sample(frac=1.0, random_state=seed + 4).reset_index(drop=True)
#    seen_test_df = seen_test_df.sample(frac=1.0, random_state=seed + 5).reset_index(drop=True)
#    unseen_test_df = unseen_test_df.sample(frac=1.0, random_state=seed + 6).reset_index(drop=True)
#    combined_test_df = combined_test_df.sample(frac=1.0, random_state=seed + 7).reset_index(drop=True)
#    info = {
#        "split_type": split_type,
#        "unseen_actions": sorted(unseen_actions),
#        "unseen_actors": sorted(unseen_actors),
#        "unseen_styles": sorted(unseen_styles),
#        "include_seen_healthy_in_unseen_style_test": bool(include_seen_healthy_in_unseen_style_test),
#        "marginal_split_info": marginal_split_info,
#    }
#    return train_df, val_df, seen_test_df, unseen_test_df, combined_test_df, info
#def _concat_unique_rows(dfs: Sequence[pd.DataFrame], seed: int) -> pd.DataFrame:
#    """Concatenate dataframes and remove duplicate CSV rows using original_index when available."""
#    non_empty = [d.copy() for d in dfs if d is not None and len(d) > 0]
#    if not non_empty:
#        return pd.DataFrame()
#    out = pd.concat(non_empty, axis=0)
#    if "original_index" in out.columns:
#        out = out.drop_duplicates(subset=["original_index"], keep="first")
#    else:
#        out = out.drop_duplicates(keep="first")
#    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
#def _balance_eval_binary(df: pd.DataFrame, label_col: str, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#    """Downsample an evaluation bucket to equal normal/anomaly counts when both labels exist."""
#    if len(df) == 0:
#        return df.copy(), {"balanced": False, "reason": "empty", "n_before": 0, "n_after": 0}
#    counts_before = {int(k): int(v) for k, v in df[label_col].astype(int).value_counts().sort_index().to_dict().items()}
#    n0 = counts_before.get(0, 0)
#    n1 = counts_before.get(1, 0)
#    if n0 == 0 or n1 == 0:
#        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True), {
#            "balanced": False,
#            "reason": "needs both labels",
#            "n_before": int(len(df)),
#            "n_after": int(len(df)),
#            "counts_before": counts_before,
#            "counts_after": counts_before,
#        }
#    n = min(n0, n1)
#    normal = df[df[label_col].astype(int) == 0].sample(n=n, replace=False, random_state=seed)
#    anomaly = df[df[label_col].astype(int) == 1].sample(n=n, replace=False, random_state=seed + 1)
#    out = pd.concat([normal, anomaly], axis=0).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
#    counts_after = {int(k): int(v) for k, v in out[label_col].astype(int).value_counts().sort_index().to_dict().items()}
#    return out, {
#        "balanced": True,
#        "n_before": int(len(df)),
#        "n_after": int(len(out)),
#        "counts_before": counts_before,
#        "counts_after": counts_after,
#        "n_kept_per_label": int(n),
#    }
#def build_unseen_test_buckets(
#    balanced_df: pd.DataFrame,
#    seen_test_df: pd.DataFrame,
#    action_col: str,
#    condition_col: str,
#    actor_col: str,
#    label_col: str,
#    healthy_condition: str,
#    unseen_actions: Sequence[str],
#    unseen_actors: Sequence[str],
#    unseen_styles: Sequence[str],
#    seed: int,
#    balance_binary: bool = True,
#) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
#    """Build separate unseen evaluation buckets.
#    The training split removes the union of requested unseen actions/actors/styles. This helper
#    creates more interpretable test subsets from that held-out space:
#      - unseen_action: action unseen, actor/style seen
#      - unseen_actor: actor unseen, action/style seen
#      - unseen_style: anomaly style unseen, action/actor seen; normal support is held-out healthy
#        from the seen test split, preferably excluding unseen actions/actors.
#      - unseen_action_actor, unseen_action_style, unseen_actor_style, unseen_action_actor_style
#      - unseen_any_combined: OR bucket over all requested unseen dimensions, with healthy support
#        for style-only anomalies.
#    For style buckets, the normal class is always healthy. Healthy is a seen style, but the rows are
#    held out from training/validation either through the action/actor holdout or from the seen test split.
#    """
#    work = balanced_df.copy()
#    seen = seen_test_df.copy()
#    for d in (work, seen):
#        if len(d) == 0:
#            continue
#        d[action_col] = d[action_col].astype(str).map(normalize_action_text)
#        d[condition_col] = d[condition_col].astype(str).map(normalize_action_text)
#        d[actor_col] = d[actor_col].astype(str).map(normalize_action_text)
#    Aset = set(parse_list_arg(unseen_actions))
#    Rset = set(parse_list_arg(unseen_actors))
#    Sset = set(parse_list_arg(unseen_styles))
#    healthy_condition = normalize_action_text(healthy_condition)
#    A = work[action_col].isin(Aset) if Aset else pd.Series(False, index=work.index)
#    R = work[actor_col].isin(Rset) if Rset else pd.Series(False, index=work.index)
#    S = work[condition_col].isin(Sset) if Sset else pd.Series(False, index=work.index)
#    H = (work[condition_col] == healthy_condition) & (work[label_col].astype(int) == 0)
#    Y1 = work[label_col].astype(int) == 1
#    seen_A = seen[action_col].isin(Aset) if Aset and len(seen) else pd.Series(False, index=seen.index)
#    seen_R = seen[actor_col].isin(Rset) if Rset and len(seen) else pd.Series(False, index=seen.index)
#    seen_H = (seen[condition_col] == healthy_condition) & (seen[label_col].astype(int) == 0) if len(seen) else pd.Series(False, index=seen.index)
#    buckets_raw: Dict[str, pd.DataFrame] = {}
#    if Aset:
#        buckets_raw["unseen_action"] = work[A & ~R & ~S].copy()
#    if Rset:
#        buckets_raw["unseen_actor"] = work[R & ~A & ~S].copy()
#    if Sset:
#        # Style-only: anomalies with unseen style, but seen action/actor if possible.
#        style_anom = work[S & ~A & ~R & Y1].copy()
#        normal_support = seen[seen_H & ~seen_A & ~seen_R].copy()
#        if len(normal_support) == 0:
#            normal_support = seen[seen_H].copy()
#        buckets_raw["unseen_style"] = _concat_unique_rows([normal_support, style_anom], seed + 300)
#    if Aset and Rset:
#        buckets_raw["unseen_action_actor"] = work[A & R & ~S].copy()
#    if Aset and Sset:
#        action_style_anom = work[A & S & ~R & Y1].copy()
#        # Healthy normal support has unseen action and healthy style; actor is seen if possible.
#        action_style_normal = work[A & ~R & H].copy()
#        if len(action_style_normal) == 0:
#            action_style_normal = work[A & H].copy()
#        buckets_raw["unseen_action_style"] = _concat_unique_rows([action_style_normal, action_style_anom], seed + 301)
#    if Rset and Sset:
#        actor_style_anom = work[R & S & ~A & Y1].copy()
#        actor_style_normal = work[R & ~A & H].copy()
#        if len(actor_style_normal) == 0:
#            actor_style_normal = work[R & H].copy()
#        buckets_raw["unseen_actor_style"] = _concat_unique_rows([actor_style_normal, actor_style_anom], seed + 302)
#    if Aset and Rset and Sset:
#        ars_anom = work[A & R & S & Y1].copy()
#        # Healthy style is seen, but action+actor are unseen.
#        ars_normal = work[A & R & H].copy()
#        buckets_raw["unseen_action_actor_style"] = _concat_unique_rows([ars_normal, ars_anom], seed + 303)
#    if Aset or Rset or Sset:
#        # Any-combined: all held-out action/actor rows plus unseen-style anomalies.
#        any_anom = work[((A | R | S) & Y1)].copy()
#        any_normal_parts = []
#        if Aset or Rset:
#            any_normal_parts.append(work[(A | R) & H].copy())
#        if Sset:
#            # Adds held-out healthy support for style-only anomalies; prefer seen action/actor.
#            style_normal = seen[seen_H & ~seen_A & ~seen_R].copy()
#            if len(style_normal) == 0:
#                style_normal = seen[seen_H].copy()
#            any_normal_parts.append(style_normal)
#        buckets_raw["unseen_any_combined"] = _concat_unique_rows(any_normal_parts + [any_anom], seed + 304)
#    buckets: Dict[str, pd.DataFrame] = {}
#    info: Dict[str, Any] = {}
#    for name, raw in buckets_raw.items():
#        raw = raw.copy()
#        if len(raw) == 0:
#            info[name] = {"n_raw": 0, "saved": False, "reason": "empty bucket"}
#            continue
#        if balance_binary:
#            final, bal_info = _balance_eval_binary(raw, label_col, seed + 400 + len(info))
#        else:
#            final = raw.sample(frac=1.0, random_state=seed + 400 + len(info)).reset_index(drop=True)
#            bal_info = {"balanced": False, "reason": "disabled"}
#        buckets[name] = final
#        info[name] = {
#            "saved": True,
#            "n_raw": int(len(raw)),
#            "n_final": int(len(final)),
#            "label_counts": {int(k): int(v) for k, v in final[label_col].astype(int).value_counts().sort_index().to_dict().items()},
#            "condition_counts": final[condition_col].value_counts().sort_index().to_dict() if condition_col in final.columns else {},
#            "action_counts": final[action_col].value_counts().sort_index().to_dict() if action_col in final.columns else {},
#            "actor_counts": final[actor_col].value_counts().sort_index().to_dict() if actor_col in final.columns else {},
#            "balance_info": bal_info,
#        }
#    return buckets, info
#def true_unseen_action_split(
#    df: pd.DataFrame,
#    action_norm_col: str,
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
#) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
#    """Split so selected actions are never used in training or validation.
#    Returns:
#      train_df: seen actions only
#      val_df: seen actions only
#      seen_test_df: held-out samples from seen actions
#      unseen_test_df: all samples from unseen actions
#      combined_test_df: seen_test + unseen_test
#      unseen_actions: normalized action names held out
#    """
#    unseen = select_unseen_actions(
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
#    unseen_set = set(unseen)
#    unseen_test_df = df[df[action_norm_col].isin(unseen_set)].copy()
#    seen_df = df[~df[action_norm_col].isin(unseen_set)].copy()
#    if len(seen_df) == 0:
#        raise ValueError("All actions were selected as unseen; no seen-action data remains for training.")
#    if len(unseen_test_df) == 0:
#        raise ValueError("No unseen test rows were selected.")
#    if not set(seen_df[label_col].astype(int).unique()).issuperset({0, 1}):
#        raise ValueError("Seen training pool must contain both labels 0 and 1.")
#    if condition_col in seen_df.columns:
#        stratify_cols = [action_norm_col, condition_col, label_col]
#    else:
#        stratify_cols = [action_norm_col, label_col]
#    train_df, val_df, seen_test_df = stratified_group_split(
#        seen_df,
#        stratify_cols=stratify_cols,
#        test_fraction=test_fraction,
#        val_fraction=val_fraction,
#        seed=seed,
#    )
#    combined_test_df = pd.concat([seen_test_df, unseen_test_df], axis=0).sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
#    unseen_test_df = unseen_test_df.sample(frac=1.0, random_state=seed + 4).reset_index(drop=True)
#    seen_test_df = seen_test_df.sample(frac=1.0, random_state=seed + 5).reset_index(drop=True)
#    return train_df, val_df, seen_test_df, unseen_test_df, combined_test_df, unseen
## -----------------------------
## -----------------------------
## AA-MotionCLIP: frozen backbones + residual adapters
## -----------------------------
#def count_parameters(module: nn.Module) -> Dict[str, int]:
#    return {
#        "total": int(sum(p.numel() for p in module.parameters())),
#        "trainable": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
#    }
#class ResidualAdapter(nn.Module):
#    """AA-CLIP-style residual adapter operating on the final feature dimension."""
#    def __init__(self, dim: int, ratio: float = 0.1) -> None:
#        super().__init__()
#        self.ratio = float(ratio)
#        self.linear = nn.Linear(dim, dim)
#        self.activation = nn.GELU()
#        self.norm = nn.LayerNorm(dim)
#        nn.init.normal_(self.linear.weight, mean=0.0, std=1e-3)
#        nn.init.zeros_(self.linear.bias)
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        residual = self.norm(self.activation(self.linear(x)))
#        return (1.0 - self.ratio) * x + self.ratio * residual
#def _primary_tensor(output: Any) -> torch.Tensor:
#    if torch.is_tensor(output):
#        return output
#    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
#        return output[0]
#    raise TypeError(f"Unsupported transformer layer output type: {type(output)}")
#def _replace_primary_tensor(output: Any, new_tensor: torch.Tensor) -> Any:
#    if torch.is_tensor(output):
#        return new_tensor
#    if isinstance(output, tuple):
#        return (new_tensor, *output[1:])
#    if isinstance(output, list):
#        return [new_tensor, *output[1:]]
#    raise TypeError(f"Unsupported transformer layer output type: {type(output)}")
#def build_motionclip_encoder(
#    checkpoint_path: str,
#    device: torch.device,
#    num_frames: int = 60,
#    njoints: int = 25,
#    nfeats: int = 6,
#    latent_dim: int = 512,
#    ff_size: int = 1024,
#    num_layers: int = 8,
#    num_heads: int = 4,
#    dropout: float = 0.1,
#) -> nn.Module:
#    """Construct and load the pretrained MotionCLIP encoder."""
#    encoder = Encoder_TRANSFORMER(
#        modeltype="motionclip",
#        njoints=njoints,
#        nfeats=nfeats,
#        num_frames=num_frames,
#        num_classes=1,
#        translation=True,
#        pose_rep="rot6d",
#        glob=True,
#        glob_rot=[math.pi, 0.0, 0.0],
#        latent_dim=latent_dim,
#        ff_size=ff_size,
#        num_layers=num_layers,
#        num_heads=num_heads,
#        dropout=dropout,
#        ablation=None,
#        activation="gelu",
#    )
#    raw = torch_load_compat(checkpoint_path, map_location="cpu")
#    state = find_state_dict(raw)
#    state = strip_module_prefix(state)
#    encoder_state: Dict[str, torch.Tensor] = {}
#    if any(k.startswith("encoder.") for k in state):
#        for k, v in state.items():
#            if k.startswith("encoder."):
#                encoder_state[k[len("encoder."):]] = v
#    else:
#        encoder_state = state
#    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
#    if unexpected:
#        raise RuntimeError(f"Unexpected MotionCLIP encoder keys: {unexpected[:30]}")
#    if missing:
#        print(f"[WARN] Missing MotionCLIP encoder keys ({len(missing)}): {missing[:30]}")
#    encoder = encoder.to(device).float()
#    encoder.eval()
#    for p in encoder.parameters():
#        p.requires_grad = False
#    return encoder
#def motion_batch_dict(motion: torch.Tensor) -> Dict[str, torch.Tensor]:
#    motion = motion.float()
#    x = motion.permute(0, 2, 3, 1).contiguous()  # [B,J,F,T]
#    batch_size, num_frames = motion.shape[0], motion.shape[1]
#    lengths = torch.full((batch_size,), num_frames, dtype=torch.long, device=motion.device)
#    mask = torch.arange(num_frames, device=motion.device).unsqueeze(0) < lengths.unsqueeze(1)
#    return {
#        "x": x,
#        "y": torch.zeros(batch_size, dtype=torch.long, device=motion.device),
#        "lengths": lengths,
#        "mask": mask,
#    }
#def encode_frozen_motion(model: nn.Module, motion: torch.Tensor) -> torch.Tensor:
#    out = model(motion_batch_dict(motion))
#    if not isinstance(out, dict) or "mu" not in out:
#        raise RuntimeError("Expected MotionCLIP encoder output dictionary containing 'mu'.")
#    return F.normalize(out["mu"].float(), dim=-1)
#class AATextEncoder(nn.Module):
#    """OpenAI CLIP text encoder with trainable shallow residual adapters."""
#    def __init__(
#        self,
#        clip_model_name: str,
#        device: torch.device,
#        adapter_layer_count: int = 3,
#        adapter_ratio: float = 0.1,
#        train_text_projection: bool = True,
#    ) -> None:
#        super().__init__()
#        try:
#            import clip
#        except ImportError as exc:
#            raise ImportError(
#                "OpenAI CLIP is required. Install it with:\n"
#                "  pip install git+https://github.com/openai/CLIP.git"
#            ) from exc
#        self.clip_package = clip
#        self.clip_model_name = clip_model_name
#        model, _ = clip.load(clip_model_name, device=device)
#        self.model = model.float()
#        for p in self.model.parameters():
#            p.requires_grad = False
#        blocks = getattr(getattr(self.model, "transformer", None), "resblocks", None)
#        if blocks is None:
#            raise AttributeError("Could not find CLIP text transformer blocks at model.transformer.resblocks.")
#        width = int(self.model.ln_final.weight.numel())
#        n_layers = len(blocks)
#        n_adapt = min(max(0, int(adapter_layer_count)), n_layers)
#        self.adapter_indices = list(range(n_adapt))
#        self.adapters = nn.ModuleDict({str(i): ResidualAdapter(width, adapter_ratio) for i in self.adapter_indices}).to(device)
#        # The hook flag lets us export *true* original CLIP features with all
#        # residual adapters bypassed. This is required for a valid before/after
#        # comparison such as AA-CLIP Figures 2 and 3.
#        self.adapters_enabled = True
#        self._handles: List[Any] = []
#        for i in self.adapter_indices:
#            self._handles.append(blocks[i].register_forward_hook(self._make_hook(i)))
#        self.train_text_projection = bool(train_text_projection)
#        if self.train_text_projection:
#            self.model.text_projection.requires_grad = True
#        self.device = device
#        self.model.eval()
#    def _make_hook(self, index: int):
#        def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
#            if not self.adapters_enabled:
#                return output
#            x = _primary_tensor(output)
#            x = self.adapters[str(index)](x)
#            return _replace_primary_tensor(output, x)
#        return hook
#    def set_adapters_enabled(self, enabled: bool) -> None:
#        """Enable adapted AA-CLIP text features or bypass adapters for original CLIP."""
#        self.adapters_enabled = bool(enabled)
#    def tokenize(self, texts: Sequence[str]) -> torch.Tensor:
#        return self.clip_package.tokenize(list(texts), truncate=True)
#    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
#        features = self.model.encode_text(tokens.to(self.device)).float()
#        return F.normalize(features, dim=-1)
#    def trainable_parameters(self) -> List[nn.Parameter]:
#        params = [p for p in self.adapters.parameters() if p.requires_grad]
#        if self.train_text_projection and self.model.text_projection.requires_grad:
#            params.append(self.model.text_projection)
#        return params
#    def adaptation_state_dict(self) -> Dict[str, Any]:
#        return {
#            "adapters": self.adapters.state_dict(),
#            "text_projection": self.model.text_projection.detach().cpu(),
#            "adapter_indices": self.adapter_indices,
#            "clip_model_name": self.clip_model_name,
#            "train_text_projection": self.train_text_projection,
#        }
#    def load_adaptation_state_dict(self, state: Dict[str, Any]) -> None:
#        self.adapters.load_state_dict(state["adapters"], strict=True)
#        if "text_projection" in state:
#            with torch.no_grad():
#                self.model.text_projection.copy_(state["text_projection"].to(self.model.text_projection.device))
#    def freeze_adaptation(self) -> None:
#        for p in self.parameters():
#            p.requires_grad = False
#        self.eval()
#    def train(self, mode: bool = True):
#        super().train(mode)
#        # Keep the pretrained CLIP backbone deterministic; only adapters/projection learn.
#        self.model.eval()
#        self.adapters.train(mode)
#        return self
#class AAMotionEncoder(nn.Module):
#    """Frozen MotionCLIP encoder with residual adapters and multi-level projectors."""
#    def __init__(
#        self,
#        base_encoder: nn.Module,
#        latent_dim: int = 512,
#        adapter_layers: Sequence[int] = (0, 1, 2, 3, 4, 5),
#        feature_layers: Sequence[int] = (1, 3, 5, 7),
#        adapter_ratio: float = 0.1,
#        pooling: str = "prefix_mean",
#        feature_fusion_ratio: float = 0.1,
#    ) -> None:
#        super().__init__()
#        self.base_encoder = base_encoder
#        for p in self.base_encoder.parameters():
#            p.requires_grad = False
#        self.base_encoder.eval()
#        seq_encoder = getattr(self.base_encoder, "seqTransEncoder", None)
#        layers = getattr(seq_encoder, "layers", None)
#        if layers is None:
#            raise AttributeError("Could not find MotionCLIP transformer layers at encoder.seqTransEncoder.layers.")
#        self.num_layers = len(layers)
#        self.latent_dim = int(latent_dim)
#        self.pooling = pooling
#        self.feature_fusion_ratio = float(feature_fusion_ratio)
#        if not 0.0 <= self.feature_fusion_ratio <= 1.0:
#            raise ValueError("feature_fusion_ratio must be in [0, 1].")
#        adapter_indices = sorted({int(i) for i in adapter_layers})
#        feature_indices = sorted({int(i) for i in feature_layers})
#        invalid = [i for i in adapter_indices + feature_indices if i < 0 or i >= self.num_layers]
#        if invalid:
#            raise ValueError(f"Invalid MotionCLIP layer indices {invalid}; model has {self.num_layers} layers.")
#        self.adapter_indices = adapter_indices
#        self.feature_indices = feature_indices
#        self.adapters = nn.ModuleDict({str(i): ResidualAdapter(self.latent_dim, adapter_ratio) for i in adapter_indices})
#        pooled_dim = self.latent_dim * 2 if pooling == "prefix_mean" else self.latent_dim
#        self.projectors = nn.ModuleDict({
#            str(i): nn.Linear(pooled_dim, self.latent_dim, bias=False)
#            for i in feature_indices
#        })
#        for projector in self.projectors.values():
#            nn.init.normal_(projector.weight, mean=0.0, std=1e-3)
#        self.global_projector = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
#        nn.init.eye_(self.global_projector.weight)
#        self._captured: Dict[int, torch.Tensor] = {}
#        self._handles: List[Any] = []
#        hooked = sorted(set(adapter_indices) | set(feature_indices))
#        for i in hooked:
#            self._handles.append(layers[i].register_forward_hook(self._make_hook(i)))
#    def _make_hook(self, index: int):
#        def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
#            x = _primary_tensor(output)
#            if index in self.adapter_indices:
#                x = self.adapters[str(index)](x)
#            if index in self.feature_indices:
#                self._captured[index] = x
#            return _replace_primary_tensor(output, x)
#        return hook
#    def _as_batch_first(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
#        if x.ndim != 3:
#            raise ValueError(f"Expected 3D transformer feature, got shape {tuple(x.shape)}")
#        if x.shape[0] == batch_size:
#            return x
#        if x.shape[1] == batch_size:
#            return x.transpose(0, 1)
#        raise ValueError(
#            f"Cannot infer batch dimension from transformer feature {tuple(x.shape)} and batch size {batch_size}."
#        )
#    def _pool(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
#        x = self._as_batch_first(x, batch_size)  # [B,S,D]
#        prefix = x[:, 0]
#        temporal = x[:, 1:].mean(dim=1) if x.shape[1] > 1 else prefix
#        if self.pooling == "prefix":
#            return prefix
#        if self.pooling == "temporal_mean":
#            return temporal
#        if self.pooling == "prefix_mean":
#            return torch.cat([prefix, temporal], dim=-1)
#        raise ValueError(f"Unsupported pooling mode: {self.pooling}")
#    def forward(self, motion: torch.Tensor) -> torch.Tensor:
#        self._captured = {}
#        out = self.base_encoder(motion_batch_dict(motion))
#        if not isinstance(out, dict) or "mu" not in out:
#            raise RuntimeError("Expected MotionCLIP encoder output dictionary containing 'mu'.")
#        mu = out["mu"].float()
#        global_feature = self.global_projector(mu)
#        feature_pieces: List[torch.Tensor] = []
#        for i in self.feature_indices:
#            if i not in self._captured:
#                raise RuntimeError(f"Motion feature hook for layer {i} did not run.")
#            pooled = self._pool(self._captured[i], motion.shape[0])
#            feature_pieces.append(self.projectors[str(i)](pooled))
#        if feature_pieces:
#            multi_level = torch.stack(feature_pieces, dim=0).mean(dim=0)
#            fused = (1.0 - self.feature_fusion_ratio) * global_feature + self.feature_fusion_ratio * multi_level
#        else:
#            fused = global_feature
#        return F.normalize(fused, dim=-1)
#    def adaptation_state_dict(self) -> Dict[str, Any]:
#        return {
#            "adapters": self.adapters.state_dict(),
#            "projectors": self.projectors.state_dict(),
#            "global_projector": self.global_projector.state_dict(),
#            "adapter_indices": self.adapter_indices,
#            "feature_indices": self.feature_indices,
#            "pooling": self.pooling,
#            "feature_fusion_ratio": self.feature_fusion_ratio,
#            "latent_dim": self.latent_dim,
#        }
#    def load_adaptation_state_dict(self, state: Dict[str, Any]) -> None:
#        self.adapters.load_state_dict(state["adapters"], strict=True)
#        self.projectors.load_state_dict(state["projectors"], strict=True)
#        self.global_projector.load_state_dict(state["global_projector"], strict=True)
#    def trainable_parameters(self) -> List[nn.Parameter]:
#        return [p for p in self.parameters() if p.requires_grad]
#    def train(self, mode: bool = True):
#        super().train(mode)
#        self.base_encoder.eval()
#        self.adapters.train(mode)
#        self.projectors.train(mode)
#        self.global_projector.train(mode)
#        return self
## -----------------------------
## Prompt ensembles and anchor caches
## -----------------------------
#@dataclass
#class AnchorCache:
#    action_names: List[str]
#    condition_names: List[str]
#    action_to_idx: Dict[str, int]
#    condition_to_idx: Dict[str, int]
#    normal: torch.Tensor             # [A,D]
#    anomaly: torch.Tensor            # [A,D]
#    style: torch.Tensor              # [A,C,D]
#    anomaly_styles: List[str]
#class PromptTokenBank:
#    def __init__(
#        self,
#        text_encoder: AATextEncoder,
#        actions: Sequence[str],
#        conditions: Sequence[str],
#        healthy_condition: str,
#        normal_templates: Sequence[str],
#        anomaly_templates: Sequence[str],
#    ) -> None:
#        self.actions = sorted({normalize_action_text(a) for a in actions})
#        self.healthy_condition = normalize_action_text(healthy_condition)
#        condition_set = {normalize_action_text(c) for c in conditions}
#        condition_set.add(self.healthy_condition)
#        self.conditions = [self.healthy_condition] + sorted(c for c in condition_set if c != self.healthy_condition)
#        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
#        self.condition_to_idx = {c: i for i, c in enumerate(self.conditions)}
#        self.normal_templates = list(normal_templates)
#        self.anomaly_templates = list(anomaly_templates)
#        if not self.normal_templates or not self.anomaly_templates:
#            raise ValueError("At least one normal and one anomaly prompt template are required.")
#        if len(self.normal_templates) != len(self.anomaly_templates):
#            raise ValueError(
#                "Normal and anomaly prompt template lists must have equal length so the prompt tensor is dense."
#            )
#        self.prompt_texts: Dict[str, Dict[str, List[str]]] = {}
#        normal_tokens: List[torch.Tensor] = []
#        style_tokens: List[torch.Tensor] = []
#        for action in self.actions:
#            self.prompt_texts[action] = {}
#            normal_texts = [t.format(action=action, condition=self.healthy_condition) for t in self.normal_templates]
#            self.prompt_texts[action][self.healthy_condition] = normal_texts
#            normal_tokens.append(text_encoder.tokenize(normal_texts))
#            per_condition = []
#            for condition in self.conditions:
#                if condition == self.healthy_condition:
#                    texts = normal_texts
#                    tokens = text_encoder.tokenize(texts)
#                else:
#                    texts = [t.format(action=action, condition=condition) for t in self.anomaly_templates]
#                    tokens = text_encoder.tokenize(texts)
#                self.prompt_texts[action][condition] = texts
#                per_condition.append(tokens)
#            style_tokens.append(torch.stack(per_condition, dim=0))
#        self.normal_tokens = torch.stack(normal_tokens, dim=0)  # [A,Pn,77]
#        self.style_tokens = torch.stack(style_tokens, dim=0)    # [A,C,P?,77]
#        self.num_templates = int(self.normal_tokens.shape[1])
#    def encode_per_template(
#        self,
#        text_encoder: AATextEncoder,
#        device: torch.device,
#        detach: bool = True,
#    ) -> torch.Tensor:
#        """Encode every rendered prompt without averaging templates.
#        Returns:
#            Tensor [num_actions, num_conditions, num_templates, embedding_dim].
#        """
#        A, C, P, L = self.style_tokens.shape
#        flat_tokens = self.style_tokens.reshape(A * C * P, L).to(device)
#        features = text_encoder.encode_tokens(flat_tokens).reshape(A, C, P, -1)
#        features = F.normalize(features, dim=-1)
#        return features.detach() if detach else features
#    def encode(
#        self,
#        text_encoder: AATextEncoder,
#        anomaly_styles: Sequence[str],
#        device: torch.device,
#        detach: bool = False,
#        template_indices: Optional[Sequence[int]] = None,
#    ) -> AnchorCache:
#        anomaly_styles_n = sorted({normalize_action_text(s) for s in anomaly_styles})
#        if self.healthy_condition in anomaly_styles_n:
#            anomaly_styles_n.remove(self.healthy_condition)
#        missing = sorted(set(anomaly_styles_n) - set(self.conditions))
#        if missing:
#            raise ValueError(f"Prompt bank is missing anomaly styles: {missing}")
#        if template_indices is None:
#            normal_token_tensor = self.normal_tokens
#            style_token_tensor = self.style_tokens
#        else:
#            chosen = sorted({int(i) for i in template_indices})
#            if not chosen:
#                raise ValueError("template_indices must contain at least one template index.")
#            invalid = [i for i in chosen if i < 0 or i >= self.num_templates]
#            if invalid:
#                raise ValueError(f"Invalid prompt template indices: {invalid}; valid range is 0..{self.num_templates - 1}.")
#            index_tensor = torch.tensor(chosen, dtype=torch.long)
#            normal_token_tensor = self.normal_tokens.index_select(1, index_tensor)
#            style_token_tensor = self.style_tokens.index_select(2, index_tensor)
#        A, P, L = normal_token_tensor.shape
#        normal_tokens = normal_token_tensor.reshape(A * P, L).to(device)
#        normal = text_encoder.encode_tokens(normal_tokens).reshape(A, P, -1).mean(dim=1)
#        normal = F.normalize(normal, dim=-1)
#        C = len(self.conditions)
#        style_flat = style_token_tensor.reshape(A * C * P, L).to(device)
#        style = text_encoder.encode_tokens(style_flat).reshape(A, C, P, -1).mean(dim=2)
#        style = F.normalize(style, dim=-1)
#        anomaly_indices = [self.condition_to_idx[s] for s in anomaly_styles_n]
#        if not anomaly_indices:
#            raise ValueError("No anomaly styles were supplied for anomaly-anchor construction.")
#        anomaly = F.normalize(style[:, anomaly_indices].mean(dim=1), dim=-1)
#        if detach:
#            normal = normal.detach()
#            anomaly = anomaly.detach()
#            style = style.detach()
#        return AnchorCache(
#            action_names=self.actions,
#            condition_names=self.conditions,
#            action_to_idx=self.action_to_idx,
#            condition_to_idx=self.condition_to_idx,
#            normal=normal,
#            anomaly=anomaly,
#            style=style,
#            anomaly_styles=anomaly_styles_n,
#        )
#def anchor_diagnostics(cache: AnchorCache) -> Dict[str, Any]:
#    pair_cos = (cache.normal * cache.anomaly).sum(dim=-1)
#    return {
#        "normal_anomaly_cosine_mean": float(pair_cos.mean().detach().cpu()),
#        "normal_anomaly_cosine_std": float(pair_cos.std(unbiased=False).detach().cpu()),
#        "normal_anomaly_cosine_min": float(pair_cos.min().detach().cpu()),
#        "normal_anomaly_cosine_max": float(pair_cos.max().detach().cpu()),
#        "actions": {
#            action: float(pair_cos[i].detach().cpu()) for i, action in enumerate(cache.action_names)
#        },
#        "anomaly_styles": cache.anomaly_styles,
#    }
#def save_anchor_cache(cache: AnchorCache, path: str | Path) -> None:
#    torch.save({
#        "action_names": cache.action_names,
#        "condition_names": cache.condition_names,
#        "action_to_idx": cache.action_to_idx,
#        "condition_to_idx": cache.condition_to_idx,
#        "normal": cache.normal.detach().cpu(),
#        "anomaly": cache.anomaly.detach().cpu(),
#        "style": cache.style.detach().cpu(),
#        "anomaly_styles": cache.anomaly_styles,
#    }, path)
#def save_prompt_feature_export(
#    prompt_bank: PromptTokenBank,
#    text_encoder: AATextEncoder,
#    output_path: str | Path,
#    metadata_csv_path: str | Path,
#    device: torch.device,
#    encoder_state: str,
#    train_actions: Sequence[str],
#    train_conditions: Sequence[str],
#    train_pairs: Sequence[Tuple[str, str]],
#) -> Dict[str, Any]:
#    """Save per-template text features and metadata for AA-CLIP-style Figures 2 and 3.
#    The tensor order is [action, condition, template, embedding]. The flattened
#    feature matrix and metadata rows use exactly the same order. The export also
#    contains a full cosine-similarity matrix for Figure 2 and seen/unseen labels
#    for Figure 3.
#    """
#    with torch.no_grad():
#        feature_tensor = prompt_bank.encode_per_template(text_encoder, device, detach=True).cpu()
#    A, C, P, D = feature_tensor.shape
#    flat_features = feature_tensor.reshape(A * C * P, D)
#    cosine_similarity = flat_features @ flat_features.t()
#    train_action_set = {normalize_action_text(v) for v in train_actions}
#    train_condition_set = {normalize_action_text(v) for v in train_conditions}
#    train_pair_set = {
#        (normalize_action_text(action), normalize_action_text(condition))
#        for action, condition in train_pairs
#    }
#    metadata: List[Dict[str, Any]] = []
#    for action_idx, action in enumerate(prompt_bank.actions):
#        action_seen = action in train_action_set
#        for condition_idx, condition in enumerate(prompt_bank.conditions):
#            condition_seen = condition in train_condition_set
#            pair_seen = (action, condition) in train_pair_set
#            is_normal = condition == prompt_bank.healthy_condition
#            if action_seen and condition_seen:
#                held_out_group = "seen"
#            elif not action_seen and not condition_seen:
#                held_out_group = "unseen_action_and_style"
#            elif not action_seen:
#                held_out_group = "unseen_action"
#            else:
#                held_out_group = "unseen_style"
#            prompts = prompt_bank.prompt_texts[action][condition]
#            for template_idx, prompt_text in enumerate(prompts):
#                metadata.append({
#                    "flat_index": len(metadata),
#                    "action_index": action_idx,
#                    "condition_index": condition_idx,
#                    "template_index": template_idx,
#                    "action": action,
#                    "condition": condition,
#                    "prompt_text": prompt_text,
#                    "semantic_label": "normal" if is_normal else "anomaly",
#                    "is_normal": bool(is_normal),
#                    # 'class_seen' follows the AA-CLIP paper convention where the
#                    # category/class is the object category; action is the closest
#                    # equivalent category in this MotionCLIP adaptation.
#                    "class_seen": bool(action_seen),
#                    "action_seen": bool(action_seen),
#                    "style_seen": bool(condition_seen),
#                    "action_style_pair_seen": bool(pair_seen),
#                    "held_out_group": held_out_group,
#                    "encoder_state": encoder_state,
#                })
#    metadata_df = pd.DataFrame(metadata)
#    metadata_csv_path = Path(metadata_csv_path)
#    metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)
#    metadata_df.to_csv(metadata_csv_path, index=False)
#    mean_features = F.normalize(feature_tensor.mean(dim=2), dim=-1)
#    payload = {
#        "encoder_state": encoder_state,
#        "features": flat_features,
#        "features_by_action_condition_template": feature_tensor,
#        "mean_features_by_action_condition": mean_features,
#        "cosine_similarity_matrix": cosine_similarity,
#        "metadata": metadata,
#        "metadata_csv": str(metadata_csv_path),
#        "action_names": prompt_bank.actions,
#        "condition_names": prompt_bank.conditions,
#        "healthy_condition": prompt_bank.healthy_condition,
#        "normal_templates": prompt_bank.normal_templates,
#        "anomaly_templates": prompt_bank.anomaly_templates,
#        "num_actions": A,
#        "num_conditions": C,
#        "num_templates": P,
#        "embedding_dim": D,
#        "train_actions": sorted(train_action_set),
#        "train_conditions": sorted(train_condition_set),
#        "format_version": 1,
#    }
#    output_path = Path(output_path)
#    output_path.parent.mkdir(parents=True, exist_ok=True)
#    torch.save(payload, output_path)
#    return {
#        "path": str(output_path),
#        "metadata_csv": str(metadata_csv_path),
#        "shape": [A, C, P, D],
#        "n_flat_features": int(len(metadata)),
#        "encoder_state": encoder_state,
#    }
## -----------------------------
## Dataset wrapper including actor metadata
## -----------------------------
#class AAPerMoMotionDataset(PerMoMotionDataset):
#    def __init__(self, *args: Any, actor_col: str, **kwargs: Any) -> None:
#        super().__init__(*args, **kwargs)
#        self.actor_col = actor_col
#    def __getitem__(self, idx: int) -> Dict[str, Any]:
#        item = super().__getitem__(idx)
#        row = self.df.iloc[idx]
#        item["actor"] = normalize_action_text(str(row[self.actor_col]))
#        return item
#def aa_collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#    out = collate_batch(batch)
#    out["actor"] = [b["actor"] for b in batch]
#    return out
## -----------------------------
## Metrics and persistence
## -----------------------------
#def binary_auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
#    y_true = np.asarray(y_true).astype(int)
#    scores = np.asarray(scores).astype(float)
#    pos = y_true == 1
#    neg = y_true == 0
#    n_pos, n_neg = int(pos.sum()), int(neg.sum())
#    if n_pos == 0 or n_neg == 0:
#        return float("nan")
#    order = np.argsort(scores)
#    ranks = np.empty_like(order, dtype=float)
#    ranks[order] = np.arange(1, len(scores) + 1)
#    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
#    for k, count in enumerate(counts):
#        if count > 1:
#            tied = inverse == k
#            ranks[tied] = ranks[tied].mean()
#    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
#def average_precision_fallback(y_true: np.ndarray, scores: np.ndarray) -> float:
#    y_true = np.asarray(y_true).astype(int)
#    scores = np.asarray(scores).astype(float)
#    order = np.argsort(-scores)
#    y = y_true[order]
#    total_pos = int(y.sum())
#    if total_pos == 0:
#        return float("nan")
#    tp = np.cumsum(y)
#    precision = tp / (np.arange(len(y)) + 1)
#    return float((precision * y).sum() / total_pos)
#def fpr_at_95_tpr(y_true: np.ndarray, scores: np.ndarray) -> float:
#    y_true = np.asarray(y_true).astype(int)
#    scores = np.asarray(scores).astype(float)
#    if len(np.unique(y_true)) != 2:
#        return float("nan")
#    try:
#        from sklearn.metrics import roc_curve
#        fpr, tpr, _ = roc_curve(y_true, scores)
#        eligible = np.where(tpr >= 0.95)[0]
#        return float(fpr[eligible[0]]) if len(eligible) else float("nan")
#    except Exception:
#        order = np.argsort(-scores)
#        y = y_true[order]
#        n_pos = max(1, int((y == 1).sum()))
#        n_neg = max(1, int((y == 0).sum()))
#        tp = np.cumsum(y == 1) / n_pos
#        fp = np.cumsum(y == 0) / n_neg
#        eligible = np.where(tp >= 0.95)[0]
#        return float(fp[eligible[0]]) if len(eligible) else float("nan")
#def classification_metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
#    y_true = np.asarray(y_true).astype(int)
#    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
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
#        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
#    }
#def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, criterion: str) -> Tuple[float, Dict[str, Any]]:
#    scores = np.asarray(scores, dtype=float)
#    if len(scores) == 0:
#        return 0.5, classification_metrics_at_threshold(np.asarray([], dtype=int), scores, 0.5)
#    candidates = np.unique(scores)
#    if len(candidates) > 1000:
#        candidates = np.quantile(scores, np.linspace(0.0, 1.0, 1000))
#    best_value = -float("inf")
#    best_threshold = float(candidates[0])
#    best_metrics: Optional[Dict[str, Any]] = None
#    for threshold in candidates:
#        metrics = classification_metrics_at_threshold(y_true, scores, float(threshold))
#        value = float(metrics[criterion])
#        if value > best_value:
#            best_value = value
#            best_threshold = float(threshold)
#            best_metrics = metrics
#    assert best_metrics is not None
#    return best_threshold, best_metrics
#def compute_binary_metrics(
#    y_true: np.ndarray,
#    scores: np.ndarray,
#    threshold: Optional[float] = None,
#    threshold_criterion: str = "balanced_accuracy",
#) -> Dict[str, Any]:
#    y_true = np.asarray(y_true).astype(int)
#    scores = np.asarray(scores).astype(float)
#    if len(y_true) == 0:
#        return {
#            "auroc": float("nan"), "auprc": float("nan"), "fpr_at_95_tpr": float("nan"),
#            "n_samples": 0, "n_normal": 0, "n_anomaly": 0,
#            "score_mean": float("nan"), "score_std": float("nan"),
#            **classification_metrics_at_threshold(y_true, scores, 0.5 if threshold is None else threshold),
#        }
#    try:
#        from sklearn.metrics import average_precision_score, roc_auc_score
#        auroc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
#        auprc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
#    except Exception:
#        auroc = binary_auc_rank(y_true, scores)
#        auprc = average_precision_fallback(y_true, scores)
#    if threshold is None:
#        threshold, threshold_metrics = find_best_threshold(y_true, scores, threshold_criterion)
#        source = f"best_{threshold_criterion}_on_validation"
#    else:
#        threshold_metrics = classification_metrics_at_threshold(y_true, scores, threshold)
#        source = "provided_validation_threshold"
#    return {
#        "auroc": auroc,
#        "auprc": auprc,
#        "fpr_at_95_tpr": fpr_at_95_tpr(y_true, scores),
#        "n_samples": int(len(y_true)),
#        "n_normal": int((y_true == 0).sum()),
#        "n_anomaly": int((y_true == 1).sum()),
#        "score_mean": float(scores.mean()),
#        "score_std": float(scores.std()),
#        "threshold_source": source,
#        **threshold_metrics,
#    }
#def save_stage_curves(records: List[Dict[str, Any]], stage_dir: Path) -> None:
#    save_training_curves(records, stage_dir)
#@dataclass
#class EvalOutput:
#    # loss is the full validation objective (classification + weighted contrastive).
#    loss: float
#    # classification_loss is used for checkpoint selection and Optuna because it
#    # is comparable across trials with different contrastive-loss weights.
#    classification_loss: float
#    contrastive_loss: float
#    y_true: np.ndarray
#    score: np.ndarray
#    margin: np.ndarray
#    sim_normal: np.ndarray
#    sim_anomaly: np.ndarray
#    embeddings: np.ndarray
#    paths: List[str]
#    actions: List[str]
#    conditions: List[str]
#    actors: List[str]
#    row_indices: List[int]
#def _action_indices(actions: Sequence[str], cache: AnchorCache, device: torch.device) -> torch.Tensor:
#    try:
#        return torch.tensor(
#            [cache.action_to_idx[normalize_action_text(a)] for a in actions],
#            dtype=torch.long,
#            device=device,
#        )
#    except KeyError as exc:
#        raise KeyError(f"Action {exc} is missing from the prompt bank.") from exc
#def _condition_indices(conditions: Sequence[str], cache: AnchorCache, device: torch.device) -> torch.Tensor:
#    try:
#        return torch.tensor(
#            [cache.condition_to_idx[normalize_action_text(c)] for c in conditions],
#            dtype=torch.long,
#            device=device,
#        )
#    except KeyError as exc:
#        raise KeyError(f"Condition {exc} is missing from the prompt bank.") from exc
#def binary_logits(z: torch.Tensor, actions: Sequence[str], cache: AnchorCache, temperature: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#    a_idx = _action_indices(actions, cache, z.device)
#    normal = cache.normal.index_select(0, a_idx)
#    anomaly = cache.anomaly.index_select(0, a_idx)
#    sim_n = (z * normal).sum(dim=-1)
#    sim_a = (z * anomaly).sum(dim=-1)
#    logits = torch.stack([sim_n, sim_a], dim=-1) / float(temperature)
#    return logits, sim_n, sim_a
#def exact_style_targets(actions: Sequence[str], conditions: Sequence[str], cache: AnchorCache, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
#    a_idx = _action_indices(actions, cache, device)
#    c_idx = _condition_indices(conditions, cache, device)
#    targets = cache.style[a_idx, c_idx]
#    group_ids = a_idx * len(cache.condition_names) + c_idx
#    return targets, group_ids
#def symmetric_supervised_contrastive_loss(
#    motion_z: torch.Tensor,
#    text_z: torch.Tensor,
#    group_ids: torch.Tensor,
#    temperature: float,
#) -> torch.Tensor:
#    motion_z = F.normalize(motion_z.float(), dim=-1)
#    text_z = F.normalize(text_z.float(), dim=-1)
#    logits = motion_z @ text_z.t() / float(temperature)
#    positive = group_ids[:, None].eq(group_ids[None, :]).float()
#    log_m2t = logits - torch.logsumexp(logits, dim=1, keepdim=True)
#    loss_m2t = -(positive * log_m2t).sum(dim=1) / positive.sum(dim=1).clamp_min(1.0)
#    logits_t = logits.t()
#    log_t2m = logits_t - torch.logsumexp(logits_t, dim=1, keepdim=True)
#    loss_t2m = -(positive.t() * log_t2m).sum(dim=1) / positive.t().sum(dim=1).clamp_min(1.0)
#    return 0.5 * (loss_m2t.mean() + loss_t2m.mean())
#@torch.no_grad()
#def evaluate_with_cache(
#    encoder: nn.Module,
#    loader: DataLoader,
#    cache: AnchorCache,
#    device: torch.device,
#    temperature: float,
#    contrastive_weight: float = 0.0,
#    is_base_encoder: bool = False,
#) -> EvalOutput:
#    encoder.eval()
#    all_y: List[int] = []
#    all_score: List[float] = []
#    all_margin: List[float] = []
#    all_n: List[float] = []
#    all_a: List[float] = []
#    all_z: List[np.ndarray] = []
#    paths: List[str] = []
#    actions: List[str] = []
#    conditions: List[str] = []
#    actors: List[str] = []
#    row_indices: List[int] = []
#    total_loss = 0.0
#    total_classification_loss = 0.0
#    total_contrastive_loss = 0.0
#    total_n = 0
#    for batch in loader:
#        motion = batch["motion"].to(device, non_blocking=True)
#        labels = batch["label"].to(device, non_blocking=True)
#        z = encode_frozen_motion(encoder, motion) if is_base_encoder else encoder(motion)
#        logits, sim_n, sim_a = binary_logits(z, batch["action"], cache, temperature)
#        classification_loss = F.cross_entropy(logits, labels)
#        contrastive_loss = torch.zeros((), device=device, dtype=classification_loss.dtype)
#        if contrastive_weight > 0:
#            targets, groups = exact_style_targets(batch["action"], batch["condition"], cache, device)
#            contrastive_loss = symmetric_supervised_contrastive_loss(z, targets, groups, temperature)
#        loss = classification_loss + float(contrastive_weight) * contrastive_loss
#        probability = torch.softmax(logits, dim=-1)[:, 1]
#        margin = sim_a - sim_n
#        batch_size = labels.numel()
#        total_loss += float(loss.detach().cpu()) * batch_size
#        total_classification_loss += float(classification_loss.detach().cpu()) * batch_size
#        total_contrastive_loss += float(contrastive_loss.detach().cpu()) * batch_size
#        total_n += batch_size
#        all_y.extend(labels.detach().cpu().numpy().astype(int).tolist())
#        all_score.extend(probability.detach().cpu().numpy().astype(float).tolist())
#        all_margin.extend(margin.detach().cpu().numpy().astype(float).tolist())
#        all_n.extend(sim_n.detach().cpu().numpy().astype(float).tolist())
#        all_a.extend(sim_a.detach().cpu().numpy().astype(float).tolist())
#        all_z.append(z.detach().cpu().numpy().astype(np.float32))
#        paths.extend(batch["path"])
#        actions.extend(batch["action"])
#        conditions.extend(batch["condition"])
#        actors.extend(batch["actor"])
#        row_indices.extend(batch["row_index"])
#    return EvalOutput(
#        loss=total_loss / max(1, total_n),
#        classification_loss=total_classification_loss / max(1, total_n),
#        contrastive_loss=total_contrastive_loss / max(1, total_n),
#        y_true=np.asarray(all_y, dtype=int),
#        score=np.asarray(all_score, dtype=float),
#        margin=np.asarray(all_margin, dtype=float),
#        sim_normal=np.asarray(all_n, dtype=float),
#        sim_anomaly=np.asarray(all_a, dtype=float),
#        embeddings=np.concatenate(all_z, axis=0) if all_z else np.empty((0, cache.normal.shape[-1]), dtype=np.float32),
#        paths=paths,
#        actions=actions,
#        conditions=conditions,
#        actors=actors,
#        row_indices=row_indices,
#    )
#def eval_output_dataframe(output: EvalOutput, threshold: float) -> pd.DataFrame:
#    return pd.DataFrame({
#        "row_index": output.row_indices,
#        "motion_path": output.paths,
#        "action": output.actions,
#        "condition": output.conditions,
#        "actor": output.actors,
#        "y_true_is_anomaly": output.y_true,
#        "anomaly_probability": output.score,
#        "anomaly_margin": output.margin,
#        "cosine_normal_anchor": output.sim_normal,
#        "cosine_anomaly_anchor": output.sim_anomaly,
#        "pred_is_anomaly": (output.score >= threshold).astype(int),
#    })
#def save_eval_output(output: EvalOutput, prefix: Path, threshold: float) -> pd.DataFrame:
#    df = eval_output_dataframe(output, threshold)
#    df.to_csv(str(prefix) + "_predictions.csv", index=False)
#    np.savez_compressed(
#        str(prefix) + "_embeddings.npz",
#        embeddings=output.embeddings,
#        y_true=output.y_true,
#        anomaly_probability=output.score,
#        anomaly_margin=output.margin,
#        cosine_normal_anchor=output.sim_normal,
#        cosine_anomaly_anchor=output.sim_anomaly,
#        row_index=np.asarray(output.row_indices, dtype=np.int64),
#        pred_is_anomaly=(output.score >= threshold).astype(np.int64),
#    )
#    return df
#def grouped_metrics_from_predictions(df: pd.DataFrame, group_col: str, threshold: float) -> List[Dict[str, Any]]:
#    records: List[Dict[str, Any]] = []
#    if group_col == "condition":
#        healthy = df[df["y_true_is_anomaly"].astype(int) == 0]
#        styles = sorted(df.loc[df["y_true_is_anomaly"].astype(int) == 1, "condition"].unique().tolist())
#        for style in styles:
#            subset = pd.concat([healthy, df[(df["condition"] == style) & (df["y_true_is_anomaly"].astype(int) == 1)]])
#            metrics = compute_binary_metrics(subset["y_true_is_anomaly"].to_numpy(), subset["anomaly_probability"].to_numpy(), threshold)
#            records.append({group_col: style, **metrics})
#        return records
#    for value, subset in df.groupby(group_col):
#        metrics = compute_binary_metrics(subset["y_true_is_anomaly"].to_numpy(), subset["anomaly_probability"].to_numpy(), threshold)
#        records.append({group_col: value, **metrics})
#    return records
#def save_breakdowns(df: pd.DataFrame, output_dir: Path, split_name: str, threshold: float) -> Dict[str, Any]:
#    summary: Dict[str, Any] = {}
#    for group in ["condition", "action", "actor"]:
#        records = grouped_metrics_from_predictions(df, group, threshold)
#        pd.DataFrame(records).to_csv(output_dir / f"{split_name}_metrics_by_{group}.csv", index=False)
#        summary[group] = records
#    return summary
## -----------------------------
## Stage 1 and Stage 2 training
## -----------------------------
#def train_stage1_epoch(
#    motion_encoder: nn.Module,
#    text_encoder: AATextEncoder,
#    prompt_bank: PromptTokenBank,
#    train_anomaly_styles: Sequence[str],
#    loader: DataLoader,
#    optimizer: torch.optim.Optimizer,
#    device: torch.device,
#    temperature: float,
#    disentangle_weight: float,
#    grad_clip: float,
#    use_amp: bool,
#    templates_per_batch: int,
#) -> Dict[str, float]:
#    motion_encoder.eval()
#    text_encoder.train()
#    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#    totals = {"loss": 0.0, "classification": 0.0, "disentangle": 0.0, "n": 0.0}
#    for batch in loader:
#        motion = batch["motion"].to(device, non_blocking=True)
#        labels = batch["label"].to(device, non_blocking=True)
#        optimizer.zero_grad(set_to_none=True)
#        with torch.no_grad():
#            motion_z = encode_frozen_motion(motion_encoder, motion)
#        template_count = min(max(1, int(templates_per_batch)), prompt_bank.num_templates)
#        if template_count == prompt_bank.num_templates:
#            selected_templates = list(range(prompt_bank.num_templates))
#        else:
#            selected_templates = random.sample(range(prompt_bank.num_templates), k=template_count)
#        with torch.cuda.amp.autocast(enabled=use_amp):
#            cache = prompt_bank.encode(
#                text_encoder, train_anomaly_styles, device, detach=False,
#                template_indices=selected_templates,
#            )
#            logits, _, _ = binary_logits(motion_z, batch["action"], cache, temperature)
#            classification = F.cross_entropy(logits, labels)
#            disentangle = ((cache.normal * cache.anomaly).sum(dim=-1).abs() ** 2).mean()
#            loss = classification + float(disentangle_weight) * disentangle
#        scaler.scale(loss).backward()
#        if grad_clip > 0:
#            scaler.unscale_(optimizer)
#            torch.nn.utils.clip_grad_norm_(text_encoder.trainable_parameters(), grad_clip)
#        scaler.step(optimizer)
#        scaler.update()
#        n = labels.numel()
#        totals["loss"] += float(loss.detach().cpu()) * n
#        totals["classification"] += float(classification.detach().cpu()) * n
#        totals["disentangle"] += float(disentangle.detach().cpu()) * n
#        totals["n"] += n
#    return {k: v / max(1.0, totals["n"]) for k, v in totals.items() if k != "n"}
#def train_stage2_epoch(
#    motion_model: AAMotionEncoder,
#    cache: AnchorCache,
#    loader: DataLoader,
#    optimizer: torch.optim.Optimizer,
#    device: torch.device,
#    temperature: float,
#    contrastive_weight: float,
#    grad_clip: float,
#    use_amp: bool,
#) -> Dict[str, float]:
#    motion_model.train()
#    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#    totals = {"loss": 0.0, "classification": 0.0, "contrastive": 0.0, "n": 0.0}
#    for batch in loader:
#        motion = batch["motion"].to(device, non_blocking=True)
#        labels = batch["label"].to(device, non_blocking=True)
#        optimizer.zero_grad(set_to_none=True)
#        with torch.cuda.amp.autocast(enabled=use_amp):
#            z = motion_model(motion)
#            logits, _, _ = binary_logits(z, batch["action"], cache, temperature)
#            classification = F.cross_entropy(logits, labels)
#            target_text, groups = exact_style_targets(batch["action"], batch["condition"], cache, device)
#            contrastive = symmetric_supervised_contrastive_loss(z, target_text, groups, temperature)
#            loss = classification + float(contrastive_weight) * contrastive
#        scaler.scale(loss).backward()
#        if grad_clip > 0:
#            scaler.unscale_(optimizer)
#            torch.nn.utils.clip_grad_norm_(motion_model.trainable_parameters(), grad_clip)
#        scaler.step(optimizer)
#        scaler.update()
#        n = labels.numel()
#        totals["loss"] += float(loss.detach().cpu()) * n
#        totals["classification"] += float(classification.detach().cpu()) * n
#        totals["contrastive"] += float(contrastive.detach().cpu()) * n
#        totals["n"] += n
#    return {k: v / max(1.0, totals["n"]) for k, v in totals.items() if k != "n"}
#def should_stop_without_improvement(epoch: int, best_epoch: int, patience: int) -> bool:
#    return patience > 0 and best_epoch > 0 and epoch - best_epoch >= patience
## -----------------------------
## Main experiment
## -----------------------------
#def main() -> None:
#    parser = argparse.ArgumentParser(
#        description="Two-stage AA-CLIP adaptation of pretrained MotionCLIP for PerMo Condition anomaly detection."
#    )
#    # Data
#    parser.add_argument("--csv_path", required=True)
#    parser.add_argument("--output_dir", required=True)
#    parser.add_argument("--path_col", default="motion_path")
#    parser.add_argument("--action_col", default="action_label")
#    parser.add_argument("--condition_col", default="condition_label")
#    parser.add_argument("--actor_col", default="actor_label")
#    parser.add_argument("--label_col", default="is_anomaly")
#    parser.add_argument("--motion_key", default="auto")
#    parser.add_argument("--num_frames", type=int, default=60)
#    parser.add_argument("--njoints", type=int, default=25)
#    parser.add_argument("--nfeats", type=int, default=6)
#    # Split
#    parser.add_argument("--test_fraction", type=float, default=0.20)
#    parser.add_argument("--val_fraction", type=float, default=0.10)
#    parser.add_argument("--seed", type=int, default=42)
#    parser.add_argument("--split_seed", type=int, default=None)
#    parser.add_argument("--train_seed", type=int, default=None)
#    parser.add_argument("--normal_target", type=int, default=200)
#    parser.add_argument("--anomaly_target", type=int, default=200)
#    parser.add_argument("--unseen_actions", nargs="*", default=[])
#    parser.add_argument("--unseen_actors", nargs="*", default=[])
#    parser.add_argument("--unseen_styles", nargs="*", default=[])
#    parser.add_argument("--no_seen_healthy_in_unseen_style_test", action="store_true")
#    # Backbones
#    parser.add_argument("--project_root", default="", help="Parent directory containing the MotionCLIP package folder.")
#    parser.add_argument("--checkpoint", required=True, help="Pretrained MotionCLIP checkpoint.")
#    parser.add_argument("--clip_model", default="ViT-B/32")
#    parser.add_argument("--latent_dim", type=int, default=512)
#    parser.add_argument("--ff_size", type=int, default=1024)
#    parser.add_argument("--motion_num_layers", type=int, default=8)
#    parser.add_argument("--motion_num_heads", type=int, default=4)
#    parser.add_argument("--motion_dropout", type=float, default=0.1)
#    # Prompting
#    parser.add_argument("--healthy_condition", default="healthy")
#    parser.add_argument(
#        "--normal_prompt_templates", nargs="+",
#        default=["healthy {action}"],
#    )
#    parser.add_argument(
#        "--anomaly_prompt_templates", nargs="+",
#        default=["{condition} {action}"],
#    )
#    # AA adapters
#    parser.add_argument("--adapter_ratio", type=float, default=0.1)
#    parser.add_argument("--text_adapter_layer_count", type=int, default=3)
#    parser.add_argument("--no_train_text_projection", action="store_true")
#    parser.add_argument("--motion_adapter_layers", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
#    parser.add_argument("--motion_feature_layers", nargs="*", type=int, default=[1, 3, 5, 7])
#    parser.add_argument("--motion_pooling", choices=["prefix", "temporal_mean", "prefix_mean"], default="prefix_mean")
#    parser.add_argument("--motion_feature_fusion_ratio", type=float, default=0.1)
#    # Training
#    parser.add_argument("--stage1_epochs", type=int, default=5)
#    parser.add_argument("--stage2_epochs", type=int, default=20)
#    parser.add_argument("--stage1_lr", type=float, default=1e-5)
#    parser.add_argument("--stage2_lr", type=float, default=5e-4)
#    parser.add_argument("--stage1_weight_decay", type=float, default=1e-4)
#    parser.add_argument("--stage2_weight_decay", type=float, default=1e-4)
#    parser.add_argument("--stage1_disentangle_weight", type=float, default=0.1)
#    parser.add_argument(
#        "--stage1_templates_per_batch", type=int, default=1,
#        help=("Number of prompt templates encoded with gradients in each Stage-1 batch. "
#              "Using 1 cycles randomly across the full prompt ensemble and is recommended for 10 GB GPUs. "
#              "Validation and final anchors still average all templates."),
#    )
#    parser.add_argument("--stage2_contrastive_weight", type=float, default=0.1)
#    parser.add_argument("--temperature", type=float, default=0.07)
#    parser.add_argument("--batch_size", type=int, default=32)
#    parser.add_argument("--num_workers", type=int, default=4)
#    parser.add_argument("--grad_clip", type=float, default=1.0)
#    parser.add_argument("--amp", action="store_true")
#    parser.add_argument("--early_stopping_patience", type=int, default=5)
#    parser.add_argument("--threshold_criterion", choices=["balanced_accuracy", "f1", "accuracy"], default="balanced_accuracy")
#    parser.add_argument("--disable_prompted_unseen_style_eval", action="store_true")
#    args = parser.parse_args()
#    args.split_seed = args.seed if args.split_seed is None else args.split_seed
#    args.train_seed = args.seed if args.train_seed is None else args.train_seed
#    set_seed(args.train_seed)
#    output_dir = ensure_dir(args.output_dir)
#    split_dir = ensure_dir(output_dir / "splits")
#    stage1_dir = ensure_dir(output_dir / "stage1_text")
#    figure_export_dir = ensure_dir(stage1_dir / "figure2_3_exports")
#    stage2_dir = ensure_dir(output_dir / "stage2_motion")
#    test_dir = ensure_dir(output_dir / "test")
#    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    save_json(vars(args), output_dir / "args.json")
#    # Metadata and leakage-safe splits
#    df = pd.read_csv(args.csv_path).copy()
#    df["original_index"] = np.arange(len(df))
#    required = [args.path_col, args.action_col, args.condition_col, args.actor_col, args.label_col]
#    missing = [c for c in required if c not in df.columns]
#    if missing:
#        raise ValueError(f"CSV is missing columns {missing}. Available columns: {list(df.columns)}")
#    df[args.label_col] = df[args.label_col].astype(int)
#    if not set(df[args.label_col].unique()).issubset({0, 1}):
#        raise ValueError(f"{args.label_col} must contain binary labels 0/1.")
#    df["_action_norm"] = df[args.action_col].map(normalize_action_text)
#    df["_condition_norm"] = df[args.condition_col].map(normalize_action_text)
#    df[args.actor_col] = df[args.actor_col].astype(str).map(normalize_action_text)
#    args.healthy_condition = normalize_action_text(args.healthy_condition)
#    balanced_df, balanced_info = build_balanced_condition_subset(
#        df=df,
#        action_col="_action_norm",
#        condition_col="_condition_norm",
#        actor_col=args.actor_col,
#        label_col=args.label_col,
#        healthy_condition=args.healthy_condition,
#        normal_target=args.normal_target,
#        anomaly_target=args.anomaly_target,
#        seed=args.split_seed,
#    )
#    train_df, val_df, seen_test_df, unseen_union_df, combined_test_df, split_info = split_balanced_condition_experiment(
#        df=balanced_df,
#        action_col="_action_norm",
#        condition_col="_condition_norm",
#        actor_col=args.actor_col,
#        label_col=args.label_col,
#        healthy_condition=args.healthy_condition,
#        test_fraction=args.test_fraction,
#        val_fraction=args.val_fraction,
#        seed=args.split_seed,
#        unseen_actions=args.unseen_actions,
#        unseen_actors=args.unseen_actors,
#        unseen_styles=args.unseen_styles,
#        include_seen_healthy_in_unseen_style_test=not args.no_seen_healthy_in_unseen_style_test,
#    )
#    extra_buckets, extra_bucket_info = build_unseen_test_buckets(
#        balanced_df=balanced_df,
#        seen_test_df=seen_test_df,
#        action_col="_action_norm",
#        condition_col="_condition_norm",
#        actor_col=args.actor_col,
#        label_col=args.label_col,
#        healthy_condition=args.healthy_condition,
#        unseen_actions=args.unseen_actions,
#        unseen_actors=args.unseen_actors,
#        unseen_styles=args.unseen_styles,
#        seed=args.split_seed,
#        balance_binary=True,
#    )
#    requested_actions = set(parse_list_arg(args.unseen_actions))
#    requested_actors = set(parse_list_arg(args.unseen_actors))
#    requested_styles = set(parse_list_arg(args.unseen_styles))
#    leakage = {
#        "actions": sorted((set(train_df["_action_norm"]) | set(val_df["_action_norm"])) & requested_actions),
#        "actors": sorted((set(train_df[args.actor_col]) | set(val_df[args.actor_col])) & requested_actors),
#        "styles": sorted((set(train_df["_condition_norm"]) | set(val_df["_condition_norm"])) & requested_styles),
#    }
#    if any(leakage.values()):
#        raise RuntimeError(f"Unseen data leakage detected: {leakage}")
#    split_frames = {
#        "train": train_df,
#        "val": val_df,
#        "test_seen": seen_test_df,
#        "test_unseen_union": unseen_union_df,
#        "test_combined": combined_test_df,
#        **{f"test_{name}": frame for name, frame in extra_buckets.items()},
#    }
#    for name, frame in split_frames.items():
#        frame.to_csv(split_dir / f"{name}.csv", index=False)
#    split_summary = {
#        "split_type": split_info["split_type"],
#        "balanced_subset_info": balanced_info,
#        "counts": {name: int(len(frame)) for name, frame in split_frames.items()},
#        "label_counts": {
#            name: {str(k): int(v) for k, v in frame[args.label_col].value_counts().sort_index().to_dict().items()}
#            for name, frame in split_frames.items()
#        },
#        "train_styles": sorted(train_df["_condition_norm"].unique().tolist()),
#        "train_actions": sorted(train_df["_action_norm"].unique().tolist()),
#        "train_actors": sorted(train_df[args.actor_col].unique().tolist()),
#        "unseen_actions": sorted(requested_actions),
#        "unseen_actors": sorted(requested_actors),
#        "unseen_styles": sorted(requested_styles),
#        "extra_bucket_info": extra_bucket_info,
#        "leakage_check": leakage,
#        "leakage_check_passed": True,
#    }
#    save_json(split_summary, output_dir / "split_summary.json")
#    expected_shape = (args.num_frames, args.njoints, args.nfeats)
#    def make_dataset(frame: pd.DataFrame) -> AAPerMoMotionDataset:
#        return AAPerMoMotionDataset(
#            frame,
#            args.path_col,
#            args.action_col,
#            args.condition_col,
#            args.label_col,
#            args.motion_key,
#            expected_shape,
#            actor_col=args.actor_col,
#        )
#    train_ds = make_dataset(train_df)
#    train_sampler = BalancedBinaryBatchSampler(
#        train_df[args.label_col].astype(int).to_numpy(), args.batch_size, args.train_seed, False
#    )
#    train_loader = DataLoader(
#        train_ds, batch_sampler=train_sampler, num_workers=args.num_workers,
#        pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch,
#    )
#    train_eval_loader = DataLoader(
#        train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
#        pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch,
#    )
#    eval_loaders: Dict[str, DataLoader] = {
#        "train": train_eval_loader,
#        "val": DataLoader(make_dataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch),
#        "test_seen": DataLoader(make_dataset(seen_test_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch),
#        "test_unseen_union": DataLoader(make_dataset(unseen_union_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch),
#        "test_combined": DataLoader(make_dataset(combined_test_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch),
#    }
#    for name, frame in extra_buckets.items():
#        eval_loaders[f"test_{name}"] = DataLoader(
#            make_dataset(frame), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
#            pin_memory=torch.cuda.is_available(), collate_fn=aa_collate_batch,
#        )
#    if args.project_root:
#        sys.path.insert(0, str(Path(args.project_root).resolve()))
#    global Encoder_TRANSFORMER
#    from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER
#    base_motion = build_motionclip_encoder(
#        args.checkpoint, device, args.num_frames, args.njoints, args.nfeats,
#        args.latent_dim, args.ff_size, args.motion_num_layers, args.motion_num_heads, args.motion_dropout,
#    )
#    text_encoder = AATextEncoder(
#        args.clip_model, device, args.text_adapter_layer_count, args.adapter_ratio,
#        train_text_projection=not args.no_train_text_projection,
#    )
#    all_actions = balanced_df["_action_norm"].unique().tolist()
#    all_conditions = balanced_df["_condition_norm"].unique().tolist()
#    prompt_bank = PromptTokenBank(
#        text_encoder, all_actions, all_conditions, args.healthy_condition,
#        args.normal_prompt_templates, args.anomaly_prompt_templates,
#    )
#    save_json(prompt_bank.prompt_texts, output_dir / "prompts.json")
#    train_anomaly_styles = sorted(
#        c for c in train_df["_condition_norm"].unique().tolist() if c != args.healthy_condition
#    )
#    if not train_anomaly_styles:
#        raise ValueError("The training split contains no anomaly styles.")
#    # Stage 1: anomaly-aware text anchors
#    print("[INFO] Stage 1: training text residual adapters against frozen MotionCLIP features.")
#    train_actions_for_export = sorted(train_df["_action_norm"].unique().tolist())
#    train_conditions_for_export = sorted(train_df["_condition_norm"].unique().tolist())
#    train_pairs_for_export = sorted({
#        (str(action), str(condition))
#        for action, condition in zip(train_df["_action_norm"], train_df["_condition_norm"])
#    })
#    # Export true original CLIP features. The adapters are explicitly bypassed;
#    # merely exporting before training would not be sufficient because randomly
#    # initialized residual adapters already alter the forward pass.
#    text_encoder.set_adapters_enabled(False)
#    with torch.no_grad():
#        initial_cache = prompt_bank.encode(text_encoder, train_anomaly_styles, device, detach=True)
#    save_anchor_cache(initial_cache, figure_export_dir / "original_clip_anchor_cache.pt")
#    save_json(anchor_diagnostics(initial_cache), stage1_dir / "anchors_before_training.json")
#    original_prompt_export = save_prompt_feature_export(
#        prompt_bank=prompt_bank,
#        text_encoder=text_encoder,
#        output_path=figure_export_dir / "original_text_features_per_template.pt",
#        metadata_csv_path=figure_export_dir / "original_text_features_metadata.csv",
#        device=device,
#        encoder_state="original_clip",
#        train_actions=train_actions_for_export,
#        train_conditions=train_conditions_for_export,
#        train_pairs=train_pairs_for_export,
#    )
#    text_encoder.set_adapters_enabled(True)
#    stage1_trainable_count = int(sum(p.numel() for p in text_encoder.trainable_parameters()))
#    optimizer1 = torch.optim.AdamW(
#        text_encoder.trainable_parameters(), lr=args.stage1_lr, weight_decay=args.stage1_weight_decay
#    )
#    best_stage1_loss = float("inf")
#    best_stage1_auroc = float("nan")
#    best_stage1_epoch = -1
#    stage1_records: List[Dict[str, Any]] = []
#    stage1_ckpt_path = checkpoint_dir / "stage1_text_adapters_best.pt"
#    for epoch in range(1, args.stage1_epochs + 1):
#        started = time.time()
#        train_sampler.set_epoch(epoch)
#        losses = train_stage1_epoch(
#            base_motion, text_encoder, prompt_bank, train_anomaly_styles, train_loader,
#            optimizer1, device, args.temperature, args.stage1_disentangle_weight,
#            args.grad_clip, args.amp, args.stage1_templates_per_batch,
#        )
#        with torch.no_grad():
#            cache = prompt_bank.encode(text_encoder, train_anomaly_styles, device, detach=True)
#        train_output = evaluate_with_cache(base_motion, train_eval_loader, cache, device, args.temperature, is_base_encoder=True)
#        val_output = evaluate_with_cache(base_motion, eval_loaders["val"], cache, device, args.temperature, is_base_encoder=True)
#        train_metrics = compute_binary_metrics(train_output.y_true, train_output.score, None, args.threshold_criterion)
#        val_metrics = compute_binary_metrics(val_output.y_true, val_output.score, None, args.threshold_criterion)
#        record = {
#            "epoch": epoch,
#            "train_loss": losses["loss"],
#            "train_classification_loss": losses["classification"],
#            "train_disentangle_loss": losses["disentangle"],
#            "val_loss": val_output.classification_loss,
#            "val_total_loss": val_output.loss,
#            "train_auroc": train_metrics["auroc"],
#            "val_auroc": val_metrics["auroc"],
#            "train_auprc": train_metrics["auprc"],
#            "val_auprc": val_metrics["auprc"],
#            "val_f1": val_metrics["f1"],
#            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
#            "normal_anomaly_anchor_cosine": anchor_diagnostics(cache)["normal_anomaly_cosine_mean"],
#            "seconds": time.time() - started,
#        }
#        stage1_records.append(record)
#        save_stage_curves(stage1_records, stage1_dir)
#        print(
#            f"[STAGE1 {epoch:03d}] loss={losses['loss']:.4f} "
#            f"dis={losses['disentangle']:.4f} val_loss={val_output.classification_loss:.6f} "
#            f"val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f}"
#        )
#        current = val_output.classification_loss
#        if np.isfinite(current) and current < best_stage1_loss:
#            best_stage1_loss = float(current)
#            best_stage1_auroc = float(val_metrics["auroc"])
#            best_stage1_epoch = epoch
#            torch.save({
#                "epoch": epoch,
#                "text_adaptation": text_encoder.adaptation_state_dict(),
#                "best_val_loss": best_stage1_loss,
#                "best_val_metrics": val_metrics,
#                "train_anomaly_styles": train_anomaly_styles,
#                "args": vars(args),
#            }, stage1_ckpt_path)
#        if should_stop_without_improvement(epoch, best_stage1_epoch, args.early_stopping_patience):
#            print(f"[INFO] Stage 1 early stopping at epoch {epoch}.")
#            break
#    if not stage1_ckpt_path.exists():
#        torch.save({
#            "epoch": args.stage1_epochs,
#            "text_adaptation": text_encoder.adaptation_state_dict(),
#            "best_val_metrics": {},
#            "train_anomaly_styles": train_anomaly_styles,
#            "args": vars(args),
#        }, stage1_ckpt_path)
#    stage1_ckpt = torch_load_compat(stage1_ckpt_path, map_location=device)
#    text_encoder.load_adaptation_state_dict(stage1_ckpt["text_adaptation"])
#    text_encoder.freeze_adaptation()
#    text_encoder.set_adapters_enabled(True)
#    with torch.no_grad():
#        strict_cache = prompt_bank.encode(text_encoder, train_anomaly_styles, device, detach=True)
#    save_anchor_cache(strict_cache, stage1_dir / "strict_anchor_cache.pt")
#    save_anchor_cache(strict_cache, figure_export_dir / "adapted_anchor_cache.pt")
#    # Compatibility alias for scripts that expect this historical filename.
#    save_anchor_cache(strict_cache, output_dir / "text_prompt_cache.pt")
#    save_json(anchor_diagnostics(strict_cache), stage1_dir / "anchors_after_training.json")
#    adapted_prompt_export = save_prompt_feature_export(
#        prompt_bank=prompt_bank,
#        text_encoder=text_encoder,
#        output_path=figure_export_dir / "adapted_text_features_per_template.pt",
#        metadata_csv_path=figure_export_dir / "adapted_text_features_metadata.csv",
#        device=device,
#        encoder_state="adapted_aaclip",
#        train_actions=train_actions_for_export,
#        train_conditions=train_conditions_for_export,
#        train_pairs=train_pairs_for_export,
#    )
#    figure_export_manifest = {
#        "purpose": "Inputs for AA-CLIP-style Figure 2 similarity heatmaps and Figure 3 t-SNE plots.",
#        "original_features_are_true_original_clip": True,
#        "original": original_prompt_export,
#        "adapted": adapted_prompt_export,
#        "original_anchor_cache": str(figure_export_dir / "original_clip_anchor_cache.pt"),
#        "adapted_anchor_cache": str(figure_export_dir / "adapted_anchor_cache.pt"),
#        "text_prompt_cache_alias": str(output_dir / "text_prompt_cache.pt"),
#        "figure2_guidance": (
#            "Use cosine_similarity_matrix with metadata ordering, or compute cosine similarity "
#            "from mean_features_by_action_condition for a compact normal/anomaly heatmap."
#        ),
#        "figure3_guidance": (
#            "Run t-SNE jointly on original and adapted per-template features using the same random seed. "
#            "Use semantic_label for normal/anomaly and class_seen/action_seen or style_seen for seen/unseen."
#        ),
#    }
#    save_json(figure_export_manifest, figure_export_dir / "manifest.json")
#    # Stage 2: align frozen MotionCLIP through motion adapters
#    print("[INFO] Stage 2: training MotionCLIP residual adapters/projectors against fixed text anchors.")
#    motion_model = AAMotionEncoder(
#        base_motion,
#        latent_dim=args.latent_dim,
#        adapter_layers=args.motion_adapter_layers,
#        feature_layers=args.motion_feature_layers,
#        adapter_ratio=args.adapter_ratio,
#        pooling=args.motion_pooling,
#        feature_fusion_ratio=args.motion_feature_fusion_ratio,
#    ).to(device)
#    optimizer2 = torch.optim.AdamW(
#        motion_model.trainable_parameters(), lr=args.stage2_lr, weight_decay=args.stage2_weight_decay
#    )
#    best_stage2_loss = float("inf")
#    best_stage2_auroc = float("nan")
#    best_stage2_epoch = -1
#    best_threshold = 0.5
#    stage2_records: List[Dict[str, Any]] = []
#    stage2_ckpt_path = checkpoint_dir / "stage2_motion_adapters_best.pt"
#    for epoch in range(1, args.stage2_epochs + 1):
#        started = time.time()
#        train_sampler.set_epoch(10000 + epoch)
#        losses = train_stage2_epoch(
#            motion_model, strict_cache, train_loader, optimizer2, device,
#            args.temperature, args.stage2_contrastive_weight, args.grad_clip, args.amp,
#        )
#        train_output = evaluate_with_cache(
#            motion_model, train_eval_loader, strict_cache, device, args.temperature,
#            args.stage2_contrastive_weight, is_base_encoder=False,
#        )
#        val_output = evaluate_with_cache(
#            motion_model, eval_loaders["val"], strict_cache, device, args.temperature,
#            args.stage2_contrastive_weight, is_base_encoder=False,
#        )
#        train_metrics = compute_binary_metrics(train_output.y_true, train_output.score, None, args.threshold_criterion)
#        val_metrics = compute_binary_metrics(val_output.y_true, val_output.score, None, args.threshold_criterion)
#        record = {
#            "epoch": epoch,
#            "train_loss": losses["loss"],
#            "train_classification_loss": losses["classification"],
#            "train_contrastive_loss": losses["contrastive"],
#            "val_loss": val_output.classification_loss,
#            "val_total_loss": val_output.loss,
#            "val_contrastive_loss": val_output.contrastive_loss,
#            "train_auroc": train_metrics["auroc"],
#            "val_auroc": val_metrics["auroc"],
#            "train_auprc": train_metrics["auprc"],
#            "val_auprc": val_metrics["auprc"],
#            "val_f1": val_metrics["f1"],
#            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
#            "val_threshold": val_metrics["threshold"],
#            "seconds": time.time() - started,
#        }
#        stage2_records.append(record)
#        save_stage_curves(stage2_records, stage2_dir)
#        print(
#            f"[STAGE2 {epoch:03d}] loss={losses['loss']:.4f} "
#            f"contrast={losses['contrastive']:.4f} val_loss={val_output.classification_loss:.6f} "
#            f"val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f} "
#            f"threshold={val_metrics['threshold']:.4f}"
#        )
#        current = val_output.classification_loss
#        if np.isfinite(current) and current < best_stage2_loss:
#            best_stage2_loss = float(current)
#            best_stage2_auroc = float(val_metrics["auroc"])
#            best_stage2_epoch = epoch
#            best_threshold = float(val_metrics["threshold"])
#            torch.save({
#                "epoch": epoch,
#                "motion_adaptation": motion_model.adaptation_state_dict(),
#                "best_val_loss": best_stage2_loss,
#                "best_val_metrics": val_metrics,
#                "threshold": best_threshold,
#                "strict_anchor_cache_path": str(stage1_dir / "strict_anchor_cache.pt"),
#                "args": vars(args),
#            }, stage2_ckpt_path)
#        if should_stop_without_improvement(epoch, best_stage2_epoch, args.early_stopping_patience):
#            print(f"[INFO] Stage 2 early stopping at epoch {epoch}.")
#            break
#    if not stage2_ckpt_path.exists():
#        torch.save({
#            "epoch": args.stage2_epochs,
#            "motion_adaptation": motion_model.adaptation_state_dict(),
#            "best_val_metrics": {},
#            "threshold": best_threshold,
#            "strict_anchor_cache_path": str(stage1_dir / "strict_anchor_cache.pt"),
#            "args": vars(args),
#        }, stage2_ckpt_path)
#    stage2_ckpt = torch_load_compat(stage2_ckpt_path, map_location=device)
#    motion_model.load_adaptation_state_dict(stage2_ckpt["motion_adaptation"])
#    best_threshold = float(stage2_ckpt.get("threshold", 0.5))
#    motion_model.eval()
#    # Final strict inference on every saved split/bucket
#    strict_metrics: Dict[str, Any] = {}
#    strict_breakdowns: Dict[str, Any] = {}
#    for name, loader in eval_loaders.items():
#        output = evaluate_with_cache(
#            motion_model, loader, strict_cache, device, args.temperature,
#            args.stage2_contrastive_weight, is_base_encoder=False,
#        )
#        metrics = compute_binary_metrics(output.y_true, output.score, best_threshold, args.threshold_criterion)
#        metrics["loss"] = output.classification_loss
#        metrics["total_loss"] = output.loss
#        metrics["contrastive_loss"] = output.contrastive_loss
#        strict_metrics[name] = metrics
#        prediction_df = save_eval_output(output, test_dir / f"strict_{name}", best_threshold)
#        strict_breakdowns[name] = save_breakdowns(prediction_df, test_dir, f"strict_{name}", best_threshold)
#    # Optional prompted unseen-style inference. The held-out style names are introduced only here.
#    prompted_metrics: Dict[str, Any] = {}
#    if requested_styles and not args.disable_prompted_unseen_style_eval:
#        with torch.no_grad():
#            prompted_cache = prompt_bank.encode(text_encoder, sorted(requested_styles), device, detach=True)
#        save_anchor_cache(prompted_cache, stage1_dir / "prompted_unseen_style_anchor_cache.pt")
#        save_json(anchor_diagnostics(prompted_cache), stage1_dir / "prompted_unseen_style_anchor_diagnostics.json")
#        style_loader_names = [name for name in eval_loaders if "style" in name]
#        for name in style_loader_names:
#            output = evaluate_with_cache(
#                motion_model, eval_loaders[name], prompted_cache, device, args.temperature,
#                0.0, is_base_encoder=False,
#            )
#            metrics = compute_binary_metrics(output.y_true, output.score, best_threshold, args.threshold_criterion)
#            metrics["loss"] = output.classification_loss
#            metrics["total_loss"] = output.loss
#            metrics["contrastive_loss"] = output.contrastive_loss
#            prompted_metrics[name] = metrics
#            save_eval_output(output, test_dir / f"prompted_{name}", best_threshold)
#    model_info = {
#        "base_motion_encoder": count_parameters(base_motion),
#        "text_model_total": count_parameters(text_encoder),
#        "text_adapter_trainable_during_stage1": stage1_trainable_count,
#        "motion_adapter_model": count_parameters(motion_model),
#        "motion_trainable": int(sum(p.numel() for p in motion_model.trainable_parameters())),
#        "text_adapter_layers": text_encoder.adapter_indices,
#        "motion_adapter_layers": motion_model.adapter_indices,
#        "motion_feature_layers": motion_model.feature_indices,
#        "frozen_backbone_policy": "All original CLIP and MotionCLIP backbone parameters remain frozen.",
#    }
#    save_json(model_info, output_dir / "model_info.json")
#    final_summary = {
#        "method": "AA-MotionCLIP two-stage residual-adapter adaptation",
#        "stage1": {
#            "best_epoch": best_stage1_epoch,
#            "best_val_loss": best_stage1_loss,
#            "best_val_auroc": best_stage1_auroc,
#            "checkpoint": str(stage1_ckpt_path),
#            "anchor_diagnostics": anchor_diagnostics(strict_cache),
#        },
#        "stage2": {
#            "best_epoch": best_stage2_epoch,
#            "best_val_loss": best_stage2_loss,
#            "best_val_auroc": best_stage2_auroc,
#            "validation_threshold": best_threshold,
#            "checkpoint": str(stage2_ckpt_path),
#        },
#        "strict_metrics": strict_metrics,
#        "prompted_unseen_style_metrics": prompted_metrics,
#        "strict_breakdowns": strict_breakdowns,
#        "split_summary": split_summary,
#        "model_info": model_info,
#        "figure2_3_exports": figure_export_manifest,
#        "important_interpretation": {
#            "strict_unseen_style": "The anomaly anchor contains only seen training anomaly styles.",
#            "prompted_unseen_style": "Held-out style names are used only at inference to construct the anomaly anchor.",
#        },
#    }
#    save_json(final_summary, output_dir / "metrics.json")
#    print("[DONE] Strict combined-test metrics:")
#    print(json.dumps(strict_metrics.get("test_combined", {}), indent=2, sort_keys=True))
#    if prompted_metrics:
#        print("[DONE] Prompted unseen-style AUROCs:")
#        print(json.dumps({k: v.get("auroc") for k, v in prompted_metrics.items()}, indent=2, sort_keys=True))
#    print(f"[DONE] All outputs saved to {output_dir}")
#if __name__ == "__main__":
#    main()
