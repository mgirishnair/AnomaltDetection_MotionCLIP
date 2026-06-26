#!/usr/bin/env python3
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
    examples are seen, without changing the train/val/test split or model initialization.
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
    conditions: Sequence[str],
    text_encoder: FrozenCLIPTextEncoder,
    normal_prompt_template: str,
    condition_prompt_template: str,
    healthy_condition: str,
    device: torch.device,
) -> Tuple[Dict[str, int], Dict[str, int], torch.Tensor, Dict[str, Dict[str, str]]]:
    """Build prompt embeddings for every condition/style per action.
    Healthy prompts use normal_prompt_template, e.g. "healthy {action}".
    Non-healthy prompts use condition_prompt_template, e.g. "{condition} {action}".
    text_feats has shape [num_actions, num_conditions, dim].
    """
    actions = sorted({normalize_action_text(a) for a in actions})
    healthy_condition = normalize_action_text(healthy_condition)
    conditions_set = {normalize_action_text(c) for c in conditions}
    conditions_set.add(healthy_condition)
    conditions_sorted = [healthy_condition] + sorted(c for c in conditions_set if c != healthy_condition)
    action_to_idx = {a: i for i, a in enumerate(actions)}
    condition_to_idx = {c: i for i, c in enumerate(conditions_sorted)}
    prompt_info: Dict[str, Dict[str, str]] = {}
    texts = []
    for action in actions:
        prompt_info[action] = {}
        for condition in conditions_sorted:
            if condition == healthy_condition:
                prompt = normal_prompt_template.format(action=action, condition=condition)
            else:
                prompt = condition_prompt_template.format(action=action, condition=condition)
            prompt_info[action][condition] = prompt
            texts.append(prompt)
    text_feats = text_encoder.encode(texts)  # [A*C, D]
    text_feats = text_feats.reshape(len(actions), len(conditions_sorted), -1).to(device)
    return action_to_idx, condition_to_idx, text_feats, prompt_info
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
    if len(y_true) == 0:
        return {
            "auroc": float("nan"), "auprc": float("nan"), "n_samples": 0,
            "n_normal": 0, "n_anomaly": 0, "score_mean": float("nan"), "score_std": float("nan"),
            "threshold_source": "empty_split", "threshold": float(threshold or 0.0),
            "accuracy": float("nan"), "balanced_accuracy": float("nan"), "precision": float("nan"),
            "recall": float("nan"), "specificity": float("nan"), "f1": float("nan"),
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        }
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
    s_anomaly_max: np.ndarray
    embeddings: np.ndarray
    paths: List[str]
    actions: List[str]
    conditions: List[str]
    max_anomaly_conditions: List[str]
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
    """Return logits against all condition/style prompts for each sample's action.
    text_feats shape: [num_actions, num_conditions, dim]
    output logits shape: [B, num_conditions]
    """
    z = encode_motion_auto(motion_encoder, motion)
    z = F.normalize(z.float(), dim=-1)
    idx = torch.tensor([action_to_idx[normalize_action_text(a)] for a in actions], dtype=torch.long, device=motion.device)
    prompts = text_feats[idx]  # [B,C,D]
    if z.shape[-1] != prompts.shape[-1]:
        raise RuntimeError(
            f"Motion embedding dim ({z.shape[-1]}) does not match text embedding dim ({prompts.shape[-1]}). "
            "Check --latent_dim and CLIP model."
        )
    logits = torch.bmm(prompts, z.unsqueeze(-1)).squeeze(-1) / temperature  # [B,C]
    return logits, z
def target_text_embeddings_and_group_ids_for_batch(
    actions: Sequence[str],
    conditions: Sequence[str],
    action_to_idx: Dict[str, int],
    condition_to_idx: Dict[str, int],
    text_feats: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return matching text embeddings and group ids for condition-per-action prompts."""
    action_idx = torch.tensor(
        [action_to_idx[normalize_action_text(a)] for a in actions],
        dtype=torch.long,
        device=device,
    )
    condition_idx = torch.tensor(
        [condition_to_idx[normalize_action_text(c)] for c in conditions],
        dtype=torch.long,
        device=device,
    )
    target_text = text_feats[action_idx, condition_idx]  # [B, D]
    group_ids = action_idx * len(condition_to_idx) + condition_idx
    return target_text, group_ids
def symmetric_motion_text_contrastive_loss(
    motion_z: torch.Tensor,
    text_z: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Bidirectional supervised contrastive loss between motions and matching text prompts.
    Samples with the same action and healthy/condition label are treated as positives.
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
    condition_to_idx: Dict[str, int],
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
            target_text, group_ids = target_text_embeddings_and_group_ids_for_batch(
                batch["action"], batch["condition"], action_to_idx, condition_to_idx, text_feats, motion.device
            )
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
    condition_to_idx: Dict[str, int],
    text_feats: torch.Tensor,
    temperature: float,
    healthy_condition: str,
    compute_contrastive_loss: bool = True,
) -> EvalOutput:
    motion_encoder.eval()
    losses = []
    y_true = []
    scores = []
    probs = []
    s_h = []
    s_anom = []
    embeddings = []
    paths = []
    actions = []
    conditions = []
    max_anomaly_conditions = []
    row_indices = []
    healthy_idx = condition_to_idx[normalize_action_text(healthy_condition)]
    idx_to_condition = {v: k for k, v in condition_to_idx.items()}
    anomaly_indices = [i for c, i in condition_to_idx.items() if c != normalize_action_text(healthy_condition)]
    if not anomaly_indices:
        raise ValueError("Need at least one non-healthy condition/style prompt for anomaly scoring.")
    anomaly_indices_t = torch.tensor(anomaly_indices, dtype=torch.long, device=device)
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits, z = logits_from_motion_and_prompts(
            motion_encoder, motion, batch["action"], action_to_idx, text_feats, temperature
        )  # [B, C]
        if compute_contrastive_loss:
            target_text, group_ids = target_text_embeddings_and_group_ids_for_batch(
                batch["action"], batch["condition"], action_to_idx, condition_to_idx, text_feats, motion.device
            )
            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
            losses.append(float(loss.detach().cpu()) * labels.numel())
        healthy_logits = logits[:, healthy_idx]
        anomaly_logits_all = logits.index_select(dim=1, index=anomaly_indices_t)
        anomaly_max_logits, anomaly_argmax = anomaly_logits_all.max(dim=1)
        chosen_anomaly_indices = anomaly_indices_t[anomaly_argmax].detach().cpu().numpy().astype(int).tolist()
        chosen_anomaly_conditions = [idx_to_condition[i] for i in chosen_anomaly_indices]
        # anomaly score: positive means closer to the strongest non-healthy condition prompt than healthy prompt
        score = anomaly_max_logits - healthy_logits
        two_class_logits = torch.stack([healthy_logits, anomaly_max_logits], dim=1)
        soft = torch.softmax(two_class_logits, dim=-1)
        embeddings.append(z.detach().cpu().numpy().astype(np.float32))
        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        scores.extend(score.cpu().numpy().astype(float).tolist())
        probs.extend(soft[:, 1].cpu().numpy().astype(float).tolist())
        s_h.extend(healthy_logits.cpu().numpy().astype(float).tolist())
        s_anom.extend(anomaly_max_logits.cpu().numpy().astype(float).tolist())
        paths.extend(batch["path"])
        actions.extend(batch["action"])
        conditions.extend(batch["condition"])
        max_anomaly_conditions.extend(chosen_anomaly_conditions)
        row_indices.extend(batch["row_index"])
    total_n = max(1, len(y_true))
    avg_loss = sum(losses) / total_n if losses else float("nan")
    return EvalOutput(
        loss=avg_loss,
        y_true=np.asarray(y_true, dtype=int),
        score=np.asarray(scores, dtype=float),
        prob_anomaly=np.asarray(probs, dtype=float),
        s_healthy=np.asarray(s_h, dtype=float),
        s_anomaly_max=np.asarray(s_anom, dtype=float),
        embeddings=np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 512), dtype=np.float32),
        paths=paths,
        actions=actions,
        conditions=conditions,
        max_anomaly_conditions=max_anomaly_conditions,
        row_indices=row_indices,
    )
def save_predictions(eval_out: EvalOutput, path: str | Path, threshold: float) -> None:
    pred = (eval_out.score >= threshold).astype(int)
    out_df = pd.DataFrame({
        "row_index": eval_out.row_indices,
        "motion_path": eval_out.paths,
        "action": eval_out.actions,
        "condition": eval_out.conditions,
        "y_true_is_anomaly": eval_out.y_true,
        "anomaly_score_max_non_healthy_minus_healthy": eval_out.score,
        "anomaly_score_flawed_minus_healthy": eval_out.score,  # compatibility alias
        "prob_anomaly": eval_out.prob_anomaly,
        "logit_healthy": eval_out.s_healthy,
        "logit_max_non_healthy": eval_out.s_anomaly_max,
        "max_non_healthy_condition": eval_out.max_anomaly_conditions,
        "pred_is_anomaly": pred,
    })
    out_df.to_csv(path, index=False)
def save_embeddings(eval_out: EvalOutput, npz_path: str | Path, metadata_csv_path: str | Path, threshold: float) -> None:
    """Save normalized MotionCLIP embeddings and metadata for plotting/debugging.
    NPZ keys:
      embeddings: [N, D] normalized motion embeddings used for prompt scoring
      y_true: [N] 0=healthy, 1=non-healthy/anomaly
      score: [N] max_non_healthy_minus_healthy anomaly score
      prob_anomaly: [N] softmax probability for non-healthy/anomaly prompt
      logit_healthy/logit_max_non_healthy: [N] prompt logits
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
        logit_max_non_healthy=eval_out.s_anomaly_max.astype(np.float32),
        row_index=np.asarray(eval_out.row_indices, dtype=np.int64),
        pred_is_anomaly=pred.astype(np.int64),
    )
    meta_df = pd.DataFrame({
        "embedding_index": np.arange(len(eval_out.row_indices), dtype=int),
        "row_index": eval_out.row_indices,
        "motion_path": eval_out.paths,
        "action": eval_out.actions,
        "condition": eval_out.conditions,
        "y_true_is_anomaly": eval_out.y_true,
        "anomaly_score_max_non_healthy_minus_healthy": eval_out.score,
        "anomaly_score_flawed_minus_healthy": eval_out.score,  # compatibility alias
        "prob_anomaly": eval_out.prob_anomaly,
        "logit_healthy": eval_out.s_healthy,
        "logit_max_non_healthy": eval_out.s_anomaly_max,
        "max_non_healthy_condition": eval_out.max_anomaly_conditions,
        "pred_is_anomaly": pred,
    })
    meta_df.to_csv(metadata_csv_path, index=False)
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MotionCLIP with healthy/condition action prompts on PerMo.")
    # Data
    parser.add_argument("--csv_path", required=True, help="Path to PerMo metadata CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory where outputs are saved.")
    parser.add_argument("--path_col", default="motion_path")
    parser.add_argument("--action_col", default="action_label")
    parser.add_argument("--condition_col", default="condition_label")
    parser.add_argument("--label_col", default="is_anomaly")
    parser.add_argument("--motion_key", default="auto", help="NPZ key. Use 'auto' to infer.")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--njoints", type=int, default=25)
    parser.add_argument("--nfeats", type=int, default=6)
    # Split / balanced PerMo Condition experiments
    parser.add_argument("--test_fraction", type=float, default=0.20, help="Test fraction for seen samples; default gives 80/20 train/test before validation is carved out of train.")
    parser.add_argument("--val_fraction", type=float, default=0.10, help="Validation fraction taken from the non-test seen pool.")
    parser.add_argument("--seed", type=int, default=42, help="Backward-compatible fallback seed. Used when --split_seed or --train_seed are not provided.")
    parser.add_argument("--split_seed", type=int, default=None, help="Controls dataset subset selection, train/val/test splitting, and evaluation bucket sampling. Defaults to --seed.")
    parser.add_argument("--train_seed", type=int, default=None, help="Controls training randomness, PyTorch/NumPy/Python RNGs, and balanced train batch sampling. Defaults to --seed.")
    parser.add_argument("--order_seed", type=int, default=None, help="Controls only the order/batch arrangement of selected training samples. Defaults to --train_seed.")
    parser.add_argument("--actor_col", default="actor_label", help="CSV column containing actor identity. Required for actor-balanced sampling.")
    parser.add_argument("--normal_target", type=int, default=200, help="Number of healthy/normal samples to keep.")
    parser.add_argument("--anomaly_target", type=int, default=200, help="Number of non-healthy/anomaly samples to keep.")
    parser.add_argument("--unseen_actions", nargs="*", default=[], help="Action names to keep completely out of train/val; comma-separated values are also accepted.")
    parser.add_argument("--unseen_actors", nargs="*", default=[], help="Actor IDs/names to keep completely out of train/val; comma-separated values are also accepted.")
    parser.add_argument("--unseen_styles", nargs="*", default=[], help="Non-healthy condition styles to keep completely out of train/val.")
    parser.add_argument("--no_seen_healthy_in_unseen_style_test", action="store_true", help="For style-only holdout, do not add held-out healthy seen-style samples to the unseen-style test set.")
    # MotionCLIP model
    parser.add_argument("--project_root", default="", help="Parent directory containing the MotionCLIP folder.")
    parser.add_argument("--checkpoint", required=True, help="Pretrained MotionCLIP checkpoint.")
    parser.add_argument("--trainable_layers", type=int, default=2)
    # Text prompts
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--healthy_condition", default="healthy", help="Condition/style name treated as normal.")
    parser.add_argument("--normal_prompt_template", default="healthy {action}")
    parser.add_argument("--condition_prompt_template", default="{condition} {action}", help="Prompt template for non-healthy condition/style prompts.")
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
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.train_seed is None:
        args.train_seed = args.seed
    if args.order_seed is None:
        args.order_seed = args.train_seed
    # Training randomness is kept separate from split/sampling randomness.
    # order_seed is intentionally not passed to set_seed so changing it only
    # changes the balanced training sample order, not initialization/dropout RNGs.
    set_seed(args.train_seed)
    output_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_json(vars(args), output_dir / "args.json")
    # Load metadata
    df = pd.read_csv(args.csv_path)
    df = df.copy()
    df["original_index"] = np.arange(len(df))
    required_cols = [args.path_col, args.action_col, args.condition_col, args.actor_col, args.label_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")
    # Normalize labels/actions
    df[args.label_col] = df[args.label_col].astype(int)
    if not set(df[args.label_col].unique()).issubset({0, 1}):
        raise ValueError(f"{args.label_col} must contain only 0/1 labels.")
    df["_action_norm"] = df[args.action_col].map(normalize_action_text)
    df["_condition_norm"] = df[args.condition_col].map(normalize_action_text)
    args.healthy_condition = normalize_action_text(args.healthy_condition)
    # Basic file existence check
    missing_paths = [p for p in df[args.path_col].head(20).tolist() if not Path(str(p)).exists()]
    if missing_paths:
        print("[WARN] Some example motion files do not exist from this machine.")
        print("       This is okay only if you are testing the script outside the data machine.")
        print(f"       First missing example: {missing_paths[0]}")
    # Build the balanced 200 healthy + 200 anomaly subset first.
    balanced_df, balanced_subset_info = build_balanced_condition_subset(
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
    train_df, val_df, seen_test_df, unseen_test_df, test_df, split_mode_info = split_balanced_condition_experiment(
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
    # Leakage checks for all unseen dimensions.
    train_actions = set(train_df["_action_norm"].unique().tolist())
    val_actions = set(val_df["_action_norm"].unique().tolist())
    train_actors = set(train_df[args.actor_col].astype(str).map(normalize_action_text).unique().tolist())
    val_actors = set(val_df[args.actor_col].astype(str).map(normalize_action_text).unique().tolist())
    train_styles = set(train_df["_condition_norm"].unique().tolist())
    val_styles = set(val_df["_condition_norm"].unique().tolist())
    requested_unseen_actions = set(parse_list_arg(args.unseen_actions))
    requested_unseen_actors = set(parse_list_arg(args.unseen_actors))
    requested_unseen_styles = set(parse_list_arg(args.unseen_styles))
    action_leakage = sorted((train_actions | val_actions) & requested_unseen_actions)
    actor_leakage = sorted((train_actors | val_actors) & requested_unseen_actors)
    style_leakage = sorted((train_styles | val_styles) & requested_unseen_styles)
    if action_leakage or actor_leakage or style_leakage:
        raise RuntimeError(
            f"Unseen leakage detected: actions={action_leakage}, actors={actor_leakage}, styles={style_leakage}"
        )
    extra_test_buckets, extra_test_bucket_info = build_unseen_test_buckets(
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
    test_balance_info = {
        "enabled": True,
        "note": "The whole experiment is balanced before splitting: 200 healthy and 200 anomaly samples. The anomaly subset is marginally balanced over style, action, and actor. Evaluation buckets are downsampled to equal normal/anomaly counts when both labels exist.",
        "balanced_subset_info": balanced_subset_info,
        "extra_test_bucket_info": extra_test_bucket_info,
    }
    train_df.to_csv(output_dir / "split_train.csv", index=False)
    val_df.to_csv(output_dir / "split_val.csv", index=False)
    seen_test_df.to_csv(output_dir / "split_test_seen.csv", index=False)
    unseen_test_df.to_csv(output_dir / "split_test_unseen_union.csv", index=False)
    test_df.to_csv(output_dir / "split_test_combined.csv", index=False)
    # Compatibility filenames from older versions.
    seen_test_df.to_csv(output_dir / "split_test_seen_actions.csv", index=False)
    unseen_test_df.to_csv(output_dir / "split_test_unseen_actions.csv", index=False)
    test_df.to_csv(output_dir / "split_test.csv", index=False)
    for bucket_name, bucket_df in extra_test_buckets.items():
        bucket_df.to_csv(output_dir / f"split_test_{bucket_name}.csv", index=False)
    split_summary = {
        "split_type": split_mode_info["split_type"],
        "total_original_rows": int(len(df)),
        "total_balanced_subset_rows": int(len(balanced_df)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "seen_test": int(len(seen_test_df)),
        "unseen_union_test": int(len(unseen_test_df)),
        "combined_test": int(len(test_df)),
        "extra_test_buckets": {k: int(len(v)) for k, v in extra_test_buckets.items()},
        "extra_test_bucket_info": extra_test_bucket_info,
        "train_label_counts": train_df[args.label_col].value_counts().to_dict(),
        "val_label_counts": val_df[args.label_col].value_counts().to_dict(),
        "seen_test_label_counts": seen_test_df[args.label_col].value_counts().to_dict(),
        "unseen_union_test_label_counts": unseen_test_df[args.label_col].value_counts().to_dict(),
        "combined_test_label_counts": test_df[args.label_col].value_counts().to_dict(),
        "train_condition_counts": train_df["_condition_norm"].value_counts().sort_index().to_dict(),
        "test_condition_counts": test_df["_condition_norm"].value_counts().sort_index().to_dict(),
        "test_balance_info": test_balance_info,
        "all_actions": sorted(balanced_df["_action_norm"].unique().tolist()),
        "all_conditions": sorted(balanced_df["_condition_norm"].unique().tolist()),
        "all_actors": sorted(balanced_df[args.actor_col].astype(str).map(normalize_action_text).unique().tolist()),
        "healthy_condition": args.healthy_condition,
        "seen_train_val_actions": sorted((train_actions | val_actions)),
        "seen_train_val_actors": sorted((train_actors | val_actors)),
        "seen_train_val_styles": sorted((train_styles | val_styles)),
        "heldout_unseen_actions": split_mode_info["unseen_actions"],
        "heldout_unseen_actors": split_mode_info["unseen_actors"],
        "heldout_unseen_styles": split_mode_info["unseen_styles"],
        "unseen_leakage_check_passed": True,
    }
    save_json(split_summary, output_dir / "split_summary.json")
    print("[INFO] Split summary:", split_summary)
    expected_shape = (args.num_frames, args.njoints, args.nfeats)
    train_ds = PerMoMotionDataset(train_df, args.path_col, args.action_col, args.condition_col, args.label_col, args.motion_key, expected_shape)
    val_ds = PerMoMotionDataset(val_df, args.path_col, args.action_col, args.condition_col, args.label_col, args.motion_key, expected_shape)
    test_ds = PerMoMotionDataset(test_df, args.path_col, args.action_col, args.condition_col, args.label_col, args.motion_key, expected_shape)
    seen_test_ds = PerMoMotionDataset(seen_test_df, args.path_col, args.action_col, args.condition_col, args.label_col, args.motion_key, expected_shape)
    unseen_test_ds = PerMoMotionDataset(unseen_test_df, args.path_col, args.action_col, args.condition_col, args.label_col, args.motion_key, expected_shape)
    train_labels = train_df[args.label_col].astype(int).to_numpy()
    train_batch_sampler = BalancedBinaryBatchSampler(
        labels=train_labels,
        batch_size=args.batch_size,
        seed=args.train_seed,
        drop_last=False,
        order_seed=args.order_seed,
    )
    sampler_info = {
        "type": "BalancedBinaryBatchSampler",
        "purpose": "class-aware healthy/condition training batches",
        "batch_size": int(args.batch_size),
        "seed": int(args.train_seed),
        "order_seed": int(args.order_seed),
        "healthy_per_batch": int(train_batch_sampler.n0),
        "flawed_per_batch": int(train_batch_sampler.n1),
        "num_batches_per_epoch": int(len(train_batch_sampler)),
        "train_label_counts": {
            "healthy_0": int((train_labels == 0).sum()),
            "flawed_1": int((train_labels == 1).sum()),
        },
    }
    save_json(sampler_info, output_dir / "train_sampler_info.json")
    print("[INFO] Train sampler info:", sampler_info)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    seen_test_loader = DataLoader(
        seen_test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    unseen_test_loader = DataLoader(
        unseen_test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    extra_test_loaders: Dict[str, DataLoader] = {}
    for bucket_name, bucket_df in extra_test_buckets.items():
        bucket_ds = PerMoMotionDataset(
            bucket_df,
            args.path_col,
            args.action_col,
            args.condition_col,
            args.label_col,
            args.motion_key,
            expected_shape,
        )
        extra_test_loaders[bucket_name] = DataLoader(
            bucket_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
    # Model
    if args.project_root:
        sys.path.insert(0, str(Path(args.project_root).resolve()))
    global Encoder_TRANSFORMER
    from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER
    motion_encoder = build_motionclip_encoder(args.checkpoint, device)
    save_json(
        {"loaded": True, "checkpoint_path": args.checkpoint, "loader": "MotionCLIP encoder loader from finetune_unsupervised_updated.py"},
        output_dir / "checkpoint_load_info.json",
    )
    unfreeze_info = freeze_encoder_except_last_layers(
        motion_encoder,
        num_trainable_blocks=args.trainable_layers,
    )
    save_json(unfreeze_info, output_dir / "unfreeze_info.json")
    print("[INFO] Unfreeze info:", unfreeze_info)
    # Text features
    print("[INFO] Loading frozen CLIP text encoder...")
    text_encoder = FrozenCLIPTextEncoder(args.clip_model, device)
    all_actions = df[args.action_col].map(normalize_action_text).tolist()
    all_conditions = df[args.condition_col].map(normalize_action_text).tolist()
    action_to_idx, condition_to_idx, text_feats, prompt_info = build_prompt_cache(
        all_actions,
        all_conditions,
        text_encoder,
        args.normal_prompt_template,
        args.condition_prompt_template,
        args.healthy_condition,
        device,
    )
    save_json(prompt_info, output_dir / "prompts.json")
    torch.save(
        {
            "action_to_idx": action_to_idx,
            "condition_to_idx": condition_to_idx,
            "text_feats": text_feats.detach().cpu(),
            "prompt_info": prompt_info,
            "healthy_condition": args.healthy_condition,
        },
        output_dir / "text_prompt_cache.pt",
    )
    # Loss/optim
    class_weights = make_class_weights(train_df[args.label_col].tolist(), device, args.class_weight)
    if class_weights is not None:
        print("[INFO] Class weights [healthy, flawed]:", class_weights.detach().cpu().tolist())
        print("[INFO] Contrastive training uses motion-text positives grouped by action and healthy/condition label; class weights are saved but not applied to the loss.")
    optimizer = torch.optim.AdamW(
        [p for p in motion_encoder.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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
            condition_to_idx=condition_to_idx,
            text_feats=text_feats,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
        )
        train_eval_out = evaluate(
            motion_encoder=motion_encoder,
            loader=train_eval_loader,
            device=device,
            action_to_idx=action_to_idx,
            condition_to_idx=condition_to_idx,
            text_feats=text_feats,
            temperature=args.temperature,
            healthy_condition=args.healthy_condition,
            compute_contrastive_loss=False,
        )
        train_metrics_epoch = compute_binary_metrics(
            train_eval_out.y_true,
            train_eval_out.score,
            threshold=None,
            threshold_criterion=args.threshold_criterion,
        )
        val_out = evaluate(
            motion_encoder=motion_encoder,
            loader=val_loader,
            device=device,
            action_to_idx=action_to_idx,
            condition_to_idx=condition_to_idx,
            text_feats=text_feats,
            temperature=args.temperature,
            healthy_condition=args.healthy_condition,
            compute_contrastive_loss=True,
        )
        val_metrics = compute_binary_metrics(
            val_out.y_true,
            val_out.score,
            threshold=None,
            threshold_criterion=args.threshold_criterion,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_out.loss,
            "train_auroc": train_metrics_epoch["auroc"],
            "val_auroc": val_metrics["auroc"],
            "train_auprc": train_metrics_epoch["auprc"],
            "val_auprc": val_metrics["auprc"],
            "train_f1": train_metrics_epoch["f1"],
            "val_f1": val_metrics["f1"],
            "train_balanced_accuracy": train_metrics_epoch["balanced_accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "train_threshold": train_metrics_epoch["threshold"],
            "val_threshold": val_metrics["threshold"],
            "seconds": time.time() - t0,
        }
        epoch_records.append(record)
        save_training_curves(epoch_records, output_dir)
        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_out.loss:.4f} "
            f"train_auroc={train_metrics_epoch['auroc']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} thr={val_metrics['threshold']:.4f}"
        )
        current_auroc = val_metrics["auroc"]
        if np.isfinite(current_auroc) and current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_epoch = epoch
            best_threshold = float(val_metrics["threshold"])
            torch.save(
                {
                    "epoch": epoch,
                    "motion_encoder_state_dict": motion_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "action_to_idx": action_to_idx,
                    "condition_to_idx": condition_to_idx,
                    "prompt_info": prompt_info,
                    "text_feats": text_feats.detach().cpu(),
                    "best_val_metrics": val_metrics,
                    "unfreeze_info": unfreeze_info,
                },
                ckpt_dir / "best_model.pt",
            )
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
    train_out = evaluate(
        motion_encoder, train_eval_loader, device, action_to_idx, condition_to_idx, text_feats, args.temperature, args.healthy_condition, True
    )
    val_out = evaluate(
        motion_encoder, val_loader, device, action_to_idx, condition_to_idx, text_feats, args.temperature, args.healthy_condition, True
    )
    test_out = evaluate(
        motion_encoder, test_loader, device, action_to_idx, condition_to_idx, text_feats, args.temperature, args.healthy_condition, True
    )
    seen_test_out = evaluate(
        motion_encoder, seen_test_loader, device, action_to_idx, condition_to_idx, text_feats, args.temperature, args.healthy_condition, True
    )
    unseen_test_out = evaluate(
        motion_encoder, unseen_test_loader, device, action_to_idx, condition_to_idx, text_feats, args.temperature, args.healthy_condition, True
    )
    train_metrics = compute_binary_metrics(train_out.y_true, train_out.score, threshold=best_threshold)
    val_metrics_final = compute_binary_metrics(val_out.y_true, val_out.score, threshold=best_threshold)
    test_metrics = compute_binary_metrics(test_out.y_true, test_out.score, threshold=best_threshold)
    seen_test_metrics = compute_binary_metrics(seen_test_out.y_true, seen_test_out.score, threshold=best_threshold)
    unseen_union_test_metrics = compute_binary_metrics(unseen_test_out.y_true, unseen_test_out.score, threshold=best_threshold)
    train_metrics["loss"] = train_out.loss
    val_metrics_final["loss"] = val_out.loss
    test_metrics["loss"] = test_out.loss
    seen_test_metrics["loss"] = seen_test_out.loss
    unseen_union_test_metrics["loss"] = unseen_test_out.loss
    extra_test_metrics: Dict[str, Dict[str, Any]] = {}
    for bucket_name, bucket_loader in extra_test_loaders.items():
        bucket_out = evaluate(
            motion_encoder,
            bucket_loader,
            device,
            action_to_idx,
            condition_to_idx,
            text_feats,
            args.temperature,
            args.healthy_condition,
            True,
        )
        bucket_metrics = compute_binary_metrics(bucket_out.y_true, bucket_out.score, threshold=best_threshold)
        bucket_metrics["loss"] = bucket_out.loss
        extra_test_metrics[bucket_name] = bucket_metrics
        save_predictions(bucket_out, output_dir / f"test_predictions_{bucket_name}.csv", threshold=best_threshold)
        save_embeddings(
            bucket_out,
            output_dir / f"test_embeddings_{bucket_name}.npz",
            output_dir / f"test_embeddings_{bucket_name}_metadata.csv",
            threshold=best_threshold,
        )
    save_predictions(train_out, output_dir / "train_predictions.csv", threshold=best_threshold)
    save_predictions(val_out, output_dir / "val_predictions.csv", threshold=best_threshold)
    save_predictions(test_out, output_dir / "test_predictions_combined.csv", threshold=best_threshold)
    save_predictions(seen_test_out, output_dir / "test_predictions_seen.csv", threshold=best_threshold)
    save_predictions(unseen_test_out, output_dir / "test_predictions_unseen_union.csv", threshold=best_threshold)
    # Compatibility filenames from older versions.
    save_predictions(seen_test_out, output_dir / "test_predictions_seen_actions.csv", threshold=best_threshold)
    save_predictions(unseen_test_out, output_dir / "test_predictions_unseen_actions.csv", threshold=best_threshold)
    save_predictions(test_out, output_dir / "test_predictions.csv", threshold=best_threshold)
    save_embeddings(train_out, output_dir / "train_embeddings.npz", output_dir / "train_embeddings_metadata.csv", threshold=best_threshold)
    save_embeddings(val_out, output_dir / "val_embeddings.npz", output_dir / "val_embeddings_metadata.csv", threshold=best_threshold)
    save_embeddings(test_out, output_dir / "test_embeddings_combined.npz", output_dir / "test_embeddings_combined_metadata.csv", threshold=best_threshold)
    save_embeddings(seen_test_out, output_dir / "test_embeddings_seen.npz", output_dir / "test_embeddings_seen_metadata.csv", threshold=best_threshold)
    save_embeddings(unseen_test_out, output_dir / "test_embeddings_unseen_union.npz", output_dir / "test_embeddings_unseen_union_metadata.csv", threshold=best_threshold)
    # Compatibility filenames from older versions.
    save_embeddings(seen_test_out, output_dir / "test_embeddings_seen_actions.npz", output_dir / "test_embeddings_seen_actions_metadata.csv", threshold=best_threshold)
    save_embeddings(unseen_test_out, output_dir / "test_embeddings_unseen_actions.npz", output_dir / "test_embeddings_unseen_actions_metadata.csv", threshold=best_threshold)
    save_embeddings(test_out, output_dir / "test_embeddings.npz", output_dir / "test_embeddings_metadata.csv", threshold=best_threshold)
    final_summary = {
        "best_epoch": best_epoch,
        "best_val_auroc_during_training": best_val_auroc,
        "threshold_selected_on_validation": best_threshold,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics_final,
        "test_metrics": test_metrics,
        "seen_test_metrics": seen_test_metrics,
        "unseen_union_test_metrics": unseen_union_test_metrics,
        "extra_unseen_test_metrics": extra_test_metrics,
        "split_summary": split_summary,
        "prompt_templates": {
            "healthy_condition": args.healthy_condition,
            "normal": args.normal_prompt_template,
            "condition": args.condition_prompt_template,
        },
        "output_files": {
            "best_checkpoint": str(best_ckpt_path),
            "epoch_metrics": str(output_dir / "epoch_metrics.csv"),
            "test_predictions_combined": str(output_dir / "test_predictions_combined.csv"),
            "test_predictions_seen": str(output_dir / "test_predictions_seen.csv"),
            "test_predictions_unseen_union": str(output_dir / "test_predictions_unseen_union.csv"),
            "extra_test_predictions": {k: str(output_dir / f"test_predictions_{k}.csv") for k in extra_test_metrics},
            "train_embeddings": str(output_dir / "train_embeddings.npz"),
            "val_embeddings": str(output_dir / "val_embeddings.npz"),
            "test_embeddings_combined": str(output_dir / "test_embeddings_combined.npz"),
            "test_embeddings_seen": str(output_dir / "test_embeddings_seen.npz"),
            "test_embeddings_unseen_union": str(output_dir / "test_embeddings_unseen_union.npz"),
            "extra_test_embeddings": {k: str(output_dir / f"test_embeddings_{k}.npz") for k in extra_test_metrics},
            "metrics": str(output_dir / "metrics.json"),
            "training_history_npz": str(output_dir / "training_history.npz"),
            "loss_curves": str(output_dir / "loss_curves.png"),
            "validation_metrics_plot": str(output_dir / "validation_metrics.png"),
            "auroc_curves": str(output_dir / "auroc_curves.png"),
        },
    }
    save_json(final_summary, output_dir / "metrics.json")
    print("[DONE] Final combined test metrics:")
    print(json.dumps(test_metrics, indent=2, sort_keys=True))
    print("[DONE] Seen test metrics:")
    print(json.dumps(seen_test_metrics, indent=2, sort_keys=True))
    print("[DONE] Unseen-union test metrics:")
    print(json.dumps(unseen_union_test_metrics, indent=2, sort_keys=True))
    if extra_test_metrics:
        print("[DONE] Extra unseen test buckets:")
        print(json.dumps({k: v.get("auroc") for k, v in extra_test_metrics.items()}, indent=2, sort_keys=True))
    print(f"[DONE] Outputs saved to: {output_dir}")
if __name__ == "__main__":
    main()
#
##!/usr/bin/env python3
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
#    conditions: Sequence[str],
#    text_encoder: FrozenCLIPTextEncoder,
#    normal_prompt_template: str,
#    condition_prompt_template: str,
#    healthy_condition: str,
#    device: torch.device,
#) -> Tuple[Dict[str, int], Dict[str, int], torch.Tensor, Dict[str, Dict[str, str]]]:
#    """Build prompt embeddings for every condition/style per action.
#    Healthy prompts use normal_prompt_template, e.g. "healthy {action}".
#    Non-healthy prompts use condition_prompt_template, e.g. "{condition} {action}".
#    text_feats has shape [num_actions, num_conditions, dim].
#    """
#    actions = sorted({normalize_action_text(a) for a in actions})
#    healthy_condition = normalize_action_text(healthy_condition)
#    conditions_set = {normalize_action_text(c) for c in conditions}
#    conditions_set.add(healthy_condition)
#    conditions_sorted = [healthy_condition] + sorted(c for c in conditions_set if c != healthy_condition)
#    action_to_idx = {a: i for i, a in enumerate(actions)}
#    condition_to_idx = {c: i for i, c in enumerate(conditions_sorted)}
#    prompt_info: Dict[str, Dict[str, str]] = {}
#    texts = []
#    for action in actions:
#        prompt_info[action] = {}
#        for condition in conditions_sorted:
#            if condition == healthy_condition:
#                prompt = normal_prompt_template.format(action=action, condition=condition)
#            else:
#                prompt = condition_prompt_template.format(action=action, condition=condition)
#            prompt_info[action][condition] = prompt
#            texts.append(prompt)
#    text_feats = text_encoder.encode(texts)  # [A*C, D]
#    text_feats = text_feats.reshape(len(actions), len(conditions_sorted), -1).to(device)
#    return action_to_idx, condition_to_idx, text_feats, prompt_info
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
#    if len(y_true) == 0:
#        return {
#            "auroc": float("nan"), "auprc": float("nan"), "n_samples": 0,
#            "n_normal": 0, "n_anomaly": 0, "score_mean": float("nan"), "score_std": float("nan"),
#            "threshold_source": "empty_split", "threshold": float(threshold or 0.0),
#            "accuracy": float("nan"), "balanced_accuracy": float("nan"), "precision": float("nan"),
#            "recall": float("nan"), "specificity": float("nan"), "f1": float("nan"),
#            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
#        }
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
#    s_anomaly_max: np.ndarray
#    embeddings: np.ndarray
#    paths: List[str]
#    actions: List[str]
#    conditions: List[str]
#    max_anomaly_conditions: List[str]
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
#    """Return logits against all condition/style prompts for each sample's action.
#    text_feats shape: [num_actions, num_conditions, dim]
#    output logits shape: [B, num_conditions]
#    """
#    z = encode_motion_auto(motion_encoder, motion)
#    z = F.normalize(z.float(), dim=-1)
#    idx = torch.tensor([action_to_idx[normalize_action_text(a)] for a in actions], dtype=torch.long, device=motion.device)
#    prompts = text_feats[idx]  # [B,C,D]
#    if z.shape[-1] != prompts.shape[-1]:
#        raise RuntimeError(
#            f"Motion embedding dim ({z.shape[-1]}) does not match text embedding dim ({prompts.shape[-1]}). "
#            "Check --latent_dim and CLIP model."
#        )
#    logits = torch.bmm(prompts, z.unsqueeze(-1)).squeeze(-1) / temperature  # [B,C]
#    return logits, z
#def target_text_embeddings_and_group_ids_for_batch(
#    actions: Sequence[str],
#    conditions: Sequence[str],
#    action_to_idx: Dict[str, int],
#    condition_to_idx: Dict[str, int],
#    text_feats: torch.Tensor,
#    device: torch.device,
#) -> Tuple[torch.Tensor, torch.Tensor]:
#    """Return matching text embeddings and group ids for condition-per-action prompts."""
#    action_idx = torch.tensor(
#        [action_to_idx[normalize_action_text(a)] for a in actions],
#        dtype=torch.long,
#        device=device,
#    )
#    condition_idx = torch.tensor(
#        [condition_to_idx[normalize_action_text(c)] for c in conditions],
#        dtype=torch.long,
#        device=device,
#    )
#    target_text = text_feats[action_idx, condition_idx]  # [B, D]
#    group_ids = action_idx * len(condition_to_idx) + condition_idx
#    return target_text, group_ids
#def symmetric_motion_text_contrastive_loss(
#    motion_z: torch.Tensor,
#    text_z: torch.Tensor,
#    group_ids: torch.Tensor,
#    temperature: float,
#) -> torch.Tensor:
#    """Bidirectional supervised contrastive loss between motions and matching text prompts.
#    Samples with the same action and healthy/condition label are treated as positives.
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
#    condition_to_idx: Dict[str, int],
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
#            target_text, group_ids = target_text_embeddings_and_group_ids_for_batch(
#                batch["action"], batch["condition"], action_to_idx, condition_to_idx, text_feats, motion.device
#            )
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
#    condition_to_idx: Dict[str, int],
#    text_feats: torch.Tensor,
#    temperature: float,
#    healthy_condition: str,
#    compute_contrastive_loss: bool = True,
#) -> EvalOutput:
#    motion_encoder.eval()
#    losses = []
#    y_true = []
#    scores = []
#    probs = []
#    s_h = []
#    s_anom = []
#    embeddings = []
#    paths = []
#    actions = []
#    conditions = []
#    max_anomaly_conditions = []
#    row_indices = []
#    healthy_idx = condition_to_idx[normalize_action_text(healthy_condition)]
#    idx_to_condition = {v: k for k, v in condition_to_idx.items()}
#    anomaly_indices = [i for c, i in condition_to_idx.items() if c != normalize_action_text(healthy_condition)]
#    if not anomaly_indices:
#        raise ValueError("Need at least one non-healthy condition/style prompt for anomaly scoring.")
#    anomaly_indices_t = torch.tensor(anomaly_indices, dtype=torch.long, device=device)
#    for batch in loader:
#        motion = batch["motion"].to(device, non_blocking=True)
#        labels = batch["label"].to(device, non_blocking=True)
#        logits, z = logits_from_motion_and_prompts(
#            motion_encoder, motion, batch["action"], action_to_idx, text_feats, temperature
#        )  # [B, C]
#        if compute_contrastive_loss:
#            target_text, group_ids = target_text_embeddings_and_group_ids_for_batch(
#                batch["action"], batch["condition"], action_to_idx, condition_to_idx, text_feats, motion.device
#            )
#            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
#            losses.append(float(loss.detach().cpu()) * labels.numel())
#        healthy_logits = logits[:, healthy_idx]
#        anomaly_logits_all = logits.index_select(dim=1, index=anomaly_indices_t)
#        anomaly_max_logits, anomaly_argmax = anomaly_logits_all.max(dim=1)
#        chosen_anomaly_indices = anomaly_indices_t[anomaly_argmax].detach().cpu().numpy().astype(int).tolist()
#        chosen_anomaly_conditions = [idx_to_condition[i] for i in chosen_anomaly_indices]
#        # anomaly score: positive means closer to the strongest non-healthy condition prompt than healthy prompt
#        score = anomaly_max_logits - healthy_logits
#        two_class_logits = torch.stack([healthy_logits, anomaly_max_logits], dim=1)
#        soft = torch.softmax(two_class_logits, dim=-1)
#        embeddings.append(z.detach().cpu().numpy().astype(np.float32))
#        y_true.extend(labels.cpu().numpy().astype(int).tolist())
#        scores.extend(score.cpu().numpy().astype(float).tolist())
#        probs.extend(soft[:, 1].cpu().numpy().astype(float).tolist())
#        s_h.extend(healthy_logits.cpu().numpy().astype(float).tolist())
#        s_anom.extend(anomaly_max_logits.cpu().numpy().astype(float).tolist())
#        paths.extend(batch["path"])
#        actions.extend(batch["action"])
#        conditions.extend(batch["condition"])
#        max_anomaly_conditions.extend(chosen_anomaly_conditions)
#        row_indices.extend(batch["row_index"])
#    total_n = max(1, len(y_true))
#    avg_loss = sum(losses) / total_n if losses else float("nan")
#    return EvalOutput(
#        loss=avg_loss,
#        y_true=np.asarray(y_true, dtype=int),
#        score=np.asarray(scores, dtype=float),
#        prob_anomaly=np.asarray(probs, dtype=float),
#        s_healthy=np.asarray(s_h, dtype=float),
#        s_anomaly_max=np.asarray(s_anom, dtype=float),
#        embeddings=np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 512), dtype=np.float32),
#        paths=paths,
#        actions=actions,
#        conditions=conditions,
#        max_anomaly_conditions=max_anomaly_conditions,
#        row_indices=row_indices,
#    )
#def save_predictions(eval_out: EvalOutput, path: str | Path, threshold: float) -> None:
#    pred = (eval_out.score >= threshold).astype(int)
#    out_df = pd.DataFrame({
#        "row_index": eval_out.row_indices,
#        "motion_path": eval_out.paths,
#        "action": eval_out.actions,
#        "condition": eval_out.conditions,
#        "y_true_is_anomaly": eval_out.y_true,
#        "anomaly_score_max_non_healthy_minus_healthy": eval_out.score,
#        "anomaly_score_flawed_minus_healthy": eval_out.score,  # compatibility alias
#        "prob_anomaly": eval_out.prob_anomaly,
#        "logit_healthy": eval_out.s_healthy,
#        "logit_max_non_healthy": eval_out.s_anomaly_max,
#        "max_non_healthy_condition": eval_out.max_anomaly_conditions,
#        "pred_is_anomaly": pred,
#    })
#    out_df.to_csv(path, index=False)
#def save_embeddings(eval_out: EvalOutput, npz_path: str | Path, metadata_csv_path: str | Path, threshold: float) -> None:
#    """Save normalized MotionCLIP embeddings and metadata for plotting/debugging.
#    NPZ keys:
#      embeddings: [N, D] normalized motion embeddings used for prompt scoring
#      y_true: [N] 0=healthy, 1=non-healthy/anomaly
#      score: [N] max_non_healthy_minus_healthy anomaly score
#      prob_anomaly: [N] softmax probability for non-healthy/anomaly prompt
#      logit_healthy/logit_max_non_healthy: [N] prompt logits
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
#        logit_max_non_healthy=eval_out.s_anomaly_max.astype(np.float32),
#        row_index=np.asarray(eval_out.row_indices, dtype=np.int64),
#        pred_is_anomaly=pred.astype(np.int64),
#    )
#    meta_df = pd.DataFrame({
#        "embedding_index": np.arange(len(eval_out.row_indices), dtype=int),
#        "row_index": eval_out.row_indices,
#        "motion_path": eval_out.paths,
#        "action": eval_out.actions,
#        "condition": eval_out.conditions,
#        "y_true_is_anomaly": eval_out.y_true,
#        "anomaly_score_max_non_healthy_minus_healthy": eval_out.score,
#        "anomaly_score_flawed_minus_healthy": eval_out.score,  # compatibility alias
#        "prob_anomaly": eval_out.prob_anomaly,
#        "logit_healthy": eval_out.s_healthy,
#        "logit_max_non_healthy": eval_out.s_anomaly_max,
#        "max_non_healthy_condition": eval_out.max_anomaly_conditions,
#        "pred_is_anomaly": pred,
#    })
#    meta_df.to_csv(metadata_csv_path, index=False)
#def main() -> None:
#    parser = argparse.ArgumentParser(description="Fine-tune MotionCLIP with healthy/condition action prompts on PerMo.")
#    # Data
#    parser.add_argument("--csv_path", required=True, help="Path to PerMo metadata CSV.")
#    parser.add_argument("--output_dir", required=True, help="Directory where outputs are saved.")
#    parser.add_argument("--path_col", default="motion_path")
#    parser.add_argument("--action_col", default="action_label")
#    parser.add_argument("--condition_col", default="condition_label")
#    parser.add_argument("--label_col", default="is_anomaly")
#    parser.add_argument("--motion_key", default="auto", help="NPZ key. Use 'auto' to infer.")
#    parser.add_argument("--num_frames", type=int, default=60)
#    parser.add_argument("--njoints", type=int, default=25)
#    parser.add_argument("--nfeats", type=int, default=6)
#    # Split / balanced PerMo Condition experiments
#    parser.add_argument("--test_fraction", type=float, default=0.20, help="Test fraction for seen samples; default gives 80/20 train/test before validation is carved out of train.")
#    parser.add_argument("--val_fraction", type=float, default=0.10, help="Validation fraction taken from the non-test seen pool.")
#    parser.add_argument("--seed", type=int, default=42, help="Backward-compatible fallback seed. Used when --split_seed or --train_seed are not provided.")
#    parser.add_argument("--split_seed", type=int, default=None, help="Controls dataset subset selection, train/val/test splitting, and evaluation bucket sampling. Defaults to --seed.")
#    parser.add_argument("--train_seed", type=int, default=None, help="Controls training randomness, PyTorch/NumPy/Python RNGs, and balanced train batch sampling. Defaults to --seed.")
#    parser.add_argument("--actor_col", default="actor_label", help="CSV column containing actor identity. Required for actor-balanced sampling.")
#    parser.add_argument("--normal_target", type=int, default=200, help="Number of healthy/normal samples to keep.")
#    parser.add_argument("--anomaly_target", type=int, default=200, help="Number of non-healthy/anomaly samples to keep.")
#    parser.add_argument("--unseen_actions", nargs="*", default=[], help="Action names to keep completely out of train/val; comma-separated values are also accepted.")
#    parser.add_argument("--unseen_actors", nargs="*", default=[], help="Actor IDs/names to keep completely out of train/val; comma-separated values are also accepted.")
#    parser.add_argument("--unseen_styles", nargs="*", default=[], help="Non-healthy condition styles to keep completely out of train/val.")
#    parser.add_argument("--no_seen_healthy_in_unseen_style_test", action="store_true", help="For style-only holdout, do not add held-out healthy seen-style samples to the unseen-style test set.")
#    # MotionCLIP model
#    parser.add_argument("--project_root", default="", help="Parent directory containing the MotionCLIP folder.")
#    parser.add_argument("--checkpoint", required=True, help="Pretrained MotionCLIP checkpoint.")
#    parser.add_argument("--trainable_layers", type=int, default=2)
#    # Text prompts
#    parser.add_argument("--clip_model", default="ViT-B/32")
#    parser.add_argument("--healthy_condition", default="healthy", help="Condition/style name treated as normal.")
#    parser.add_argument("--normal_prompt_template", default="healthy {action}")
#    parser.add_argument("--condition_prompt_template", default="{condition} {action}", help="Prompt template for non-healthy condition/style prompts.")
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
#    if args.split_seed is None:
#        args.split_seed = args.seed
#    if args.train_seed is None:
#        args.train_seed = args.seed
#    # Training randomness is kept separate from split/sampling randomness.
#    set_seed(args.train_seed)
#    output_dir = ensure_dir(args.output_dir)
#    ckpt_dir = ensure_dir(output_dir / "checkpoints")
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    save_json(vars(args), output_dir / "args.json")
#    # Load metadata
#    df = pd.read_csv(args.csv_path)
#    df = df.copy()
#    df["original_index"] = np.arange(len(df))
#    required_cols = [args.path_col, args.action_col, args.condition_col, args.actor_col, args.label_col]
#    missing = [c for c in required_cols if c not in df.columns]
#    if missing:
#        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")
#    # Normalize labels/actions
#    df[args.label_col] = df[args.label_col].astype(int)
#    if not set(df[args.label_col].unique()).issubset({0, 1}):
#        raise ValueError(f"{args.label_col} must contain only 0/1 labels.")
#    df["_action_norm"] = df[args.action_col].map(normalize_action_text)
#    df["_condition_norm"] = df[args.condition_col].map(normalize_action_text)
#    args.healthy_condition = normalize_action_text(args.healthy_condition)
#    # Basic file existence check
#    missing_paths = [p for p in df[args.path_col].head(20).tolist() if not Path(str(p)).exists()]
#    if missing_paths:
#        print("[WARN] Some example motion files do not exist from this machine.")
#        print("       This is okay only if you are testing the script outside the data machine.")
#        print(f"       First missing example: {missing_paths[0]}")
#    # Build the balanced 200 healthy + 200 anomaly subset first.
#    balanced_df, balanced_subset_info = build_balanced_condition_subset(
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
#    train_df, val_df, seen_test_df, unseen_test_df, test_df, split_mode_info = split_balanced_condition_experiment(
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
#    # Leakage checks for all unseen dimensions.
#    train_actions = set(train_df["_action_norm"].unique().tolist())
#    val_actions = set(val_df["_action_norm"].unique().tolist())
#    train_actors = set(train_df[args.actor_col].astype(str).map(normalize_action_text).unique().tolist())
#    val_actors = set(val_df[args.actor_col].astype(str).map(normalize_action_text).unique().tolist())
#    train_styles = set(train_df["_condition_norm"].unique().tolist())
#    val_styles = set(val_df["_condition_norm"].unique().tolist())
#    requested_unseen_actions = set(parse_list_arg(args.unseen_actions))
#    requested_unseen_actors = set(parse_list_arg(args.unseen_actors))
#    requested_unseen_styles = set(parse_list_arg(args.unseen_styles))
#    action_leakage = sorted((train_actions | val_actions) & requested_unseen_actions)
#    actor_leakage = sorted((train_actors | val_actors) & requested_unseen_actors)
#    style_leakage = sorted((train_styles | val_styles) & requested_unseen_styles)
#    if action_leakage or actor_leakage or style_leakage:
#        raise RuntimeError(
#            f"Unseen leakage detected: actions={action_leakage}, actors={actor_leakage}, styles={style_leakage}"
#        )
#    extra_test_buckets, extra_test_bucket_info = build_unseen_test_buckets(
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
#    test_balance_info = {
#        "enabled": True,
#        "note": "The whole experiment is balanced before splitting: 200 healthy and 200 anomaly samples. The anomaly subset is marginally balanced over style, action, and actor. Evaluation buckets are downsampled to equal normal/anomaly counts when both labels exist.",
#        "balanced_subset_info": balanced_subset_info,
#        "extra_test_bucket_info": extra_test_bucket_info,
#    }
#    train_df.to_csv(output_dir / "split_train.csv", index=False)
#    val_df.to_csv(output_dir / "split_val.csv", index=False)
#    seen_test_df.to_csv(output_dir / "split_test_seen.csv", index=False)
#    unseen_test_df.to_csv(output_dir / "split_test_unseen_union.csv", index=False)
#    test_df.to_csv(output_dir / "split_test_combined.csv", index=False)
#    # Compatibility filenames from older versions.
#    seen_test_df.to_csv(output_dir / "split_test_seen_actions.csv", index=False)
#    unseen_test_df.to_csv(output_dir / "split_test_unseen_actions.csv", index=False)
#    test_df.to_csv(output_dir / "split_test.csv", index=False)
#    for bucket_name, bucket_df in extra_test_buckets.items():
#        bucket_df.to_csv(output_dir / f"split_test_{bucket_name}.csv", index=False)
#    split_summary = {
#        "split_type": split_mode_info["split_type"],
#        "total_original_rows": int(len(df)),
#        "total_balanced_subset_rows": int(len(balanced_df)),
#        "train": int(len(train_df)),
#        "val": int(len(val_df)),
#        "seen_test": int(len(seen_test_df)),
#        "unseen_union_test": int(len(unseen_test_df)),
#        "combined_test": int(len(test_df)),
#        "extra_test_buckets": {k: int(len(v)) for k, v in extra_test_buckets.items()},
#        "extra_test_bucket_info": extra_test_bucket_info,
#        "train_label_counts": train_df[args.label_col].value_counts().to_dict(),
#        "val_label_counts": val_df[args.label_col].value_counts().to_dict(),
#        "seen_test_label_counts": seen_test_df[args.label_col].value_counts().to_dict(),
#        "unseen_union_test_label_counts": unseen_test_df[args.label_col].value_counts().to_dict(),
#        "combined_test_label_counts": test_df[args.label_col].value_counts().to_dict(),
#        "train_condition_counts": train_df["_condition_norm"].value_counts().sort_index().to_dict(),
#        "test_condition_counts": test_df["_condition_norm"].value_counts().sort_index().to_dict(),
#        "test_balance_info": test_balance_info,
#        "all_actions": sorted(balanced_df["_action_norm"].unique().tolist()),
#        "all_conditions": sorted(balanced_df["_condition_norm"].unique().tolist()),
#        "all_actors": sorted(balanced_df[args.actor_col].astype(str).map(normalize_action_text).unique().tolist()),
#        "healthy_condition": args.healthy_condition,
#        "seen_train_val_actions": sorted((train_actions | val_actions)),
#        "seen_train_val_actors": sorted((train_actors | val_actors)),
#        "seen_train_val_styles": sorted((train_styles | val_styles)),
#        "heldout_unseen_actions": split_mode_info["unseen_actions"],
#        "heldout_unseen_actors": split_mode_info["unseen_actors"],
#        "heldout_unseen_styles": split_mode_info["unseen_styles"],
#        "unseen_leakage_check_passed": True,
#    }
#    save_json(split_summary, output_dir / "split_summary.json")
#    print("[INFO] Split summary:", split_summary)
#    expected_shape = (args.num_frames, args.njoints, args.nfeats)
#    train_ds = PerMoMotionDataset(train_df, args.path_col, args.action_col, args.condition_col, args.label_col, args.motion_key, expected_shape)
#    val_ds = PerMoMotionDataset(val_df, args.path_col, args.action_col, args.condition_col, args.label_col, args.motion_key, expected_shape)
#    test_ds = PerMoMotionDataset(test_df, args.path_col, args.action_col, args.condition_col, args.label_col, args.motion_key, expected_shape)
#    seen_test_ds = PerMoMotionDataset(seen_test_df, args.path_col, args.action_col, args.condition_col, args.label_col, args.motion_key, expected_shape)
#    unseen_test_ds = PerMoMotionDataset(unseen_test_df, args.path_col, args.action_col, args.condition_col, args.label_col, args.motion_key, expected_shape)
#    train_labels = train_df[args.label_col].astype(int).to_numpy()
#    train_batch_sampler = BalancedBinaryBatchSampler(
#        labels=train_labels,
#        batch_size=args.batch_size,
#        seed=args.train_seed,
#        drop_last=False,
#    )
#    sampler_info = {
#        "type": "BalancedBinaryBatchSampler",
#        "purpose": "class-aware healthy/condition training batches",
#        "batch_size": int(args.batch_size),
#        "healthy_per_batch": int(train_batch_sampler.n0),
#        "flawed_per_batch": int(train_batch_sampler.n1),
#        "num_batches_per_epoch": int(len(train_batch_sampler)),
#        "train_label_counts": {
#            "healthy_0": int((train_labels == 0).sum()),
#            "flawed_1": int((train_labels == 1).sum()),
#        },
#    }
#    save_json(sampler_info, output_dir / "train_sampler_info.json")
#    print("[INFO] Train sampler info:", sampler_info)
#    train_loader = DataLoader(
#        train_ds,
#        batch_sampler=train_batch_sampler,
#        num_workers=args.num_workers,
#        pin_memory=torch.cuda.is_available(),
#        collate_fn=collate_batch,
#    )
#    train_eval_loader = DataLoader(
#        train_ds,
#        batch_size=args.batch_size,
#        shuffle=False,
#        num_workers=args.num_workers,
#        pin_memory=torch.cuda.is_available(),
#        collate_fn=collate_batch,
#    )
#    val_loader = DataLoader(
#        val_ds,
#        batch_size=args.batch_size,
#        shuffle=False,
#        num_workers=args.num_workers,
#        pin_memory=torch.cuda.is_available(),
#        collate_fn=collate_batch,
#    )
#    test_loader = DataLoader(
#        test_ds,
#        batch_size=args.batch_size,
#        shuffle=False,
#        num_workers=args.num_workers,
#        pin_memory=torch.cuda.is_available(),
#        collate_fn=collate_batch,
#    )
#    seen_test_loader = DataLoader(
#        seen_test_ds,
#        batch_size=args.batch_size,
#        shuffle=False,
#        num_workers=args.num_workers,
#        pin_memory=torch.cuda.is_available(),
#        collate_fn=collate_batch,
#    )
#    unseen_test_loader = DataLoader(
#        unseen_test_ds,
#        batch_size=args.batch_size,
#        shuffle=False,
#        num_workers=args.num_workers,
#        pin_memory=torch.cuda.is_available(),
#        collate_fn=collate_batch,
#    )
#    extra_test_loaders: Dict[str, DataLoader] = {}
#    for bucket_name, bucket_df in extra_test_buckets.items():
#        bucket_ds = PerMoMotionDataset(
#            bucket_df,
#            args.path_col,
#            args.action_col,
#            args.condition_col,
#            args.label_col,
#            args.motion_key,
#            expected_shape,
#        )
#        extra_test_loaders[bucket_name] = DataLoader(
#            bucket_ds,
#            batch_size=args.batch_size,
#            shuffle=False,
#            num_workers=args.num_workers,
#            pin_memory=torch.cuda.is_available(),
#            collate_fn=collate_batch,
#        )
#    # Model
#    if args.project_root:
#        sys.path.insert(0, str(Path(args.project_root).resolve()))
#    global Encoder_TRANSFORMER
#    from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER
#    motion_encoder = build_motionclip_encoder(args.checkpoint, device)
#    save_json(
#        {"loaded": True, "checkpoint_path": args.checkpoint, "loader": "MotionCLIP encoder loader from finetune_unsupervised_updated.py"},
#        output_dir / "checkpoint_load_info.json",
#    )
#    unfreeze_info = freeze_encoder_except_last_layers(
#        motion_encoder,
#        num_trainable_blocks=args.trainable_layers,
#    )
#    save_json(unfreeze_info, output_dir / "unfreeze_info.json")
#    print("[INFO] Unfreeze info:", unfreeze_info)
#    # Text features
#    print("[INFO] Loading frozen CLIP text encoder...")
#    text_encoder = FrozenCLIPTextEncoder(args.clip_model, device)
#    all_actions = df[args.action_col].map(normalize_action_text).tolist()
#    all_conditions = df[args.condition_col].map(normalize_action_text).tolist()
#    action_to_idx, condition_to_idx, text_feats, prompt_info = build_prompt_cache(
#        all_actions,
#        all_conditions,
#        text_encoder,
#        args.normal_prompt_template,
#        args.condition_prompt_template,
#        args.healthy_condition,
#        device,
#    )
#    save_json(prompt_info, output_dir / "prompts.json")
#    torch.save(
#        {
#            "action_to_idx": action_to_idx,
#            "condition_to_idx": condition_to_idx,
#            "text_feats": text_feats.detach().cpu(),
#            "prompt_info": prompt_info,
#            "healthy_condition": args.healthy_condition,
#        },
#        output_dir / "text_prompt_cache.pt",
#    )
#    # Loss/optim
#    class_weights = make_class_weights(train_df[args.label_col].tolist(), device, args.class_weight)
#    if class_weights is not None:
#        print("[INFO] Class weights [healthy, flawed]:", class_weights.detach().cpu().tolist())
#        print("[INFO] Contrastive training uses motion-text positives grouped by action and healthy/condition label; class weights are saved but not applied to the loss.")
#    optimizer = torch.optim.AdamW(
#        [p for p in motion_encoder.parameters() if p.requires_grad],
#        lr=args.lr,
#        weight_decay=args.weight_decay,
#    )
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
#            condition_to_idx=condition_to_idx,
#            text_feats=text_feats,
#            temperature=args.temperature,
#            grad_clip=args.grad_clip,
#            use_amp=args.amp,
#        )
#        train_eval_out = evaluate(
#            motion_encoder=motion_encoder,
#            loader=train_eval_loader,
#            device=device,
#            action_to_idx=action_to_idx,
#            condition_to_idx=condition_to_idx,
#            text_feats=text_feats,
#            temperature=args.temperature,
#            healthy_condition=args.healthy_condition,
#            compute_contrastive_loss=False,
#        )
#        train_metrics_epoch = compute_binary_metrics(
#            train_eval_out.y_true,
#            train_eval_out.score,
#            threshold=None,
#            threshold_criterion=args.threshold_criterion,
#        )
#        val_out = evaluate(
#            motion_encoder=motion_encoder,
#            loader=val_loader,
#            device=device,
#            action_to_idx=action_to_idx,
#            condition_to_idx=condition_to_idx,
#            text_feats=text_feats,
#            temperature=args.temperature,
#            healthy_condition=args.healthy_condition,
#            compute_contrastive_loss=True,
#        )
#        val_metrics = compute_binary_metrics(
#            val_out.y_true,
#            val_out.score,
#            threshold=None,
#            threshold_criterion=args.threshold_criterion,
#        )
#        record = {
#            "epoch": epoch,
#            "train_loss": train_loss,
#            "val_loss": val_out.loss,
#            "train_auroc": train_metrics_epoch["auroc"],
#            "val_auroc": val_metrics["auroc"],
#            "train_auprc": train_metrics_epoch["auprc"],
#            "val_auprc": val_metrics["auprc"],
#            "train_f1": train_metrics_epoch["f1"],
#            "val_f1": val_metrics["f1"],
#            "train_balanced_accuracy": train_metrics_epoch["balanced_accuracy"],
#            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
#            "train_threshold": train_metrics_epoch["threshold"],
#            "val_threshold": val_metrics["threshold"],
#            "seconds": time.time() - t0,
#        }
#        epoch_records.append(record)
#        save_training_curves(epoch_records, output_dir)
#        print(
#            f"[EPOCH {epoch:03d}] "
#            f"train_loss={train_loss:.4f} val_loss={val_out.loss:.4f} "
#            f"train_auroc={train_metrics_epoch['auroc']:.4f} "
#            f"val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f} "
#            f"val_f1={val_metrics['f1']:.4f} thr={val_metrics['threshold']:.4f}"
#        )
#        current_auroc = val_metrics["auroc"]
#        if np.isfinite(current_auroc) and current_auroc > best_val_auroc:
#            best_val_auroc = current_auroc
#            best_epoch = epoch
#            best_threshold = float(val_metrics["threshold"])
#            torch.save(
#                {
#                    "epoch": epoch,
#                    "motion_encoder_state_dict": motion_encoder.state_dict(),
#                    "optimizer_state_dict": optimizer.state_dict(),
#                    "args": vars(args),
#                    "action_to_idx": action_to_idx,
#                    "condition_to_idx": condition_to_idx,
#                    "prompt_info": prompt_info,
#                    "text_feats": text_feats.detach().cpu(),
#                    "best_val_metrics": val_metrics,
#                    "unfreeze_info": unfreeze_info,
#                },
#                ckpt_dir / "best_model.pt",
#            )
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
#    train_out = evaluate(
#        motion_encoder, train_eval_loader, device, action_to_idx, condition_to_idx, text_feats, args.temperature, args.healthy_condition, True
#    )
#    val_out = evaluate(
#        motion_encoder, val_loader, device, action_to_idx, condition_to_idx, text_feats, args.temperature, args.healthy_condition, True
#    )
#    test_out = evaluate(
#        motion_encoder, test_loader, device, action_to_idx, condition_to_idx, text_feats, args.temperature, args.healthy_condition, True
#    )
#    seen_test_out = evaluate(
#        motion_encoder, seen_test_loader, device, action_to_idx, condition_to_idx, text_feats, args.temperature, args.healthy_condition, True
#    )
#    unseen_test_out = evaluate(
#        motion_encoder, unseen_test_loader, device, action_to_idx, condition_to_idx, text_feats, args.temperature, args.healthy_condition, True
#    )
#    train_metrics = compute_binary_metrics(train_out.y_true, train_out.score, threshold=best_threshold)
#    val_metrics_final = compute_binary_metrics(val_out.y_true, val_out.score, threshold=best_threshold)
#    test_metrics = compute_binary_metrics(test_out.y_true, test_out.score, threshold=best_threshold)
#    seen_test_metrics = compute_binary_metrics(seen_test_out.y_true, seen_test_out.score, threshold=best_threshold)
#    unseen_union_test_metrics = compute_binary_metrics(unseen_test_out.y_true, unseen_test_out.score, threshold=best_threshold)
#    train_metrics["loss"] = train_out.loss
#    val_metrics_final["loss"] = val_out.loss
#    test_metrics["loss"] = test_out.loss
#    seen_test_metrics["loss"] = seen_test_out.loss
#    unseen_union_test_metrics["loss"] = unseen_test_out.loss
#    extra_test_metrics: Dict[str, Dict[str, Any]] = {}
#    for bucket_name, bucket_loader in extra_test_loaders.items():
#        bucket_out = evaluate(
#            motion_encoder,
#            bucket_loader,
#            device,
#            action_to_idx,
#            condition_to_idx,
#            text_feats,
#            args.temperature,
#            args.healthy_condition,
#            True,
#        )
#        bucket_metrics = compute_binary_metrics(bucket_out.y_true, bucket_out.score, threshold=best_threshold)
#        bucket_metrics["loss"] = bucket_out.loss
#        extra_test_metrics[bucket_name] = bucket_metrics
#        save_predictions(bucket_out, output_dir / f"test_predictions_{bucket_name}.csv", threshold=best_threshold)
#        save_embeddings(
#            bucket_out,
#            output_dir / f"test_embeddings_{bucket_name}.npz",
#            output_dir / f"test_embeddings_{bucket_name}_metadata.csv",
#            threshold=best_threshold,
#        )
#    save_predictions(train_out, output_dir / "train_predictions.csv", threshold=best_threshold)
#    save_predictions(val_out, output_dir / "val_predictions.csv", threshold=best_threshold)
#    save_predictions(test_out, output_dir / "test_predictions_combined.csv", threshold=best_threshold)
#    save_predictions(seen_test_out, output_dir / "test_predictions_seen.csv", threshold=best_threshold)
#    save_predictions(unseen_test_out, output_dir / "test_predictions_unseen_union.csv", threshold=best_threshold)
#    # Compatibility filenames from older versions.
#    save_predictions(seen_test_out, output_dir / "test_predictions_seen_actions.csv", threshold=best_threshold)
#    save_predictions(unseen_test_out, output_dir / "test_predictions_unseen_actions.csv", threshold=best_threshold)
#    save_predictions(test_out, output_dir / "test_predictions.csv", threshold=best_threshold)
#    save_embeddings(train_out, output_dir / "train_embeddings.npz", output_dir / "train_embeddings_metadata.csv", threshold=best_threshold)
#    save_embeddings(val_out, output_dir / "val_embeddings.npz", output_dir / "val_embeddings_metadata.csv", threshold=best_threshold)
#    save_embeddings(test_out, output_dir / "test_embeddings_combined.npz", output_dir / "test_embeddings_combined_metadata.csv", threshold=best_threshold)
#    save_embeddings(seen_test_out, output_dir / "test_embeddings_seen.npz", output_dir / "test_embeddings_seen_metadata.csv", threshold=best_threshold)
#    save_embeddings(unseen_test_out, output_dir / "test_embeddings_unseen_union.npz", output_dir / "test_embeddings_unseen_union_metadata.csv", threshold=best_threshold)
#    # Compatibility filenames from older versions.
#    save_embeddings(seen_test_out, output_dir / "test_embeddings_seen_actions.npz", output_dir / "test_embeddings_seen_actions_metadata.csv", threshold=best_threshold)
#    save_embeddings(unseen_test_out, output_dir / "test_embeddings_unseen_actions.npz", output_dir / "test_embeddings_unseen_actions_metadata.csv", threshold=best_threshold)
#    save_embeddings(test_out, output_dir / "test_embeddings.npz", output_dir / "test_embeddings_metadata.csv", threshold=best_threshold)
#    final_summary = {
#        "best_epoch": best_epoch,
#        "best_val_auroc_during_training": best_val_auroc,
#        "threshold_selected_on_validation": best_threshold,
#        "train_metrics": train_metrics,
#        "val_metrics": val_metrics_final,
#        "test_metrics": test_metrics,
#        "seen_test_metrics": seen_test_metrics,
#        "unseen_union_test_metrics": unseen_union_test_metrics,
#        "extra_unseen_test_metrics": extra_test_metrics,
#        "split_summary": split_summary,
#        "prompt_templates": {
#            "healthy_condition": args.healthy_condition,
#            "normal": args.normal_prompt_template,
#            "condition": args.condition_prompt_template,
#        },
#        "output_files": {
#            "best_checkpoint": str(best_ckpt_path),
#            "epoch_metrics": str(output_dir / "epoch_metrics.csv"),
#            "test_predictions_combined": str(output_dir / "test_predictions_combined.csv"),
#            "test_predictions_seen": str(output_dir / "test_predictions_seen.csv"),
#            "test_predictions_unseen_union": str(output_dir / "test_predictions_unseen_union.csv"),
#            "extra_test_predictions": {k: str(output_dir / f"test_predictions_{k}.csv") for k in extra_test_metrics},
#            "train_embeddings": str(output_dir / "train_embeddings.npz"),
#            "val_embeddings": str(output_dir / "val_embeddings.npz"),
#            "test_embeddings_combined": str(output_dir / "test_embeddings_combined.npz"),
#            "test_embeddings_seen": str(output_dir / "test_embeddings_seen.npz"),
#            "test_embeddings_unseen_union": str(output_dir / "test_embeddings_unseen_union.npz"),
#            "extra_test_embeddings": {k: str(output_dir / f"test_embeddings_{k}.npz") for k in extra_test_metrics},
#            "metrics": str(output_dir / "metrics.json"),
#            "training_history_npz": str(output_dir / "training_history.npz"),
#            "loss_curves": str(output_dir / "loss_curves.png"),
#            "validation_metrics_plot": str(output_dir / "validation_metrics.png"),
#            "auroc_curves": str(output_dir / "auroc_curves.png"),
#        },
#    }
#    save_json(final_summary, output_dir / "metrics.json")
#    print("[DONE] Final combined test metrics:")
#    print(json.dumps(test_metrics, indent=2, sort_keys=True))
#    print("[DONE] Seen test metrics:")
#    print(json.dumps(seen_test_metrics, indent=2, sort_keys=True))
#    print("[DONE] Unseen-union test metrics:")
#    print(json.dumps(unseen_union_test_metrics, indent=2, sort_keys=True))
#    if extra_test_metrics:
#        print("[DONE] Extra unseen test buckets:")
#        print(json.dumps({k: v.get("auroc") for k, v in extra_test_metrics.items()}, indent=2, sort_keys=True))
#    print(f"[DONE] Outputs saved to: {output_dir}")
#if __name__ == "__main__":
#    main()
