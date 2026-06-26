#!/usr/bin/env python3
"""
Fine-tune MotionCLIP for PerMo with one chosen normal style/condition and one chosen anomaly style/condition.
Main changes vs. healthy-vs-all scripts:
  1. You explicitly choose the metadata class column, normal class, and anomaly class.
     Example: --class_col condition_label --normal_class Healthy --anomaly_class Head-aching
  2. Only rows from those two classes are used. Labels are assigned internally:
       normal_class  -> 0
       anomaly_class -> 1
  3. Default split is 80/20 train/test, stratified by action + label where possible.
  4. Optional full hold-out of actions and/or actors:
       --unseen_actions Walk Run
       --unseen_actors actor_01 actor_02
     Any matching rows are removed from train/val and evaluated as separate unseen test sets.
  5. Saves the same important outputs as the condition-prompt unseen script:
       split CSVs, metrics.json, predictions, embeddings, metadata CSVs,
       training curves, prompt cache, best checkpoint, and seen/unseen test metrics.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
# -----------------------------
# Utilities
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
    def _default(o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_default)
def norm_text(text: Any) -> str:
    text = str(text).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())
def parse_list(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for v in values:
        if v is None:
            continue
        # Allow both: --unseen_actions Walk Run and --unseen_actions "Walk,Run"
        for part in str(v).split(","):
            part = norm_text(part)
            if part:
                out.append(part)
    return sorted(set(out))
def read_list_file(path: str) -> List[str]:
    if not path:
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            out.append(norm_text(line.split(",")[0]))
    return sorted(set(out))
def save_training_curves(epoch_records: List[Dict[str, Any]], output_dir: str | Path) -> None:
    if not epoch_records:
        return
    output_dir = Path(output_dir)
    hist_df = pd.DataFrame(epoch_records)
    hist_df.to_csv(output_dir / "epoch_metrics.csv", index=False)
    np.savez(
        output_dir / "training_history.npz",
        **{c: hist_df[c].to_numpy() for c in hist_df.columns if pd.api.types.is_numeric_dtype(hist_df[c])},
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib, skipping plots: {exc}")
        return
    def _plot(cols: Sequence[str], filename: str, ylabel: str) -> None:
        cols = [c for c in cols if c in hist_df.columns]
        if not cols:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in cols:
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
# -----------------------------
# Dataset
# -----------------------------
class PerMoMotionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        action_col: str,
        class_col: str,
        label_col: str,
        motion_key: str = "auto",
        expected_shape: Tuple[int, int, int] = (60, 25, 6),
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.path_col = path_col
        self.action_col = action_col
        self.class_col = class_col
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
        preferred = ["motion", "motions", "x", "X", "data", "arr_0", "poses", "pose", "rot6d", "features", "joints", "input"]
        for key in preferred:
            if key in data.files and np.issubdtype(data[key].dtype, np.number):
                return data[key]
        numeric = [k for k in data.files if np.issubdtype(data[k].dtype, np.number)]
        if not numeric:
            raise KeyError(f"No numeric arrays found in {path}. Keys: {data.files}")
        return data[numeric[0]]
    def _standardize_motion_shape(self, arr: np.ndarray, path: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        while arr.ndim > 3 and 1 in arr.shape:
            arr = np.squeeze(arr, axis=arr.shape.index(1))
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D motion array [T,J,F], got shape {arr.shape} in {path}")
        T, J, Fdim = self.expected_shape
        if arr.shape == (T, J, Fdim):
            return arr
        candidates = {
            (J, Fdim, T): (2, 0, 1),
            (Fdim, T, J): (1, 2, 0),
            (T, Fdim, J): (0, 2, 1),
            (J, T, Fdim): (1, 0, 2),
        }
        if arr.shape in candidates:
            return np.transpose(arr, candidates[arr.shape])
        shape = list(arr.shape)
        try:
            return np.transpose(arr, (shape.index(T), shape.index(J), shape.index(Fdim)))
        except ValueError as e:
            raise ValueError(
                f"Cannot convert motion shape {arr.shape} to expected {(T, J, Fdim)} for {path}. "
                "Pass --num_frames/--njoints/--nfeats or adapt _standardize_motion_shape()."
            ) from e
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        with np.load(path, allow_pickle=False) as data:
            arr = self._pick_npz_array(data, path)
        arr = self._standardize_motion_shape(arr, path)
        return {
            "motion": torch.from_numpy(arr),
            "action": norm_text(row[self.action_col]),
            "class_name": norm_text(row[self.class_col]),
            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),
            "path": path,
            "row_index": int(row.get("original_index", idx)),
        }
def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "motion": torch.stack([b["motion"] for b in batch], dim=0),
        "action": [b["action"] for b in batch],
        "class_name": [b["class_name"] for b in batch],
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
        "row_index": [b["row_index"] for b in batch],
    }
class BalancedBinaryBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: Sequence[int], batch_size: int, seed: int = 42, drop_last: bool = False) -> None:
        if batch_size < 2:
            raise ValueError("BalancedBinaryBatchSampler requires batch_size >= 2.")
        labels_np = np.asarray(labels).astype(int)
        unique = set(labels_np.tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"Expected binary labels 0/1, got {sorted(unique)}")
        self.indices_by_class = {0: np.where(labels_np == 0)[0], 1: np.where(labels_np == 1)[0]}
        if len(self.indices_by_class[0]) == 0 or len(self.indices_by_class[1]) == 0:
            raise ValueError("Training split needs at least one normal and one anomaly sample.")
        self.batch_size = int(batch_size)
        self.n0 = self.batch_size // 2
        self.n1 = self.batch_size - self.n0
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.num_batches = int(max(
            math.ceil(len(self.indices_by_class[0]) / self.n0),
            math.ceil(len(self.indices_by_class[1]) / self.n1),
        ))
    def __len__(self) -> int:
        return self.num_batches
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
    def _sample(self, cls: int, n_total: int, rng: np.random.Generator) -> np.ndarray:
        pool = self.indices_by_class[cls]
        if n_total <= len(pool):
            return rng.permutation(pool)[:n_total]
        return np.concatenate([rng.permutation(pool), rng.choice(pool, size=n_total - len(pool), replace=True)])
    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        labels0 = self._sample(0, self.num_batches * self.n0, rng)
        labels1 = self._sample(1, self.num_batches * self.n1, rng)
        for batch_idx in range(self.num_batches):
            b0 = labels0[batch_idx * self.n0:(batch_idx + 1) * self.n0]
            b1 = labels1[batch_idx * self.n1:(batch_idx + 1) * self.n1]
            batch = np.concatenate([b0, b1])
            rng.shuffle(batch)
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch.astype(int).tolist()
# -----------------------------
# Splitting
# -----------------------------
def split_random_stratified_80_20(
    df: pd.DataFrame,
    stratify_cols: Sequence[str],
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns train/val/test. Test is a disjoint 20% row holdout.
    val_fraction is taken from the remaining train pool. Set --val_fraction 0 to skip validation split;
    the script will then use the test threshold only if needed, but checkpoint selection is weaker.
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
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
        n_val = int(round(len(rem_idx) * val_fraction))
        if len(rem_idx) >= 5 and val_fraction > 0:
            n_val = max(1, n_val)
        n_val = min(n_val, max(0, len(rem_idx) - 1))
        val_idx = rem_idx[:n_val]
        train_idx = rem_idx[n_val:]
        train_parts.append(df.loc[train_idx])
        val_parts.append(df.loc[val_idx])
        test_parts.append(df.loc[test_idx])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True) if val_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    if len(val_df) == 0:
        # Use a small deterministic slice from train as validation, because the training loop needs validation for checkpoint/threshold.
        val_df = train_df.groupby("label", group_keys=False).sample(frac=0.10, random_state=seed + 3).reset_index(drop=True)
        train_df = train_df.drop(index=val_df.index, errors="ignore").reset_index(drop=True)
    return train_df, val_df, test_df
def balance_binary_split(df: pd.DataFrame, label_col: str, seed: int, split_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    counts_before = df[label_col].astype(int).value_counts().to_dict()
    n0 = int((df[label_col].astype(int) == 0).sum())
    n1 = int((df[label_col].astype(int) == 1).sum())
    if n0 == 0 or n1 == 0:
        return df.reset_index(drop=True), {
            "split_name": split_name,
            "balanced": False,
            "reason": "split does not contain both labels",
            "counts_before": {int(k): int(v) for k, v in counts_before.items()},
        }
    n = min(n0, n1)
    d0 = df[df[label_col].astype(int) == 0].sample(n=n, random_state=seed)
    d1 = df[df[label_col].astype(int) == 1].sample(n=n, random_state=seed + 1)
    out = pd.concat([d0, d1]).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return out, {
        "split_name": split_name,
        "balanced": True,
        "seed": int(seed),
        "n_before": int(len(df)),
        "n_after": int(len(out)),
        "counts_before": {int(k): int(v) for k, v in counts_before.items()},
        "counts_after": {int(k): int(v) for k, v in out[label_col].astype(int).value_counts().to_dict().items()},
        "n_kept_per_label": int(n),
    }
def make_splits(args: argparse.Namespace, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    unseen_actions = sorted(set(parse_list(args.unseen_actions) + read_list_file(args.unseen_actions_file)))
    unseen_actors = sorted(set(parse_list(args.unseen_actors) + read_list_file(args.unseen_actors_file)))
    if unseen_actions:
        missing = sorted(set(unseen_actions) - set(df["_action_norm"].unique()))
        if missing:
            raise ValueError(f"Requested unseen actions not found: {missing}. Available: {sorted(df['_action_norm'].unique())}")
    if unseen_actors:
        if not args.actor_col:
            raise ValueError("--unseen_actors was given, so --actor_col must also be given.")
        missing = sorted(set(unseen_actors) - set(df["_actor_norm"].unique()))
        if missing:
            raise ValueError(f"Requested unseen actors not found: {missing}. Available examples: {sorted(df['_actor_norm'].unique())[:30]}")
    unseen_action_df = df[df["_action_norm"].isin(unseen_actions)].copy() if unseen_actions else pd.DataFrame(columns=df.columns)
    unseen_actor_df = df[df["_actor_norm"].isin(unseen_actors)].copy() if unseen_actors else pd.DataFrame(columns=df.columns)
    # Anything matching either holdout condition is excluded from seen train/val/test.
    holdout_mask = pd.Series(False, index=df.index)
    if unseen_actions:
        holdout_mask |= df["_action_norm"].isin(unseen_actions)
    if unseen_actors:
        holdout_mask |= df["_actor_norm"].isin(unseen_actors)
    seen_pool = df[~holdout_mask].copy()
    if len(seen_pool) == 0:
        raise ValueError("No seen pool remains after action/actor holdout.")
    if not {0, 1}.issubset(set(seen_pool["label"].astype(int).unique())):
        raise ValueError("Seen train pool must contain both normal and anomaly labels.")
    stratify_cols = ["_action_norm", "label"]
    if args.actor_col and args.stratify_by_actor:
        stratify_cols = ["_action_norm", "_actor_norm", "label"]
    train_df, val_df, seen_test_df = split_random_stratified_80_20(
        seen_pool,
        stratify_cols=stratify_cols,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    test_sets: Dict[str, pd.DataFrame] = {"seen_test": seen_test_df}
    if unseen_actions:
        test_sets["unseen_action_test"] = unseen_action_df.reset_index(drop=True)
    if unseen_actors:
        test_sets["unseen_actor_test"] = unseen_actor_df.reset_index(drop=True)
    if unseen_actions and unseen_actors:
        both = df[df["_action_norm"].isin(unseen_actions) & df["_actor_norm"].isin(unseen_actors)].copy()
        if len(both):
            test_sets["unseen_action_actor_test"] = both.reset_index(drop=True)
    combined = pd.concat(list(test_sets.values()), axis=0).drop_duplicates(subset=["original_index"]).sample(frac=1.0, random_state=args.seed + 50).reset_index(drop=True)
    test_sets["combined_test"] = combined
    # Leakage checks
    train_actions = set(train_df["_action_norm"].unique())
    val_actions = set(val_df["_action_norm"].unique())
    train_actors = set(train_df["_actor_norm"].unique()) if args.actor_col else set()
    val_actors = set(val_df["_actor_norm"].unique()) if args.actor_col else set()
    leakage = {
        "unseen_actions_in_train_val": sorted(set(unseen_actions) & (train_actions | val_actions)),
        "unseen_actors_in_train_val": sorted(set(unseen_actors) & (train_actors | val_actors)),
    }
    if leakage["unseen_actions_in_train_val"] or leakage["unseen_actors_in_train_val"]:
        raise RuntimeError(f"Holdout leakage detected: {leakage}")
    balance_info = {"enabled": bool(args.balance_test_sets)}
    if args.balance_test_sets:
        balanced_sets: Dict[str, pd.DataFrame] = {}
        for name, split_df in test_sets.items():
            if len(split_df) == 0:
                balanced_sets[name] = split_df
                continue
            bdf, info = balance_binary_split(split_df, "label", args.seed + hash(name) % 10000, name)
            balanced_sets[name] = bdf
            balance_info[name] = info
        test_sets = balanced_sets
    summary = {
        "split_type": "specific_normal_anomaly_with_optional_action_actor_holdout",
        "normal_class": args.normal_class,
        "anomaly_class": args.anomaly_class,
        "class_col": args.class_col,
        "test_fraction": float(args.test_fraction),
        "val_fraction_from_train_pool": float(args.val_fraction),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test_sets": {name: int(len(split_df)) for name, split_df in test_sets.items()},
        "train_label_counts": train_df["label"].value_counts().to_dict(),
        "val_label_counts": val_df["label"].value_counts().to_dict(),
        "test_label_counts": {name: split_df["label"].value_counts().to_dict() for name, split_df in test_sets.items()},
        "heldout_unseen_actions": unseen_actions,
        "heldout_unseen_actors": unseen_actors,
        "leakage_check": leakage,
        "test_balance_info": balance_info,
        "seen_actions_train_val": sorted((set(train_df["_action_norm"].unique()) | set(val_df["_action_norm"].unique()))),
        "seen_actors_train_val": sorted((set(train_df["_actor_norm"].unique()) | set(val_df["_actor_norm"].unique()))) if args.actor_col else [],
    }
    return train_df, val_df, test_sets, summary
def apply_few_shot_training(
    train_df: pd.DataFrame,
    test_sets: Dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Restrict supervised finetuning to a tiny number of examples.
    If --few_shot_k > 0, the script samples only K rows from the original
    training split, either per action+label or per label. The unused training
    rows are moved into a test split, so they are never seen during training.
    This gives a low-shot setting like:
      1 healthy walk + 1 anomaly walk + ... for training,
      everything else for testing.
    """
    k = int(getattr(args, "few_shot_k", 0))
    if k <= 0:
        return train_df, test_sets, {"enabled": False}
    unit = getattr(args, "few_shot_unit", "per_action_class")
    require_exact = bool(getattr(args, "require_exact_few_shot", False))
    seed = int(args.seed)
    if unit == "per_action_class":
        group_cols = ["_action_norm", "label"]
    elif unit == "per_class":
        group_cols = ["label"]
    else:
        raise ValueError(f"Unknown few_shot_unit={unit!r}")
    selected_parts = []
    skipped_groups = []
    group_counts = {}
    for group_key, group in train_df.groupby(group_cols, dropna=False):
        group_counts[str(group_key)] = int(len(group))
        if len(group) < k:
            if require_exact:
                raise ValueError(
                    f"Few-shot group {group_key} has only {len(group)} samples, "
                    f"but --few_shot_k={k}. Use fewer shots or remove --require_exact_few_shot."
                )
            skipped_groups.append({"group": str(group_key), "available": int(len(group)), "used": int(len(group))})
            selected_parts.append(group)
        else:
            selected_parts.append(group.sample(n=k, replace=False, random_state=seed + abs(hash(str(group_key))) % 100000))
    few_train_df = pd.concat(selected_parts, axis=0).sample(frac=1.0, random_state=seed + 700).reset_index(drop=True)
    selected_indices = set(few_train_df["original_index"].astype(int).tolist())
    unused_train_df = train_df[~train_df["original_index"].astype(int).isin(selected_indices)].copy().reset_index(drop=True)
    new_test_sets = dict(test_sets)
    if len(unused_train_df):
        new_test_sets["fewshot_unused_train_pool_test"] = unused_train_df
    # Recreate combined_test after adding unused training rows.
    combined = pd.concat(list(new_test_sets.values()), axis=0).drop_duplicates(subset=["original_index"]).sample(
        frac=1.0, random_state=seed + 701
    ).reset_index(drop=True)
    new_test_sets["combined_test"] = combined
    # Optionally balance all test sets after adding the unused training pool.
    balance_info = {"enabled": bool(args.balance_test_sets), "after_few_shot": True}
    if args.balance_test_sets:
        balanced_sets: Dict[str, pd.DataFrame] = {}
        for name, split_df in new_test_sets.items():
            if len(split_df) == 0:
                balanced_sets[name] = split_df
                continue
            bdf, info = balance_binary_split(split_df, "label", seed + 800 + abs(hash(name)) % 10000, name)
            balanced_sets[name] = bdf
            balance_info[name] = info
        new_test_sets = balanced_sets
    info = {
        "enabled": True,
        "few_shot_k": k,
        "few_shot_unit": unit,
        "require_exact_few_shot": require_exact,
        "original_train_size": int(len(train_df)),
        "fewshot_train_size": int(len(few_train_df)),
        "unused_train_pool_test_size": int(len(unused_train_df)),
        "fewshot_train_label_counts": few_train_df["label"].value_counts().to_dict(),
        "unused_train_pool_label_counts": unused_train_df["label"].value_counts().to_dict() if len(unused_train_df) else {},
        "group_counts_before_sampling": group_counts,
        "groups_with_less_than_k": skipped_groups,
        "test_balance_info_after_fewshot": balance_info,
    }
    return few_train_df, new_test_sets, info
# -----------------------------
# MotionCLIP loading/freezing
# -----------------------------
def build_motionclip_encoder(checkpoint_path: str, device: torch.device) -> nn.Module:
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
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys: {unexpected}")
    if missing:
        print("[WARN] Missing encoder keys:", missing)
    return encoder.to(device)
def freeze_encoder_except_last_layers(encoder: nn.Module, num_trainable_blocks: int = 2) -> Dict[str, Any]:
    for p in encoder.parameters():
        p.requires_grad = False
    unfrozen_layer_indices: List[int] = []
    if hasattr(encoder, "seqTransEncoder") and hasattr(encoder.seqTransEncoder, "layers"):
        layers = encoder.seqTransEncoder.layers
        n = min(num_trainable_blocks, len(layers))
        start = len(layers) - n
        for i, layer in enumerate(layers[start:], start=start):
            for p in layer.parameters():
                p.requires_grad = True
            unfrozen_layer_indices.append(i)
        if getattr(encoder.seqTransEncoder, "norm", None) is not None:
            for p in encoder.seqTransEncoder.norm.parameters():
                p.requires_grad = True
    else:
        print("[WARN] Could not find encoder.seqTransEncoder.layers; encoder may remain frozen.")
    trainable_names = [name for name, p in encoder.named_parameters() if p.requires_grad]
    return {
        "num_trainable_blocks_requested": int(num_trainable_blocks),
        "unfrozen_layer_indices": unfrozen_layer_indices,
        "num_trainable_params": int(sum(p.numel() for p in encoder.parameters() if p.requires_grad)),
        "num_total_params": int(sum(p.numel() for p in encoder.parameters())),
        "trainable_param_names_first_100": trainable_names[:100],
    }
def encode_motion_auto(model: nn.Module, motion: torch.Tensor) -> torch.Tensor:
    motion = motion.float()
    x = motion.permute(0, 2, 3, 1).contiguous()  # [B,25,6,60]
    B, T = motion.shape[0], motion.shape[1]
    lengths = torch.full((B,), T, dtype=torch.long, device=motion.device)
    mask = torch.arange(T, device=motion.device).unsqueeze(0) < lengths.unsqueeze(1)
    out = model({"x": x, "y": torch.zeros(B, dtype=torch.long, device=motion.device), "lengths": lengths, "mask": mask})
    if not isinstance(out, dict) or "mu" not in out:
        raise RuntimeError("Expected MotionCLIP encoder output dict with key 'mu'.")
    return out["mu"]
# -----------------------------
# Text prompts
# -----------------------------
class FrozenCLIPTextEncoder:
    def __init__(self, clip_model_name: str, device: torch.device):
        try:
            import clip
        except ImportError as e:
            raise ImportError("Could not import `clip`. Install OpenAI CLIP first.") from e
        self.clip = clip
        self.model, _ = clip.load(clip_model_name, device=device)
        self.model = self.model.float().eval()
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
            feats.append(F.normalize(f, dim=-1).cpu())
        return torch.cat(feats, dim=0)
def build_prompt_cache(
    actions: Sequence[str],
    text_encoder: FrozenCLIPTextEncoder,
    normal_class: str,
    anomaly_class: str,
    normal_prompt_template: str,
    anomaly_prompt_template: str,
    device: torch.device,
) -> Tuple[Dict[str, int], torch.Tensor, Dict[str, Dict[str, str]]]:
    actions = sorted({norm_text(a) for a in actions})
    action_to_idx = {a: i for i, a in enumerate(actions)}
    prompt_info: Dict[str, Dict[str, str]] = {}
    texts: List[str] = []
    for action in actions:
        normal_prompt = normal_prompt_template.format(action=action, normal_class=normal_class, anomaly_class=anomaly_class, class_name=normal_class)
        anomaly_prompt = anomaly_prompt_template.format(action=action, normal_class=normal_class, anomaly_class=anomaly_class, class_name=anomaly_class)
        prompt_info[action] = {"normal": normal_prompt, "anomaly": anomaly_prompt}
        texts.extend([normal_prompt, anomaly_prompt])
    text_feats = text_encoder.encode(texts).reshape(len(actions), 2, -1).to(device)
    return action_to_idx, text_feats, prompt_info
# -----------------------------
# Metrics
# -----------------------------
def binary_auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum(); n_neg = neg.sum()
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
    total_pos = y.sum()
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / total_pos)
def classification_metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
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
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, criterion: str = "f1") -> Tuple[float, Dict[str, Any]]:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 0.0, {}
    candidates = np.unique(scores)
    if len(candidates) > 1000:
        candidates = np.quantile(scores, np.linspace(0, 1, 1000))
    best_thr, best_metrics, best_value = float(candidates[0]), None, -float("inf")
    for thr in candidates:
        m = classification_metrics_at_threshold(y_true, scores, float(thr))
        value = m.get(criterion, m["f1"])
        if value > best_value:
            best_thr, best_metrics, best_value = float(thr), m, value
    assert best_metrics is not None
    return best_thr, best_metrics
def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: Optional[float] = None, threshold_criterion: str = "f1") -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auroc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
        auprc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        auroc = binary_auc_rank(y_true, scores)
        auprc = average_precision_fallback(y_true, scores)
    if threshold is None:
        threshold, threshold_metrics = find_best_threshold(y_true, scores, threshold_criterion)
        source = f"best_{threshold_criterion}_on_this_split"
    else:
        threshold_metrics = classification_metrics_at_threshold(y_true, scores, threshold)
        source = "provided"
    return {
        "auroc": auroc,
        "auprc": auprc,
        "n_samples": int(len(y_true)),
        "n_normal": int((y_true == 0).sum()),
        "n_anomaly": int((y_true == 1).sum()),
        "score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
        "score_std": float(np.std(scores)) if len(scores) else float("nan"),
        "threshold_source": source,
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
    logit_normal: np.ndarray
    logit_anomaly: np.ndarray
    embeddings: np.ndarray
    paths: List[str]
    actions: List[str]
    class_names: List[str]
    row_indices: List[int]
def logits_from_motion_and_prompts(
    motion_encoder: nn.Module,
    motion: torch.Tensor,
    actions: Sequence[str],
    action_to_idx: Dict[str, int],
    text_feats: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z = F.normalize(encode_motion_auto(motion_encoder, motion).float(), dim=-1)
    idx = torch.tensor([action_to_idx[norm_text(a)] for a in actions], dtype=torch.long, device=motion.device)
    prompts = text_feats[idx]  # [B,2,D]
    if z.shape[-1] != prompts.shape[-1]:
        raise RuntimeError(f"Motion embedding dim {z.shape[-1]} does not match text dim {prompts.shape[-1]}.")
    logits = torch.bmm(prompts, z.unsqueeze(-1)).squeeze(-1) / temperature
    return logits, z
def target_text_embeddings_and_group_ids(
    actions: Sequence[str],
    labels: torch.Tensor,
    action_to_idx: Dict[str, int],
    text_feats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    action_idx = torch.tensor([action_to_idx[norm_text(a)] for a in actions], dtype=torch.long, device=labels.device)
    labels = labels.long()
    target_text = text_feats[action_idx, labels]
    group_ids = action_idx * 2 + labels
    return target_text, group_ids
def symmetric_motion_text_contrastive_loss(motion_z: torch.Tensor, text_z: torch.Tensor, group_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    motion_z = F.normalize(motion_z.float(), dim=-1)
    text_z = F.normalize(text_z.float(), dim=-1)
    logits = motion_z @ text_z.t() / temperature
    pos = group_ids[:, None].eq(group_ids[None, :]).float()
    log_prob_m2t = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_m2t = -(pos * log_prob_m2t).sum(dim=1) / pos.sum(dim=1).clamp_min(1.0)
    log_prob_t2m = logits.t() - torch.logsumexp(logits.t(), dim=1, keepdim=True)
    loss_t2m = -(pos.t() * log_prob_t2m).sum(dim=1) / pos.t().sum(dim=1).clamp_min(1.0)
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
    total_loss, total_n = 0.0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            z = encode_motion_auto(motion_encoder, motion)
            target_text, group_ids = target_text_embeddings_and_group_ids(batch["action"], labels, action_to_idx, text_feats)
            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in motion_encoder.parameters() if p.requires_grad], grad_clip)
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
    losses: List[float] = []
    y_true: List[int] = []
    scores: List[float] = []
    probs: List[float] = []
    logit_normal: List[float] = []
    logit_anomaly: List[float] = []
    embeddings: List[np.ndarray] = []
    paths: List[str] = []
    actions: List[str] = []
    class_names: List[str] = []
    row_indices: List[int] = []
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits, z = logits_from_motion_and_prompts(motion_encoder, motion, batch["action"], action_to_idx, text_feats, temperature)
        if compute_contrastive_loss:
            target_text, group_ids = target_text_embeddings_and_group_ids(batch["action"], labels, action_to_idx, text_feats)
            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
            losses.append(float(loss.detach().cpu()) * labels.numel())
        soft = torch.softmax(logits, dim=-1)
        score = logits[:, 1] - logits[:, 0]
        embeddings.append(z.detach().cpu().numpy().astype(np.float32))
        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        scores.extend(score.cpu().numpy().astype(float).tolist())
        probs.extend(soft[:, 1].cpu().numpy().astype(float).tolist())
        logit_normal.extend(logits[:, 0].cpu().numpy().astype(float).tolist())
        logit_anomaly.extend(logits[:, 1].cpu().numpy().astype(float).tolist())
        paths.extend(batch["path"])
        actions.extend(batch["action"])
        class_names.extend(batch["class_name"])
        row_indices.extend(batch["row_index"])
    total_n = max(1, len(y_true))
    return EvalOutput(
        loss=sum(losses) / total_n if losses else float("nan"),
        y_true=np.asarray(y_true, dtype=int),
        score=np.asarray(scores, dtype=float),
        prob_anomaly=np.asarray(probs, dtype=float),
        logit_normal=np.asarray(logit_normal, dtype=float),
        logit_anomaly=np.asarray(logit_anomaly, dtype=float),
        embeddings=np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 512), dtype=np.float32),
        paths=paths,
        actions=actions,
        class_names=class_names,
        row_indices=row_indices,
    )
def save_predictions(eval_out: EvalOutput, path: str | Path, threshold: float) -> None:
    pred = (eval_out.score >= threshold).astype(int)
    pd.DataFrame({
        "row_index": eval_out.row_indices,
        "motion_path": eval_out.paths,
        "action": eval_out.actions,
        "class_name": eval_out.class_names,
        "y_true_is_anomaly": eval_out.y_true,
        "anomaly_score_anomaly_minus_normal": eval_out.score,
        "anomaly_score_flawed_minus_healthy": eval_out.score,  # compatibility alias
        "prob_anomaly": eval_out.prob_anomaly,
        "logit_normal": eval_out.logit_normal,
        "logit_anomaly": eval_out.logit_anomaly,
        "logit_healthy": eval_out.logit_normal,  # compatibility alias
        "logit_flawed": eval_out.logit_anomaly,  # compatibility alias
        "pred_is_anomaly": pred,
    }).to_csv(path, index=False)
def save_embeddings(eval_out: EvalOutput, npz_path: str | Path, metadata_csv_path: str | Path, threshold: float) -> None:
    pred = (eval_out.score >= threshold).astype(int)
    np.savez_compressed(
        npz_path,
        embeddings=eval_out.embeddings.astype(np.float32),
        y_true=eval_out.y_true.astype(np.int64),
        score=eval_out.score.astype(np.float32),
        prob_anomaly=eval_out.prob_anomaly.astype(np.float32),
        logit_normal=eval_out.logit_normal.astype(np.float32),
        logit_anomaly=eval_out.logit_anomaly.astype(np.float32),
        row_index=np.asarray(eval_out.row_indices, dtype=np.int64),
        pred_is_anomaly=pred.astype(np.int64),
    )
    pd.DataFrame({
        "embedding_index": np.arange(len(eval_out.row_indices), dtype=int),
        "row_index": eval_out.row_indices,
        "motion_path": eval_out.paths,
        "action": eval_out.actions,
        "class_name": eval_out.class_names,
        "y_true_is_anomaly": eval_out.y_true,
        "anomaly_score_anomaly_minus_normal": eval_out.score,
        "anomaly_score_flawed_minus_healthy": eval_out.score,
        "prob_anomaly": eval_out.prob_anomaly,
        "logit_normal": eval_out.logit_normal,
        "logit_anomaly": eval_out.logit_anomaly,
        "pred_is_anomaly": pred,
    }).to_csv(metadata_csv_path, index=False)
# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MotionCLIP on one normal style/condition vs one anomaly style/condition.")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--path_col", default="motion_path")
    parser.add_argument("--action_col", default="action_label")
    parser.add_argument("--class_col", default="condition_label", help="Column containing the style/condition class to use.")
    parser.add_argument("--actor_col", default="", help="Optional actor column, required only for --unseen_actors.")
    parser.add_argument("--normal_class", required=True, help="Specific style/condition treated as normal.")
    parser.add_argument("--anomaly_class", required=True, help="Specific style/condition treated as anomaly.")
    parser.add_argument("--motion_key", default="auto")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--njoints", type=int, default=25)
    parser.add_argument("--nfeats", type=int, default=6)
    parser.add_argument("--test_fraction", type=float, default=0.20, help="Default 80/20 train/test split for seen pool.")
    parser.add_argument("--val_fraction", type=float, default=0.10, help="Validation fraction from the 80% train pool.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few_shot_k", type=int, default=0, help="Use only K training examples in the selected unit. 0 disables few-shot.")
    parser.add_argument("--few_shot_unit", choices=["per_action_class", "per_class"], default="per_action_class", help="per_action_class means K normal and K anomaly samples per action; per_class means K normal and K anomaly total.")
    parser.add_argument("--require_exact_few_shot", action="store_true", help="Error if any few-shot group has fewer than K samples.")
    parser.add_argument("--zero_shot", action="store_true", help="Do not finetune MotionCLIP. Only evaluate pretrained MotionCLIP with the selected prompts.")
    parser.add_argument("--stratify_by_actor", action="store_true", help="When actor_col is given, stratify row split by action+actor+label.")
    parser.add_argument("--balance_test_sets", action="store_true", default=True)
    parser.add_argument("--no_balance_test_sets", action="store_false", dest="balance_test_sets")
    parser.add_argument("--unseen_actions", nargs="*", default=[], help="Optional action names to keep completely unseen from train/val.")
    parser.add_argument("--unseen_actions_file", default="")
    parser.add_argument("--unseen_actors", nargs="*", default=[], help="Optional actor IDs/names to keep completely unseen from train/val.")
    parser.add_argument("--unseen_actors_file", default="")
    parser.add_argument("--project_root", default="", help="Parent directory containing the MotionCLIP folder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trainable_layers", type=int, default=2)
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--normal_prompt_template", default="{normal_class} {action}")
    parser.add_argument("--anomaly_prompt_template", default="{anomaly_class} {action}")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--threshold_criterion", default="f1", choices=["f1", "balanced_accuracy", "accuracy"])
    args = parser.parse_args()
    args.normal_class = norm_text(args.normal_class)
    args.anomaly_class = norm_text(args.anomaly_class)
    if args.normal_class == args.anomaly_class:
        raise ValueError("normal_class and anomaly_class must be different.")
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_json(vars(args), output_dir / "args.json")
    df = pd.read_csv(args.csv_path).copy()
    df["original_index"] = np.arange(len(df))
    required = [args.path_col, args.action_col, args.class_col]
    if args.actor_col:
        required.append(args.actor_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")
    df["_action_norm"] = df[args.action_col].map(norm_text)
    df["_class_norm"] = df[args.class_col].map(norm_text)
    if args.actor_col:
        df["_actor_norm"] = df[args.actor_col].map(norm_text)
    else:
        df["_actor_norm"] = ""
    available_classes = sorted(df["_class_norm"].dropna().unique().tolist())
    missing_classes = sorted({args.normal_class, args.anomaly_class} - set(available_classes))
    if missing_classes:
        raise ValueError(f"Requested class(es) not found in {args.class_col}: {missing_classes}. Available: {available_classes}")
    df = df[df["_class_norm"].isin([args.normal_class, args.anomaly_class])].copy()
    df["label"] = (df["_class_norm"] == args.anomaly_class).astype(int)
    if not {0, 1}.issubset(set(df["label"].unique())):
        raise ValueError("Filtered data does not contain both normal and anomaly rows.")
    # Basic path check only on examples.
    missing_paths = [p for p in df[args.path_col].head(20).tolist() if not Path(str(p)).exists()]
    if missing_paths:
        print("[WARN] Some example motion files do not exist from this machine.")
        print(f"       First missing example: {missing_paths[0]}")
    train_df, val_df, test_sets, split_summary = make_splits(args, df)
    train_df, test_sets, few_shot_info = apply_few_shot_training(train_df, test_sets, args)
    split_summary["few_shot_info"] = few_shot_info
    split_summary["train"] = int(len(train_df))
    split_summary["test_sets"] = {name: int(len(split_df)) for name, split_df in test_sets.items()}
    split_summary["train_label_counts"] = train_df["label"].value_counts().to_dict()
    split_summary["test_label_counts"] = {name: split_df["label"].value_counts().to_dict() for name, split_df in test_sets.items()}
    train_df.to_csv(output_dir / "split_train.csv", index=False)
    val_df.to_csv(output_dir / "split_val.csv", index=False)
    for name, split_df in test_sets.items():
        split_df.to_csv(output_dir / f"split_{name}.csv", index=False)
    test_sets["combined_test"].to_csv(output_dir / "split_test.csv", index=False)  # compatibility
    save_json(split_summary, output_dir / "split_summary.json")
    print("[INFO] Split summary:", json.dumps(split_summary, indent=2))
    expected_shape = (args.num_frames, args.njoints, args.nfeats)
    train_ds = PerMoMotionDataset(train_df, args.path_col, args.action_col, args.class_col, "label", args.motion_key, expected_shape)
    val_ds = PerMoMotionDataset(val_df, args.path_col, args.action_col, args.class_col, "label", args.motion_key, expected_shape)
    test_datasets = {
        name: PerMoMotionDataset(split_df, args.path_col, args.action_col, args.class_col, "label", args.motion_key, expected_shape)
        for name, split_df in test_sets.items()
    }
    train_labels = train_df["label"].astype(int).to_numpy()
    train_batch_sampler = BalancedBinaryBatchSampler(train_labels, batch_size=args.batch_size, seed=args.seed)
    sampler_info = {
        "type": "BalancedBinaryBatchSampler",
        "batch_size": int(args.batch_size),
        "normal_per_batch": int(train_batch_sampler.n0),
        "anomaly_per_batch": int(train_batch_sampler.n1),
        "num_batches_per_epoch": int(len(train_batch_sampler)),
        "train_label_counts": {"normal_0": int((train_labels == 0).sum()), "anomaly_1": int((train_labels == 1).sum())},
    }
    save_json(sampler_info, output_dir / "train_sampler_info.json")
    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
        for name, ds in test_datasets.items()
    }
    if args.project_root:
        sys.path.insert(0, str(Path(args.project_root).resolve()))
    global Encoder_TRANSFORMER
    from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER
    motion_encoder = build_motionclip_encoder(args.checkpoint, device)
    save_json({"loaded": True, "checkpoint_path": args.checkpoint}, output_dir / "checkpoint_load_info.json")
    unfreeze_info = freeze_encoder_except_last_layers(motion_encoder, args.trainable_layers)
    save_json(unfreeze_info, output_dir / "unfreeze_info.json")
    print("[INFO] Unfreeze info:", unfreeze_info)
    print("[INFO] Loading frozen CLIP text encoder...")
    text_encoder = FrozenCLIPTextEncoder(args.clip_model, device)
    all_actions_for_prompts = df[args.action_col].map(norm_text).tolist()
    action_to_idx, text_feats, prompt_info = build_prompt_cache(
        all_actions_for_prompts,
        text_encoder,
        args.normal_class,
        args.anomaly_class,
        args.normal_prompt_template,
        args.anomaly_prompt_template,
        device,
    )
    save_json(prompt_info, output_dir / "prompts.json")
    torch.save({"action_to_idx": action_to_idx, "text_feats": text_feats.detach().cpu(), "prompt_info": prompt_info}, output_dir / "text_prompt_cache.pt")
    optimizer = torch.optim.AdamW([p for p in motion_encoder.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    best_val_auroc = -float("inf")
    best_epoch = -1
    best_threshold = 0.0
    epoch_records: List[Dict[str, Any]] = []
    if args.zero_shot:
        print("[INFO] Zero-shot mode: skipping finetuning and evaluating pretrained MotionCLIP with prompts only.")
        val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, text_feats, args.temperature, compute_contrastive_loss=True)
        val_metrics = compute_binary_metrics(val_out.y_true, val_out.score, threshold=None, threshold_criterion=args.threshold_criterion)
        best_val_auroc = val_metrics["auroc"]
        best_epoch = 0
        best_threshold = float(val_metrics["threshold"])
        save_predictions(val_out, output_dir / "val_predictions_best.csv", best_threshold)
    else:
        if args.epochs <= 0:
            raise ValueError("Use --zero_shot for no-training evaluation, or set --epochs > 0.")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_batch_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(motion_encoder, train_loader, optimizer, device, action_to_idx, text_feats, args.temperature, args.grad_clip, args.amp)
            train_eval_out = evaluate(motion_encoder, train_eval_loader, device, action_to_idx, text_feats, args.temperature, compute_contrastive_loss=False)
            train_metrics_epoch = compute_binary_metrics(train_eval_out.y_true, train_eval_out.score, threshold=None, threshold_criterion=args.threshold_criterion)
            val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, text_feats, args.temperature, compute_contrastive_loss=True)
            val_metrics = compute_binary_metrics(val_out.y_true, val_out.score, threshold=None, threshold_criterion=args.threshold_criterion)
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
                f"[EPOCH {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_out.loss:.4f} "
                f"train_auroc={train_metrics_epoch['auroc']:.4f} val_auroc={val_metrics['auroc']:.4f} "
                f"val_auprc={val_metrics['auprc']:.4f} val_f1={val_metrics['f1']:.4f} thr={val_metrics['threshold']:.4f}"
            )
            if np.isfinite(val_metrics["auroc"]) and val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
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
                save_predictions(val_out, output_dir / "val_predictions_best.csv", best_threshold)
    best_ckpt_path = ckpt_dir / "best_model.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        motion_encoder.load_state_dict(best_ckpt["motion_encoder_state_dict"], strict=True)
        best_threshold = float(best_ckpt["best_val_metrics"]["threshold"])
    else:
        print("[WARN] No best checkpoint saved. Testing final epoch model.")
    # Final eval and saving
    train_out = evaluate(motion_encoder, train_eval_loader, device, action_to_idx, text_feats, args.temperature, True)
    val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, text_feats, args.temperature, True)
    eval_outputs: Dict[str, EvalOutput] = {name: evaluate(motion_encoder, loader, device, action_to_idx, text_feats, args.temperature, True) for name, loader in test_loaders.items()}
    train_metrics = compute_binary_metrics(train_out.y_true, train_out.score, threshold=best_threshold)
    val_metrics_final = compute_binary_metrics(val_out.y_true, val_out.score, threshold=best_threshold)
    train_metrics["loss"] = train_out.loss
    val_metrics_final["loss"] = val_out.loss
    save_predictions(train_out, output_dir / "train_predictions.csv", best_threshold)
    save_predictions(val_out, output_dir / "val_predictions.csv", best_threshold)
    save_embeddings(train_out, output_dir / "train_embeddings.npz", output_dir / "train_embeddings_metadata.csv", best_threshold)
    save_embeddings(val_out, output_dir / "val_embeddings.npz", output_dir / "val_embeddings_metadata.csv", best_threshold)
    test_metrics_by_name: Dict[str, Any] = {}
    output_files: Dict[str, str] = {
        "best_checkpoint": str(best_ckpt_path),
        "epoch_metrics": str(output_dir / "epoch_metrics.csv"),
        "metrics": str(output_dir / "metrics.json"),
        "training_history_npz": str(output_dir / "training_history.npz"),
        "loss_curves": str(output_dir / "loss_curves.png"),
        "validation_metrics_plot": str(output_dir / "validation_metrics.png"),
        "auroc_curves": str(output_dir / "auroc_curves.png"),
        "train_embeddings": str(output_dir / "train_embeddings.npz"),
        "val_embeddings": str(output_dir / "val_embeddings.npz"),
    }
    for name, out in eval_outputs.items():
        m = compute_binary_metrics(out.y_true, out.score, threshold=best_threshold)
        m["loss"] = out.loss
        test_metrics_by_name[name] = m
        pred_path = output_dir / f"test_predictions_{name}.csv"
        emb_path = output_dir / f"test_embeddings_{name}.npz"
        meta_path = output_dir / f"test_embeddings_{name}_metadata.csv"
        save_predictions(out, pred_path, best_threshold)
        save_embeddings(out, emb_path, meta_path, best_threshold)
        output_files[f"test_predictions_{name}"] = str(pred_path)
        output_files[f"test_embeddings_{name}"] = str(emb_path)
        output_files[f"test_embeddings_{name}_metadata"] = str(meta_path)
    # Compatibility names = combined test
    if "combined_test" in eval_outputs:
        save_predictions(eval_outputs["combined_test"], output_dir / "test_predictions.csv", best_threshold)
        save_embeddings(eval_outputs["combined_test"], output_dir / "test_embeddings.npz", output_dir / "test_embeddings_metadata.csv", best_threshold)
        output_files["test_predictions"] = str(output_dir / "test_predictions.csv")
        output_files["test_embeddings"] = str(output_dir / "test_embeddings.npz")
        output_files["test_embeddings_metadata"] = str(output_dir / "test_embeddings_metadata.csv")
    final_summary = {
        "best_epoch": best_epoch,
        "best_val_auroc_during_training": best_val_auroc,
        "threshold_selected_on_validation": best_threshold,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics_final,
        "test_metrics": test_metrics_by_name.get("combined_test", {}),
        "test_metrics_by_split": test_metrics_by_name,
        "split_summary": split_summary,
        "prompt_templates": {"normal": args.normal_prompt_template, "anomaly": args.anomaly_prompt_template},
        "zero_shot": bool(args.zero_shot),
        "few_shot_info": few_shot_info,
        "normal_class": args.normal_class,
        "anomaly_class": args.anomaly_class,
        "output_files": output_files,
    }
    save_json(final_summary, output_dir / "metrics.json")
    print("[DONE] Combined test metrics:")
    print(json.dumps(test_metrics_by_name.get("combined_test", {}), indent=2, sort_keys=True))
    print(f"[DONE] Outputs saved to: {output_dir}")
if __name__ == "__main__":
    main()
