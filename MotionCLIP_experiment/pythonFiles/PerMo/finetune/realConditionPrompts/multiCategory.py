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
# Text prompts for unseen anomaly-style transfer
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
    class_names: Sequence[str],
    text_encoder: FrozenCLIPTextEncoder,
    normal_prompt_classes: Sequence[str],
    normal_prompt_template: str,
    anomaly_prompt_template: str,
    device: torch.device,
) -> Tuple[Dict[str, int], Dict[str, int], torch.Tensor, Dict[str, Dict[str, str]]]:
    """Build text prompts for all action/style pairs.

    Any class in normal_prompt_classes is formatted with normal_prompt_template.
    All other classes are formatted with anomaly_prompt_template.
    This allows cross-parent tests such as train normal=Healthy and test normal=Happy.
    """
    actions = sorted({norm_text(a) for a in actions})
    class_names = sorted({norm_text(c) for c in class_names})
    normal_prompt_classes = sorted({norm_text(c) for c in normal_prompt_classes})
    for c in normal_prompt_classes:
        if c not in class_names:
            class_names.append(c)
    # Put normal prompt classes first for readability.
    class_names = normal_prompt_classes + sorted(c for c in set(class_names) if c not in set(normal_prompt_classes))
    action_to_idx = {a: i for i, a in enumerate(actions)}
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    texts: List[str] = []
    prompt_info: Dict[str, Dict[str, str]] = {}
    first_normal = normal_prompt_classes[0] if normal_prompt_classes else "normal"
    for action in actions:
        prompt_info[action] = {}
        for cls in class_names:
            if cls in normal_prompt_classes:
                prompt = normal_prompt_template.format(action=action, normal_class=cls, class_name=cls)
            else:
                prompt = anomaly_prompt_template.format(action=action, anomaly_class=cls, class_name=cls, normal_class=first_normal)
            prompt_info[action][cls] = prompt
            texts.append(prompt)
    text_feats = text_encoder.encode(texts).reshape(len(actions), len(class_names), -1).to(device)
    return action_to_idx, class_to_idx, text_feats, prompt_info


# -----------------------------
# Multi-style split logic
# -----------------------------

def _parse_style_specs(values: Optional[Sequence[str]]) -> List[str]:
    """Parse style specs.

    Supported forms:
      Healthy
      condition_label:Healthy
      condition:Healthy

    If --category_col is not used, only the style name part is used.
    If --category_col is used, category:style restricts selection to that category.
    """
    return parse_list(values)


def _style_spec_to_key(spec: str, has_category: bool) -> str:
    spec = norm_text(spec)
    if has_category and ":" in spec:
        left, right = spec.split(":", 1)
        return f"{norm_text(left)}::{norm_text(right)}"
    if has_category:
        # No category given: match this style in any category via class-only logic elsewhere.
        return norm_text(spec)
    return norm_text(spec.split(":", 1)[-1])


def _mask_from_style_specs(df: pd.DataFrame, specs: Sequence[str], has_category: bool) -> pd.Series:
    """Build a row mask from style specs.

    With --category_col, specs may be either category:style or plain style.
    Plain style matches all categories with that class/style name.
    """
    mask = pd.Series(False, index=df.index)
    for raw in specs:
        spec = norm_text(raw)
        if not spec:
            continue
        if has_category and ":" in spec:
            category, style = spec.split(":", 1)
            mask |= (df["_category_norm"] == norm_text(category)) & (df["_class_norm"] == norm_text(style))
        else:
            style = spec.split(":", 1)[-1]
            mask |= df["_class_norm"] == norm_text(style)
    return mask


def _read_style_specs_from_file(path: str) -> List[str]:
    if not path:
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Preserve category:style, but allow csv first field for simple lists.
            out.append(norm_text(line.split(",")[0]))
    return sorted(set(out))


def make_transfer_splits(args: argparse.Namespace, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Create train/val/test splits for multi-style normal-vs-anomaly experiments.

    Normal and anomaly styles can come from one or more parent categories. If your CSV
    has a category column, pass --category_col and use category-qualified specs such as:
        --normal_classes "condition:Healthy" "emotion:Happy"
        --anomaly_classes "condition:Drunken" "emotion:Sad"

    If no --category_col is given, specs are matched only against --class_col.

    Optional --unseen_actions and --unseen_actors are removed from train/val/seen_test
    and evaluated as separate test splits using all selected normal/anomaly styles.
    """
    has_category = bool(getattr(args, "category_col", ""))

    normal_specs = sorted(set(
        _parse_style_specs(getattr(args, "normal_classes", []))
        + _read_style_specs_from_file(getattr(args, "normal_classes_file", ""))
    ))
    anomaly_specs = sorted(set(
        _parse_style_specs(getattr(args, "anomaly_classes", []))
        + _read_style_specs_from_file(getattr(args, "anomaly_classes_file", ""))
    ))

    # Backward-compatible aliases.
    if getattr(args, "normal_class", ""):
        normal_specs.append(norm_text(args.normal_class))
    if getattr(args, "anomaly_class", ""):
        anomaly_specs.append(norm_text(args.anomaly_class))
    if getattr(args, "train_anomaly_classes", []):
        anomaly_specs.extend(_parse_style_specs(args.train_anomaly_classes))
    if getattr(args, "train_anomaly_classes_file", ""):
        anomaly_specs.extend(_read_style_specs_from_file(args.train_anomaly_classes_file))
    normal_specs = sorted(set(normal_specs))
    anomaly_specs = sorted(set(anomaly_specs))

    if not normal_specs:
        raise ValueError("Give at least one --normal_classes value, e.g. Healthy Happy or condition:Healthy emotion:Happy.")
    if not anomaly_specs:
        raise ValueError("Give at least one --anomaly_classes value, e.g. Drunken Sad or condition:Drunken emotion:Sad.")

    normal_mask = _mask_from_style_specs(df, normal_specs, has_category)
    anomaly_mask = _mask_from_style_specs(df, anomaly_specs, has_category)
    overlap_mask = normal_mask & anomaly_mask
    if overlap_mask.any():
        examples = df.loc[overlap_mask, [c for c in [getattr(args, "category_col", ""), args.class_col] if c]].drop_duplicates().head(20).to_dict("records")
        raise ValueError(f"Some rows match both normal and anomaly style specs. Examples: {examples}")

    selected_mask = normal_mask | anomaly_mask
    if not selected_mask.any():
        available = sorted(df["_class_norm"].dropna().unique().tolist())
        raise ValueError(f"No rows matched the requested styles. Available classes include: {available[:80]}")

    df = df[selected_mask].copy()
    df["label"] = np.where(anomaly_mask.loc[df.index], 1, 0).astype(int)
    df["style_role"] = np.where(df["label"].astype(int) == 0, "normal_class", "anomaly_class")
    if has_category:
        df["style_key"] = df["_category_norm"] + "::" + df["_class_norm"]
    else:
        df["style_key"] = df["_class_norm"]

    if not {0, 1}.issubset(set(df["label"].astype(int).unique())):
        raise ValueError("Selected rows must contain at least one normal style and one anomaly style.")

    unseen_actions = sorted(set(parse_list(args.unseen_actions) + read_list_file(args.unseen_actions_file)))
    unseen_actors = sorted(set(parse_list(args.unseen_actors) + read_list_file(args.unseen_actors_file)))

    if unseen_actions:
        missing_actions = sorted(set(unseen_actions) - set(df["_action_norm"].unique()))
        if missing_actions:
            raise ValueError(f"Requested unseen actions not found in selected styles: {missing_actions}. Available: {sorted(df['_action_norm'].unique())}")
    if unseen_actors:
        if not args.actor_col:
            raise ValueError("--unseen_actors was given, so --actor_col must also be given.")
        missing_actors = sorted(set(unseen_actors) - set(df["_actor_norm"].unique()))
        if missing_actors:
            raise ValueError(f"Requested unseen actors not found in selected styles: {missing_actors}. Available examples: {sorted(df['_actor_norm'].unique())[:30]}")

    action_holdout_mask = df["_action_norm"].isin(unseen_actions) if unseen_actions else pd.Series(False, index=df.index)
    actor_holdout_mask = df["_actor_norm"].isin(unseen_actors) if unseen_actors else pd.Series(False, index=df.index)
    holdout_mask = action_holdout_mask | actor_holdout_mask

    seen_pool = df[~holdout_mask].copy()
    if len(seen_pool) == 0:
        raise ValueError("No seen training pool remains after optional action/actor holdout.")
    if not {0, 1}.issubset(set(seen_pool["label"].astype(int).unique())):
        raise ValueError("Training pool must contain at least one normal and one anomaly sample after optional holdouts.")

    # Stratify by action + style + label so seen_test contains held-out rows, not held-out styles.
    stratify_cols = ["_action_norm", "style_key", "label"]
    if args.actor_col and args.stratify_by_actor:
        stratify_cols = ["_action_norm", "_actor_norm", "style_key", "label"]

    train_df, val_df, seen_test_df = split_random_stratified_80_20(
        seen_pool,
        stratify_cols=stratify_cols,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    test_sets: Dict[str, pd.DataFrame] = {"seen_test": seen_test_df.reset_index(drop=True)}
    if unseen_actions:
        test_sets["unseen_action_test"] = df[action_holdout_mask].copy().reset_index(drop=True)
    if unseen_actors:
        test_sets["unseen_actor_test"] = df[actor_holdout_mask].copy().reset_index(drop=True)
    if unseen_actions and unseen_actors:
        both = df[action_holdout_mask & actor_holdout_mask].copy()
        if len(both):
            test_sets["unseen_action_actor_test"] = both.reset_index(drop=True)

    combined = pd.concat(list(test_sets.values()), axis=0).drop_duplicates(subset=["original_index"])
    combined = combined.sample(frac=1.0, random_state=args.seed + 50).reset_index(drop=True)
    test_sets["combined_test"] = combined

    leakage = {
        "unseen_actions_in_train_val": sorted(set(unseen_actions) & (set(train_df["_action_norm"].unique()) | set(val_df["_action_norm"].unique()))),
        "unseen_actors_in_train_val": sorted(set(unseen_actors) & (set(train_df["_actor_norm"].unique()) | set(val_df["_actor_norm"].unique()))) if args.actor_col else [],
    }
    if leakage["unseen_actions_in_train_val"] or leakage["unseen_actors_in_train_val"]:
        raise RuntimeError(f"Holdout leakage detected: {leakage}")

    balance_info: Dict[str, Any] = {"enabled": bool(args.balance_test_sets)}
    if args.balance_test_sets:
        balanced: Dict[str, pd.DataFrame] = {}
        for name, split_df in test_sets.items():
            if len(split_df) == 0:
                balanced[name] = split_df
                continue
            bdf, info = balance_binary_split(split_df, "label", args.seed + (abs(hash(name)) % 10000), name)
            balanced[name] = bdf
            balance_info[name] = info
        test_sets = balanced

    normal_classes_for_prompts = sorted(df.loc[df["label"].astype(int) == 0, "_class_norm"].unique().tolist())
    anomaly_classes_for_prompts = sorted(df.loc[df["label"].astype(int) == 1, "_class_norm"].unique().tolist())
    prompt_classes = normal_classes_for_prompts + [c for c in anomaly_classes_for_prompts if c not in set(normal_classes_for_prompts)]

    summary = {
        "split_type": "multi_style_normal_vs_multi_style_anomaly_with_optional_action_actor_holdout",
        "class_col": args.class_col,
        "category_col": getattr(args, "category_col", ""),
        "normal_style_specs_requested": normal_specs,
        "anomaly_style_specs_requested": anomaly_specs,
        "normal_classes": normal_classes_for_prompts,
        "anomaly_classes": anomaly_classes_for_prompts,
        "test_fraction_for_seen_styles": float(args.test_fraction),
        "val_fraction_from_seen_train_pool": float(args.val_fraction),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test_sets": {name: int(len(split_df)) for name, split_df in test_sets.items()},
        "train_label_counts": train_df["label"].value_counts().to_dict(),
        "val_label_counts": val_df["label"].value_counts().to_dict(),
        "test_label_counts": {name: split_df["label"].value_counts().to_dict() for name, split_df in test_sets.items()},
        "train_class_counts": train_df["_class_norm"].value_counts().to_dict(),
        "val_class_counts": val_df["_class_norm"].value_counts().to_dict(),
        "test_class_counts": {name: split_df["_class_norm"].value_counts().to_dict() for name, split_df in test_sets.items()},
        "train_style_key_counts": train_df["style_key"].value_counts().to_dict(),
        "val_style_key_counts": val_df["style_key"].value_counts().to_dict(),
        "test_style_key_counts": {name: split_df["style_key"].value_counts().to_dict() for name, split_df in test_sets.items()},
        "heldout_unseen_actions": unseen_actions,
        "heldout_unseen_actors": unseen_actors,
        "leakage_check": leakage,
        "test_balance_info": balance_info,
        "classes_used_for_prompts": prompt_classes,
        "normal_prompt_classes": normal_classes_for_prompts,
        "actions_used_for_prompts": sorted(df["_action_norm"].unique().tolist()),
    }
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_sets, summary


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
    return {"threshold": float(threshold), "accuracy": float(accuracy), "balanced_accuracy": float(0.5 * (recall + specificity)), "precision": float(precision), "recall": float(recall), "specificity": float(specificity), "f1": float(f1), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


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
    return {"auroc": auroc, "auprc": auprc, "n_samples": int(len(y_true)), "n_normal": int((y_true == 0).sum()), "n_anomaly": int((y_true == 1).sum()), "score_mean": float(np.mean(scores)) if len(scores) else float("nan"), "score_std": float(np.std(scores)) if len(scores) else float("nan"), "threshold_source": source, **threshold_metrics}


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
    max_normal_prompt_class: List[str]
    max_anomaly_prompt_class: List[str]
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
    prompts = text_feats[idx]  # [B,C,D]
    if z.shape[-1] != prompts.shape[-1]:
        raise RuntimeError(f"Motion embedding dim {z.shape[-1]} does not match text dim {prompts.shape[-1]}.")
    logits = torch.bmm(prompts, z.unsqueeze(-1)).squeeze(-1) / temperature
    return logits, z


def target_text_embeddings_and_group_ids(
    actions: Sequence[str],
    class_names: Sequence[str],
    action_to_idx: Dict[str, int],
    class_to_idx: Dict[str, int],
    text_feats: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    action_idx = torch.tensor([action_to_idx[norm_text(a)] for a in actions], dtype=torch.long, device=device)
    class_idx = torch.tensor([class_to_idx[norm_text(c)] for c in class_names], dtype=torch.long, device=device)
    target_text = text_feats[action_idx, class_idx]
    group_ids = action_idx * len(class_to_idx) + class_idx
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
    class_to_idx: Dict[str, int],
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
            target_text, group_ids = target_text_embeddings_and_group_ids(batch["action"], batch["class_name"], action_to_idx, class_to_idx, text_feats, motion.device)
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
    class_to_idx: Dict[str, int],
    text_feats: torch.Tensor,
    normal_prompt_classes: Sequence[str],
    anomaly_prompt_classes: Sequence[str],
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
    max_norm_classes: List[str] = []
    max_anom_classes: List[str] = []
    row_indices: List[int] = []

    normal_prompt_classes = [norm_text(c) for c in normal_prompt_classes]
    anomaly_prompt_classes = [norm_text(c) for c in anomaly_prompt_classes]
    normal_indices = [class_to_idx[c] for c in normal_prompt_classes if c in class_to_idx]
    anomaly_indices = [class_to_idx[c] for c in anomaly_prompt_classes if c in class_to_idx and c not in set(normal_prompt_classes)]
    if not normal_indices:
        raise ValueError("No normal prompt classes available for scoring.")
    if not anomaly_indices:
        raise ValueError("No anomaly prompt classes available for scoring.")
    normal_indices_t = torch.tensor(normal_indices, dtype=torch.long, device=device)
    anomaly_indices_t = torch.tensor(anomaly_indices, dtype=torch.long, device=device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits, z = logits_from_motion_and_prompts(motion_encoder, motion, batch["action"], action_to_idx, text_feats, temperature)
        if compute_contrastive_loss:
            target_text, group_ids = target_text_embeddings_and_group_ids(batch["action"], batch["class_name"], action_to_idx, class_to_idx, text_feats, motion.device)
            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
            losses.append(float(loss.detach().cpu()) * labels.numel())

        norm_logits_all = logits.index_select(1, normal_indices_t)
        normal_max_logits, normal_argmax = norm_logits_all.max(dim=1)
        anom_logits_all = logits.index_select(1, anomaly_indices_t)
        anom_max_logits, anom_argmax = anom_logits_all.max(dim=1)

        chosen_norm_indices = normal_indices_t[normal_argmax].detach().cpu().numpy().astype(int).tolist()
        chosen_anom_indices = anomaly_indices_t[anom_argmax].detach().cpu().numpy().astype(int).tolist()
        chosen_norm_classes = [idx_to_class[i] for i in chosen_norm_indices]
        chosen_anom_classes = [idx_to_class[i] for i in chosen_anom_indices]

        score = anom_max_logits - normal_max_logits
        two_class_logits = torch.stack([normal_max_logits, anom_max_logits], dim=1)
        soft = torch.softmax(two_class_logits, dim=-1)

        embeddings.append(z.detach().cpu().numpy().astype(np.float32))
        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        scores.extend(score.cpu().numpy().astype(float).tolist())
        probs.extend(soft[:, 1].cpu().numpy().astype(float).tolist())
        logit_normal.extend(normal_max_logits.cpu().numpy().astype(float).tolist())
        logit_anomaly.extend(anom_max_logits.cpu().numpy().astype(float).tolist())
        paths.extend(batch["path"])
        actions.extend(batch["action"])
        class_names.extend(batch["class_name"])
        max_norm_classes.extend(chosen_norm_classes)
        max_anom_classes.extend(chosen_anom_classes)
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
        max_normal_prompt_class=max_norm_classes,
        max_anomaly_prompt_class=max_anom_classes,
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
        "anomaly_score_max_anomaly_prompt_minus_normal": eval_out.score,
        "anomaly_score_flawed_minus_healthy": eval_out.score,
        "prob_anomaly": eval_out.prob_anomaly,
        "logit_normal": eval_out.logit_normal,
        "max_normal_prompt_class": eval_out.max_normal_prompt_class,
        "logit_max_anomaly_prompt": eval_out.logit_anomaly,
        "logit_healthy": eval_out.logit_normal,
        "logit_flawed": eval_out.logit_anomaly,
        "max_anomaly_prompt_class": eval_out.max_anomaly_prompt_class,
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
        logit_max_anomaly_prompt=eval_out.logit_anomaly.astype(np.float32),
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
        "anomaly_score_max_anomaly_prompt_minus_normal": eval_out.score,
        "anomaly_score_flawed_minus_healthy": eval_out.score,
        "prob_anomaly": eval_out.prob_anomaly,
        "logit_normal": eval_out.logit_normal,
        "max_normal_prompt_class": eval_out.max_normal_prompt_class,
        "logit_max_anomaly_prompt": eval_out.logit_anomaly,
        "max_anomaly_prompt_class": eval_out.max_anomaly_prompt_class,
        "pred_is_anomaly": pred,
    }).to_csv(metadata_csv_path, index=False)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MotionCLIP for unseen anomaly-style transfer on PerMo.")

    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--path_col", default="motion_path")
    parser.add_argument("--action_col", default="action_label")
    parser.add_argument("--class_col", default="condition_label", help="Column containing the style/condition/emotion name.")
    parser.add_argument("--category_col", default="", help="Optional parent-category column. If given, style specs may be category:style, e.g. condition:Healthy emotion:Happy.")
    parser.add_argument("--actor_col", default="", help="Optional actor column, required only for --unseen_actors.")

    parser.add_argument("--normal_classes", nargs="*", default=[], help="One or more styles treated as normal. Supports plain style names or category:style when --category_col is given.")
    parser.add_argument("--normal_classes_file", default="", help="Optional text file with one normal style spec per line.")
    parser.add_argument("--anomaly_classes", nargs="*", default=[], help="One or more styles treated as anomaly. Supports plain style names or category:style when --category_col is given.")
    parser.add_argument("--anomaly_classes_file", default="", help="Optional text file with one anomaly style spec per line.")

    # Backward-compatible aliases.
    parser.add_argument("--normal_class", default="", help="Compatibility alias for one normal class.")
    parser.add_argument("--anomaly_class", default="", help="Compatibility alias for one anomaly class.")
    parser.add_argument("--train_anomaly_classes", nargs="*", default=[], help="Compatibility alias; added to --anomaly_classes.")
    parser.add_argument("--train_anomaly_classes_file", default="", help="Compatibility alias; added to --anomaly_classes_file.")

    parser.add_argument("--motion_key", default="auto")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--njoints", type=int, default=25)
    parser.add_argument("--nfeats", type=int, default=6)

    parser.add_argument("--test_fraction", type=float, default=0.20, help="80/20 split for normal + train-anomaly seen styles.")
    parser.add_argument("--val_fraction", type=float, default=0.10, help="Validation fraction from seen-style train pool.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify_by_actor", action="store_true")
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
    args.normal_classes = [norm_text(x) for x in args.normal_classes]
    args.anomaly_classes = [norm_text(x) for x in args.anomaly_classes]
    args.normal_class = norm_text(args.normal_class) if args.normal_class else ""
    args.anomaly_class = norm_text(args.anomaly_class) if args.anomaly_class else ""

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_json(vars(args), output_dir / "args.json")

    df = pd.read_csv(args.csv_path).copy()
    df["original_index"] = np.arange(len(df))
    required = [args.path_col, args.action_col, args.class_col]
    if args.category_col:
        required.append(args.category_col)
    if args.actor_col:
        required.append(args.actor_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    df["_action_norm"] = df[args.action_col].map(norm_text)
    df["_class_norm"] = df[args.class_col].map(norm_text)
    df["_category_norm"] = df[args.category_col].map(norm_text) if args.category_col else ""
    df["_actor_norm"] = df[args.actor_col].map(norm_text) if args.actor_col else ""

    missing_paths = [p for p in df[args.path_col].head(20).tolist() if not Path(str(p)).exists()]
    if missing_paths:
        print("[WARN] Some example motion files do not exist from this machine.")
        print(f"       First missing example: {missing_paths[0]}")

    train_df, val_df, test_sets, split_summary = make_transfer_splits(args, df)
    save_json(split_summary, output_dir / "split_summary.json")
    print("[INFO] Split summary:")
    print(json.dumps(split_summary, indent=2, sort_keys=True))

    train_df.to_csv(output_dir / "split_train.csv", index=False)
    val_df.to_csv(output_dir / "split_val.csv", index=False)
    for name, split_df in test_sets.items():
        split_df.to_csv(output_dir / f"split_{name}.csv", index=False)
    if "combined_test" in test_sets:
        test_sets["combined_test"].to_csv(output_dir / "split_test.csv", index=False)

    expected_shape = (args.num_frames, args.njoints, args.nfeats)
    train_ds = PerMoMotionDataset(train_df, args.path_col, args.action_col, args.class_col, "label", args.motion_key, expected_shape)
    val_ds = PerMoMotionDataset(val_df, args.path_col, args.action_col, args.class_col, "label", args.motion_key, expected_shape)
    test_datasets = {name: PerMoMotionDataset(split_df, args.path_col, args.action_col, args.class_col, "label", args.motion_key, expected_shape) for name, split_df in test_sets.items()}

    train_labels = train_df["label"].astype(int).to_numpy()
    train_batch_sampler = BalancedBinaryBatchSampler(train_labels, args.batch_size, seed=args.seed, drop_last=False)
    sampler_info = {"type": "BalancedBinaryBatchSampler", "batch_size": int(args.batch_size), "normal_per_batch": int(train_batch_sampler.n0), "anomaly_per_batch": int(train_batch_sampler.n1), "num_batches_per_epoch": int(len(train_batch_sampler)), "train_label_counts": {"normal_0": int((train_labels == 0).sum()), "anomaly_1": int((train_labels == 1).sum())}}
    save_json(sampler_info, output_dir / "train_sampler_info.json")

    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
    test_loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch) for name, ds in test_datasets.items()}

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
    prompt_classes = split_summary["classes_used_for_prompts"]
    normal_prompt_classes = split_summary.get("normal_prompt_classes", [args.normal_class])
    action_to_idx, class_to_idx, text_feats, prompt_info = build_prompt_cache(
        split_summary["actions_used_for_prompts"], prompt_classes, text_encoder, normal_prompt_classes,
        args.normal_prompt_template, args.anomaly_prompt_template, device,
    )
    save_json(prompt_info, output_dir / "prompts.json")
    torch.save({"action_to_idx": action_to_idx, "class_to_idx": class_to_idx, "text_feats": text_feats.detach().cpu(), "prompt_info": prompt_info, "normal_prompt_classes": normal_prompt_classes, "anomaly_prompt_classes": [c for c in prompt_classes if c not in set(normal_prompt_classes)]}, output_dir / "text_prompt_cache.pt")

    anomaly_prompt_classes = [c for c in prompt_classes if c not in set(normal_prompt_classes)]
    optimizer = torch.optim.AdamW([p for p in motion_encoder.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    best_val_auroc = -float("inf")
    best_epoch = -1
    best_threshold = 0.0
    epoch_records: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_batch_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(motion_encoder, train_loader, optimizer, device, action_to_idx, class_to_idx, text_feats, args.temperature, args.grad_clip, args.amp)
        train_eval_out = evaluate(motion_encoder, train_eval_loader, device, action_to_idx, class_to_idx, text_feats, normal_prompt_classes, anomaly_prompt_classes, args.temperature, compute_contrastive_loss=False)
        train_metrics_epoch = compute_binary_metrics(train_eval_out.y_true, train_eval_out.score, threshold=None, threshold_criterion=args.threshold_criterion)
        val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, class_to_idx, text_feats, normal_prompt_classes, anomaly_prompt_classes, args.temperature, compute_contrastive_loss=True)
        val_metrics = compute_binary_metrics(val_out.y_true, val_out.score, threshold=None, threshold_criterion=args.threshold_criterion)

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_out.loss, "train_auroc": train_metrics_epoch["auroc"], "val_auroc": val_metrics["auroc"], "train_auprc": train_metrics_epoch["auprc"], "val_auprc": val_metrics["auprc"], "train_f1": train_metrics_epoch["f1"], "val_f1": val_metrics["f1"], "train_balanced_accuracy": train_metrics_epoch["balanced_accuracy"], "val_balanced_accuracy": val_metrics["balanced_accuracy"], "train_threshold": train_metrics_epoch["threshold"], "val_threshold": val_metrics["threshold"], "seconds": time.time() - t0}
        epoch_records.append(record)
        save_training_curves(epoch_records, output_dir)
        print(f"[EPOCH {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_out.loss:.4f} train_auroc={train_metrics_epoch['auroc']:.4f} val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f} val_f1={val_metrics['f1']:.4f} thr={val_metrics['threshold']:.4f}")

        if np.isfinite(val_metrics["auroc"]) and val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            best_threshold = float(val_metrics["threshold"])
            torch.save({"epoch": epoch, "motion_encoder_state_dict": motion_encoder.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args), "action_to_idx": action_to_idx, "class_to_idx": class_to_idx, "prompt_info": prompt_info, "text_feats": text_feats.detach().cpu(), "best_val_metrics": val_metrics, "unfreeze_info": unfreeze_info}, ckpt_dir / "best_model.pt")
            save_predictions(val_out, output_dir / "val_predictions_best.csv", best_threshold)

    best_ckpt_path = ckpt_dir / "best_model.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        motion_encoder.load_state_dict(best_ckpt["motion_encoder_state_dict"], strict=True)
        best_threshold = float(best_ckpt["best_val_metrics"]["threshold"])
    else:
        print("[WARN] No best checkpoint saved. Testing final epoch model.")

    train_out = evaluate(motion_encoder, train_eval_loader, device, action_to_idx, class_to_idx, text_feats, normal_prompt_classes, anomaly_prompt_classes, args.temperature, True)
    val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, class_to_idx, text_feats, normal_prompt_classes, anomaly_prompt_classes, args.temperature, True)
    eval_outputs = {name: evaluate(motion_encoder, loader, device, action_to_idx, class_to_idx, text_feats, normal_prompt_classes, anomaly_prompt_classes, args.temperature, True) for name, loader in test_loaders.items()}

    train_metrics = compute_binary_metrics(train_out.y_true, train_out.score, threshold=best_threshold); train_metrics["loss"] = train_out.loss
    val_metrics_final = compute_binary_metrics(val_out.y_true, val_out.score, threshold=best_threshold); val_metrics_final["loss"] = val_out.loss

    save_predictions(train_out, output_dir / "train_predictions.csv", best_threshold)
    save_predictions(val_out, output_dir / "val_predictions.csv", best_threshold)
    save_embeddings(train_out, output_dir / "train_embeddings.npz", output_dir / "train_embeddings_metadata.csv", best_threshold)
    save_embeddings(val_out, output_dir / "val_embeddings.npz", output_dir / "val_embeddings_metadata.csv", best_threshold)

    test_metrics_by_name: Dict[str, Any] = {}
    output_files: Dict[str, str] = {"best_checkpoint": str(best_ckpt_path), "epoch_metrics": str(output_dir / "epoch_metrics.csv"), "metrics": str(output_dir / "metrics.json"), "training_history_npz": str(output_dir / "training_history.npz"), "loss_curves": str(output_dir / "loss_curves.png"), "validation_metrics_plot": str(output_dir / "validation_metrics.png"), "auroc_curves": str(output_dir / "auroc_curves.png"), "train_embeddings": str(output_dir / "train_embeddings.npz"), "val_embeddings": str(output_dir / "val_embeddings.npz")}
    for name, out in eval_outputs.items():
        m = compute_binary_metrics(out.y_true, out.score, threshold=best_threshold); m["loss"] = out.loss
        test_metrics_by_name[name] = m
        pred_path = output_dir / f"test_predictions_{name}.csv"
        emb_path = output_dir / f"test_embeddings_{name}.npz"
        meta_path = output_dir / f"test_embeddings_{name}_metadata.csv"
        save_predictions(out, pred_path, best_threshold)
        save_embeddings(out, emb_path, meta_path, best_threshold)
        output_files[f"test_predictions_{name}"] = str(pred_path)
        output_files[f"test_embeddings_{name}"] = str(emb_path)
        output_files[f"test_embeddings_{name}_metadata"] = str(meta_path)

    if "combined_test" in eval_outputs:
        save_predictions(eval_outputs["combined_test"], output_dir / "test_predictions.csv", best_threshold)
        save_embeddings(eval_outputs["combined_test"], output_dir / "test_embeddings.npz", output_dir / "test_embeddings_metadata.csv", best_threshold)
        output_files["test_predictions"] = str(output_dir / "test_predictions.csv")
        output_files["test_embeddings"] = str(output_dir / "test_embeddings.npz")
        output_files["test_embeddings_metadata"] = str(output_dir / "test_embeddings_metadata.csv")

    final_summary = {"best_epoch": best_epoch, "best_val_auroc_during_training": best_val_auroc, "threshold_selected_on_validation": best_threshold, "train_metrics": train_metrics, "val_metrics": val_metrics_final, "test_metrics": test_metrics_by_name.get("combined_test", {}), "test_metrics_by_split": test_metrics_by_name, "split_summary": split_summary, "prompt_templates": {"normal": args.normal_prompt_template, "anomaly": args.anomaly_prompt_template}, "normal_classes": split_summary["normal_classes"], "normal_prompt_classes_used_for_scoring": normal_prompt_classes, "anomaly_classes": split_summary["anomaly_classes"], "anomaly_prompt_classes_used_for_scoring": anomaly_prompt_classes, "output_files": output_files}
    save_json(final_summary, output_dir / "metrics.json")
    print("[DONE] Combined test metrics:")
    print(json.dumps(test_metrics_by_name.get("combined_test", {}), indent=2, sort_keys=True))
    print("[DONE] Unseen anomaly-style test metrics:")
    print(json.dumps(test_metrics_by_name.get("unseen_anomaly_style_test", {}), indent=2, sort_keys=True))
    print(f"[DONE] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

#
##!/usr/bin/env python3
#"""
#Fine-tune MotionCLIP for PerMo with one chosen normal style/condition and one chosen anomaly style/condition.
#Main changes vs. healthy-vs-all scripts:
#  1. You explicitly choose the metadata class column, normal class, and anomaly class.
#     Example: --class_col condition_label --normal_class Healthy --anomaly_class Head-aching
#  2. Only rows from those two classes are used. Labels are assigned internally:
#       normal_class  -> 0
#       anomaly_class -> 1
#  3. Default split is 80/20 train/test, stratified by action + label where possible.
#  4. Optional full hold-out of actions and/or actors:
#       --unseen_actions Walk Run
#       --unseen_actors actor_01 actor_02
#     Any matching rows are removed from train/val and evaluated as separate unseen test sets.
#  5. Saves the same important outputs as the condition-prompt unseen script:
#       split CSVs, metrics.json, predictions, embeddings, metadata CSVs,
#       training curves, prompt cache, best checkpoint, and seen/unseen test metrics.
#"""
#from __future__ import annotations
#import argparse
#import json
#import math
#import random
#import sys
#import time
#from dataclasses import dataclass
#from pathlib import Path
#from typing import Any, Dict, List, Optional, Sequence, Tuple
#import numpy as np
#import pandas as pd
#import torch
#import torch.nn.functional as F
#from torch import nn
#from torch.utils.data import DataLoader, Dataset, Sampler
## -----------------------------
## Utilities
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
#    def _default(o: Any):
#        if isinstance(o, (np.integer,)):
#            return int(o)
#        if isinstance(o, (np.floating,)):
#            return float(o)
#        if isinstance(o, np.ndarray):
#            return o.tolist()
#        return str(o)
#    with open(path, "w", encoding="utf-8") as f:
#        json.dump(obj, f, indent=2, sort_keys=True, default=_default)
#def norm_text(text: Any) -> str:
#    text = str(text).strip().lower().replace("_", " ").replace("-", " ")
#    return " ".join(text.split())
#def parse_list(values: Optional[Sequence[str]]) -> List[str]:
#    if not values:
#        return []
#    out: List[str] = []
#    for v in values:
#        if v is None:
#            continue
#        # Allow both: --unseen_actions Walk Run and --unseen_actions "Walk,Run"
#        for part in str(v).split(","):
#            part = norm_text(part)
#            if part:
#                out.append(part)
#    return sorted(set(out))
#def read_list_file(path: str) -> List[str]:
#    if not path:
#        return []
#    out: List[str] = []
#    with open(path, "r", encoding="utf-8") as f:
#        for raw in f:
#            line = raw.strip()
#            if not line or line.startswith("#"):
#                continue
#            out.append(norm_text(line.split(",")[0]))
#    return sorted(set(out))
#def save_training_curves(epoch_records: List[Dict[str, Any]], output_dir: str | Path) -> None:
#    if not epoch_records:
#        return
#    output_dir = Path(output_dir)
#    hist_df = pd.DataFrame(epoch_records)
#    hist_df.to_csv(output_dir / "epoch_metrics.csv", index=False)
#    np.savez(
#        output_dir / "training_history.npz",
#        **{c: hist_df[c].to_numpy() for c in hist_df.columns if pd.api.types.is_numeric_dtype(hist_df[c])},
#    )
#    try:
#        import matplotlib
#        matplotlib.use("Agg")
#        import matplotlib.pyplot as plt
#    except Exception as exc:
#        print(f"[WARN] Could not import matplotlib, skipping plots: {exc}")
#        return
#    def _plot(cols: Sequence[str], filename: str, ylabel: str) -> None:
#        cols = [c for c in cols if c in hist_df.columns]
#        if not cols:
#            return
#        fig, ax = plt.subplots(figsize=(8, 5))
#        for c in cols:
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
## -----------------------------
## Dataset
## -----------------------------
#class PerMoMotionDataset(Dataset):
#    def __init__(
#        self,
#        df: pd.DataFrame,
#        path_col: str,
#        action_col: str,
#        class_col: str,
#        label_col: str,
#        motion_key: str = "auto",
#        expected_shape: Tuple[int, int, int] = (60, 25, 6),
#    ) -> None:
#        self.df = df.reset_index(drop=True).copy()
#        self.path_col = path_col
#        self.action_col = action_col
#        self.class_col = class_col
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
#        preferred = ["motion", "motions", "x", "X", "data", "arr_0", "poses", "pose", "rot6d", "features", "joints", "input"]
#        for key in preferred:
#            if key in data.files and np.issubdtype(data[key].dtype, np.number):
#                return data[key]
#        numeric = [k for k in data.files if np.issubdtype(data[k].dtype, np.number)]
#        if not numeric:
#            raise KeyError(f"No numeric arrays found in {path}. Keys: {data.files}")
#        return data[numeric[0]]
#    def _standardize_motion_shape(self, arr: np.ndarray, path: str) -> np.ndarray:
#        arr = np.asarray(arr, dtype=np.float32)
#        while arr.ndim > 3 and 1 in arr.shape:
#            arr = np.squeeze(arr, axis=arr.shape.index(1))
#        if arr.ndim != 3:
#            raise ValueError(f"Expected 3D motion array [T,J,F], got shape {arr.shape} in {path}")
#        T, J, Fdim = self.expected_shape
#        if arr.shape == (T, J, Fdim):
#            return arr
#        candidates = {
#            (J, Fdim, T): (2, 0, 1),
#            (Fdim, T, J): (1, 2, 0),
#            (T, Fdim, J): (0, 2, 1),
#            (J, T, Fdim): (1, 0, 2),
#        }
#        if arr.shape in candidates:
#            return np.transpose(arr, candidates[arr.shape])
#        shape = list(arr.shape)
#        try:
#            return np.transpose(arr, (shape.index(T), shape.index(J), shape.index(Fdim)))
#        except ValueError as e:
#            raise ValueError(
#                f"Cannot convert motion shape {arr.shape} to expected {(T, J, Fdim)} for {path}. "
#                "Pass --num_frames/--njoints/--nfeats or adapt _standardize_motion_shape()."
#            ) from e
#    def __getitem__(self, idx: int) -> Dict[str, Any]:
#        row = self.df.iloc[idx]
#        path = str(row[self.path_col])
#        with np.load(path, allow_pickle=False) as data:
#            arr = self._pick_npz_array(data, path)
#        arr = self._standardize_motion_shape(arr, path)
#        return {
#            "motion": torch.from_numpy(arr),
#            "action": norm_text(row[self.action_col]),
#            "class_name": norm_text(row[self.class_col]),
#            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),
#            "path": path,
#            "row_index": int(row.get("original_index", idx)),
#        }
#def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#    return {
#        "motion": torch.stack([b["motion"] for b in batch], dim=0),
#        "action": [b["action"] for b in batch],
#        "class_name": [b["class_name"] for b in batch],
#        "label": torch.stack([b["label"] for b in batch], dim=0),
#        "path": [b["path"] for b in batch],
#        "row_index": [b["row_index"] for b in batch],
#    }
#class BalancedBinaryBatchSampler(Sampler[List[int]]):
#    def __init__(self, labels: Sequence[int], batch_size: int, seed: int = 42, drop_last: bool = False) -> None:
#        if batch_size < 2:
#            raise ValueError("BalancedBinaryBatchSampler requires batch_size >= 2.")
#        labels_np = np.asarray(labels).astype(int)
#        unique = set(labels_np.tolist())
#        if not unique.issubset({0, 1}):
#            raise ValueError(f"Expected binary labels 0/1, got {sorted(unique)}")
#        self.indices_by_class = {0: np.where(labels_np == 0)[0], 1: np.where(labels_np == 1)[0]}
#        if len(self.indices_by_class[0]) == 0 or len(self.indices_by_class[1]) == 0:
#            raise ValueError("Training split needs at least one normal and one anomaly sample.")
#        self.batch_size = int(batch_size)
#        self.n0 = self.batch_size // 2
#        self.n1 = self.batch_size - self.n0
#        self.seed = int(seed)
#        self.drop_last = bool(drop_last)
#        self.epoch = 0
#        self.num_batches = int(max(
#            math.ceil(len(self.indices_by_class[0]) / self.n0),
#            math.ceil(len(self.indices_by_class[1]) / self.n1),
#        ))
#    def __len__(self) -> int:
#        return self.num_batches
#    def set_epoch(self, epoch: int) -> None:
#        self.epoch = int(epoch)
#    def _sample(self, cls: int, n_total: int, rng: np.random.Generator) -> np.ndarray:
#        pool = self.indices_by_class[cls]
#        if n_total <= len(pool):
#            return rng.permutation(pool)[:n_total]
#        return np.concatenate([rng.permutation(pool), rng.choice(pool, size=n_total - len(pool), replace=True)])
#    def __iter__(self):
#        rng = np.random.default_rng(self.seed + self.epoch)
#        labels0 = self._sample(0, self.num_batches * self.n0, rng)
#        labels1 = self._sample(1, self.num_batches * self.n1, rng)
#        for batch_idx in range(self.num_batches):
#            b0 = labels0[batch_idx * self.n0:(batch_idx + 1) * self.n0]
#            b1 = labels1[batch_idx * self.n1:(batch_idx + 1) * self.n1]
#            batch = np.concatenate([b0, b1])
#            rng.shuffle(batch)
#            if self.drop_last and len(batch) < self.batch_size:
#                continue
#            yield batch.astype(int).tolist()
## -----------------------------
## Splitting
## -----------------------------
#def split_random_stratified_80_20(
#    df: pd.DataFrame,
#    stratify_cols: Sequence[str],
#    test_fraction: float,
#    val_fraction: float,
#    seed: int,
#) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#    """Returns train/val/test. Test is a disjoint 20% row holdout.
#    val_fraction is taken from the remaining train pool. Set --val_fraction 0 to skip validation split;
#    the script will then use the test threshold only if needed, but checkpoint selection is weaker.
#    """
#    rng = np.random.default_rng(seed)
#    train_parts, val_parts, test_parts = [], [], []
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
#        n_val = int(round(len(rem_idx) * val_fraction))
#        if len(rem_idx) >= 5 and val_fraction > 0:
#            n_val = max(1, n_val)
#        n_val = min(n_val, max(0, len(rem_idx) - 1))
#        val_idx = rem_idx[:n_val]
#        train_idx = rem_idx[n_val:]
#        train_parts.append(df.loc[train_idx])
#        val_parts.append(df.loc[val_idx])
#        test_parts.append(df.loc[test_idx])
#    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
#    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True) if val_parts else pd.DataFrame(columns=df.columns)
#    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
#    if len(val_df) == 0:
#        # Use a small deterministic slice from train as validation, because the training loop needs validation for checkpoint/threshold.
#        val_df = train_df.groupby("label", group_keys=False).sample(frac=0.10, random_state=seed + 3).reset_index(drop=True)
#        train_df = train_df.drop(index=val_df.index, errors="ignore").reset_index(drop=True)
#    return train_df, val_df, test_df
#def balance_binary_split(df: pd.DataFrame, label_col: str, seed: int, split_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#    counts_before = df[label_col].astype(int).value_counts().to_dict()
#    n0 = int((df[label_col].astype(int) == 0).sum())
#    n1 = int((df[label_col].astype(int) == 1).sum())
#    if n0 == 0 or n1 == 0:
#        return df.reset_index(drop=True), {
#            "split_name": split_name,
#            "balanced": False,
#            "reason": "split does not contain both labels",
#            "counts_before": {int(k): int(v) for k, v in counts_before.items()},
#        }
#    n = min(n0, n1)
#    d0 = df[df[label_col].astype(int) == 0].sample(n=n, random_state=seed)
#    d1 = df[df[label_col].astype(int) == 1].sample(n=n, random_state=seed + 1)
#    out = pd.concat([d0, d1]).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
#    return out, {
#        "split_name": split_name,
#        "balanced": True,
#        "seed": int(seed),
#        "n_before": int(len(df)),
#        "n_after": int(len(out)),
#        "counts_before": {int(k): int(v) for k, v in counts_before.items()},
#        "counts_after": {int(k): int(v) for k, v in out[label_col].astype(int).value_counts().to_dict().items()},
#        "n_kept_per_label": int(n),
#    }
#def make_splits(args: argparse.Namespace, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
#    unseen_actions = sorted(set(parse_list(args.unseen_actions) + read_list_file(args.unseen_actions_file)))
#    unseen_actors = sorted(set(parse_list(args.unseen_actors) + read_list_file(args.unseen_actors_file)))
#    if unseen_actions:
#        missing = sorted(set(unseen_actions) - set(df["_action_norm"].unique()))
#        if missing:
#            raise ValueError(f"Requested unseen actions not found: {missing}. Available: {sorted(df['_action_norm'].unique())}")
#    if unseen_actors:
#        if not args.actor_col:
#            raise ValueError("--unseen_actors was given, so --actor_col must also be given.")
#        missing = sorted(set(unseen_actors) - set(df["_actor_norm"].unique()))
#        if missing:
#            raise ValueError(f"Requested unseen actors not found: {missing}. Available examples: {sorted(df['_actor_norm'].unique())[:30]}")
#    unseen_action_df = df[df["_action_norm"].isin(unseen_actions)].copy() if unseen_actions else pd.DataFrame(columns=df.columns)
#    unseen_actor_df = df[df["_actor_norm"].isin(unseen_actors)].copy() if unseen_actors else pd.DataFrame(columns=df.columns)
#    # Anything matching either holdout condition is excluded from seen train/val/test.
#    holdout_mask = pd.Series(False, index=df.index)
#    if unseen_actions:
#        holdout_mask |= df["_action_norm"].isin(unseen_actions)
#    if unseen_actors:
#        holdout_mask |= df["_actor_norm"].isin(unseen_actors)
#    seen_pool = df[~holdout_mask].copy()
#    if len(seen_pool) == 0:
#        raise ValueError("No seen pool remains after action/actor holdout.")
#    if not {0, 1}.issubset(set(seen_pool["label"].astype(int).unique())):
#        raise ValueError("Seen train pool must contain both normal and anomaly labels.")
#    stratify_cols = ["_action_norm", "label"]
#    if args.actor_col and args.stratify_by_actor:
#        stratify_cols = ["_action_norm", "_actor_norm", "label"]
#    train_df, val_df, seen_test_df = split_random_stratified_80_20(
#        seen_pool,
#        stratify_cols=stratify_cols,
#        test_fraction=args.test_fraction,
#        val_fraction=args.val_fraction,
#        seed=args.seed,
#    )
#    test_sets: Dict[str, pd.DataFrame] = {"seen_test": seen_test_df}
#    if unseen_actions:
#        test_sets["unseen_action_test"] = unseen_action_df.reset_index(drop=True)
#    if unseen_actors:
#        test_sets["unseen_actor_test"] = unseen_actor_df.reset_index(drop=True)
#    if unseen_actions and unseen_actors:
#        both = df[df["_action_norm"].isin(unseen_actions) & df["_actor_norm"].isin(unseen_actors)].copy()
#        if len(both):
#            test_sets["unseen_action_actor_test"] = both.reset_index(drop=True)
#    combined = pd.concat(list(test_sets.values()), axis=0).drop_duplicates(subset=["original_index"]).sample(frac=1.0, random_state=args.seed + 50).reset_index(drop=True)
#    test_sets["combined_test"] = combined
#    # Leakage checks
#    train_actions = set(train_df["_action_norm"].unique())
#    val_actions = set(val_df["_action_norm"].unique())
#    train_actors = set(train_df["_actor_norm"].unique()) if args.actor_col else set()
#    val_actors = set(val_df["_actor_norm"].unique()) if args.actor_col else set()
#    leakage = {
#        "unseen_actions_in_train_val": sorted(set(unseen_actions) & (train_actions | val_actions)),
#        "unseen_actors_in_train_val": sorted(set(unseen_actors) & (train_actors | val_actors)),
#    }
#    if leakage["unseen_actions_in_train_val"] or leakage["unseen_actors_in_train_val"]:
#        raise RuntimeError(f"Holdout leakage detected: {leakage}")
#    balance_info = {"enabled": bool(args.balance_test_sets)}
#    if args.balance_test_sets:
#        balanced_sets: Dict[str, pd.DataFrame] = {}
#        for name, split_df in test_sets.items():
#            if len(split_df) == 0:
#                balanced_sets[name] = split_df
#                continue
#            bdf, info = balance_binary_split(split_df, "label", args.seed + hash(name) % 10000, name)
#            balanced_sets[name] = bdf
#            balance_info[name] = info
#        test_sets = balanced_sets
#    summary = {
#        "split_type": "specific_normal_anomaly_with_optional_action_actor_holdout",
#        "normal_class": args.normal_class,
#        "anomaly_class": args.anomaly_class,
#        "class_col": args.class_col,
#        "test_fraction": float(args.test_fraction),
#        "val_fraction_from_train_pool": float(args.val_fraction),
#        "train": int(len(train_df)),
#        "val": int(len(val_df)),
#        "test_sets": {name: int(len(split_df)) for name, split_df in test_sets.items()},
#        "train_label_counts": train_df["label"].value_counts().to_dict(),
#        "val_label_counts": val_df["label"].value_counts().to_dict(),
#        "test_label_counts": {name: split_df["label"].value_counts().to_dict() for name, split_df in test_sets.items()},
#        "heldout_unseen_actions": unseen_actions,
#        "heldout_unseen_actors": unseen_actors,
#        "leakage_check": leakage,
#        "test_balance_info": balance_info,
#        "seen_actions_train_val": sorted((set(train_df["_action_norm"].unique()) | set(val_df["_action_norm"].unique()))),
#        "seen_actors_train_val": sorted((set(train_df["_actor_norm"].unique()) | set(val_df["_actor_norm"].unique()))) if args.actor_col else [],
#    }
#    return train_df, val_df, test_sets, summary
## -----------------------------
## MotionCLIP loading/freezing
## -----------------------------
#def build_motionclip_encoder(checkpoint_path: str, device: torch.device) -> nn.Module:
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
#    encoder_state = {k[len("encoder."):]: v for k, v in ckpt.items() if k.startswith("encoder.")}
#    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
#    if unexpected:
#        raise RuntimeError(f"Unexpected encoder keys: {unexpected}")
#    if missing:
#        print("[WARN] Missing encoder keys:", missing)
#    return encoder.to(device)
#def freeze_encoder_except_last_layers(encoder: nn.Module, num_trainable_blocks: int = 2) -> Dict[str, Any]:
#    for p in encoder.parameters():
#        p.requires_grad = False
#    unfrozen_layer_indices: List[int] = []
#    if hasattr(encoder, "seqTransEncoder") and hasattr(encoder.seqTransEncoder, "layers"):
#        layers = encoder.seqTransEncoder.layers
#        n = min(num_trainable_blocks, len(layers))
#        start = len(layers) - n
#        for i, layer in enumerate(layers[start:], start=start):
#            for p in layer.parameters():
#                p.requires_grad = True
#            unfrozen_layer_indices.append(i)
#        if getattr(encoder.seqTransEncoder, "norm", None) is not None:
#            for p in encoder.seqTransEncoder.norm.parameters():
#                p.requires_grad = True
#    else:
#        print("[WARN] Could not find encoder.seqTransEncoder.layers; encoder may remain frozen.")
#    trainable_names = [name for name, p in encoder.named_parameters() if p.requires_grad]
#    return {
#        "num_trainable_blocks_requested": int(num_trainable_blocks),
#        "unfrozen_layer_indices": unfrozen_layer_indices,
#        "num_trainable_params": int(sum(p.numel() for p in encoder.parameters() if p.requires_grad)),
#        "num_total_params": int(sum(p.numel() for p in encoder.parameters())),
#        "trainable_param_names_first_100": trainable_names[:100],
#    }
#def encode_motion_auto(model: nn.Module, motion: torch.Tensor) -> torch.Tensor:
#    motion = motion.float()
#    x = motion.permute(0, 2, 3, 1).contiguous()  # [B,25,6,60]
#    B, T = motion.shape[0], motion.shape[1]
#    lengths = torch.full((B,), T, dtype=torch.long, device=motion.device)
#    mask = torch.arange(T, device=motion.device).unsqueeze(0) < lengths.unsqueeze(1)
#    out = model({"x": x, "y": torch.zeros(B, dtype=torch.long, device=motion.device), "lengths": lengths, "mask": mask})
#    if not isinstance(out, dict) or "mu" not in out:
#        raise RuntimeError("Expected MotionCLIP encoder output dict with key 'mu'.")
#    return out["mu"]
## -----------------------------
## Text prompts for unseen anomaly-style transfer
## -----------------------------
#class FrozenCLIPTextEncoder:
#    def __init__(self, clip_model_name: str, device: torch.device):
#        try:
#            import clip
#        except ImportError as e:
#            raise ImportError("Could not import `clip`. Install OpenAI CLIP first.") from e
#        self.clip = clip
#        self.model, _ = clip.load(clip_model_name, device=device)
#        self.model = self.model.float().eval()
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
#            feats.append(F.normalize(f, dim=-1).cpu())
#        return torch.cat(feats, dim=0)
#def build_prompt_cache(
#    actions: Sequence[str],
#    class_names: Sequence[str],
#    text_encoder: FrozenCLIPTextEncoder,
#    normal_prompt_classes: Sequence[str],
#    normal_prompt_template: str,
#    anomaly_prompt_template: str,
#    device: torch.device,
#) -> Tuple[Dict[str, int], Dict[str, int], torch.Tensor, Dict[str, Dict[str, str]]]:
#    """Build text prompts for all action/style pairs.
#    Any class in normal_prompt_classes is formatted with normal_prompt_template.
#    All other classes are formatted with anomaly_prompt_template.
#    This allows cross-parent tests such as train normal=Healthy and test normal=Happy.
#    """
#    actions = sorted({norm_text(a) for a in actions})
#    class_names = sorted({norm_text(c) for c in class_names})
#    normal_prompt_classes = sorted({norm_text(c) for c in normal_prompt_classes})
#    for c in normal_prompt_classes:
#        if c not in class_names:
#            class_names.append(c)
#    # Put normal prompt classes first for readability.
#    class_names = normal_prompt_classes + sorted(c for c in set(class_names) if c not in set(normal_prompt_classes))
#    action_to_idx = {a: i for i, a in enumerate(actions)}
#    class_to_idx = {c: i for i, c in enumerate(class_names)}
#    texts: List[str] = []
#    prompt_info: Dict[str, Dict[str, str]] = {}
#    first_normal = normal_prompt_classes[0] if normal_prompt_classes else "normal"
#    for action in actions:
#        prompt_info[action] = {}
#        for cls in class_names:
#            if cls in normal_prompt_classes:
#                prompt = normal_prompt_template.format(action=action, normal_class=cls, class_name=cls)
#            else:
#                prompt = anomaly_prompt_template.format(action=action, anomaly_class=cls, class_name=cls, normal_class=first_normal)
#            prompt_info[action][cls] = prompt
#            texts.append(prompt)
#    text_feats = text_encoder.encode(texts).reshape(len(actions), len(class_names), -1).to(device)
#    return action_to_idx, class_to_idx, text_feats, prompt_info
## -----------------------------
## Split logic for train-anomaly vs test-only anomaly styles
## -----------------------------
#def make_transfer_splits(args: argparse.Namespace, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
#    """Multi-style normal/anomaly split.
#    Use this for experiments like:
#      Normal:  Healthy Neutral Happy Elegant
#      Anomaly: Text-necked Sad Crowded Muddy-floor
#    All selected normal classes get label 0.
#    All selected anomaly classes get label 1.
#    Train/val/seen_test are made from selected rows after optional action/actor holdout.
#    Optional unseen action/actor rows are kept out of train/val and reported as separate test splits.
#    """
#    normal_classes = sorted(set(parse_list(args.normal_classes) + read_list_file(args.normal_classes_file)))
#    anomaly_classes = sorted(set(parse_list(args.anomaly_classes) + read_list_file(args.anomaly_classes_file)))
#    # Backwards-compatible aliases.
#    if getattr(args, "normal_class", ""):
#        normal_classes.append(args.normal_class)
#    if getattr(args, "anomaly_class", ""):
#        anomaly_classes.append(args.anomaly_class)
#    normal_classes = sorted(set(normal_classes))
#    anomaly_classes = sorted(set(anomaly_classes))
#    unseen_actions = sorted(set(parse_list(args.unseen_actions) + read_list_file(args.unseen_actions_file)))
#    unseen_actors = sorted(set(parse_list(args.unseen_actors) + read_list_file(args.unseen_actors_file)))
#    if not normal_classes:
#        raise ValueError("Give at least one --normal_classes value, e.g. Healthy Neutral Happy Elegant.")
#    if not anomaly_classes:
#        raise ValueError("Give at least one --anomaly_classes value, e.g. Text-necked Sad Crowded Muddy-floor.")
#    available_classes = set(df["_class_norm"].unique().tolist())
#    requested_classes = set(normal_classes) | set(anomaly_classes)
#    missing = sorted(requested_classes - available_classes)
#    if missing:
#        raise ValueError(f"Requested style classes not found in {args.class_col}: {missing}. Available: {sorted(available_classes)}")
#    overlap = sorted(set(normal_classes) & set(anomaly_classes))
#    if overlap:
#        raise ValueError(f"Classes cannot be both normal and anomaly: {overlap}")
#    if unseen_actions:
#        missing_actions = sorted(set(unseen_actions) - set(df["_action_norm"].unique()))
#        if missing_actions:
#            raise ValueError(f"Requested unseen actions not found: {missing_actions}. Available: {sorted(df['_action_norm'].unique())}")
#    if unseen_actors:
#        if not args.actor_col:
#            raise ValueError("--unseen_actors was given, so --actor_col must also be given.")
#        missing_actors = sorted(set(unseen_actors) - set(df["_actor_norm"].unique()))
#        if missing_actors:
#            raise ValueError(f"Requested unseen actors not found: {missing_actors}. Available examples: {sorted(df['_actor_norm'].unique())[:30]}")
#    df = df[df["_class_norm"].isin(requested_classes)].copy()
#    df["label"] = df["_class_norm"].isin(anomaly_classes).astype(int)
#    df["style_role"] = np.where(df["_class_norm"].isin(normal_classes), "normal_class", "anomaly_class")
#    action_holdout_mask = df["_action_norm"].isin(unseen_actions) if unseen_actions else pd.Series(False, index=df.index)
#    actor_holdout_mask = df["_actor_norm"].isin(unseen_actors) if unseen_actors else pd.Series(False, index=df.index)
#    optional_holdout_mask = action_holdout_mask | actor_holdout_mask
#    seen_pool = df[~optional_holdout_mask].copy()
#    if not {0, 1}.issubset(set(seen_pool["label"].astype(int).unique())):
#        raise ValueError("Training pool must contain at least one normal class and one anomaly class after optional holdouts.")
#    stratify_cols = ["_action_norm", "_class_norm", "label"]
#    if args.actor_col and args.stratify_by_actor:
#        stratify_cols = ["_action_norm", "_actor_norm", "_class_norm", "label"]
#    train_df, val_df, seen_test_df = split_random_stratified_80_20(
#        seen_pool,
#        stratify_cols=stratify_cols,
#        test_fraction=args.test_fraction,
#        val_fraction=args.val_fraction,
#        seed=args.seed,
#    )
#    test_sets: Dict[str, pd.DataFrame] = {"seen_test": seen_test_df.reset_index(drop=True)}
#    if unseen_actions:
#        test_sets["unseen_action_test"] = df[action_holdout_mask].copy().reset_index(drop=True)
#    if unseen_actors:
#        test_sets["unseen_actor_test"] = df[actor_holdout_mask].copy().reset_index(drop=True)
#    if unseen_actions and unseen_actors:
#        both = df[action_holdout_mask & actor_holdout_mask].copy()
#        if len(both):
#            test_sets["unseen_action_actor_test"] = both.reset_index(drop=True)
#    combined = pd.concat(list(test_sets.values()), axis=0).drop_duplicates(subset=["original_index"])
#    combined = combined.sample(frac=1.0, random_state=args.seed + 50).reset_index(drop=True)
#    test_sets["combined_test"] = combined
#    leakage = {
#        "unseen_actions_in_train_val": sorted(set(unseen_actions) & (set(train_df["_action_norm"].unique()) | set(val_df["_action_norm"].unique()))),
#        "unseen_actors_in_train_val": sorted(set(unseen_actors) & (set(train_df["_actor_norm"].unique()) | set(val_df["_actor_norm"].unique()))),
#    }
#    if any(leakage.values()):
#        raise RuntimeError(f"Leakage detected: {leakage}")
#    balance_info: Dict[str, Any] = {"enabled": bool(args.balance_test_sets)}
#    if args.balance_test_sets:
#        balanced: Dict[str, pd.DataFrame] = {}
#        for name, split_df in test_sets.items():
#            if len(split_df) == 0:
#                balanced[name] = split_df
#                continue
#            bdf, info = balance_binary_split(split_df, "label", args.seed + (abs(hash(name)) % 10000), name)
#            balanced[name] = bdf
#            balance_info[name] = info
#        test_sets = balanced
#    summary = {
#        "split_type": "multi_style_normal_vs_multi_style_anomaly",
#        "class_col": args.class_col,
#        "normal_classes": normal_classes,
#        "anomaly_classes": anomaly_classes,
#        "test_fraction_for_seen_styles": float(args.test_fraction),
#        "val_fraction_from_seen_train_pool": float(args.val_fraction),
#        "train": int(len(train_df)),
#        "val": int(len(val_df)),
#        "test_sets": {name: int(len(split_df)) for name, split_df in test_sets.items()},
#        "train_label_counts": train_df["label"].value_counts().to_dict(),
#        "val_label_counts": val_df["label"].value_counts().to_dict(),
#        "test_label_counts": {name: split_df["label"].value_counts().to_dict() for name, split_df in test_sets.items()},
#        "train_class_counts": train_df["_class_norm"].value_counts().to_dict(),
#        "val_class_counts": val_df["_class_norm"].value_counts().to_dict(),
#        "test_class_counts": {name: split_df["_class_norm"].value_counts().to_dict() for name, split_df in test_sets.items()},
#        "heldout_unseen_actions": unseen_actions,
#        "heldout_unseen_actors": unseen_actors,
#        "leakage_check": leakage,
#        "test_balance_info": balance_info,
#        "classes_used_for_prompts": normal_classes + anomaly_classes,
#        "normal_prompt_classes": normal_classes,
#        "actions_used_for_prompts": sorted(df["_action_norm"].unique().tolist()),
#    }
#    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_sets, summary
## -----------------------------
## Metrics
## -----------------------------
#def binary_auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
#    y_true = np.asarray(y_true).astype(int)
#    scores = np.asarray(scores).astype(float)
#    pos = y_true == 1
#    neg = y_true == 0
#    n_pos = pos.sum(); n_neg = neg.sum()
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
#    total_pos = y.sum()
#    if total_pos == 0:
#        return float("nan")
#    tp = np.cumsum(y)
#    precision = tp / (np.arange(len(y)) + 1)
#    return float((precision * y).sum() / total_pos)
#def classification_metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
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
#    return {"threshold": float(threshold), "accuracy": float(accuracy), "balanced_accuracy": float(0.5 * (recall + specificity)), "precision": float(precision), "recall": float(recall), "specificity": float(specificity), "f1": float(f1), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
#def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, criterion: str = "f1") -> Tuple[float, Dict[str, Any]]:
#    scores = np.asarray(scores, dtype=float)
#    if len(scores) == 0:
#        return 0.0, {}
#    candidates = np.unique(scores)
#    if len(candidates) > 1000:
#        candidates = np.quantile(scores, np.linspace(0, 1, 1000))
#    best_thr, best_metrics, best_value = float(candidates[0]), None, -float("inf")
#    for thr in candidates:
#        m = classification_metrics_at_threshold(y_true, scores, float(thr))
#        value = m.get(criterion, m["f1"])
#        if value > best_value:
#            best_thr, best_metrics, best_value = float(thr), m, value
#    assert best_metrics is not None
#    return best_thr, best_metrics
#def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: Optional[float] = None, threshold_criterion: str = "f1") -> Dict[str, Any]:
#    y_true = np.asarray(y_true).astype(int)
#    scores = np.asarray(scores).astype(float)
#    try:
#        from sklearn.metrics import average_precision_score, roc_auc_score
#        auroc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
#        auprc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
#    except Exception:
#        auroc = binary_auc_rank(y_true, scores)
#        auprc = average_precision_fallback(y_true, scores)
#    if threshold is None:
#        threshold, threshold_metrics = find_best_threshold(y_true, scores, threshold_criterion)
#        source = f"best_{threshold_criterion}_on_this_split"
#    else:
#        threshold_metrics = classification_metrics_at_threshold(y_true, scores, threshold)
#        source = "provided"
#    return {"auroc": auroc, "auprc": auprc, "n_samples": int(len(y_true)), "n_normal": int((y_true == 0).sum()), "n_anomaly": int((y_true == 1).sum()), "score_mean": float(np.mean(scores)) if len(scores) else float("nan"), "score_std": float(np.std(scores)) if len(scores) else float("nan"), "threshold_source": source, **threshold_metrics}
## -----------------------------
## Train/evaluate
## -----------------------------
#@dataclass
#class EvalOutput:
#    loss: float
#    y_true: np.ndarray
#    score: np.ndarray
#    prob_anomaly: np.ndarray
#    logit_normal: np.ndarray
#    logit_anomaly: np.ndarray
#    embeddings: np.ndarray
#    paths: List[str]
#    actions: List[str]
#    class_names: List[str]
#    max_normal_prompt_class: List[str]
#    max_anomaly_prompt_class: List[str]
#    row_indices: List[int]
#def logits_from_motion_and_prompts(
#    motion_encoder: nn.Module,
#    motion: torch.Tensor,
#    actions: Sequence[str],
#    action_to_idx: Dict[str, int],
#    text_feats: torch.Tensor,
#    temperature: float,
#) -> Tuple[torch.Tensor, torch.Tensor]:
#    z = F.normalize(encode_motion_auto(motion_encoder, motion).float(), dim=-1)
#    idx = torch.tensor([action_to_idx[norm_text(a)] for a in actions], dtype=torch.long, device=motion.device)
#    prompts = text_feats[idx]  # [B,C,D]
#    if z.shape[-1] != prompts.shape[-1]:
#        raise RuntimeError(f"Motion embedding dim {z.shape[-1]} does not match text dim {prompts.shape[-1]}.")
#    logits = torch.bmm(prompts, z.unsqueeze(-1)).squeeze(-1) / temperature
#    return logits, z
#def target_text_embeddings_and_group_ids(
#    actions: Sequence[str],
#    class_names: Sequence[str],
#    action_to_idx: Dict[str, int],
#    class_to_idx: Dict[str, int],
#    text_feats: torch.Tensor,
#    device: torch.device,
#) -> Tuple[torch.Tensor, torch.Tensor]:
#    action_idx = torch.tensor([action_to_idx[norm_text(a)] for a in actions], dtype=torch.long, device=device)
#    class_idx = torch.tensor([class_to_idx[norm_text(c)] for c in class_names], dtype=torch.long, device=device)
#    target_text = text_feats[action_idx, class_idx]
#    group_ids = action_idx * len(class_to_idx) + class_idx
#    return target_text, group_ids
#def symmetric_motion_text_contrastive_loss(motion_z: torch.Tensor, text_z: torch.Tensor, group_ids: torch.Tensor, temperature: float) -> torch.Tensor:
#    motion_z = F.normalize(motion_z.float(), dim=-1)
#    text_z = F.normalize(text_z.float(), dim=-1)
#    logits = motion_z @ text_z.t() / temperature
#    pos = group_ids[:, None].eq(group_ids[None, :]).float()
#    log_prob_m2t = logits - torch.logsumexp(logits, dim=1, keepdim=True)
#    loss_m2t = -(pos * log_prob_m2t).sum(dim=1) / pos.sum(dim=1).clamp_min(1.0)
#    log_prob_t2m = logits.t() - torch.logsumexp(logits.t(), dim=1, keepdim=True)
#    loss_t2m = -(pos.t() * log_prob_t2m).sum(dim=1) / pos.t().sum(dim=1).clamp_min(1.0)
#    return 0.5 * (loss_m2t.mean() + loss_t2m.mean())
#def train_one_epoch(
#    motion_encoder: nn.Module,
#    loader: DataLoader,
#    optimizer: torch.optim.Optimizer,
#    device: torch.device,
#    action_to_idx: Dict[str, int],
#    class_to_idx: Dict[str, int],
#    text_feats: torch.Tensor,
#    temperature: float,
#    grad_clip: float,
#    use_amp: bool,
#) -> float:
#    motion_encoder.train()
#    total_loss, total_n = 0.0, 0
#    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#    for batch in loader:
#        motion = batch["motion"].to(device, non_blocking=True)
#        labels = batch["label"].to(device, non_blocking=True)
#        optimizer.zero_grad(set_to_none=True)
#        with torch.cuda.amp.autocast(enabled=use_amp):
#            z = encode_motion_auto(motion_encoder, motion)
#            target_text, group_ids = target_text_embeddings_and_group_ids(batch["action"], batch["class_name"], action_to_idx, class_to_idx, text_feats, motion.device)
#            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
#        scaler.scale(loss).backward()
#        if grad_clip > 0:
#            scaler.unscale_(optimizer)
#            torch.nn.utils.clip_grad_norm_([p for p in motion_encoder.parameters() if p.requires_grad], grad_clip)
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
#    class_to_idx: Dict[str, int],
#    text_feats: torch.Tensor,
#    normal_prompt_classes: Sequence[str],
#    anomaly_prompt_classes: Sequence[str],
#    temperature: float,
#    compute_contrastive_loss: bool = True,
#) -> EvalOutput:
#    motion_encoder.eval()
#    losses: List[float] = []
#    y_true: List[int] = []
#    scores: List[float] = []
#    probs: List[float] = []
#    logit_normal: List[float] = []
#    logit_anomaly: List[float] = []
#    embeddings: List[np.ndarray] = []
#    paths: List[str] = []
#    actions: List[str] = []
#    class_names: List[str] = []
#    max_norm_classes: List[str] = []
#    max_anom_classes: List[str] = []
#    row_indices: List[int] = []
#    normal_prompt_classes = [norm_text(c) for c in normal_prompt_classes]
#    anomaly_prompt_classes = [norm_text(c) for c in anomaly_prompt_classes]
#    normal_indices = [class_to_idx[c] for c in normal_prompt_classes if c in class_to_idx]
#    anomaly_indices = [class_to_idx[c] for c in anomaly_prompt_classes if c in class_to_idx and c not in set(normal_prompt_classes)]
#    if not normal_indices:
#        raise ValueError("No normal prompt classes available for scoring.")
#    if not anomaly_indices:
#        raise ValueError("No anomaly prompt classes available for scoring.")
#    normal_indices_t = torch.tensor(normal_indices, dtype=torch.long, device=device)
#    anomaly_indices_t = torch.tensor(anomaly_indices, dtype=torch.long, device=device)
#    idx_to_class = {v: k for k, v in class_to_idx.items()}
#    for batch in loader:
#        motion = batch["motion"].to(device, non_blocking=True)
#        labels = batch["label"].to(device, non_blocking=True)
#        logits, z = logits_from_motion_and_prompts(motion_encoder, motion, batch["action"], action_to_idx, text_feats, temperature)
#        if compute_contrastive_loss:
#            target_text, group_ids = target_text_embeddings_and_group_ids(batch["action"], batch["class_name"], action_to_idx, class_to_idx, text_feats, motion.device)
#            loss = symmetric_motion_text_contrastive_loss(z, target_text, group_ids, temperature)
#            losses.append(float(loss.detach().cpu()) * labels.numel())
#        norm_logits_all = logits.index_select(1, normal_indices_t)
#        normal_max_logits, normal_argmax = norm_logits_all.max(dim=1)
#        anom_logits_all = logits.index_select(1, anomaly_indices_t)
#        anom_max_logits, anom_argmax = anom_logits_all.max(dim=1)
#        chosen_norm_indices = normal_indices_t[normal_argmax].detach().cpu().numpy().astype(int).tolist()
#        chosen_anom_indices = anomaly_indices_t[anom_argmax].detach().cpu().numpy().astype(int).tolist()
#        chosen_norm_classes = [idx_to_class[i] for i in chosen_norm_indices]
#        chosen_anom_classes = [idx_to_class[i] for i in chosen_anom_indices]
#        score = anom_max_logits - normal_max_logits
#        two_class_logits = torch.stack([normal_max_logits, anom_max_logits], dim=1)
#        soft = torch.softmax(two_class_logits, dim=-1)
#        embeddings.append(z.detach().cpu().numpy().astype(np.float32))
#        y_true.extend(labels.cpu().numpy().astype(int).tolist())
#        scores.extend(score.cpu().numpy().astype(float).tolist())
#        probs.extend(soft[:, 1].cpu().numpy().astype(float).tolist())
#        logit_normal.extend(normal_max_logits.cpu().numpy().astype(float).tolist())
#        logit_anomaly.extend(anom_max_logits.cpu().numpy().astype(float).tolist())
#        paths.extend(batch["path"])
#        actions.extend(batch["action"])
#        class_names.extend(batch["class_name"])
#        max_norm_classes.extend(chosen_norm_classes)
#        max_anom_classes.extend(chosen_anom_classes)
#        row_indices.extend(batch["row_index"])
#    total_n = max(1, len(y_true))
#    return EvalOutput(
#        loss=sum(losses) / total_n if losses else float("nan"),
#        y_true=np.asarray(y_true, dtype=int),
#        score=np.asarray(scores, dtype=float),
#        prob_anomaly=np.asarray(probs, dtype=float),
#        logit_normal=np.asarray(logit_normal, dtype=float),
#        logit_anomaly=np.asarray(logit_anomaly, dtype=float),
#        embeddings=np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 512), dtype=np.float32),
#        paths=paths,
#        actions=actions,
#        class_names=class_names,
#        max_normal_prompt_class=max_norm_classes,
#        max_anomaly_prompt_class=max_anom_classes,
#        row_indices=row_indices,
#    )
#def save_predictions(eval_out: EvalOutput, path: str | Path, threshold: float) -> None:
#    pred = (eval_out.score >= threshold).astype(int)
#    pd.DataFrame({
#        "row_index": eval_out.row_indices,
#        "motion_path": eval_out.paths,
#        "action": eval_out.actions,
#        "class_name": eval_out.class_names,
#        "y_true_is_anomaly": eval_out.y_true,
#        "anomaly_score_max_anomaly_prompt_minus_normal": eval_out.score,
#        "anomaly_score_flawed_minus_healthy": eval_out.score,
#        "prob_anomaly": eval_out.prob_anomaly,
#        "logit_normal": eval_out.logit_normal,
#        "max_normal_prompt_class": eval_out.max_normal_prompt_class,
#        "logit_max_anomaly_prompt": eval_out.logit_anomaly,
#        "logit_healthy": eval_out.logit_normal,
#        "logit_flawed": eval_out.logit_anomaly,
#        "max_anomaly_prompt_class": eval_out.max_anomaly_prompt_class,
#        "pred_is_anomaly": pred,
#    }).to_csv(path, index=False)
#def save_embeddings(eval_out: EvalOutput, npz_path: str | Path, metadata_csv_path: str | Path, threshold: float) -> None:
#    pred = (eval_out.score >= threshold).astype(int)
#    np.savez_compressed(
#        npz_path,
#        embeddings=eval_out.embeddings.astype(np.float32),
#        y_true=eval_out.y_true.astype(np.int64),
#        score=eval_out.score.astype(np.float32),
#        prob_anomaly=eval_out.prob_anomaly.astype(np.float32),
#        logit_normal=eval_out.logit_normal.astype(np.float32),
#        logit_max_anomaly_prompt=eval_out.logit_anomaly.astype(np.float32),
#        row_index=np.asarray(eval_out.row_indices, dtype=np.int64),
#        pred_is_anomaly=pred.astype(np.int64),
#    )
#    pd.DataFrame({
#        "embedding_index": np.arange(len(eval_out.row_indices), dtype=int),
#        "row_index": eval_out.row_indices,
#        "motion_path": eval_out.paths,
#        "action": eval_out.actions,
#        "class_name": eval_out.class_names,
#        "y_true_is_anomaly": eval_out.y_true,
#        "anomaly_score_max_anomaly_prompt_minus_normal": eval_out.score,
#        "anomaly_score_flawed_minus_healthy": eval_out.score,
#        "prob_anomaly": eval_out.prob_anomaly,
#        "logit_normal": eval_out.logit_normal,
#        "max_normal_prompt_class": eval_out.max_normal_prompt_class,
#        "logit_max_anomaly_prompt": eval_out.logit_anomaly,
#        "max_anomaly_prompt_class": eval_out.max_anomaly_prompt_class,
#        "pred_is_anomaly": pred,
#    }).to_csv(metadata_csv_path, index=False)
## -----------------------------
## Main
## -----------------------------
#def main() -> None:
#    parser = argparse.ArgumentParser(description="Fine-tune MotionCLIP for unseen anomaly-style transfer on PerMo.")
#    parser.add_argument("--csv_path", required=True)
#    parser.add_argument("--output_dir", required=True)
#    parser.add_argument("--path_col", default="motion_path")
#    parser.add_argument("--action_col", default="action_label")
#    parser.add_argument("--class_col", default="condition_label", help="Column containing the style/condition class to use.")
#    parser.add_argument("--actor_col", default="", help="Optional actor column, required only for --unseen_actors.")
#    parser.add_argument("--normal_class", required=True, help="Train normal style/class, e.g. Healthy.")
#    parser.add_argument("--test_normal_classes", nargs="*", default=[], help="Normal classes used only in test, e.g. Happy for Condition -> Emotion transfer. Defaults to --normal_class.")
#    parser.add_argument("--test_normal_classes_file", default="")
#    parser.add_argument("--train_anomaly_classes", nargs="*", default=[], help="Anomaly styles/classes allowed in train/val, e.g. Drunken Exhausted.")
#    parser.add_argument("--train_anomaly_classes_file", default="")
#    parser.add_argument("--test_anomaly_classes", nargs="*", default=[], help="Anomaly styles/classes kept completely test-only, e.g. Head-aching Text-necked.")
#    parser.add_argument("--test_anomaly_classes_file", default="")
#    # Backward-compatible alias: if user gives --anomaly_class, treat it as train anomaly.
#    parser.add_argument("--anomaly_class", default="", help="Compatibility alias for a single train anomaly class.")
#    parser.add_argument("--motion_key", default="auto")
#    parser.add_argument("--num_frames", type=int, default=60)
#    parser.add_argument("--njoints", type=int, default=25)
#    parser.add_argument("--nfeats", type=int, default=6)
#    parser.add_argument("--test_fraction", type=float, default=0.20, help="80/20 split for normal + train-anomaly seen styles.")
#    parser.add_argument("--val_fraction", type=float, default=0.10, help="Validation fraction from seen-style train pool.")
#    parser.add_argument("--seed", type=int, default=42)
#    parser.add_argument("--stratify_by_actor", action="store_true")
#    parser.add_argument("--balance_test_sets", action="store_true", default=True)
#    parser.add_argument("--no_balance_test_sets", action="store_false", dest="balance_test_sets")
#    parser.add_argument("--unseen_actions", nargs="*", default=[], help="Optional action names to keep completely unseen from train/val.")
#    parser.add_argument("--unseen_actions_file", default="")
#    parser.add_argument("--unseen_actors", nargs="*", default=[], help="Optional actor IDs/names to keep completely unseen from train/val.")
#    parser.add_argument("--unseen_actors_file", default="")
#    parser.add_argument("--project_root", default="", help="Parent directory containing the MotionCLIP folder.")
#    parser.add_argument("--checkpoint", required=True)
#    parser.add_argument("--trainable_layers", type=int, default=2)
#    parser.add_argument("--clip_model", default="ViT-B/32")
#    parser.add_argument("--normal_prompt_template", default="{normal_class} {action}")
#    parser.add_argument("--anomaly_prompt_template", default="{anomaly_class} {action}")
#    parser.add_argument("--epochs", type=int, default=20)
#    parser.add_argument("--batch_size", type=int, default=32)
#    parser.add_argument("--num_workers", type=int, default=4)
#    parser.add_argument("--lr", type=float, default=1e-5)
#    parser.add_argument("--weight_decay", type=float, default=1e-2)
#    parser.add_argument("--temperature", type=float, default=0.07)
#    parser.add_argument("--grad_clip", type=float, default=1.0)
#    parser.add_argument("--amp", action="store_true")
#    parser.add_argument("--threshold_criterion", default="f1", choices=["f1", "balanced_accuracy", "accuracy"])
#    args = parser.parse_args()
#    args.normal_classes = [norm_text(x) for x in args.normal_classes]
#    args.anomaly_classes = [norm_text(x) for x in args.anomaly_classes]
#    args.normal_class = norm_text(args.normal_class) if args.normal_class else ""
#    args.anomaly_class = norm_text(args.anomaly_class) if args.anomaly_class else ""
#    set_seed(args.seed)
#    output_dir = ensure_dir(args.output_dir)
#    ckpt_dir = ensure_dir(output_dir / "checkpoints")
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    save_json(vars(args), output_dir / "args.json")
#    df = pd.read_csv(args.csv_path).copy()
#    df["original_index"] = np.arange(len(df))
#    required = [args.path_col, args.action_col, args.class_col]
#    if args.actor_col:
#        required.append(args.actor_col)
#    missing = [c for c in required if c not in df.columns]
#    if missing:
#        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")
#    df["_action_norm"] = df[args.action_col].map(norm_text)
#    df["_class_norm"] = df[args.class_col].map(norm_text)
#    df["_actor_norm"] = df[args.actor_col].map(norm_text) if args.actor_col else ""
#    missing_paths = [p for p in df[args.path_col].head(20).tolist() if not Path(str(p)).exists()]
#    if missing_paths:
#        print("[WARN] Some example motion files do not exist from this machine.")
#        print(f"       First missing example: {missing_paths[0]}")
#    train_df, val_df, test_sets, split_summary = make_transfer_splits(args, df)
#    save_json(split_summary, output_dir / "split_summary.json")
#    print("[INFO] Split summary:")
#    print(json.dumps(split_summary, indent=2, sort_keys=True))
#    train_df.to_csv(output_dir / "split_train.csv", index=False)
#    val_df.to_csv(output_dir / "split_val.csv", index=False)
#    for name, split_df in test_sets.items():
#        split_df.to_csv(output_dir / f"split_{name}.csv", index=False)
#    if "combined_test" in test_sets:
#        test_sets["combined_test"].to_csv(output_dir / "split_test.csv", index=False)
#    expected_shape = (args.num_frames, args.njoints, args.nfeats)
#    train_ds = PerMoMotionDataset(train_df, args.path_col, args.action_col, args.class_col, "label", args.motion_key, expected_shape)
#    val_ds = PerMoMotionDataset(val_df, args.path_col, args.action_col, args.class_col, "label", args.motion_key, expected_shape)
#    test_datasets = {name: PerMoMotionDataset(split_df, args.path_col, args.action_col, args.class_col, "label", args.motion_key, expected_shape) for name, split_df in test_sets.items()}
#    train_labels = train_df["label"].astype(int).to_numpy()
#    train_batch_sampler = BalancedBinaryBatchSampler(train_labels, args.batch_size, seed=args.seed, drop_last=False)
#    sampler_info = {"type": "BalancedBinaryBatchSampler", "batch_size": int(args.batch_size), "normal_per_batch": int(train_batch_sampler.n0), "anomaly_per_batch": int(train_batch_sampler.n1), "num_batches_per_epoch": int(len(train_batch_sampler)), "train_label_counts": {"normal_0": int((train_labels == 0).sum()), "anomaly_1": int((train_labels == 1).sum())}}
#    save_json(sampler_info, output_dir / "train_sampler_info.json")
#    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
#    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
#    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch)
#    test_loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch) for name, ds in test_datasets.items()}
#    if args.project_root:
#        sys.path.insert(0, str(Path(args.project_root).resolve()))
#    global Encoder_TRANSFORMER
#    from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER
#    motion_encoder = build_motionclip_encoder(args.checkpoint, device)
#    save_json({"loaded": True, "checkpoint_path": args.checkpoint}, output_dir / "checkpoint_load_info.json")
#    unfreeze_info = freeze_encoder_except_last_layers(motion_encoder, args.trainable_layers)
#    save_json(unfreeze_info, output_dir / "unfreeze_info.json")
#    print("[INFO] Unfreeze info:", unfreeze_info)
#    print("[INFO] Loading frozen CLIP text encoder...")
#    text_encoder = FrozenCLIPTextEncoder(args.clip_model, device)
#    prompt_classes = split_summary["classes_used_for_prompts"]
#    normal_prompt_classes = split_summary.get("normal_prompt_classes", [args.normal_class])
#    action_to_idx, class_to_idx, text_feats, prompt_info = build_prompt_cache(
#        split_summary["actions_used_for_prompts"], prompt_classes, text_encoder, normal_prompt_classes,
#        args.normal_prompt_template, args.anomaly_prompt_template, device,
#    )
#    save_json(prompt_info, output_dir / "prompts.json")
#    torch.save({"action_to_idx": action_to_idx, "class_to_idx": class_to_idx, "text_feats": text_feats.detach().cpu(), "prompt_info": prompt_info, "normal_prompt_classes": normal_prompt_classes, "anomaly_prompt_classes": [c for c in prompt_classes if c not in set(normal_prompt_classes)]}, output_dir / "text_prompt_cache.pt")
#    anomaly_prompt_classes = [c for c in prompt_classes if c not in set(normal_prompt_classes)]
#    optimizer = torch.optim.AdamW([p for p in motion_encoder.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
#    best_val_auroc = -float("inf")
#    best_epoch = -1
#    best_threshold = 0.0
#    epoch_records: List[Dict[str, Any]] = []
#    for epoch in range(1, args.epochs + 1):
#        t0 = time.time()
#        train_batch_sampler.set_epoch(epoch)
#        train_loss = train_one_epoch(motion_encoder, train_loader, optimizer, device, action_to_idx, class_to_idx, text_feats, args.temperature, args.grad_clip, args.amp)
#        train_eval_out = evaluate(motion_encoder, train_eval_loader, device, action_to_idx, class_to_idx, text_feats, normal_prompt_classes, anomaly_prompt_classes, args.temperature, compute_contrastive_loss=False)
#        train_metrics_epoch = compute_binary_metrics(train_eval_out.y_true, train_eval_out.score, threshold=None, threshold_criterion=args.threshold_criterion)
#        val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, class_to_idx, text_feats, normal_prompt_classes, anomaly_prompt_classes, args.temperature, compute_contrastive_loss=True)
#        val_metrics = compute_binary_metrics(val_out.y_true, val_out.score, threshold=None, threshold_criterion=args.threshold_criterion)
#        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_out.loss, "train_auroc": train_metrics_epoch["auroc"], "val_auroc": val_metrics["auroc"], "train_auprc": train_metrics_epoch["auprc"], "val_auprc": val_metrics["auprc"], "train_f1": train_metrics_epoch["f1"], "val_f1": val_metrics["f1"], "train_balanced_accuracy": train_metrics_epoch["balanced_accuracy"], "val_balanced_accuracy": val_metrics["balanced_accuracy"], "train_threshold": train_metrics_epoch["threshold"], "val_threshold": val_metrics["threshold"], "seconds": time.time() - t0}
#        epoch_records.append(record)
#        save_training_curves(epoch_records, output_dir)
#        print(f"[EPOCH {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_out.loss:.4f} train_auroc={train_metrics_epoch['auroc']:.4f} val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f} val_f1={val_metrics['f1']:.4f} thr={val_metrics['threshold']:.4f}")
#        if np.isfinite(val_metrics["auroc"]) and val_metrics["auroc"] > best_val_auroc:
#            best_val_auroc = val_metrics["auroc"]
#            best_epoch = epoch
#            best_threshold = float(val_metrics["threshold"])
#            torch.save({"epoch": epoch, "motion_encoder_state_dict": motion_encoder.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args), "action_to_idx": action_to_idx, "class_to_idx": class_to_idx, "prompt_info": prompt_info, "text_feats": text_feats.detach().cpu(), "best_val_metrics": val_metrics, "unfreeze_info": unfreeze_info}, ckpt_dir / "best_model.pt")
#            save_predictions(val_out, output_dir / "val_predictions_best.csv", best_threshold)
#    best_ckpt_path = ckpt_dir / "best_model.pt"
#    if best_ckpt_path.exists():
#        best_ckpt = torch.load(best_ckpt_path, map_location=device)
#        motion_encoder.load_state_dict(best_ckpt["motion_encoder_state_dict"], strict=True)
#        best_threshold = float(best_ckpt["best_val_metrics"]["threshold"])
#    else:
#        print("[WARN] No best checkpoint saved. Testing final epoch model.")
#    train_out = evaluate(motion_encoder, train_eval_loader, device, action_to_idx, class_to_idx, text_feats, normal_prompt_classes, anomaly_prompt_classes, args.temperature, True)
#    val_out = evaluate(motion_encoder, val_loader, device, action_to_idx, class_to_idx, text_feats, normal_prompt_classes, anomaly_prompt_classes, args.temperature, True)
#    eval_outputs = {name: evaluate(motion_encoder, loader, device, action_to_idx, class_to_idx, text_feats, normal_prompt_classes, anomaly_prompt_classes, args.temperature, True) for name, loader in test_loaders.items()}
#    train_metrics = compute_binary_metrics(train_out.y_true, train_out.score, threshold=best_threshold); train_metrics["loss"] = train_out.loss
#    val_metrics_final = compute_binary_metrics(val_out.y_true, val_out.score, threshold=best_threshold); val_metrics_final["loss"] = val_out.loss
#    save_predictions(train_out, output_dir / "train_predictions.csv", best_threshold)
#    save_predictions(val_out, output_dir / "val_predictions.csv", best_threshold)
#    save_embeddings(train_out, output_dir / "train_embeddings.npz", output_dir / "train_embeddings_metadata.csv", best_threshold)
#    save_embeddings(val_out, output_dir / "val_embeddings.npz", output_dir / "val_embeddings_metadata.csv", best_threshold)
#    test_metrics_by_name: Dict[str, Any] = {}
#    output_files: Dict[str, str] = {"best_checkpoint": str(best_ckpt_path), "epoch_metrics": str(output_dir / "epoch_metrics.csv"), "metrics": str(output_dir / "metrics.json"), "training_history_npz": str(output_dir / "training_history.npz"), "loss_curves": str(output_dir / "loss_curves.png"), "validation_metrics_plot": str(output_dir / "validation_metrics.png"), "auroc_curves": str(output_dir / "auroc_curves.png"), "train_embeddings": str(output_dir / "train_embeddings.npz"), "val_embeddings": str(output_dir / "val_embeddings.npz")}
#    for name, out in eval_outputs.items():
#        m = compute_binary_metrics(out.y_true, out.score, threshold=best_threshold); m["loss"] = out.loss
#        test_metrics_by_name[name] = m
#        pred_path = output_dir / f"test_predictions_{name}.csv"
#        emb_path = output_dir / f"test_embeddings_{name}.npz"
#        meta_path = output_dir / f"test_embeddings_{name}_metadata.csv"
#        save_predictions(out, pred_path, best_threshold)
#        save_embeddings(out, emb_path, meta_path, best_threshold)
#        output_files[f"test_predictions_{name}"] = str(pred_path)
#        output_files[f"test_embeddings_{name}"] = str(emb_path)
#        output_files[f"test_embeddings_{name}_metadata"] = str(meta_path)
#    if "combined_test" in eval_outputs:
#        save_predictions(eval_outputs["combined_test"], output_dir / "test_predictions.csv", best_threshold)
#        save_embeddings(eval_outputs["combined_test"], output_dir / "test_embeddings.npz", output_dir / "test_embeddings_metadata.csv", best_threshold)
#        output_files["test_predictions"] = str(output_dir / "test_predictions.csv")
#        output_files["test_embeddings"] = str(output_dir / "test_embeddings.npz")
#        output_files["test_embeddings_metadata"] = str(output_dir / "test_embeddings_metadata.csv")
#    final_summary = {"best_epoch": best_epoch, "best_val_auroc_during_training": best_val_auroc, "threshold_selected_on_validation": best_threshold, "train_metrics": train_metrics, "val_metrics": val_metrics_final, "test_metrics": test_metrics_by_name.get("combined_test", {}), "test_metrics_by_split": test_metrics_by_name, "split_summary": split_summary, "prompt_templates": {"normal": args.normal_prompt_template, "anomaly": args.anomaly_prompt_template}, "normal_classes": split_summary["normal_classes"], "normal_prompt_classes_used_for_scoring": normal_prompt_classes, "anomaly_classes": split_summary["anomaly_classes"], "anomaly_prompt_classes_used_for_scoring": anomaly_prompt_classes, "output_files": output_files}
#    save_json(final_summary, output_dir / "metrics.json")
#    print("[DONE] Combined test metrics:")
#    print(json.dumps(test_metrics_by_name.get("combined_test", {}), indent=2, sort_keys=True))
#    print("[DONE] Unseen anomaly-style test metrics:")
#    print(json.dumps(test_metrics_by_name.get("unseen_anomaly_style_test", {}), indent=2, sort_keys=True))
#    print(f"[DONE] Outputs saved to: {output_dir}")
#if __name__ == "__main__":
#    main()
