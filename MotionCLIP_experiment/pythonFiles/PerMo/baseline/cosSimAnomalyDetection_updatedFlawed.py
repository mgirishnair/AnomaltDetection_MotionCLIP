#!/usr/bin/env python3
"""
Frozen MotionCLIP prompt-semantic baseline for PerMo Condition anomaly detection.
Core idea
---------
For each sample, use its action label to create action-specific condition prompts:
    healthy <action>
    drunken <action>
    exhausted <action>
    arm aching <action>
    leg aching <action>
    head aching <action>
    text necked <action>
    ...
Then compare the MotionCLIP motion embedding against:
    sim_healthy = cos(motion, "healthy <action>")
    sim_nonhealthy_max = max_{condition != healthy} cos(motion, "<condition> <action>")
Anomaly score:
    score = sim_nonhealthy_max - sim_healthy
If score > 0, the motion is closer to a non-healthy prompt than to the healthy prompt,
so it is predicted as anomalous. AUROC is computed using this continuous score and
CSV ground truth `is_anomaly`.
Expected CSV columns
--------------------
Required:
    motion_path, action_label, condition_label, is_anomaly
Optional/recommended:
    condition_prompt_label
Example:
    motion_path,action_label,condition_label,condition_prompt_label,is_anomaly
    /scratch/.../Exhausted_Hop_A01_003.npz,hop,Exhausted,exhausted,1
    /scratch/.../Healthy_Hop_A01_003.npz,hop,Healthy,healthy,0
Outputs
-------
Inside <output_dir>/<run_name>/:
    config.json
    dataset_summary.json
    prompts_used.txt
    prompt_lookup.json
    per_sample_scores.csv
    metrics_summary.json / metrics_summary.csv
    per_action_metrics.csv
    per_condition_score_summary.csv
    text_prompt_embeddings.npz
    plotting_data_all.npz
    cosine_similarity_matrix.csv
    plot_arrays/*.npz
    plots/*.png
"""
import os
import sys
import json
import math
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
# ==================================================
# Utilities
# ==================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def save_json(obj: Dict[str, Any], save_path: str) -> None:
    with open(save_path, "w") as f:
        json.dump(obj, f, indent=2)
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
def normalize_text_label(x: Any) -> str:
    return str(x).strip().replace("_", " ").replace("-", " ").lower()
def add_motionclip_to_path(motionclip_repo: str):
    """
    Expected import:
        from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER
    For this to work, the parent directory of MotionCLIP must be on sys.path.
    """
    motionclip_repo = os.path.abspath(motionclip_repo)
    parent_dir = os.path.dirname(motionclip_repo)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER
    return Encoder_TRANSFORMER
# ==================================================
# CLIP text encoder
# ==================================================
class CLIPTextEncoder:
    """
    Wrapper for OpenAI CLIP or OpenCLIP.
    It first tries:
        import clip
    If that fails, it tries:
        import open_clip
    Output:
        L2-normalized text embeddings.
    """
    def __init__(self, device: str, clip_model_name: str = "ViT-B/32"):
        self.device = device
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.clip_module = None
        try:
            import clip
            self.backend = "clip"
            self.clip_module = clip
            self.model, _ = clip.load(clip_model_name, device=device, jit=False)
            self.model=self.model.float()
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            print(f"Loaded OpenAI CLIP text encoder: {clip_model_name}")
            return
        except Exception as e:
            print(f"OpenAI clip load failed: {e}")
        try:
            import open_clip
            self.backend = "open_clip"
            open_clip_name = clip_model_name.replace("/", "-")
            self.model, _, _ = open_clip.create_model_and_transforms(
                open_clip_name,
                pretrained="openai",
                device=device,
            )
            self.tokenizer = open_clip.get_tokenizer(open_clip_name)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            print(f"Loaded OpenCLIP text encoder: {open_clip_name}")
            return
        except Exception as e:
            print(f"OpenCLIP load failed: {e}")
        raise RuntimeError(
            "Could not load either `clip` or `open_clip`. "
            "Install one of them in the same environment as MotionCLIP."
        )
    @torch.no_grad()
    def encode(self, prompts: List[str], batch_size: int = 128) -> torch.Tensor:
        all_embs = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            if self.backend == "clip":
                tokens = self.clip_module.tokenize(batch_prompts).to(self.device)
                embs = self.model.encode_text(tokens).float()
            elif self.backend == "open_clip":
                tokens = self.tokenizer(batch_prompts).to(self.device)
                embs = self.model.encode_text(tokens).float()
            else:
                raise RuntimeError("Unknown CLIP backend.")
            embs = F.normalize(embs, dim=1)
            all_embs.append(embs.detach().cpu())
        return torch.cat(all_embs, dim=0)
# ==================================================
# MotionCLIP encoder
# ==================================================
def build_motionclip_encoder(
    checkpoint_path: str,
    motionclip_repo: str,
    device: str,
) -> nn.Module:
    Encoder_TRANSFORMER = add_motionclip_to_path(motionclip_repo)
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
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {checkpoint_path}")
    has_encoder_prefix = any(k.startswith("encoder.") for k in ckpt.keys())
    if has_encoder_prefix:
        encoder_state = {
            k[len("encoder."):]: v
            for k, v in ckpt.items()
            if k.startswith("encoder.")
        }
    else:
        encoder_state = ckpt
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        print("\nWarning: unexpected encoder keys:")
        print(unexpected)
    if missing:
        print("\nWarning: missing encoder keys:")
        print(missing)
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder
class MotionCLIPEncoderOnly(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.encoder(batch)
        z = out["mu"]
        z = F.normalize(z.float(), dim=1)
        return z
# ==================================================
# PerMo dataset
# ==================================================
def resolve_motion_path(path_value: str, path_root: Optional[str]) -> str:
    p = Path(str(path_value))
    if p.is_absolute():
        return str(p)
    if path_root is None:
        return str(p.resolve())
    return str((Path(path_root) / p).resolve())
def load_motion_npz(path: str, npz_key: str = "auto") -> np.ndarray:
    """
    Loads one converted MotionCLIP-format PerMo motion.
    Expected canonical format:
        [60, 25, 6]
    Also supports:
        [1, 60, 25, 6]
        [25, 6, 60]
        [60, 150]
    """
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.keys())
        if npz_key != "auto":
            if npz_key not in keys:
                raise KeyError(
                    f"Key '{npz_key}' not found in {path}. "
                    f"Available keys: {keys}"
                )
            arr = data[npz_key]
        else:
            preferred_keys = [
                "motion",
                "rot6d",
                "poses",
                "pose",
                "x",
                "X",
                "arr_0",
            ]
            found_key = None
            for k in preferred_keys:
                if k in keys:
                    found_key = k
                    break
            if found_key is None:
                if len(keys) == 1:
                    found_key = keys[0]
                else:
                    raise KeyError(
                        f"Could not infer motion key in {path}. "
                        f"Available keys: {keys}. Use --npz_key."
                    )
            arr = data[found_key]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape == (60, 25, 6):
        return arr
    if arr.shape == (25, 6, 60):
        return np.transpose(arr, (2, 0, 1)).copy()
    if arr.ndim == 2 and arr.shape == (60, 150):
        return arr.reshape(60, 25, 6)
    raise ValueError(
        f"Unsupported motion shape in {path}: {arr.shape}. "
        "Expected [60,25,6], [25,6,60], or [60,150]."
    )
class PerMoMotionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_root: Optional[str],
        npz_key: str = "auto",
    ):
        self.df = df.reset_index(drop=True)
        self.path_root = path_root
        self.npz_key = npz_key
    def __len__(self) -> int:
        return len(self.df)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        motion_path = resolve_motion_path(row["motion_path"], self.path_root)
        pose = load_motion_npz(motion_path, self.npz_key)   # [60, 25, 6]
        pose = np.transpose(pose, (1, 2, 0)).copy()         # [25, 6, 60]
        return {
            "x": torch.from_numpy(pose),
            "y": torch.tensor(0, dtype=torch.long),
            "lengths": torch.tensor(60, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }
def collate_motionclip(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    x = torch.stack([b["x"] for b in batch], dim=0)              # [B, 25, 6, 60]
    y = torch.stack([b["y"] for b in batch], dim=0)              # [B]
    lengths = torch.stack([b["lengths"] for b in batch], dim=0)  # [B]
    indices = torch.stack([b["index"] for b in batch], dim=0)    # [B]
    T = x.shape[-1]
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "x": x,
        "y": y,
        "lengths": lengths,
        "mask": mask,
        "index": indices,
    }
# ==================================================
# Prompts
# ==================================================
def make_prompt_bank(
    actions: List[str],
    condition_prompt_labels: List[str],
    normal_prompt_label: str = "healthy",
    anomaly_prompt_label: str = "flawed",
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Creates one normal prompt and one shared anomaly prompt per action:
        healthy <action>
        flawed <action>
    The original non-healthy condition/style names from the CSV are NOT used
    as prompt labels. They are all collapsed to the shared anomaly prompt.
    """
    prompts: List[str] = []
    lookup: Dict[str, Dict[str, str]] = {}
    normal_prompt_label = normalize_text_label(normal_prompt_label)
    anomaly_prompt_label = normalize_text_label(anomaly_prompt_label)
    csv_conditions = sorted(set(normalize_text_label(c) for c in condition_prompt_labels))
    if normal_prompt_label not in csv_conditions:
        raise ValueError(
            f"Normal prompt label '{normal_prompt_label}' not found in condition_prompt_label values: {csv_conditions}"
        )
    # Only these two prompt labels are encoded per action.
    conditions = [normal_prompt_label, anomaly_prompt_label]
    for action in sorted(set(actions)):
        action = normalize_text_label(action)
        lookup[action] = {}
        for cond in conditions:
            prompt = f"{cond} {action}"
            lookup[action][cond] = prompt
            prompts.append(prompt)
    prompts = sorted(set(prompts))
    return prompts, lookup
# ==================================================
# Metrics and plots
# ==================================================
def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    thresholds = np.unique(scores)
    if len(thresholds) == 0:
        return {"threshold": 0.0, "f1": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0}
    best = {"threshold": float(thresholds[0]), "f1": -1.0, "accuracy": 0.0, "balanced_accuracy": 0.0}
    for t in thresholds:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)
        bacc = balanced_accuracy_score(y_true, pred)
        if f1 > best["f1"]:
            best = {
                "threshold": float(t),
                "f1": float(f1),
                "accuracy": float(acc),
                "balanced_accuracy": float(bacc),
            }
    return best
def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    prefix: str,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    result: Dict[str, Any] = {
        f"{prefix}_num_samples": int(len(y_true)),
        f"{prefix}_num_normal": int((y_true == 0).sum()),
        f"{prefix}_num_anomaly": int((y_true == 1).sum()),
    }
    if len(np.unique(y_true)) < 2:
        result[f"{prefix}_auroc"] = None
        result[f"{prefix}_aupr"] = None
        result[f"{prefix}_best_threshold"] = None
        result[f"{prefix}_best_f1"] = None
        result[f"{prefix}_best_accuracy"] = None
        result[f"{prefix}_best_balanced_accuracy"] = None
        return result
    result[f"{prefix}_auroc"] = float(roc_auc_score(y_true, scores))
    result[f"{prefix}_aupr"] = float(average_precision_score(y_true, scores))
    best = best_f1_threshold(y_true, scores)
    result[f"{prefix}_best_threshold"] = best["threshold"]
    result[f"{prefix}_best_f1"] = best["f1"]
    result[f"{prefix}_best_accuracy"] = best["accuracy"]
    result[f"{prefix}_best_balanced_accuracy"] = best["balanced_accuracy"]
    # Natural argmax decision: anomaly if max_nonhealthy > healthy, i.e. score > 0.
    pred_zero_threshold = (scores > 0.0).astype(int)
    result[f"{prefix}_zero_threshold_accuracy"] = float(accuracy_score(y_true, pred_zero_threshold))
    result[f"{prefix}_zero_threshold_balanced_accuracy"] = float(
        balanced_accuracy_score(y_true, pred_zero_threshold)
    )
    result[f"{prefix}_zero_threshold_f1"] = float(f1_score(y_true, pred_zero_threshold, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, pred_zero_threshold, labels=[0, 1]).ravel()
    result[f"{prefix}_zero_threshold_tn"] = int(tn)
    result[f"{prefix}_zero_threshold_fp"] = int(fp)
    result[f"{prefix}_zero_threshold_fn"] = int(fn)
    result[f"{prefix}_zero_threshold_tp"] = int(tp)
    return result
def plot_score_histogram(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    plt.figure(figsize=(8, 5))
    plt.hist(scores[y_true == 0], bins=40, alpha=0.6, label="Healthy / normal")
    plt.hist(scores[y_true == 1], bins=40, alpha=0.6, label="Non-healthy / anomaly")
    plt.axvline(0.0, linestyle="--", label="argmax boundary: score = 0")
    plt.xlabel("Anomaly score = max_nonhealthy_similarity - healthy_similarity")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
def plot_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, scores)
    aupr = average_precision_score(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AUPR = {aupr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
def plot_similarity_histogram(
    y_true: np.ndarray,
    values: np.ndarray,
    title: str,
    xlabel: str,
    save_path: str,
    bins: int = 50,
) -> None:
    """Histogram of one cosine similarity column, split by healthy/flawed labels."""
    y_true = np.asarray(y_true).astype(int)
    values = np.asarray(values).astype(float)
    plt.figure(figsize=(9, 5.5))
    plt.hist(values[y_true == 0], bins=bins, alpha=0.6, label="healthy motions")
    plt.hist(values[y_true == 1], bins=bins, alpha=0.6, label="flawed motions")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
def plot_similarity_scatter(scores_df: pd.DataFrame, save_path: str) -> None:
    """Scatter plot directly showing the decision boundary sim_max_nonhealthy = sim_healthy."""
    y_true = scores_df["is_anomaly"].values.astype(int)
    x = scores_df["sim_healthy_action"].values.astype(float)
    y = scores_df["sim_max_nonhealthy_action"].values.astype(float)
    plt.figure(figsize=(7, 7))
    plt.scatter(x[y_true == 0], y[y_true == 0], s=18, alpha=0.65, label="healthy motions")
    plt.scatter(x[y_true == 1], y[y_true == 1], s=18, alpha=0.65, label="flawed motions")
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    pad = 0.02 * max(1e-6, hi - lo)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", label="score = 0 boundary")
    plt.xlabel("cosine similarity to healthy prompt")
    plt.ylabel("cosine similarity to best non-healthy prompt")
    plt.title("Healthy prompt similarity vs best flawed/non-healthy prompt similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
def plot_boxplot_by_group(scores_df: pd.DataFrame, value_col: str, group_col: str, title: str, save_path: str) -> None:
    groups = []
    labels = []
    for name, group in scores_df.groupby(group_col):
        vals = group[value_col].dropna().astype(float).values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(str(name))
    if not groups:
        return
    plt.figure(figsize=(max(9, 0.55 * len(labels)), 5.5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
def _reduce_embeddings(method: str, X: np.ndarray, seed: int) -> Optional[np.ndarray]:
    if len(X) < 2:
        return None
    method = method.lower()
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        perplexity = min(30, max(2, (len(X) - 1) // 3))
        return TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(X)
    if method == "umap":
        try:
            import umap
        except Exception as exc:
            print(f"[WARN] Could not import umap-learn, skipping UMAP plot: {exc}")
            return None
        n_neighbors = min(15, max(2, len(X) - 1))
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors).fit_transform(X)
    raise ValueError(f"Unknown reduction method: {method}")
def save_embedding_space_plots(
    motion_embs: np.ndarray,
    text_emb_matrix: np.ndarray,
    text_prompts: List[str],
    scores_df: pd.DataFrame,
    plots_dir: str,
    arrays_dir: str,
    seed: int,
) -> None:
    """Save PCA/t-SNE/UMAP plots and the reduced coordinates used to create them."""
    ensure_dir(arrays_dir)
    X = np.vstack([motion_embs, text_emb_matrix]).astype(np.float32)
    n_motion = len(motion_embs)
    point_kind = np.array(["motion"] * n_motion + ["prompt"] * len(text_prompts), dtype=object)
    point_text = np.array([""] * n_motion + list(text_prompts), dtype=object)
    action_labels = np.concatenate([
        scores_df["action_label"].astype(str).values,
        np.array([p.split(" ")[-1] if len(p.split(" ")) else "" for p in text_prompts], dtype=object),
    ])
    is_anomaly = np.concatenate([
        scores_df["is_anomaly"].astype(int).values,
        np.full(len(text_prompts), -1, dtype=int),
    ])
    for method in ["pca", "tsne", "umap"]:
        coords = _reduce_embeddings(method, X, seed)
        if coords is None:
            continue
        np.savez_compressed(
            os.path.join(arrays_dir, f"embedding_space_{method}.npz"),
            coords=coords,
            point_kind=point_kind,
            point_text=point_text,
            action_label=action_labels,
            is_anomaly=is_anomaly,
        )
        plt.figure(figsize=(9, 7))
        # Motions colored by action.
        actions = sorted(scores_df["action_label"].astype(str).unique())
        for action in actions:
            idx = (np.arange(len(coords)) < n_motion) & (action_labels == action)
            plt.scatter(coords[idx, 0], coords[idx, 1], s=16, alpha=0.6, label=action)
        prompt_idx = np.arange(len(coords)) >= n_motion
        plt.scatter(coords[prompt_idx, 0], coords[prompt_idx, 1], s=120, marker="*", edgecolors="black", linewidths=0.6, label="text prompts")
        for i in np.where(prompt_idx)[0]:
            plt.annotate(str(point_text[i]), (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.85)
        plt.title(f"Motion + prompt embedding space ({method.upper()})")
        plt.xlabel(f"{method.upper()} 1")
        plt.ylabel(f"{method.upper()} 2")
        plt.legend(fontsize=8, loc="best", ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"embedding_space_{method}_motions_and_prompts.png"), dpi=220)
        plt.close()
def save_plotting_bundle(
    run_dir: str,
    motion_embs: np.ndarray,
    text_embs: Dict[str, np.ndarray],
    prompts: List[str],
    scores_df: pd.DataFrame,
) -> None:
    """One NPZ containing everything needed to recreate similarity and embedding-space plots."""
    text_emb_matrix = np.stack([text_embs[p] for p in prompts], axis=0).astype(np.float32)
    np.savez_compressed(
        os.path.join(run_dir, "plotting_data_all.npz"),
        motion_embeddings=motion_embs.astype(np.float32),
        text_embeddings=text_emb_matrix,
        prompt_text=np.array(prompts, dtype=object),
        motion_path=scores_df["motion_path"].astype(str).values,
        action_label=scores_df["action_label"].astype(str).values,
        condition_label=scores_df["condition_label"].astype(str).values,
        condition_prompt_label=scores_df["condition_prompt_label"].astype(str).values,
        is_anomaly=scores_df["is_anomaly"].astype(int).values,
        sim_healthy_action=scores_df["sim_healthy_action"].astype(float).values,
        sim_max_nonhealthy_action=scores_df["sim_max_nonhealthy_action"].astype(float).values,
        score_prompt_max_nonhealthy_minus_healthy=scores_df["score_prompt_max_nonhealthy_minus_healthy"].astype(float).values,
        pred_is_anomaly_prompt_argmax=scores_df["pred_is_anomaly_prompt_argmax"].astype(int).values,
    )
# ==================================================
# Embedding and scoring
# ==================================================
@torch.no_grad()
def encode_all_motions(
    model: MotionCLIPEncoderOnly,
    df: pd.DataFrame,
    path_root: Optional[str],
    npz_key: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> np.ndarray:
    dataset = PerMoMotionDataset(
        df=df,
        path_root=path_root,
        npz_key=npz_key,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_motionclip,
        persistent_workers=True if num_workers > 0 else False,
    )
    embeddings = np.zeros((len(df), 512), dtype=np.float32)
    for batch_id, batch in enumerate(loader):
        batch["x"] = batch["x"].to(device, non_blocking=True).float()
        batch["y"] = batch["y"].to(device, non_blocking=True)
        batch["lengths"] = batch["lengths"].to(device, non_blocking=True)
        batch["mask"] = batch["mask"].to(device, non_blocking=True)
        z = model(batch).detach().cpu().numpy()
        idx = batch["index"].detach().cpu().numpy()
        embeddings[idx] = z
        if (batch_id + 1) % 10 == 0:
            print(f"Encoded motion batch {batch_id + 1}/{len(loader)}")
    return embeddings
def build_text_embedding_dict(
    text_encoder: CLIPTextEncoder,
    prompts: List[str],
    text_batch_size: int,
) -> Dict[str, np.ndarray]:
    text_emb_tensor = text_encoder.encode(prompts, batch_size=text_batch_size)
    text_embs = text_emb_tensor.numpy().astype(np.float32)
    return {prompt: text_embs[i] for i, prompt in enumerate(prompts)}
def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b))
def score_samples(
    df: pd.DataFrame,
    motion_embs: np.ndarray,
    text_embs: Dict[str, np.ndarray],
    prompt_lookup: Dict[str, Dict[str, str]],
    condition_prompt_labels: List[str],
    normal_prompt_label: str = "healthy",
    anomaly_prompt_label: str = "flawed",
) -> pd.DataFrame:
    rows = []
    normal_prompt_label = normalize_text_label(normal_prompt_label)
    anomaly_prompt_label = normalize_text_label(anomaly_prompt_label)
    # Score against only two prompts per action: healthy <action> and flawed <action>.
    conditions = [normal_prompt_label, anomaly_prompt_label]
    nonhealthy_conditions = [anomaly_prompt_label]
    for i, row in df.iterrows():
        action = normalize_text_label(row["action_label"])
        z = motion_embs[i]
        if action not in prompt_lookup:
            raise KeyError(f"Action '{action}' not found in prompt lookup.")
        if normal_prompt_label not in prompt_lookup[action]:
            raise KeyError(f"Normal prompt '{normal_prompt_label}' not found for action '{action}'.")
        healthy_prompt = prompt_lookup[action][normal_prompt_label]
        sim_healthy = cosine_np(z, text_embs[healthy_prompt])
        nonhealthy_sims: Dict[str, float] = {}
        nonhealthy_prompts: Dict[str, str] = {}
        for cond in nonhealthy_conditions:
            if cond not in prompt_lookup[action]:
                continue
            prompt = prompt_lookup[action][cond]
            nonhealthy_sims[cond] = cosine_np(z, text_embs[prompt])
            nonhealthy_prompts[cond] = prompt
        if not nonhealthy_sims:
            raise ValueError("No non-healthy prompts available. Cannot compute anomaly score.")
        best_nonhealthy_condition = max(nonhealthy_sims, key=nonhealthy_sims.get)
        sim_max_nonhealthy = nonhealthy_sims[best_nonhealthy_condition]
        best_nonhealthy_prompt = nonhealthy_prompts[best_nonhealthy_condition]
        score = sim_max_nonhealthy - sim_healthy
        pred_is_anomaly = int(score > 0.0)
        pred_prompt_condition = best_nonhealthy_condition if pred_is_anomaly else normal_prompt_label
        pred_prompt = best_nonhealthy_prompt if pred_is_anomaly else healthy_prompt
        out = row.to_dict()
        out.update({
            "healthy_prompt": healthy_prompt,
            "best_nonhealthy_prompt": best_nonhealthy_prompt,
            "best_nonhealthy_condition": best_nonhealthy_condition,
            "pred_prompt_condition": pred_prompt_condition,
            "pred_prompt": pred_prompt,
            "sim_healthy_action": sim_healthy,
            "sim_max_nonhealthy_action": sim_max_nonhealthy,
            "score_prompt_max_nonhealthy_minus_healthy": score,
            "pred_is_anomaly_prompt_argmax": pred_is_anomaly,
        })
        for cond in conditions:
            prompt = prompt_lookup[action][cond]
            out[f"sim_condition::{cond}"] = cosine_np(z, text_embs[prompt])
        rows.append(out)
    return pd.DataFrame(rows)
# ==================================================
# Main
# ==================================================
def main():
    parser = argparse.ArgumentParser(
        description="Frozen MotionCLIP + condition/action prompt cosine baseline for PerMo anomaly detection."
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="CSV with motion_path, action_label, condition_label, is_anomaly.",
    )
    parser.add_argument(
        "--motionclip_repo",
        type=str,
        required=True,
        help="Path to MotionCLIP repo, e.g. /scratch/mgirishnair/Thesis/MotionCLIP",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "MotionCLIP checkpoint. If omitted, uses "
            "<motionclip_repo>/exps/paper-model/checkpoint_0100.pth.tar"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where outputs are saved.",
    )
    parser.add_argument(
        "--path_root",
        type=str,
        default=None,
        help=(
            "Only needed if motion_path in CSV is relative. "
            "Example: /scratch/mgirishnair/Thesis/PerMoConverted/Condition"
        ),
    )
    parser.add_argument(
        "--npz_key",
        type=str,
        default="auto",
        help="NPZ key containing motion. Use auto unless needed.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="condition_action_prompt_cosine_anomaly",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-B/32",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal_prompt_label", type=str, default="healthy")
    parser.add_argument("--save_embeddings", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    args.normal_prompt_label = normalize_text_label(args.normal_prompt_label)
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(
            args.motionclip_repo,
            "exps",
            "paper-model",
            "checkpoint_0100.pth.tar",
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(args.output_dir, args.run_name)
    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(run_dir)
    ensure_dir(plots_dir)
    print("=" * 80)
    print("PerMo MotionCLIP Prompt-Cosine Anomaly Baseline")
    print("=" * 80)
    print(f"Device          : {device}")
    print(f"Metadata CSV    : {args.metadata_csv}")
    print(f"MotionCLIP repo : {args.motionclip_repo}")
    print(f"Checkpoint      : {args.checkpoint_path}")
    print(f"Output dir      : {run_dir}")
    config = vars(args).copy()
    config["device"] = device
    config["baseline_type"] = "Frozen MotionCLIP encoder + frozen CLIP text encoder + healthy/flawed action prompt cosine scoring"
    config["normal_definition"] = "is_anomaly == 0, expected to correspond to Healthy"
    config["anomaly_definition"] = "is_anomaly == 1, all non-Healthy conditions collapsed to the shared prompt 'flawed <action>'"
    config["main_score"] = "score_prompt_max_nonhealthy_minus_healthy"
    config["main_score_formula"] = "cos(motion, 'flawed <action>') - cos(motion, 'healthy <action>')"
    config["argmax_decision_rule"] = "predict anomaly if score_prompt_max_nonhealthy_minus_healthy > 0"
    save_json(config, os.path.join(run_dir, "config.json"))
    shutil.copy2(args.metadata_csv, os.path.join(run_dir, "metadata_input_copy.csv"))
    df = pd.read_csv(args.metadata_csv)
    required_cols = ["motion_path", "action_label", "condition_label", "is_anomaly"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required CSV column: {col}")
    if "condition_prompt_label" not in df.columns:
        df["condition_prompt_label"] = df["condition_label"].astype(str)
    df["action_label"] = df["action_label"].apply(normalize_text_label)
    df["condition_label"] = df["condition_label"].astype(str)
    df["condition_prompt_label"] = df["condition_prompt_label"].apply(normalize_text_label)
    df["is_anomaly"] = df["is_anomaly"].astype(int)
    print("\nDataset summary:")
    print(f"Total samples: {len(df)}")
    print("\nCondition counts:")
    print(df["condition_label"].value_counts())
    print("\nCondition prompt counts:")
    print(df["condition_prompt_label"].value_counts())
    print("\nAction counts:")
    print(df["action_label"].value_counts())
    print("\nNormal/anomaly counts:")
    print(df["is_anomaly"].value_counts())
    dataset_summary = {
        "num_samples": int(len(df)),
        "condition_counts": {str(k): int(v) for k, v in df["condition_label"].value_counts().items()},
        "condition_prompt_counts": {str(k): int(v) for k, v in df["condition_prompt_label"].value_counts().items()},
        "action_counts": {str(k): int(v) for k, v in df["action_label"].value_counts().items()},
        "normal_anomaly_counts": {str(k): int(v) for k, v in df["is_anomaly"].value_counts().items()},
    }
    save_json(dataset_summary, os.path.join(run_dir, "dataset_summary.json"))
    actions = sorted(df["action_label"].unique().tolist())
    condition_prompt_labels = sorted(df["condition_prompt_label"].unique().tolist())
    prompts, prompt_lookup = make_prompt_bank(
        actions=actions,
        condition_prompt_labels=condition_prompt_labels,
        normal_prompt_label=args.normal_prompt_label,
        anomaly_prompt_label="flawed",
    )
    with open(os.path.join(run_dir, "prompts_used.txt"), "w") as f:
        for p in prompts:
            f.write(p + "\n")
    save_json(prompt_lookup, os.path.join(run_dir, "prompt_lookup.json"))
    print(f"\nNumber of unique prompts: {len(prompts)}")
    print("Example prompts:")
    for p in prompts[:20]:
        print(f"  {p}")
    print("\nLoading MotionCLIP encoder...")
    motion_encoder = build_motionclip_encoder(
        checkpoint_path=args.checkpoint_path,
        motionclip_repo=args.motionclip_repo,
        device=device,
    )
    motion_model = MotionCLIPEncoderOnly(motion_encoder).to(device)
    motion_model.eval()
    print("\nLoading CLIP text encoder...")
    text_encoder = CLIPTextEncoder(device=device, clip_model_name=args.clip_model_name)
    print("\nEncoding text prompts...")
    text_embs = build_text_embedding_dict(
        text_encoder=text_encoder,
        prompts=prompts,
        text_batch_size=args.text_batch_size,
    )
    print("\nEncoding motions...")
    motion_embs = encode_all_motions(
        model=motion_model,
        df=df,
        path_root=args.path_root,
        npz_key=args.npz_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    emb_norms = np.linalg.norm(motion_embs, axis=1)
    sanity = {
        "motion_embedding_shape": list(motion_embs.shape),
        "motion_embedding_norm_mean": float(np.mean(emb_norms)),
        "motion_embedding_norm_std": float(np.std(emb_norms)),
        "motion_embedding_norm_min": float(np.min(emb_norms)),
        "motion_embedding_norm_max": float(np.max(emb_norms)),
        "num_nan_embeddings": int(np.isnan(motion_embs).sum()),
    }
    save_json(sanity, os.path.join(run_dir, "embedding_sanity_checks.json"))
    print("\nEmbedding sanity checks:")
    print(json.dumps(sanity, indent=2))
    text_emb_matrix = np.stack([text_embs[p] for p in prompts], axis=0).astype(np.float32)
    np.savez_compressed(
        os.path.join(run_dir, "text_prompt_embeddings.npz"),
        text_embeddings=text_emb_matrix,
        prompt_text=np.array(prompts, dtype=object),
    )
    if args.save_embeddings:
        np.savez_compressed(
            os.path.join(run_dir, "motion_embeddings.npz"),
            embeddings=motion_embs,
            motion_path=df["motion_path"].astype(str).values,
            action_label=df["action_label"].astype(str).values,
            condition_label=df["condition_label"].astype(str).values,
            condition_prompt_label=df["condition_prompt_label"].astype(str).values,
            is_anomaly=df["is_anomaly"].astype(int).values,
        )
    print("\nScoring samples...")
    scores_df = score_samples(
        df=df,
        motion_embs=motion_embs,
        text_embs=text_embs,
        prompt_lookup=prompt_lookup,
        condition_prompt_labels=condition_prompt_labels,
        normal_prompt_label=args.normal_prompt_label,
        anomaly_prompt_label="flawed",
    )
    scores_csv = os.path.join(run_dir, "per_sample_scores.csv")
    scores_df.to_csv(scores_csv, index=False)
    print(f"Saved per-sample scores to: {scores_csv}")
    # Save a compact similarity matrix CSV: one row per sample, one column per prompt similarity.
    sim_cols = [c for c in scores_df.columns if c.startswith("sim_condition::")]
    similarity_matrix_csv = os.path.join(run_dir, "cosine_similarity_matrix.csv")
    scores_df[["motion_path", "action_label", "condition_label", "condition_prompt_label", "is_anomaly"] + sim_cols].to_csv(
        similarity_matrix_csv, index=False
    )
    save_plotting_bundle(
        run_dir=run_dir,
        motion_embs=motion_embs,
        text_embs=text_embs,
        prompts=prompts,
        scores_df=scores_df,
    )
    arrays_dir = os.path.join(run_dir, "plot_arrays")
    save_embedding_space_plots(
        motion_embs=motion_embs,
        text_emb_matrix=np.stack([text_embs[p] for p in prompts], axis=0).astype(np.float32),
        text_prompts=prompts,
        scores_df=scores_df,
        plots_dir=plots_dir,
        arrays_dir=arrays_dir,
        seed=args.seed,
    )
    y_true = scores_df["is_anomaly"].values.astype(int)
    score_col = "score_prompt_max_nonhealthy_minus_healthy"
    scores = scores_df[score_col].values.astype(float)
    all_metrics: Dict[str, Any] = compute_binary_metrics(
        y_true=y_true,
        scores=scores,
        prefix=score_col,
    )
    pred_zero = scores_df["pred_is_anomaly_prompt_argmax"].values.astype(int)
    all_metrics["prompt_argmax_accuracy"] = float(accuracy_score(y_true, pred_zero))
    all_metrics["prompt_argmax_balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred_zero))
    all_metrics["prompt_argmax_f1"] = float(f1_score(y_true, pred_zero, zero_division=0))
    plot_score_histogram(
        y_true=y_true,
        scores=scores,
        title=score_col,
        save_path=os.path.join(plots_dir, f"{score_col}_histogram.png"),
    )
    plot_roc_curve(
        y_true=y_true,
        scores=scores,
        title=score_col,
        save_path=os.path.join(plots_dir, f"{score_col}_roc.png"),
    )
    plot_pr_curve(
        y_true=y_true,
        scores=scores,
        title=score_col,
        save_path=os.path.join(plots_dir, f"{score_col}_pr.png"),
    )
    plot_similarity_histogram(
        y_true=y_true,
        values=scores_df["sim_healthy_action"].values.astype(float),
        title="Cosine similarity to HEALTHY prompt",
        xlabel="cosine similarity",
        save_path=os.path.join(plots_dir, "cosine_similarity_to_healthy_prompt_histogram.png"),
    )
    plot_similarity_histogram(
        y_true=y_true,
        values=scores_df["sim_max_nonhealthy_action"].values.astype(float),
        title="Cosine similarity to BEST NON-HEALTHY prompt",
        xlabel="cosine similarity",
        save_path=os.path.join(plots_dir, "cosine_similarity_to_best_nonhealthy_prompt_histogram.png"),
    )
    plot_similarity_scatter(
        scores_df=scores_df,
        save_path=os.path.join(plots_dir, "healthy_vs_best_nonhealthy_similarity_scatter.png"),
    )
    plot_boxplot_by_group(
        scores_df=scores_df,
        value_col=score_col,
        group_col="action_label",
        title="Anomaly score by action",
        save_path=os.path.join(plots_dir, "anomaly_score_by_action_boxplot.png"),
    )
    plot_boxplot_by_group(
        scores_df=scores_df,
        value_col=score_col,
        group_col="condition_label",
        title="Anomaly score by condition",
        save_path=os.path.join(plots_dir, "anomaly_score_by_condition_boxplot.png"),
    )
    for sim_col in [c for c in scores_df.columns if c.startswith("sim_condition::")]:
        cond = sim_col.split("::", 1)[1]
        safe_cond = cond.replace(" ", "_").replace("/", "_")
        plot_similarity_histogram(
            y_true=y_true,
            values=scores_df[sim_col].values.astype(float),
            title=f"Cosine similarity to prompt condition: {cond}",
            xlabel="cosine similarity",
            save_path=os.path.join(plots_dir, f"cosine_similarity_to_condition_{safe_cond}.png"),
        )
    per_action_rows = []
    for action, group in scores_df.groupby("action_label"):
        row = {
            "action_label": action,
            "num_samples": int(len(group)),
            "num_normal": int((group["is_anomaly"] == 0).sum()),
            "num_anomaly": int((group["is_anomaly"] == 1).sum()),
        }
        y_action = group["is_anomaly"].values.astype(int)
        action_scores = group[score_col].values.astype(float)
        row.update(compute_binary_metrics(y_action, action_scores, prefix=score_col))
        pred_action = group["pred_is_anomaly_prompt_argmax"].values.astype(int)
        if len(np.unique(y_action)) >= 2:
            row["prompt_argmax_accuracy"] = float(accuracy_score(y_action, pred_action))
            row["prompt_argmax_balanced_accuracy"] = float(balanced_accuracy_score(y_action, pred_action))
            row["prompt_argmax_f1"] = float(f1_score(y_action, pred_action, zero_division=0))
        else:
            row["prompt_argmax_accuracy"] = None
            row["prompt_argmax_balanced_accuracy"] = None
            row["prompt_argmax_f1"] = None
        per_action_rows.append(row)
    per_action_df = pd.DataFrame(per_action_rows)
    per_action_csv = os.path.join(run_dir, "per_action_metrics.csv")
    per_action_df.to_csv(per_action_csv, index=False)
    condition_summary_cols = [
        "sim_healthy_action",
        "sim_max_nonhealthy_action",
        "score_prompt_max_nonhealthy_minus_healthy",
        "pred_is_anomaly_prompt_argmax",
    ]
    condition_summary = (
        scores_df
        .groupby("condition_label")[condition_summary_cols]
        .agg(["mean", "std", "count"])
    )
    condition_summary_csv = os.path.join(run_dir, "per_condition_score_summary.csv")
    condition_summary.to_csv(condition_summary_csv)
    prediction_summary = (
        scores_df
        .groupby(["condition_label", "pred_prompt_condition"])
        .size()
        .reset_index(name="count")
    )
    prediction_summary_csv = os.path.join(run_dir, "condition_vs_pred_prompt_condition.csv")
    prediction_summary.to_csv(prediction_summary_csv, index=False)
    all_metrics["notes"] = {
        "baseline": "Frozen zero-shot prompt-semantic baseline. No PerMo training or fine-tuning is performed.",
        "score_prompt_max_nonhealthy_minus_healthy": (
            "cos(motion, 'flawed <action>') minus cos(motion, 'healthy <action>'). "
            "All non-healthy CSV conditions are treated as anomaly but use the same 'flawed <action>' text prompt."
        ),
        "argmax_decision": "pred_is_anomaly_prompt_argmax = 1 if score > 0, else 0",
        "normal_prompt_label": args.normal_prompt_label,
        "anomaly_label": "all non-Healthy Condition styles according to CSV is_anomaly",
    }
    metrics_json = os.path.join(run_dir, "metrics_summary.json")
    metrics_csv = os.path.join(run_dir, "metrics_summary.csv")
    save_json(all_metrics, metrics_json)
    pd.DataFrame([all_metrics]).to_csv(metrics_csv, index=False)
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Results directory           : {run_dir}")
    print(f"Per-sample scores           : {scores_csv}")
    print(f"Cosine similarity matrix    : {similarity_matrix_csv}")
    print(f"Plotting bundle NPZ         : {os.path.join(run_dir, 'plotting_data_all.npz')}")
    print(f"Prompt embeddings NPZ       : {os.path.join(run_dir, 'text_prompt_embeddings.npz')}")
    print(f"Metrics summary JSON        : {metrics_json}")
    print(f"Metrics summary CSV         : {metrics_csv}")
    print(f"Per-action metrics          : {per_action_csv}")
    print(f"Per-condition summary       : {condition_summary_csv}")
    print(f"Prediction summary          : {prediction_summary_csv}")
    print(f"Plots directory             : {plots_dir}")
    print("\nMain metrics:")
    for k, v in all_metrics.items():
        if "auroc" in k or "aupr" in k or "accuracy" in k or "f1" in k:
            print(f"{k}: {v}")
if __name__ == "__main__":
    main()
