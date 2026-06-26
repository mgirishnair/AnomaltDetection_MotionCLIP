#!/usr/bin/env python3
"""
Plot an AA-CLIP Fig.2-style cosine-similarity heatmap using average anomaly anchors.
Inputs are the anchor caches exported by train_AACLIP.py:
  - original_clip_anchor_cache.pt
  - adapted_anchor_cache.pt
Each cache must contain:
  normal:  [num_actions, dim]
  anomaly: [num_actions, dim]
  action_names: list[str]
The plotted matrix is:
  [normal anchors for all actions ; average anomaly anchors for all actions]
against itself.
This shows whether text adaptation separates normal anchors from the averaged anomaly anchors.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def torch_load(path: str | Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)
def load_anchor_cache(path: str | Path) -> Tuple[torch.Tensor, List[str], int]:
    payload = torch_load(path)
    required = ["normal", "anomaly", "action_names"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}. Available keys: {list(payload.keys())}")
    normal = normalize_features(payload["normal"])
    anomaly = normalize_features(payload["anomaly"])
    if normal.shape != anomaly.shape:
        raise ValueError(f"normal and anomaly tensors have different shapes: {normal.shape} vs {anomaly.shape}")
    action_names = [str(a) for a in payload["action_names"]]
    if len(action_names) != normal.shape[0]:
        raise ValueError(
            f"Number of action names ({len(action_names)}) does not match tensor rows ({normal.shape[0]})."
        )
    features = torch.cat([normal, anomaly], dim=0)
    labels = [f"healthy {a}" for a in action_names] + [f"avg anomaly {a}" for a in action_names]
    return features, labels, normal.shape[0]
def cosine_matrix(features: torch.Tensor) -> np.ndarray:
    features = normalize_features(features)
    return (features @ features.T).cpu().numpy()
def save_matrix_csv(matrix: np.ndarray, labels: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [float(v) for v in row])
def save_delta_long_csv(before: np.ndarray, after: np.ndarray, labels: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_label", "col_label", "before", "after", "delta_after_minus_before"])
        for i, row_label in enumerate(labels):
            for j, col_label in enumerate(labels):
                writer.writerow([
                    row_label,
                    col_label,
                    float(before[i, j]),
                    float(after[i, j]),
                    float(after[i, j] - before[i, j]),
                ])
def add_group_labels(ax, n_normal: int, n_total: int) -> None:
    boundary = n_normal - 0.5
    ax.axhline(boundary, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(boundary, color="black", linestyle="--", linewidth=1.2)
    normal_y = 1.0 - n_normal / (2.0 * n_total)
    anomaly_y = (n_total - n_normal) / (2.0 * n_total)
    ax.text(
        -0.12,
        normal_y,
        "Normal",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"facecolor": "#dcebf7", "edgecolor": "black", "linewidth": 0.8},
    )
    ax.text(
        -0.12,
        anomaly_y,
        "Anomaly\nanchor",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"facecolor": "#f4dfa0", "edgecolor": "black", "linewidth": 0.8},
    )
def plot_single_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    n_normal: int,
    output_path: Path,
    title: str,
    vmin: float,
    vmax: float,
    dpi: int,
    annotate: bool,
) -> None:
    n_total = len(labels)
    fig, ax = plt.subplots(figsize=(max(7.5, n_total * 0.42), max(6.5, n_total * 0.38)))
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xticks(np.arange(n_total))
    ax.set_yticks(np.arange(n_total))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    add_group_labels(ax, n_normal, n_total)
    if annotate:
        midpoint = (vmin + vmax) / 2.0
        for i in range(n_total):
            for j in range(n_total):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="white" if value > midpoint else "black",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Cosine similarity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
def plot_side_by_side(
    before: np.ndarray,
    after: np.ndarray,
    labels: List[str],
    n_normal: int,
    output_path: Path,
    vmin: float,
    vmax: float,
    dpi: int,
    annotate: bool,
) -> None:
    n_total = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_total * 0.78), max(6.5, n_total * 0.38)))
    for ax, matrix, title in zip(axes, [before, after], ["Original CLIP", "After Text Adaptation"]):
        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xticks(np.arange(n_total))
        ax.set_yticks(np.arange(n_total))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        add_group_labels(ax, n_normal, n_total)
        if annotate:
            midpoint = (vmin + vmax) / 2.0
            for i in range(n_total):
                for j in range(n_total):
                    value = matrix[i, j]
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=4.0,
                        color="white" if value > midpoint else "black",
                    )
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("Cosine similarity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
def plot_delta_heatmap(
    delta: np.ndarray,
    labels: List[str],
    n_normal: int,
    output_path: Path,
    dpi: int,
    annotate: bool,
) -> None:
    n_total = len(labels)
    lim = max(0.05, float(np.nanmax(np.abs(delta))))
    fig, ax = plt.subplots(figsize=(max(7.5, n_total * 0.42), max(6.5, n_total * 0.38)))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest", aspect="equal")
    ax.set_title("Change in Cosine Similarity: After - Original", fontsize=14, pad=10)
    ax.set_xticks(np.arange(n_total))
    ax.set_yticks(np.arange(n_total))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    add_group_labels(ax, n_normal, n_total)
    if annotate:
        for i in range(n_total):
            for j in range(n_total):
                value = delta[i, j]
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=5.0, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Delta cosine similarity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
def block_summary(matrix: np.ndarray, n_normal: int) -> Dict[str, float]:
    normal = matrix[:n_normal, :n_normal]
    anomaly = matrix[n_normal:, n_normal:]
    cross = matrix[:n_normal, n_normal:]
    def offdiag_mean(x: np.ndarray) -> float:
        if x.shape[0] <= 1:
            return float("nan")
        mask = ~np.eye(x.shape[0], dtype=bool)
        return float(x[mask].mean())
    return {
        "normal_normal_offdiag_mean": offdiag_mean(normal),
        "anomaly_anchor_anomaly_anchor_offdiag_mean": offdiag_mean(anomaly),
        "normal_anomaly_anchor_mean": float(cross.mean()),
        "normal_anomaly_anchor_min": float(cross.min()),
        "normal_anomaly_anchor_max": float(cross.max()),
    }
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_anchor_cache", required=True)
    parser.add_argument("--adapted_anchor_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vmin", type=float, default=0.3)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no_annotate", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    annotate = not args.no_annotate
    original_features, labels, n_normal = load_anchor_cache(args.original_anchor_cache)
    adapted_features, adapted_labels, adapted_n_normal = load_anchor_cache(args.adapted_anchor_cache)
    if labels != adapted_labels or n_normal != adapted_n_normal:
        raise ValueError("Original and adapted anchor caches do not have the same action ordering.")
    original_matrix = cosine_matrix(original_features)
    adapted_matrix = cosine_matrix(adapted_features)
    delta = adapted_matrix - original_matrix
    save_matrix_csv(original_matrix, labels, output_dir / "original_average_anchor_similarity_matrix.csv")
    save_matrix_csv(adapted_matrix, labels, output_dir / "adapted_average_anchor_similarity_matrix.csv")
    save_matrix_csv(delta, labels, output_dir / "average_anchor_delta_matrix.csv")
    save_delta_long_csv(original_matrix, adapted_matrix, labels, output_dir / "average_anchor_similarity_changes_every_cell.csv")
    plot_single_heatmap(
        original_matrix,
        labels,
        n_normal,
        output_dir / "average_anchor_original_clip.png",
        "Original CLIP: Normal vs Average Anomaly Anchors",
        args.vmin,
        args.vmax,
        args.dpi,
        annotate,
    )
    plot_single_heatmap(
        adapted_matrix,
        labels,
        n_normal,
        output_dir / "average_anchor_after_text_adaptation.png",
        "After Text Adaptation: Normal vs Average Anomaly Anchors",
        args.vmin,
        args.vmax,
        args.dpi,
        annotate,
    )
    plot_side_by_side(
        original_matrix,
        adapted_matrix,
        labels,
        n_normal,
        output_dir / "average_anchor_fig2_original_vs_after.png",
        args.vmin,
        args.vmax,
        args.dpi,
        annotate,
    )
    plot_delta_heatmap(
        delta,
        labels,
        n_normal,
        output_dir / "average_anchor_delta_after_minus_original.png",
        args.dpi,
        annotate,
    )
    summary = {
        "original_anchor_cache": str(args.original_anchor_cache),
        "adapted_anchor_cache": str(args.adapted_anchor_cache),
        "n_actions": int(n_normal),
        "labels": labels,
        "original_block_summary": block_summary(original_matrix, n_normal),
        "adapted_block_summary": block_summary(adapted_matrix, n_normal),
        "delta_block_summary": block_summary(delta, n_normal),
    }
    with (output_dir / "average_anchor_similarity_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] Saved average-anchor Fig.2-style outputs to: {output_dir}")
if __name__ == "__main__":
    main()
