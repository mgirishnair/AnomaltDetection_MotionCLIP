#!/usr/bin/env python3
"""
Plot MotionCLIP motion embeddings together with healthy/flawed prompt embeddings.
Inputs expected from the finetuning scripts:
  - *_embeddings.npz files containing key: embeddings
  - matching *_embeddings_metadata.csv files
  - text_prompt_cache.pt containing: action_to_idx, text_feats, prompt_info
This script creates PCA, t-SNE, and UMAP plots showing the embedding space used for
prompt-similarity anomaly detection.
Example:
  python plot_permo_motion_prompt_embedding_space.py \
    --run_dir /scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/run1 \
    --output_dir /scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/run1/embedding_plots \
    --splits train val test
For true-unseen runs:
  python plot_permo_motion_prompt_embedding_space.py \
    --run_dir /path/to/run \
    --splits train val test_seen_actions test_unseen_actions test_combined
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
def normalize_action_text(text: str) -> str:
    text = str(text).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
def split_to_paths(run_dir: Path, split: str) -> Tuple[Path, Path]:
    """Map split name to embeddings NPZ and metadata CSV paths."""
    aliases = {
        "test": ("test_embeddings.npz", "test_embeddings_metadata.csv"),
        "test_combined": ("test_embeddings_combined.npz", "test_embeddings_combined_metadata.csv"),
        "test_seen_actions": ("test_embeddings_seen_actions.npz", "test_embeddings_seen_actions_metadata.csv"),
        "test_unseen_actions": ("test_embeddings_unseen_actions.npz", "test_embeddings_unseen_actions_metadata.csv"),
        "seen_actions": ("test_embeddings_seen_actions.npz", "test_embeddings_seen_actions_metadata.csv"),
        "unseen_actions": ("test_embeddings_unseen_actions.npz", "test_embeddings_unseen_actions_metadata.csv"),
        "train": ("train_embeddings.npz", "train_embeddings_metadata.csv"),
        "val": ("val_embeddings.npz", "val_embeddings_metadata.csv"),
    }
    if split in aliases:
        npz_name, csv_name = aliases[split]
    else:
        npz_name, csv_name = f"{split}_embeddings.npz", f"{split}_embeddings_metadata.csv"
    return run_dir / npz_name, run_dir / csv_name
def load_motion_embeddings(run_dir: Path, splits: Sequence[str], normalize: bool) -> pd.DataFrame:
    rows = []
    for split in splits:
        npz_path, meta_path = split_to_paths(run_dir, split)
        if not npz_path.exists():
            print(f"[WARN] Missing embeddings file for split={split}: {npz_path}")
            continue
        if not meta_path.exists():
            print(f"[WARN] Missing metadata file for split={split}: {meta_path}")
            continue
        data = np.load(npz_path, allow_pickle=False)
        if "embeddings" not in data.files:
            raise KeyError(f"{npz_path} does not contain key 'embeddings'. Keys: {data.files}")
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        if normalize:
            emb = l2_normalize(emb)
        meta = pd.read_csv(meta_path)
        if len(meta) != len(emb):
            raise ValueError(f"Length mismatch for {split}: embeddings={len(emb)}, metadata={len(meta)}")
        for i in range(len(meta)):
            action = normalize_action_text(meta.loc[i, "action"] if "action" in meta.columns else "unknown")
            y = int(meta.loc[i, "y_true_is_anomaly"] if "y_true_is_anomaly" in meta.columns else -1)
            rows.append({
                "kind": "motion",
                "split": split,
                "action": action,
                "label": y,
                "label_name": "flawed/anomaly" if y == 1 else "healthy" if y == 0 else "unknown",
                "prompt_type": "",
                "text": "",
                "embedding": emb[i],
                "score": float(meta.loc[i, "anomaly_score_flawed_minus_healthy"]) if "anomaly_score_flawed_minus_healthy" in meta.columns else np.nan,
                "pred_is_anomaly": int(meta.loc[i, "pred_is_anomaly"]) if "pred_is_anomaly" in meta.columns else -1,
                "row_index": int(meta.loc[i, "row_index"]) if "row_index" in meta.columns else i,
            })
    if not rows:
        raise FileNotFoundError("No embeddings were loaded. Check --run_dir and --splits.")
    return pd.DataFrame(rows)
def load_prompt_embeddings_from_cache(run_dir: Path, actions: Sequence[str], normalize: bool) -> pd.DataFrame:
    import torch
    cache_path = run_dir / "text_prompt_cache.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Could not find {cache_path}. This script expects the finetuning output text_prompt_cache.pt."
        )
    cache = torch.load(cache_path, map_location="cpu")
    action_to_idx: Dict[str, int] = {normalize_action_text(k): int(v) for k, v in cache["action_to_idx"].items()}
    text_feats = cache["text_feats"]
    if hasattr(text_feats, "detach"):
        text_feats = text_feats.detach().cpu().numpy()
    text_feats = np.asarray(text_feats, dtype=np.float32)  # [A, 2, D]
    if normalize:
        flat = l2_normalize(text_feats.reshape(-1, text_feats.shape[-1]))
        text_feats = flat.reshape(text_feats.shape)
    prompt_info = cache.get("prompt_info", {})
    rows = []
    missing = []
    for action in sorted({normalize_action_text(a) for a in actions}):
        if action not in action_to_idx:
            missing.append(action)
            continue
        idx = action_to_idx[action]
        normal_text = prompt_info.get(action, {}).get("normal", f"healthy {action}")
        flawed_text = prompt_info.get(action, {}).get("anomaly", f"flawed {action}")
        rows.append({
            "kind": "prompt", "split": "prompt", "action": action, "label": 0,
            "label_name": "healthy prompt", "prompt_type": "healthy", "text": normal_text,
            "embedding": text_feats[idx, 0], "score": np.nan, "pred_is_anomaly": -1, "row_index": -1,
        })
        rows.append({
            "kind": "prompt", "split": "prompt", "action": action, "label": 1,
            "label_name": "flawed prompt", "prompt_type": "flawed", "text": flawed_text,
            "embedding": text_feats[idx, 1], "score": np.nan, "pred_is_anomaly": -1, "row_index": -1,
        })
    if missing:
        print(f"[WARN] {len(missing)} actions missing from text_prompt_cache.pt. First few: {missing[:10]}")
    return pd.DataFrame(rows)
def compute_projection(method: str, X: np.ndarray, seed: int, perplexity: float, n_neighbors: int, min_dist: float) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        n = len(X)
        # sklearn requires perplexity < n_samples
        effective_perplexity = min(perplexity, max(2.0, (n - 1) / 3.0))
        return TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(X)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP is not installed. Install with: pip install umap-learn") from exc
        return umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
        ).fit_transform(X)
    raise ValueError(f"Unknown method: {method}")
def add_projection(df: pd.DataFrame, method: str, seed: int, perplexity: float, n_neighbors: int, min_dist: float) -> pd.DataFrame:
    X = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    Z = compute_projection(method, X, seed, perplexity, n_neighbors, min_dist)
    out = df.drop(columns=["embedding"]).copy()
    out[f"{method}_x"] = Z[:, 0]
    out[f"{method}_y"] = Z[:, 1]
    return out
def plot_projection(df_proj: pd.DataFrame, method: str, output_dir: Path, max_legend_actions: int = 20) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xcol = f"{method}_x"
    ycol = f"{method}_y"
    # Plot 1: healthy/flawed motion + prompts
    fig, ax = plt.subplots(figsize=(9, 7))
    motion = df_proj[df_proj["kind"] == "motion"]
    prompt = df_proj[df_proj["kind"] == "prompt"]
    healthy = motion[motion["label"] == 0]
    flawed = motion[motion["label"] == 1]
    ax.scatter(healthy[xcol], healthy[ycol], s=16, alpha=0.55, label="motion healthy")
    ax.scatter(flawed[xcol], flawed[ycol], s=16, alpha=0.55, label="motion flawed/anomaly")
    p_healthy = prompt[prompt["prompt_type"] == "healthy"]
    p_flawed = prompt[prompt["prompt_type"] == "flawed"]
    ax.scatter(p_healthy[xcol], p_healthy[ycol], s=120, marker="X", edgecolors="black", linewidths=0.8, label="prompt healthy")
    ax.scatter(p_flawed[xcol], p_flawed[ycol], s=120, marker="P", edgecolors="black", linewidths=0.8, label="prompt flawed")
    ax.set_title(f"{method.upper()}: Motion embeddings + healthy/flawed prompt embeddings")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"{method}_motion_vs_prompt_label.png", dpi=220)
    plt.close(fig)
    # Plot 2: by split, prompts highlighted
    fig, ax = plt.subplots(figsize=(9, 7))
    for split, g in motion.groupby("split"):
        ax.scatter(g[xcol], g[ycol], s=14, alpha=0.5, label=f"motion {split}")
    ax.scatter(prompt[xcol], prompt[ycol], s=115, marker="X", edgecolors="black", linewidths=0.8, label="prompts")
    ax.set_title(f"{method.upper()}: Split view + prompts")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{method}_by_split_with_prompts.png", dpi=220)
    plt.close(fig)
    # Plot 3: color by action, if not too many actions
    actions = sorted(motion["action"].dropna().unique().tolist())
    if len(actions) <= max_legend_actions:
        fig, ax = plt.subplots(figsize=(10, 8))
        for action, g in motion.groupby("action"):
            ax.scatter(g[xcol], g[ycol], s=15, alpha=0.55, label=action)
        ax.scatter(prompt[xcol], prompt[ycol], s=110, marker="X", edgecolors="black", linewidths=0.8, label="prompts")
        for _, r in prompt.iterrows():
            label = f"{r['prompt_type']}: {r['action']}"
            ax.annotate(
                label,
                (r[xcol], r[ycol]),
                fontsize=7,
                alpha=0.85,
                xytext=(4, 4),
                textcoords="offset points",
        )
        ax.set_title(f"{method.upper()}: Motion embeddings by action + prompts")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / f"{method}_by_action_with_prompts.png", dpi=220)
        plt.close(fig)
    else:
        print(f"[INFO] Skipping action legend plot for {method}: {len(actions)} actions > {max_legend_actions}")
    # Plot 4: annotate prompts only, useful for seeing where anchors are
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(healthy[xcol], healthy[ycol], s=12, alpha=0.25, label="motion healthy")
    ax.scatter(flawed[xcol], flawed[ycol], s=12, alpha=0.25, label="motion flawed/anomaly")
    ax.scatter(prompt[xcol], prompt[ycol], s=100, marker="X", edgecolors="black", linewidths=0.8, label="prompts")
    for _, r in prompt.iterrows():
        label = f"{r['prompt_type']}: {r['action']}"
        ax.annotate(label, (r[xcol], r[ycol]), fontsize=7, alpha=0.85)
    ax.set_title(f"{method.upper()}: Prompt anchors annotated")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{method}_prompt_anchors_annotated.png", dpi=220)
    plt.close(fig)
def save_nearest_prompt_report(df: pd.DataFrame, output_dir: Path) -> None:
    """For each motion embedding, compute nearest healthy/flawed prompts by cosine similarity."""
    motion = df[df["kind"] == "motion"].reset_index(drop=True)
    prompt = df[df["kind"] == "prompt"].reset_index(drop=True)
    if len(motion) == 0 or len(prompt) == 0:
        return
    X = l2_normalize(np.stack(motion["embedding"].to_numpy()).astype(np.float32))
    P = l2_normalize(np.stack(prompt["embedding"].to_numpy()).astype(np.float32))
    sims = X @ P.T
    nn_idx = sims.argmax(axis=1)
    rows = []
    for i, j in enumerate(nn_idx):
        rows.append({
            "motion_index": i,
            "split": motion.loc[i, "split"],
            "action": motion.loc[i, "action"],
            "true_label": motion.loc[i, "label"],
            "true_label_name": motion.loc[i, "label_name"],
            "nearest_prompt_action": prompt.loc[j, "action"],
            "nearest_prompt_type": prompt.loc[j, "prompt_type"],
            "nearest_prompt_text": prompt.loc[j, "text"],
            "nearest_prompt_cosine": float(sims[i, j]),
            "score": motion.loc[i, "score"],
            "pred_is_anomaly": motion.loc[i, "pred_is_anomaly"],
            "row_index": motion.loc[i, "row_index"],
        })
    pd.DataFrame(rows).to_csv(output_dir / "nearest_prompt_report.csv", index=False)
def load_selected_threshold(run_dir: Path) -> float | None:
    """Load the validation-selected threshold from metrics.json, when available."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read {metrics_path}: {exc}")
        return None
    for key in [
        "threshold_selected_on_validation",
        "test_metrics",
        "unseen_action_test_metrics",
        "seen_action_test_metrics",
        "val_metrics",
    ]:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict) and "threshold" in value:
            return float(value["threshold"])
    return None
def save_score_histograms(motion_df: pd.DataFrame, output_dir: Path, threshold: float | None = None) -> None:
    """Plot and save anomaly-score histograms for healthy vs flawed motions.
    The score is expected to be flawed-minus-healthy prompt similarity/logit.
    Higher score means more anomalous.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "score" not in motion_df.columns:
        print("[WARN] No score column found; skipping score histograms.")
        return
    motion = motion_df[(motion_df["kind"] == "motion") & np.isfinite(motion_df["score"])].copy()
    if motion.empty:
        print("[WARN] No finite motion scores found; skipping score histograms.")
        return
    score_summary = (
        motion.groupby(["split", "label", "label_name"])["score"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    score_summary.to_csv(output_dir / "score_histogram_summary.csv", index=False)
    def _plot_one(df: pd.DataFrame, filename: str, title: str) -> None:
        healthy = df[df["label"] == 0]["score"].to_numpy(dtype=float)
        flawed = df[df["label"] == 1]["score"].to_numpy(dtype=float)
        if len(healthy) == 0 or len(flawed) == 0:
            print(f"[WARN] Skipping {filename}: need both healthy and flawed samples.")
            return
        all_scores = np.concatenate([healthy, flawed])
        bins = np.linspace(float(all_scores.min()), float(all_scores.max()), 35)
        if np.allclose(bins[0], bins[-1]):
            bins = 20
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(healthy, bins=bins, alpha=0.55, density=True, label=f"healthy (n={len(healthy)})")
        ax.hist(flawed, bins=bins, alpha=0.55, density=True, label=f"flawed/anomaly (n={len(flawed)})")
        if threshold is not None:
            ax.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
        ax.set_title(title)
        ax.set_xlabel("anomaly score = flawed logit - healthy logit")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)
    _plot_one(
        motion,
        "score_histogram_healthy_vs_flawed_all_splits.png",
        "Anomaly-score histogram: healthy vs flawed/anomaly",
    )
    for split, g in motion.groupby("split"):
        safe_split = str(split).replace("/", "_").replace(" ", "_")
        _plot_one(
            g,
            f"score_histogram_healthy_vs_flawed_{safe_split}.png",
            f"Anomaly-score histogram: {split}",
        )
def save_cosine_similarity_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    motion = df[df["kind"] == "motion"].copy()
    prompt = df[df["kind"] == "prompt"].copy()

    if motion.empty or prompt.empty:
        return

    prompt_lookup = {}
    for _, r in prompt.iterrows():
        prompt_lookup[(r["action"], r["prompt_type"])] = r["embedding"]

    rows = []

    for _, r in motion.iterrows():
        action = r["action"]

        healthy_prompt = prompt_lookup.get((action, "healthy"))
        flawed_prompt = prompt_lookup.get((action, "flawed"))

        if healthy_prompt is None or flawed_prompt is None:
            continue

        m = r["embedding"]
        m = m / np.linalg.norm(m)

        hp = healthy_prompt / np.linalg.norm(healthy_prompt)
        fp = flawed_prompt / np.linalg.norm(flawed_prompt)

        sim_healthy = float(np.dot(m, hp))
        sim_flawed = float(np.dot(m, fp))

        rows.append({
            "label": r["label"],
            "sim_healthy": sim_healthy,
            "sim_flawed": sim_flawed,
            "score": sim_flawed - sim_healthy,
        })

    sims = pd.DataFrame(rows)

    sims.to_csv(
        output_dir / "cosine_similarity_summary.csv",
        index=False,
    )

    healthy = sims[sims["label"] == 0]
    flawed = sims[sims["label"] == 1]

    # similarity to healthy prompt
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        healthy["sim_healthy"],
        bins=30,
        alpha=0.6,
        density=True,
        label="healthy motions",
    )
    ax.hist(
        flawed["sim_healthy"],
        bins=30,
        alpha=0.6,
        density=True,
        label="flawed motions",
    )
    ax.set_title("Cosine similarity to HEALTHY prompt")
    ax.set_xlabel("cosine similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "cosine_similarity_to_healthy_prompt.png",
        dpi=220,
    )
    plt.close(fig)

    # similarity to flawed prompt
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        healthy["sim_flawed"],
        bins=30,
        alpha=0.6,
        density=True,
        label="healthy motions",
    )
    ax.hist(
        flawed["sim_flawed"],
        bins=30,
        alpha=0.6,
        density=True,
        label="flawed motions",
    )
    ax.set_title("Cosine similarity to FLAWED prompt")
    ax.set_xlabel("cosine similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "cosine_similarity_to_flawed_prompt.png",
        dpi=220,
    )
    plt.close(fig)

    # difference (actual anomaly score)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        healthy["score"],
        bins=30,
        alpha=0.6,
        density=True,
        label="healthy motions",
    )
    ax.hist(
        flawed["score"],
        bins=30,
        alpha=0.6,
        density=True,
        label="flawed motions",
    )
    ax.set_title("Cosine similarity difference")
    ax.set_xlabel("sim_flawed - sim_healthy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "cosine_similarity_difference.png",
        dpi=220,
    )
    plt.close(fig)
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PerMo MotionCLIP motion embeddings with healthy/flawed prompt embeddings.")
    parser.add_argument("--run_dir", required=True, help="Directory containing embeddings and text_prompt_cache.pt from finetuning.")
    parser.add_argument("--output_dir", default="", help="Where plots/CSVs are saved. Default: <run_dir>/embedding_space_plots")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to load, e.g. train val test or train val test_seen_actions test_unseen_actions test_combined")
    parser.add_argument("--methods", nargs="+", default=["pca", "tsne", "umap"], choices=["pca", "tsne", "umap"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--no_normalize", action="store_true", help="Do not L2-normalize embeddings before projection. Default normalizes because scoring uses cosine similarity.")
    parser.add_argument("--max_legend_actions", type=int, default=20)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "embedding_space_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize = not args.no_normalize
    motion_df = load_motion_embeddings(run_dir, args.splits, normalize=normalize)
    prompt_df = load_prompt_embeddings_from_cache(run_dir, motion_df["action"].tolist(), normalize=normalize)
    df = pd.concat([motion_df, prompt_df], ignore_index=True)
    # Save combined high-dimensional metadata without the actual vectors for readability.
    meta_no_vec = df.drop(columns=["embedding"]).copy()
    meta_no_vec.to_csv(output_dir / "combined_motion_prompt_metadata.csv", index=False)
    # Save combined embeddings too, useful for debugging or external tools.
    X = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    np.savez_compressed(output_dir / "combined_motion_prompt_embeddings.npz", embeddings=X)
    save_nearest_prompt_report(df, output_dir)
    selected_threshold = load_selected_threshold(run_dir)
    save_score_histograms(motion_df, output_dir, threshold=selected_threshold)
    save_cosine_similarity_histograms(df, output_dir)
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "splits_loaded": args.splits,
        "methods": args.methods,
        "n_motion": int((df["kind"] == "motion").sum()),
        "n_prompts": int((df["kind"] == "prompt").sum()),
        "n_total": int(len(df)),
        "actions": sorted(motion_df["action"].unique().tolist()),
        "normalized_before_projection": normalize,
        "selected_threshold_from_metrics_json": selected_threshold,
    }
    with open(output_dir / "plot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    for method in args.methods:
        print(f"[INFO] Computing {method.upper()} projection for {len(df)} points...")
        df_proj = add_projection(
            df=df,
            method=method,
            seed=args.seed,
            perplexity=args.tsne_perplexity,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
        )
        df_proj.to_csv(output_dir / f"{method}_coordinates.csv", index=False)
        plot_projection(df_proj, method, output_dir, max_legend_actions=args.max_legend_actions)
        print(f"[INFO] Saved {method.upper()} plots to {output_dir}")
    print(f"[DONE] Outputs saved to: {output_dir}")
if __name__ == "__main__":
    main()
