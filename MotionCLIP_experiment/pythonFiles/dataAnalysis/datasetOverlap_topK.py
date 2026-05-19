#!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import csv
import argparse
from collections import defaultdict, Counter

import numpy as np
import torch
import clip
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER


class NTURot6dDataset(Dataset):
    def __init__(self, X, y):
        assert isinstance(X, np.ndarray) or isinstance(X, np.memmap)
        assert X.ndim == 4, f"Expected [N, 60, 25, 6], got {X.shape}"
        assert X.shape[1:] == (60, 25, 6), f"Expected [N, 60, 25, 6], got {X.shape}"
        assert len(X) == len(y), "X and y length mismatch"

        self.X = X
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        pose = np.asarray(self.X[idx], dtype=np.float32)  # [60, 25, 6]
        pose = np.transpose(pose, (1, 2, 0))              # [25, 6, 60]

        return {
            "x": torch.from_numpy(pose),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "lengths": torch.tensor(60, dtype=torch.long),
        }


def collate_motionclip(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    lengths = torch.stack([b["lengths"] for b in batch], dim=0)

    T = x.shape[-1]
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "x": x,
        "y": y,
        "lengths": lengths,
        "mask": mask,
    }


def str2bool(v):
    if isinstance(v, bool):
        return v

    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_motionclip_model(checkpoint_path, device):
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

    encoder_state = {}
    for k, v in ckpt.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v

    if not encoder_state:
        encoder_state = ckpt

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

    print("\nCheckpoint loading diagnostics")
    print("------------------------------")
    print("Missing encoder keys:", len(missing))
    print("Unexpected encoder keys:", len(unexpected))

    if missing:
        print("First missing keys:", missing[:20])
    if unexpected:
        print("First unexpected keys:", unexpected[:20])

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


@torch.no_grad()
def encode_babel_labels(
    babel_labels,
    device,
    clip_path="/home/mgirishnair/.cache/clip/ViT-B-32.pt",
    use_prompt_averaging=True,
):
    clip_model, _ = clip.load(clip_path, device="cpu")
    clip_model.eval()
    clip_model = clip_model.float()

    if use_prompt_averaging:
        templates = [
            "a person is {}",
            "someone is performing {}",
            "a human action of {}",
            "a person doing {}",
            "a motion sequence of {}",
        ]

        all_embs = []

        for label in tqdm(babel_labels, desc="Encoding BABEL labels with prompt averaging"):
            texts = [template.format(label) for template in templates]
            tokens = clip.tokenize(texts).to("cpu")

            emb = clip_model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)

            emb = emb.mean(dim=0)
            emb = emb / emb.norm()

            all_embs.append(emb)

        text_emb = torch.stack(all_embs, dim=0)

    else:
        tokens = clip.tokenize(babel_labels).to("cpu")
        text_emb = clip_model.encode_text(tokens).float()
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    text_emb = text_emb.to(device)
    return text_emb


@torch.no_grad()
def encode_motion_batch(encoder, batch, device):
    batch["x"] = batch["x"].to(device).float()
    batch["lengths"] = batch["lengths"].to(device)
    batch["mask"] = batch["mask"].to(device)

    out = encoder(batch)

    if "mu" not in out:
        raise KeyError(f"Expected encoder output to contain 'mu'. Got keys: {list(out.keys())}")

    z = out["mu"].float()
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    return z


@torch.no_grad()
def print_input_diagnostics(loader, max_batches=1):
    print("\nInput diagnostics")
    print("-----------------")

    for i, batch in enumerate(loader):
        x = batch["x"].float()
        y = batch["y"]

        print("batch x shape:", tuple(x.shape))
        print("batch y shape:", tuple(y.shape))
        print("x mean:", x.mean().item())
        print("x std:", x.std().item())
        print("x min:", x.min().item())
        print("x max:", x.max().item())
        print("first labels:", y[:20].tolist())

        if i + 1 >= max_batches:
            break


@torch.no_grad()
def print_similarity_diagnostics(
    encoder,
    loader,
    babel_text_emb,
    device,
    max_batches=20,
):
    all_z = []

    for i, batch in enumerate(loader):
        z = encode_motion_batch(encoder, batch, device)
        all_z.append(z.detach().cpu())

        if i + 1 >= max_batches:
            break

    all_z = torch.cat(all_z, dim=0)
    babel_text_emb_cpu = babel_text_emb.detach().cpu()

    motion_pairwise_sim = all_z @ all_z.T
    sims = all_z @ babel_text_emb_cpu.T

    top2 = sims.topk(k=2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]

    print("\nSimilarity diagnostics")
    print("----------------------")
    print("num diagnostic samples:", all_z.shape[0])

    print("\nMotion embedding diagnostics")
    print("z shape:", tuple(all_z.shape))
    print("z std across samples:", all_z.std(dim=0).mean().item())
    print("mean pairwise motion similarity:", motion_pairwise_sim.mean().item())
    print("min pairwise motion similarity:", motion_pairwise_sim.min().item())
    print("max pairwise motion similarity:", motion_pairwise_sim.max().item())

    print("\nMotion-to-BABEL similarity diagnostics")
    print("sims shape:", tuple(sims.shape))
    print("global sim mean:", sims.mean().item())
    print("global sim std:", sims.std().item())
    print("per-sample sim std mean:", sims.std(dim=1).mean().item())
    print("per-label sim std mean:", sims.std(dim=0).mean().item())
    print("mean top1-top2 margin:", margins.mean().item())
    print("max top1-top2 margin:", margins.max().item())

    print("\nInterpretation")
    if all_z.std(dim=0).mean().item() < 0.01:
        print("WARNING: motion embeddings have very low variation across samples.")

    if motion_pairwise_sim.mean().item() > 0.95:
        print("WARNING: motion embeddings are almost identical across samples.")

    if sims.std(dim=1).mean().item() < 0.01:
        print("WARNING: BABEL similarities are nearly flat per NTU sample.")

    if margins.mean().item() < 0.01:
        print("WARNING: top-1 BABEL label is probably not meaningful.")


@torch.no_grad()
def estimate_babel_label_bias(
    encoder,
    loader,
    babel_text_emb,
    device,
):
    babel_sum = torch.zeros(babel_text_emb.shape[0], device=device)
    total_samples = 0

    for batch in tqdm(loader, desc="Estimating BABEL hub-label bias"):
        z = encode_motion_batch(encoder, batch, device)
        raw_sims = z @ babel_text_emb.T

        babel_sum += raw_sims.sum(dim=0)
        total_samples += raw_sims.shape[0]

        del z, raw_sims

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    babel_mean = babel_sum / max(1, total_samples)

    print("\nBABEL label bias diagnostics")
    print("----------------------------")
    print("total samples used:", total_samples)
    print("babel_mean shape:", tuple(babel_mean.shape))
    print("babel_mean min:", babel_mean.min().item())
    print("babel_mean max:", babel_mean.max().item())
    print("babel_mean mean:", babel_mean.mean().item())
    print("babel_mean std:", babel_mean.std().item())

    return babel_mean


def compute_entropy_from_topk(topk_vals, temperature=0.05, eps=1e-8):
    logits = topk_vals / temperature
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return entropy


def normalize_entropy(mean_entropy, top_k):
    if top_k <= 1:
        return 0.0
    return float(mean_entropy / math.log(top_k))


def assign_overlap_group(babel_likeness_score):
    if babel_likeness_score >= 0.70:
        return "high"
    elif babel_likeness_score >= 0.45:
        return "medium"
    else:
        return "low"


def write_results_csv(
    output_csv,
    class_top1_raw_sims,
    class_topk_raw_mean_sims,
    class_top1_corrected_sims,
    class_topk_corrected_mean_sims,
    class_prop_raw_above,
    class_entropy,
    class_margin_corrected,
    class_sample_argmax_indices,
    class_sum_raw_sims,
    class_sum_corrected_sims,
    class_counts,
    babel_labels,
    threshold,
    top_k,
):
    rows = []

    for cls in sorted(class_counts.keys()):
        num_samples = int(class_counts[cls])

        top1_raw_sims = np.asarray(class_top1_raw_sims[cls], dtype=np.float32)
        topk_raw_mean_sims = np.asarray(class_topk_raw_mean_sims[cls], dtype=np.float32)

        top1_corrected_sims = np.asarray(class_top1_corrected_sims[cls], dtype=np.float32)
        topk_corrected_mean_sims = np.asarray(class_topk_corrected_mean_sims[cls], dtype=np.float32)

        prop_flags = np.asarray(class_prop_raw_above[cls], dtype=np.float32)
        entropies = np.asarray(class_entropy[cls], dtype=np.float32)
        margins_corrected = np.asarray(class_margin_corrected[cls], dtype=np.float32)

        sample_argmax_indices = class_sample_argmax_indices[cls]
        sample_argmax_counter = Counter(sample_argmax_indices)

        sample_dominant_idx, sample_dominant_count = sample_argmax_counter.most_common(1)[0]
        sample_dominant_label = babel_labels[sample_dominant_idx]
        sample_dominant_prop = float(sample_dominant_count / num_samples)

        class_mean_raw_vec = class_sum_raw_sims[cls] / num_samples
        class_mean_corrected_vec = class_sum_corrected_sims[cls] / num_samples

        class_topk_idx = np.argsort(class_mean_corrected_vec)[::-1][:top_k]

        class_topk_labels = [babel_labels[i] for i in class_topk_idx]
        class_topk_corrected_scores = [float(class_mean_corrected_vec[i]) for i in class_topk_idx]
        class_topk_raw_scores = [float(class_mean_raw_vec[i]) for i in class_topk_idx]

        class_top1_label = class_topk_labels[0]
        class_top1_corrected_score = class_topk_corrected_scores[0]
        class_top1_raw_score_at_corrected_top1 = class_topk_raw_scores[0]

        mean_top1_raw_sim = float(np.mean(top1_raw_sims))
        median_top1_raw_sim = float(np.median(top1_raw_sims))
        max_top1_raw_sim = float(np.max(top1_raw_sims))

        mean_topk_raw_sim = float(np.mean(topk_raw_mean_sims))
        median_topk_raw_sim = float(np.median(topk_raw_mean_sims))

        mean_top1_corrected_sim = float(np.mean(top1_corrected_sims))
        median_top1_corrected_sim = float(np.median(top1_corrected_sims))

        mean_topk_corrected_sim = float(np.mean(topk_corrected_mean_sims))
        median_topk_corrected_sim = float(np.median(topk_corrected_mean_sims))

        prop_raw_above = float(np.mean(prop_flags))
        mean_entropy = float(np.mean(entropies))
        normalized_entropy = normalize_entropy(mean_entropy, top_k)

        mean_corrected_margin = float(np.mean(margins_corrected))
        median_corrected_margin = float(np.median(margins_corrected))

        certainty = max(0.0, min(1.0, 1.0 - normalized_entropy))

        raw_sim_component = max(0.0, min(1.0, (mean_topk_raw_sim - 0.65) / 0.15))
        margin_component = max(0.0, min(1.0, mean_corrected_margin / 0.03))

        babel_likeness_score = (
            0.45 * raw_sim_component
            + 0.30 * prop_raw_above
            + 0.15 * margin_component
            + 0.10 * certainty
        )

        overlap_group = assign_overlap_group(babel_likeness_score)

        rows.append({
            "ntu_class": int(cls),
            "num_samples": num_samples,

            "mean_top1_raw_similarity": mean_top1_raw_sim,
            "median_top1_raw_similarity": median_top1_raw_sim,
            "max_top1_raw_similarity": max_top1_raw_sim,

            "mean_topk_raw_similarity": mean_topk_raw_sim,
            "median_topk_raw_similarity": median_topk_raw_sim,

            "mean_top1_corrected_similarity": mean_top1_corrected_sim,
            "median_top1_corrected_similarity": median_top1_corrected_sim,

            "mean_topk_corrected_similarity": mean_topk_corrected_sim,
            "median_topk_corrected_similarity": median_topk_corrected_sim,

            f"proportion_top1_raw_above_{threshold}": prop_raw_above,

            "mean_topk_entropy": mean_entropy,
            "normalized_entropy_log_topk": normalized_entropy,

            "mean_corrected_top1_top2_margin": mean_corrected_margin,
            "median_corrected_top1_top2_margin": median_corrected_margin,

            "sample_argmax_dominant_babel_label": sample_dominant_label,
            "sample_argmax_dominant_babel_proportion": sample_dominant_prop,

            "class_level_top1_babel_label": class_top1_label,
            "class_level_top1_raw_score": class_top1_raw_score_at_corrected_top1,
            "class_level_top1_corrected_score": class_top1_corrected_score,

            "class_level_topk_babel_labels": " | ".join(class_topk_labels),
            "class_level_topk_raw_scores": " | ".join(f"{s:.6f}" for s in class_topk_raw_scores),
            "class_level_topk_corrected_scores": " | ".join(f"{s:.6f}" for s in class_topk_corrected_scores),

            "babel_likeness_score": float(babel_likeness_score),
            "overlap_group": overlap_group,
        })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    fieldnames = [
        "ntu_class",
        "num_samples",

        "mean_top1_raw_similarity",
        "median_top1_raw_similarity",
        "max_top1_raw_similarity",

        "mean_topk_raw_similarity",
        "median_topk_raw_similarity",

        "mean_top1_corrected_similarity",
        "median_top1_corrected_similarity",

        "mean_topk_corrected_similarity",
        "median_topk_corrected_similarity",

        f"proportion_top1_raw_above_{threshold}",

        "mean_topk_entropy",
        "normalized_entropy_log_topk",

        "mean_corrected_top1_top2_margin",
        "median_corrected_top1_top2_margin",

        "sample_argmax_dominant_babel_label",
        "sample_argmax_dominant_babel_proportion",

        "class_level_top1_babel_label",
        "class_level_top1_raw_score",
        "class_level_top1_corrected_score",

        "class_level_topk_babel_labels",
        "class_level_topk_raw_scores",
        "class_level_topk_corrected_scores",

        "babel_likeness_score",
        "overlap_group",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


@torch.no_grad()
def run_streaming_overlap(
    encoder,
    X,
    y,
    babel_text_emb,
    babel_labels,
    batch_size,
    threshold,
    output_csv,
    device,
    top_k=5,
    entropy_temperature=0.05,
    use_hub_correction=True,
    run_diagnostics=True,
):
    dataset = NTURot6dDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_motionclip,
    )

    babel_text_emb = babel_text_emb.to(device)

    if run_diagnostics:
        print_input_diagnostics(loader, max_batches=1)

        print_similarity_diagnostics(
            encoder=encoder,
            loader=loader,
            babel_text_emb=babel_text_emb,
            device=device,
            max_batches=20,
        )

    if use_hub_correction:
        babel_mean = estimate_babel_label_bias(
            encoder=encoder,
            loader=loader,
            babel_text_emb=babel_text_emb,
            device=device,
        )
    else:
        babel_mean = torch.zeros(babel_text_emb.shape[0], device=device)

    top_k = min(top_k, len(babel_labels))

    class_top1_raw_sims = defaultdict(list)
    class_topk_raw_mean_sims = defaultdict(list)

    class_top1_corrected_sims = defaultdict(list)
    class_topk_corrected_mean_sims = defaultdict(list)

    class_prop_raw_above = defaultdict(list)
    class_entropy = defaultdict(list)
    class_margin_corrected = defaultdict(list)

    class_sample_argmax_indices = defaultdict(list)

    class_sum_raw_sims = defaultdict(lambda: None)
    class_sum_corrected_sims = defaultdict(lambda: None)
    class_counts = defaultdict(int)

    for batch in tqdm(loader, desc="Streaming NTU-BABEL overlap"):
        batch_y = batch["y"].cpu().numpy()

        z = encode_motion_batch(encoder, batch, device)
        raw_sims = z @ babel_text_emb.T

        corrected_sims = raw_sims - babel_mean.unsqueeze(0)

        topk_corrected_vals, topk_idx = corrected_sims.topk(
            k=top_k,
            dim=1,
            largest=True,
        )

        topk_raw_vals = raw_sims.gather(1, topk_idx)

        top1_raw_sim = topk_raw_vals[:, 0]
        topk_raw_mean_sim = topk_raw_vals.mean(dim=1)

        top1_corrected_sim = topk_corrected_vals[:, 0]
        topk_corrected_mean_sim = topk_corrected_vals.mean(dim=1)

        if top_k >= 2:
            corrected_margin = topk_corrected_vals[:, 0] - topk_corrected_vals[:, 1]
        else:
            corrected_margin = torch.zeros_like(top1_corrected_sim)

        topk_entropy = compute_entropy_from_topk(
            topk_corrected_vals,
            temperature=entropy_temperature,
        )

        best_idx = topk_idx[:, 0]
        above_threshold = (top1_raw_sim > threshold).float()

        raw_sims_cpu = raw_sims.detach().cpu().numpy()
        corrected_sims_cpu = corrected_sims.detach().cpu().numpy()

        top1_raw_sim = top1_raw_sim.detach().cpu().numpy()
        topk_raw_mean_sim = topk_raw_mean_sim.detach().cpu().numpy()

        top1_corrected_sim = top1_corrected_sim.detach().cpu().numpy()
        topk_corrected_mean_sim = topk_corrected_mean_sim.detach().cpu().numpy()

        topk_entropy = topk_entropy.detach().cpu().numpy()
        corrected_margin = corrected_margin.detach().cpu().numpy()

        above_threshold = above_threshold.detach().cpu().numpy()
        best_idx = best_idx.detach().cpu().numpy()

        for (
            cls,
            t1_raw,
            tk_raw,
            t1_corr,
            tk_corr,
            entropy,
            margin,
            above,
            bidx,
            raw_vec,
            corrected_vec,
        ) in zip(
            batch_y,
            top1_raw_sim,
            topk_raw_mean_sim,
            top1_corrected_sim,
            topk_corrected_mean_sim,
            topk_entropy,
            corrected_margin,
            above_threshold,
            best_idx,
            raw_sims_cpu,
            corrected_sims_cpu,
        ):
            cls = int(cls)

            class_top1_raw_sims[cls].append(float(t1_raw))
            class_topk_raw_mean_sims[cls].append(float(tk_raw))

            class_top1_corrected_sims[cls].append(float(t1_corr))
            class_topk_corrected_mean_sims[cls].append(float(tk_corr))

            class_prop_raw_above[cls].append(float(above))
            class_entropy[cls].append(float(entropy))
            class_margin_corrected[cls].append(float(margin))

            class_sample_argmax_indices[cls].append(int(bidx))

            if class_sum_raw_sims[cls] is None:
                class_sum_raw_sims[cls] = raw_vec.astype(np.float64)
                class_sum_corrected_sims[cls] = corrected_vec.astype(np.float64)
            else:
                class_sum_raw_sims[cls] += raw_vec.astype(np.float64)
                class_sum_corrected_sims[cls] += corrected_vec.astype(np.float64)

            class_counts[cls] += 1

        del (
            z,
            raw_sims,
            corrected_sims,
            topk_corrected_vals,
            topk_idx,
            topk_raw_vals,
            top1_raw_sim,
            topk_raw_mean_sim,
            top1_corrected_sim,
            topk_corrected_mean_sim,
            topk_entropy,
            corrected_margin,
            above_threshold,
            best_idx,
        )

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    rows = write_results_csv(
        output_csv=output_csv,
        class_top1_raw_sims=class_top1_raw_sims,
        class_topk_raw_mean_sims=class_topk_raw_mean_sims,
        class_top1_corrected_sims=class_top1_corrected_sims,
        class_topk_corrected_mean_sims=class_topk_corrected_mean_sims,
        class_prop_raw_above=class_prop_raw_above,
        class_entropy=class_entropy,
        class_margin_corrected=class_margin_corrected,
        class_sample_argmax_indices=class_sample_argmax_indices,
        class_sum_raw_sims=class_sum_raw_sims,
        class_sum_corrected_sims=class_sum_corrected_sims,
        class_counts=class_counts,
        babel_labels=babel_labels,
        threshold=threshold,
        top_k=top_k,
    )

    return rows


def filter_by_class_chunk(X, y, chunk_id, num_chunks):
    all_classes = np.array(sorted(np.unique(y)))
    class_chunks = np.array_split(all_classes, num_chunks)

    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError(f"chunk_id must be in [0, {num_chunks - 1}], got {chunk_id}")

    selected_classes = class_chunks[chunk_id]
    keep_idx = np.where(np.isin(y, selected_classes))[0]

    print("\nClass chunking")
    print("--------------")
    print(f"All classes: {all_classes.tolist()}")
    print(f"Num chunks: {num_chunks}")
    print(f"Chunk ID: {chunk_id}")
    print(f"Selected classes: {selected_classes.tolist()}")
    print(f"Selected samples: {len(keep_idx)}")

    return X[keep_idx], y[keep_idx], selected_classes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--x_path", type=str, required=True)
    parser.add_argument("--y_path", type=str, required=True)
    parser.add_argument("--babel_labels_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--entropy_temperature", type=float, default=0.05)

    parser.add_argument(
        "--clip_path",
        type=str,
        default="/home/mgirishnair/.cache/clip/ViT-B-32.pt",
    )
    parser.add_argument("--use_prompt_averaging", type=str2bool, default=True)
    parser.add_argument("--use_hub_correction", type=str2bool, default=True)
    parser.add_argument("--run_diagnostics", type=str2bool, default=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X = np.load(args.x_path, mmap_mode="r")
    y = np.load(args.y_path)

    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)

    X, y, selected_classes = filter_by_class_chunk(
        X=X,
        y=y,
        chunk_id=args.chunk_id,
        num_chunks=args.num_chunks,
    )

    print("Chunk X shape:", X.shape)
    print("Chunk y shape:", y.shape)

    with open(args.babel_labels_path, "r") as f:
        babel_labels = [line.strip() for line in f if line.strip()]

    print("Number of BABEL labels:", len(babel_labels))
    print("Top-k:", args.top_k)
    print("Use prompt averaging:", args.use_prompt_averaging)
    print("Use hub correction:", args.use_hub_correction)
    print("Run diagnostics:", args.run_diagnostics)

    encoder = load_motionclip_model(args.checkpoint_path, device=device)

    babel_text_emb = encode_babel_labels(
        babel_labels=babel_labels,
        device=device,
        clip_path=args.clip_path,
        use_prompt_averaging=args.use_prompt_averaging,
    )

    rows = run_streaming_overlap(
        encoder=encoder,
        X=X,
        y=y,
        babel_text_emb=babel_text_emb,
        babel_labels=babel_labels,
        batch_size=args.batch_size,
        threshold=args.threshold,
        output_csv=args.output_csv,
        device=device,
        top_k=args.top_k,
        entropy_temperature=args.entropy_temperature,
        use_hub_correction=args.use_hub_correction,
        run_diagnostics=args.run_diagnostics,
    )

    groups = Counter(row["overlap_group"] for row in rows)

    high_overlap_ratio = groups.get("high", 0) / max(1, len(rows))
    avg_babel_likeness_score = float(np.mean([row["babel_likeness_score"] for row in rows]))
    avg_raw_topk_similarity = float(np.mean([row["median_topk_raw_similarity"] for row in rows]))
    avg_corrected_margin = float(np.mean([row["mean_corrected_top1_top2_margin"] for row in rows]))
    avg_entropy = float(np.mean([row["normalized_entropy_log_topk"] for row in rows]))

    print("\nSaved class-level overlap CSV to:")
    print(args.output_csv)

    print("\nOverlap group counts:")
    for k, v in groups.items():
        print(f"{k}: {v}")

    print("\nChunk-level summary")
    print("-------------------")
    print(f"Selected classes: {selected_classes.tolist()}")
    print(f"High-overlap class ratio: {high_overlap_ratio:.4f}")
    print(f"Average BABEL-likeness score: {avg_babel_likeness_score:.4f}")
    print(f"Average median raw top-k similarity: {avg_raw_topk_similarity:.4f}")
    print(f"Average corrected top1-top2 margin: {avg_corrected_margin:.6f}")
    print(f"Average normalized entropy: {avg_entropy:.4f}")

    print("\nLowest BABEL-like NTU classes in this chunk")
    print("------------------------------------------")

    rows_sorted = sorted(rows, key=lambda r: r["babel_likeness_score"])

    for row in rows_sorted[:20]:
        print(
            f"NTU {row['ntu_class']}: "
            f"score={row['babel_likeness_score']:.4f}, "
            f"group={row['overlap_group']}, "
            f"class_top1={row['class_level_top1_babel_label']}, "
            f"entropy={row['normalized_entropy_log_topk']:.4f}, "
            f"margin={row['mean_corrected_top1_top2_margin']:.6f}"
        )


if __name__ == "__main__":
    main()
