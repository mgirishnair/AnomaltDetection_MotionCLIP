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

    if missing:
        print("Warning: missing encoder keys:", missing)
    if unexpected:
        print("Warning: unexpected encoder keys:", unexpected)

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


@torch.no_grad()
def encode_babel_labels(babel_labels, device):
    clip_model, _ = clip.load("/home/mgirishnair/.cache/clip/ViT-B-32.pt", device="cpu")
    clip_model.eval()


    clip_model = clip_model.float()

    tokens = clip.tokenize(babel_labels).to("cpu")
    text_emb = clip_model.encode_text(tokens).float()

    #tokens = clip.tokenize(babel_labels).to(device)
    #text_emb = clip_model.encode_text(tokens).float()
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.to(device)

    return text_emb


@torch.no_grad()
def encode_motion_batch(encoder, batch, device):
    batch["x"] = batch["x"].to(device).float()
    batch["lengths"] = batch["lengths"].to(device)
    batch["mask"] = batch["mask"].to(device)

    out = encoder(batch)
    z = out["mu"].float()
    z = z / z.norm(dim=-1, keepdim=True)

    return z


def assign_overlap(median_sim, prop_above, threshold):
    if median_sim >= threshold and prop_above >= 0.5:
        return "high"
    elif median_sim >= 0.55 or prop_above >= 0.25:
        return "medium"
    else:
        return "low"


def write_results_csv(output_csv, class_sims, class_match_indices, babel_labels, threshold):
    rows = []

    for cls in sorted(class_sims.keys()):
        sims = np.asarray(class_sims[cls], dtype=np.float32)
        match_indices = class_match_indices[cls]

        match_counter = Counter(match_indices)
        dominant_idx, dominant_count = match_counter.most_common(1)[0]
        dominant_label = babel_labels[dominant_idx]

        num_samples = len(sims)
        mean_sim = float(np.mean(sims))
        median_sim = float(np.median(sims))
        max_sim = float(np.max(sims))
        prop_above = float(np.mean(sims > threshold))
        dominant_prop = float(dominant_count / num_samples)
        overlap_group = assign_overlap(median_sim, prop_above, threshold)

        rows.append({
            "ntu_class": int(cls),
            "num_samples": int(num_samples),
            "mean_best_similarity": mean_sim,
            "median_best_similarity": median_sim,
            "max_best_similarity": max_sim,
            f"proportion_above_{threshold}": prop_above,
            "dominant_babel_label": dominant_label,
            "dominant_babel_proportion": dominant_prop,
            "overlap_group": overlap_group,
        })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    fieldnames = [
        "ntu_class",
        "num_samples",
        "mean_best_similarity",
        "median_best_similarity",
        "max_best_similarity",
        f"proportion_above_{threshold}",
        "dominant_babel_label",
        "dominant_babel_proportion",
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
):
    dataset = NTURot6dDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_motionclip,
    )

    class_sims = defaultdict(list)
    class_match_indices = defaultdict(list)

    babel_text_emb = babel_text_emb.to(device)

    for batch in tqdm(loader, desc="Streaming NTU-BABEL overlap"):
        batch_y = batch["y"].cpu().numpy()

        z = encode_motion_batch(encoder, batch, device)      # [B, 512]
        sims = z @ babel_text_emb.T                          # [B, num_babel]
        best_sim, best_idx = sims.max(dim=1)                 # [B]

        best_sim = best_sim.detach().cpu().numpy()
        best_idx = best_idx.detach().cpu().numpy()

        for cls, sim, bidx in zip(batch_y, best_sim, best_idx):
            cls = int(cls)
            class_sims[cls].append(float(sim))
            class_match_indices[cls].append(int(bidx))

        del z, sims, best_sim, best_idx
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    rows = write_results_csv(
        output_csv=output_csv,
        class_sims=class_sims,
        class_match_indices=class_match_indices,
        babel_labels=babel_labels,
        threshold=threshold,
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

    encoder = load_motionclip_model(args.checkpoint_path, device=device)
    babel_text_emb = encode_babel_labels(babel_labels, device=device)

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
    )

    groups = Counter(row["overlap_group"] for row in rows)
    high_overlap_ratio = groups.get("high", 0) / len(rows)
    avg_median_similarity = float(np.mean([row["median_best_similarity"] for row in rows]))

    print("\nSaved class-level overlap CSV to:")
    print(args.output_csv)

    print("\nOverlap group counts:")
    for k, v in groups.items():
        print(f"{k}: {v}")

    print("\nChunk-level summary")
    print("-------------------")
    print(f"Selected classes: {selected_classes.tolist()}")
    print(f"High-overlap class ratio: {high_overlap_ratio:.4f}")
    print(f"Average median similarity: {avg_median_similarity:.4f}")


if __name__ == "__main__":
    main()
