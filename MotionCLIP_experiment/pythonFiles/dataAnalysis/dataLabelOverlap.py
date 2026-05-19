#!/usr/bin/env python
# coding: utf-8

import os
import csv
import math
import argparse

import numpy as np
import torch
import clip
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def read_label_file(path):
    labels = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


@torch.no_grad()
def encode_labels_with_clip(
    labels,
    clip_model,
    device,
    use_prompt_averaging=True,
):
    if use_prompt_averaging:
        templates = [
            "a person is {}",
            "someone is performing {}",
            "a human action of {}",
            "a person doing {}",
            "a motion sequence of {}",
        ]
    else:
        templates = ["{}"]

    all_embs = []

    for label in tqdm(labels, desc="Encoding text labels"):
        texts = [template.format(label) for template in templates]
        tokens = clip.tokenize(texts, truncate=True).to(device)

        emb = clip_model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        emb = emb.mean(dim=0)
        emb = emb / emb.norm().clamp_min(1e-8)

        all_embs.append(emb.detach().cpu())

    return torch.stack(all_embs, dim=0)


def entropy_from_topk(topk_scores, temperature=0.05, eps=1e-8):
    logits = torch.tensor(topk_scores, dtype=torch.float32) / temperature
    probs = torch.softmax(logits, dim=0)
    entropy = -torch.sum(probs * torch.log(probs + eps)).item()
    return entropy


def overlap_group(score):
    if score >= 0.70:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ntu_labels_path", type=str, required=True)
    parser.add_argument("--babel_labels_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument(
        "--clip_path",
        type=str,
        default="/home/mgirishnair/.cache/clip/ViT-B-32.pt",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--entropy_temperature", type=float, default=0.05)
    parser.add_argument("--use_prompt_averaging", type=str2bool, default=True)
    parser.add_argument("--use_hub_correction", type=str2bool, default=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    ntu_labels = read_label_file(args.ntu_labels_path)
    babel_labels = read_label_file(args.babel_labels_path)

    print("Number of NTU labels:", len(ntu_labels))
    print("Number of BABEL labels:", len(babel_labels))

    clip_model, _ = clip.load(args.clip_path, device=device)
    clip_model.eval()
    clip_model = clip_model.float()

    ntu_emb = encode_labels_with_clip(
        labels=ntu_labels,
        clip_model=clip_model,
        device=device,
        use_prompt_averaging=args.use_prompt_averaging,
    )

    babel_emb = encode_labels_with_clip(
        labels=babel_labels,
        clip_model=clip_model,
        device=device,
        use_prompt_averaging=args.use_prompt_averaging,
    )

    sims = ntu_emb @ babel_emb.T  # [num_ntu, num_babel]

    if args.use_hub_correction:
        babel_bias = sims.mean(dim=0, keepdim=True)
        corrected_sims = sims - babel_bias
    else:
        corrected_sims = sims.clone()

    top_k = min(args.top_k, len(babel_labels))

    rows = []

    for ntu_idx, ntu_label in enumerate(ntu_labels):
        raw_vec = sims[ntu_idx].numpy()
        corrected_vec = corrected_sims[ntu_idx].numpy()

        top_idx = np.argsort(corrected_vec)[::-1][:top_k]

        top_labels = [babel_labels[i] for i in top_idx]
        top_raw_scores = [float(raw_vec[i]) for i in top_idx]
        top_corrected_scores = [float(corrected_vec[i]) for i in top_idx]

        top1_raw = top_raw_scores[0]
        top1_corrected = top_corrected_scores[0]

        if top_k >= 2:
            top1_top2_margin_corrected = top_corrected_scores[0] - top_corrected_scores[1]
        else:
            top1_top2_margin_corrected = 0.0

        topk_raw_mean = float(np.mean(top_raw_scores))
        topk_corrected_mean = float(np.mean(top_corrected_scores))

        ent = entropy_from_topk(
            top_corrected_scores,
            temperature=args.entropy_temperature,
        )
        normalized_entropy = ent / math.log(top_k) if top_k > 1 else 0.0

        certainty = max(0.0, min(1.0, 1.0 - normalized_entropy))
        raw_sim_component = max(0.0, min(1.0, (topk_raw_mean - 0.65) / 0.15))
        margin_component = max(0.0, min(1.0, top1_top2_margin_corrected / 0.03))

        label_overlap_score = (
            0.60 * raw_sim_component
            + 0.25 * margin_component
            + 0.15 * certainty
        )

        rows.append({
            "ntu_index": ntu_idx,
            "ntu_label": ntu_label,

            "top1_babel_label": top_labels[0],
            "top1_raw_similarity": top1_raw,
            "top1_corrected_similarity": top1_corrected,

            "topk_babel_labels": " | ".join(top_labels),
            "topk_raw_scores": " | ".join(f"{s:.6f}" for s in top_raw_scores),
            "topk_corrected_scores": " | ".join(f"{s:.6f}" for s in top_corrected_scores),

            "topk_raw_mean": topk_raw_mean,
            "topk_corrected_mean": topk_corrected_mean,

            "top1_top2_margin_corrected": top1_top2_margin_corrected,
            "topk_entropy": ent,
            "normalized_entropy_log_topk": normalized_entropy,

            "label_overlap_score": float(label_overlap_score),
            "overlap_group": overlap_group(label_overlap_score),
        })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    fieldnames = [
        "ntu_index",
        "ntu_label",

        "top1_babel_label",
        "top1_raw_similarity",
        "top1_corrected_similarity",

        "topk_babel_labels",
        "topk_raw_scores",
        "topk_corrected_scores",

        "topk_raw_mean",
        "topk_corrected_mean",

        "top1_top2_margin_corrected",
        "topk_entropy",
        "normalized_entropy_log_topk",

        "label_overlap_score",
        "overlap_group",
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved label-text overlap CSV to:")
    print(args.output_csv)

    rows_sorted = sorted(rows, key=lambda r: r["label_overlap_score"])

    print("\nLowest BABEL-like NTU labels")
    print("----------------------------")
    for row in rows_sorted[:20]:
        print(
            f"NTU {row['ntu_index']}: {row['ntu_label']} | "
            f"score={row['label_overlap_score']:.4f} | "
            f"group={row['overlap_group']} | "
            f"top1={row['top1_babel_label']} | "
            f"raw={row['top1_raw_similarity']:.4f} | "
            f"margin={row['top1_top2_margin_corrected']:.6f} | "
            f"entropy={row['normalized_entropy_log_topk']:.4f}"
        )


if __name__ == "__main__":
    main()
