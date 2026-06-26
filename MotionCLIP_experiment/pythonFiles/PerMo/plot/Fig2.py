#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_prompt_txt(path):
    normal_prompts = []
    anomaly_prompts = []
    current_section = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            lower = line.lower()

            if lower in {"[normal]", "normal:"}:
                current_section = "normal"
                continue

            if lower in {"[anomaly]", "anomaly:", "[abnormal]", "abnormal:"}:
                current_section = "anomaly"
                continue

            if current_section == "normal":
                normal_prompts.append(line)
            elif current_section == "anomaly":
                anomaly_prompts.append(line)
            else:
                raise ValueError(
                    f"Prompt appears before [normal] or [anomaly] section: {line}"
                )

    if not normal_prompts:
        raise ValueError("No normal prompts found.")
    if not anomaly_prompts:
        raise ValueError("No anomaly prompts found.")

    return normal_prompts, anomaly_prompts


class FrozenCLIPTextEncoder:
    def __init__(self, clip_model_name, device):
        import clip

        self.clip = clip
        self.model, _ = clip.load(clip_model_name, device=device)
        self.model = self.model.float().eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device

    @torch.no_grad()
    def encode(self, texts, batch_size=256):
        feats = []

        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            tokens = self.clip.tokenize(chunk, truncate=True).to(self.device)

            text_features = self.model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)

            feats.append(text_features.cpu())

        return torch.cat(feats, dim=0)


def plot_prompt_similarity_heatmap(
    prompt_txt,
    save_path,
    clip_model="ViT-B/32",
    device="cuda",
    batch_size=256,
    title="Original CLIP",
    show_values=True,
):
    normal_prompts, anomaly_prompts = read_prompt_txt(prompt_txt)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    text_encoder = FrozenCLIPTextEncoder(
        clip_model_name=clip_model,
        device=device,
    )

    prompts = normal_prompts + anomaly_prompts
    labels = normal_prompts + anomaly_prompts

    text_features = text_encoder.encode(prompts, batch_size=batch_size)
    sim_matrix = (text_features @ text_features.T).numpy()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path.with_suffix(".npy"), sim_matrix)

    n_normal = len(normal_prompts)
    n_anomaly = len(anomaly_prompts)
    n_total = len(prompts)

    fig_w = max(12, min(42, 0.42 * n_total + 8))
    fig_h = max(10, min(42, 0.34 * n_total + 6))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        sim_matrix,
        vmin=0.3,
        vmax=1.0,
        cmap="RdYlBu_r",
    )

    ax.set_title(title, fontsize=14)

    ax.set_xticks(np.arange(n_total))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)

    # Remove default y-axis labels.
    # We draw them manually so they sit between Normal/Anomaly labels and heatmap.
    ax.set_yticks(np.arange(n_total))
    ax.set_yticklabels([])

    # Cell grid.
    ax.set_xticks(np.arange(n_total + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_total + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    split = n_normal - 0.5

    # Normal/anomaly quadrant split.
    ax.axhline(split, color="black", linestyle="--", linewidth=2.2)
    ax.axvline(split, color="black", linestyle="--", linewidth=2.2)

    # Outer dashed border.
    outer_border = plt.Rectangle(
        (-0.5, -0.5),
        n_total,
        n_total,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=2.0,
    )
    ax.add_patch(outer_border)

    # Normal-normal quadrant border.
    normal_box = plt.Rectangle(
        (-0.5, -0.5),
        n_normal,
        n_normal,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=1.4,
    )
    ax.add_patch(normal_box)

    # Anomaly-anomaly quadrant border.
    anomaly_box = plt.Rectangle(
        (n_normal - 0.5, n_normal - 0.5),
        n_anomaly,
        n_anomaly,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=1.4,
    )
    ax.add_patch(anomaly_box)

    if show_values:
        value_fontsize = 4.5 if n_total > 50 else 7
        for i in range(n_total):
            for j in range(n_total):
                ax.text(
                    j,
                    i,
                    f"{sim_matrix[i, j]:.2f}".rstrip("0").rstrip("."),
                    ha="center",
                    va="center",
                    fontsize=value_fontsize,
                    color="black",
                )

    # Extend x-axis leftwards to create space for group labels and prompt labels.
    ax.set_xlim(-4.2, n_total - 0.5)
    ax.set_ylim(n_total - 0.5, -0.5)

    # Leftmost: Normal / Anomaly label.
    group_label_x = -3.7

    ax.text(
        group_label_x,
        (n_normal - 1) / 2,
        "Normal",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
        bbox=dict(
            facecolor="#dceaf7",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.95,
        ),
    )

    ax.text(
        group_label_x,
        n_normal + (n_anomaly - 1) / 2,
        "Anomaly",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
        bbox=dict(
            facecolor="#f7df9e",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.95,
        ),
    )

    # Middle-left: prompt labels.
    prompt_label_x = -0.8

    for i, label in enumerate(labels):
        ax.text(
            prompt_label_x,
            i,
            label,
            va="center",
            ha="right",
            fontsize=6,
            color="black",
        )

    # Add a subtle vertical separator between manual labels and heatmap.
    ax.axvline(-0.5, color="black", linestyle="--", linewidth=1.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return sim_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Create AA-CLIP-style normal/anomaly prompt similarity heatmap using OpenAI CLIP."
    )

    parser.add_argument("--prompt_txt", required=True)
    parser.add_argument("--save_path", default="prompt_similarity_heatmap.png")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--title", default="Original CLIP")
    parser.add_argument("--hide_values", action="store_true")

    args = parser.parse_args()

    plot_prompt_similarity_heatmap(
        prompt_txt=args.prompt_txt,
        save_path=args.save_path,
        clip_model=args.clip_model,
        device=args.device,
        batch_size=args.batch_size,
        title=args.title,
        show_values=not args.hide_values,
    )


if __name__ == "__main__":
    main()
