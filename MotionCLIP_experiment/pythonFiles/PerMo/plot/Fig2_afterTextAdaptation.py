from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def plot_condition_action_similarity(
    export_path: str,
    output_path: str,
    title: str = "After Text Adaptation",
    vmin: float = 0.3,
    vmax: float = 1.0,
) -> None:
    try:
        payload = torch.load(
            export_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(export_path, map_location="cpu")

    if int(payload["num_templates"]) != 1:
        raise ValueError(
            "This plot expects exactly one prompt per condition-action pair."
        )

    metadata = pd.DataFrame(payload["metadata"])

    # Arrange the matrix as:
    # healthy action 1, healthy action 2, ...
    # condition 1 action 1, condition 1 action 2, ...
    metadata = metadata.sort_values(
        by=[
            "is_normal",
            "condition_index",
            "action_index",
            "template_index",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    indices = metadata["flat_index"].astype(int).to_numpy()

    features = payload["features"][indices].float()
    features = F.normalize(features, dim=-1)

    similarity = (features @ features.T).numpy()
    labels = metadata["prompt_text"].tolist()

    n_normal = int(metadata["is_normal"].sum())
    n_total = len(metadata)

    fig, ax = plt.subplots(
        figsize=(max(18, n_total * 0.28), max(16, n_total * 0.25))
    )

    image = ax.imshow(
        similarity,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title(title, fontsize=15, pad=12)

    ax.set_xticks(np.arange(n_total))
    ax.set_yticks(np.arange(n_total))

    ax.set_xticklabels(
        labels,
        rotation=55,
        ha="right",
        fontsize=5,
    )
    ax.set_yticklabels(labels, fontsize=5)

    # Normal/anomaly block separator.
    boundary = n_normal - 0.5
    ax.axhline(boundary, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(boundary, color="black", linestyle="--", linewidth=1.5)

    # Similarity numbers inside the cells.
    midpoint = (vmin + vmax) / 2

    for row in range(n_total):
        for column in range(n_total):
            value = similarity[row, column]

            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=2.5,
                color="white" if value > midpoint else "black",
            )

    # Normal/anomaly labels on the left.
    normal_y = 1.0 - n_normal / (2.0 * n_total)
    anomaly_y = (n_total - n_normal) / (2.0 * n_total)

    ax.text(
        -0.075,
        normal_y,
        "Normal",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        bbox={
            "facecolor": "#dcebf7",
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    )

    ax.text(
        -0.075,
        anomaly_y,
        "Anomaly",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        bbox={
            "facecolor": "#f4dfa0",
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.05)
    colorbar.set_label("Cosine similarity")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--export_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--title", default="After Text Adaptation")
    parser.add_argument("--vmin", type=float, default=0.3)
    parser.add_argument("--vmax", type=float, default=1.0)
    args = parser.parse_args()

    plot_condition_action_similarity(
        export_path=args.export_path,
        output_path=args.output_path,
        title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
    )
