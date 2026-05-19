import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError as e:
    raise ImportError(
        "umap-learn is required for the UMAP plot. "
        "Install it with: pip install umap-learn"
    ) from e


def load_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    required = [
        "Z_train_normal",
        "Z_test_normal",
        "Z_test_abnormal",
        "y_train_normal",
        "y_test_normal",
        "y_test_abnormal",
    ]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key in npz file: {key}")

    return {
        "Z_train_normal": data["Z_train_normal"],
        "Z_test_normal": data["Z_test_normal"],
        "Z_test_abnormal": data["Z_test_abnormal"],
        "y_train_normal": data["y_train_normal"],
        "y_test_normal": data["y_test_normal"],
        "y_test_abnormal": data["y_test_abnormal"],
        "split_name": str(data["split_name"]) if "split_name" in data else "unknown_split",
        "normal_classes": data["normal_classes"] if "normal_classes" in data else None,
    }


def fit_and_project_pca(Z_train_normal, Z_test_normal, Z_test_abnormal, fit_on="train"):
    if fit_on == "train":
        Z_fit = Z_train_normal
    elif fit_on == "all":
        Z_fit = np.concatenate([Z_train_normal, Z_test_normal, Z_test_abnormal], axis=0)
    else:
        raise ValueError("--fit_on must be 'train' or 'all'")

    pca = PCA(n_components=2)
    pca.fit(Z_fit)

    train_2d = pca.transform(Z_train_normal)
    test_normal_2d = pca.transform(Z_test_normal)
    test_abnormal_2d = pca.transform(Z_test_abnormal)

    return pca, train_2d, test_normal_2d, test_abnormal_2d


def compute_class_means(points_2d, labels):
    means = []
    counts = []

    for cls in np.unique(labels):
        mask = labels == cls
        cls_points = points_2d[mask]
        mean_xy = cls_points.mean(axis=0)

        means.append([int(cls), mean_xy[0], mean_xy[1]])
        counts.append([int(cls), int(mask.sum())])

    means = np.array(means, dtype=float)
    counts = np.array(counts, dtype=int)
    return means, counts


def counts_to_dict(counts_array):
    return {int(row[0]): int(row[1]) for row in counts_array}


def add_cov_ellipse(
    ax,
    points_2d,
    n_std=2.0,
    edgecolor=None,
    facecolor=None,
    alpha=0.18,
    linewidth=1.5,
    zorder=1,
):
    if len(points_2d) < 2:
        return

    cov = np.cov(points_2d, rowvar=False)
    mean = points_2d.mean(axis=0)

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=edgecolor,
        facecolor=facecolor if facecolor is not None else edgecolor,
        fill=True,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(ell)


def subsample_points(points_2d, labels, max_points=None, seed=42):
    if max_points is None or len(points_2d) <= max_points:
        return points_2d, labels

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points_2d), size=max_points, replace=False)
    return points_2d[idx], labels[idx]


def fit_class_gaussians(Z_train_normal, y_train_normal, reg_eps=1e-6):
    gaussian_dict = {}

    for cls in np.unique(y_train_normal):
        cls_mask = y_train_normal == cls
        Z_cls = Z_train_normal[cls_mask]

        mu = np.mean(Z_cls, axis=0)
        cov = np.cov(Z_cls, rowvar=False)

        cov = cov + reg_eps * np.eye(cov.shape[0])
        inv_cov = np.linalg.inv(cov)

        gaussian_dict[int(cls)] = {
            "mu": mu,
            "cov": cov,
            "inv_cov": inv_cov,
        }

    return gaussian_dict


def mahalanobis_to_gaussian_batch(Z, mu, inv_cov):
    diff = Z - mu
    left = diff @ inv_cov
    d2 = np.sum(left * diff, axis=1)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


def compute_min_mahalanobis_distances(Z, gaussian_dict):
    all_dists = []
    class_order = sorted(gaussian_dict.keys())

    for cls in class_order:
        mu = gaussian_dict[cls]["mu"]
        inv_cov = gaussian_dict[cls]["inv_cov"]
        d = mahalanobis_to_gaussian_batch(Z, mu, inv_cov)
        all_dists.append(d)

    all_dists = np.stack(all_dists, axis=1)  # [N, num_classes]
    min_dists = np.min(all_dists, axis=1)
    nearest_cls = np.argmin(all_dists, axis=1)

    return min_dists, nearest_cls, class_order, all_dists


def plot_distance_histogram(
    d_train_min,
    d_test_normal_min,
    d_test_abnormal_min,
    split_name,
    normal_classes,
    output_path,
):
    plt.figure(figsize=(9, 6))

    plt.hist(d_train_min, bins=50, alpha=0.35, label="train normal")
    plt.hist(d_test_normal_min, bins=50, alpha=0.45, label="test normal")
    plt.hist(d_test_abnormal_min, bins=50, alpha=0.55, label="abnormal")

    title = f"Min Mahalanobis distance histogram - {split_name}"
    if normal_classes is not None:
        title += f"\nNormal classes: {list(map(int, normal_classes))}"

    plt.title(title, fontsize=14)
    plt.xlabel("Minimum Mahalanobis distance to any normal-class Gaussian")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.25)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distance_cdf(
    d_train_min,
    d_test_normal_min,
    d_test_abnormal_min,
    split_name,
    normal_classes,
    output_path,
):
    plt.figure(figsize=(9, 6))

    def plot_cdf(data, label):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=label, linewidth=2)

    plot_cdf(d_train_min, "train normal")
    plot_cdf(d_test_normal_min, "test normal")
    plot_cdf(d_test_abnormal_min, "abnormal")

    title = f"CDF of min Mahalanobis distance - {split_name}"
    if normal_classes is not None:
        title += f"\nNormal classes: {list(map(int, normal_classes))}"

    plt.title(title, fontsize=14)
    plt.xlabel("Minimum Mahalanobis distance to any normal-class Gaussian")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True, alpha=0.25)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_umap(
    Z_train_normal,
    Z_test_normal,
    Z_test_abnormal,
    y_train_normal,
    y_test_normal,
    y_test_abnormal,
    split_name,
    normal_classes,
    output_path,
    max_abnormal_points=200,
    abnormal_seed=42,
):
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        random_state=42,
    )

    Z_all = np.concatenate([Z_train_normal, Z_test_normal, Z_test_abnormal], axis=0)
    Z_2d = reducer.fit_transform(Z_all)

    n_train = len(Z_train_normal)
    n_test_normal = len(Z_test_normal)

    train_2d = Z_2d[:n_train]
    test_normal_2d = Z_2d[n_train:n_train + n_test_normal]
    test_abnormal_2d = Z_2d[n_train + n_test_normal:]

    abn_pts_plot, _ = subsample_points(
        test_abnormal_2d,
        y_test_abnormal,
        max_points=max_abnormal_points,
        seed=abnormal_seed,
    )

    cmap = plt.get_cmap("tab10")
    normal_cls_list = sorted(np.unique(y_test_normal))
    color_map = {cls: cmap(i % 10) for i, cls in enumerate(normal_cls_list)}

    fig, ax = plt.subplots(figsize=(11, 8))

    first_train_label = True
    for cls in np.unique(y_train_normal):
        mask = y_train_normal == cls
        pts = train_2d[mask]
        c = color_map.get(int(cls), "C0")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=10,
            alpha=0.18,
            color=c,
            label="train normal" if first_train_label else None,
            zorder=2,
        )
        first_train_label = False

    first_test_label = True
    for cls in np.unique(y_test_normal):
        mask = y_test_normal == cls
        pts = test_normal_2d[mask]
        c = color_map.get(int(cls), "C1")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=22,
            alpha=0.50,
            marker="^",
            color=c,
            label="test normal" if first_test_label else None,
            zorder=3,
        )
        first_test_label = False

    ax.scatter(
        abn_pts_plot[:, 0],
        abn_pts_plot[:, 1],
        s=70,
        alpha=0.9,
        marker="x",
        color="black",
        linewidths=1.4,
        label=f"abnormal (subsampled, n={len(abn_pts_plot)})",
        zorder=5,
    )

    title = f"UMAP of MotionCLIP embeddings - {split_name}"
    if normal_classes is not None:
        title += f"\nNormal classes: {list(map(int, normal_classes))}"

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        frameon=True,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_tsne(
    Z_train_normal,
    Z_test_normal,
    Z_test_abnormal,
    y_train_normal,
    y_test_normal,
    y_test_abnormal,
    split_name,
    normal_classes,
    output_path,
    max_abnormal_points=200,
    abnormal_seed=42,
):
    Z_all = np.concatenate([Z_train_normal, Z_test_normal, Z_test_abnormal], axis=0)

    perplexity = min(30, max(5, (len(Z_all) - 1) // 3))

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )

    Z_2d = reducer.fit_transform(Z_all)

    n_train = len(Z_train_normal)
    n_test_normal = len(Z_test_normal)

    train_2d = Z_2d[:n_train]
    test_normal_2d = Z_2d[n_train:n_train + n_test_normal]
    test_abnormal_2d = Z_2d[n_train + n_test_normal:]

    abn_pts_plot, _ = subsample_points(
        test_abnormal_2d,
        y_test_abnormal,
        max_points=max_abnormal_points,
        seed=abnormal_seed,
    )

    cmap = plt.get_cmap("tab10")
    normal_cls_list = sorted(np.unique(y_test_normal))
    color_map = {cls: cmap(i % 10) for i, cls in enumerate(normal_cls_list)}

    fig, ax = plt.subplots(figsize=(11, 8))

    first_train_label = True
    for cls in np.unique(y_train_normal):
        mask = y_train_normal == cls
        pts = train_2d[mask]
        c = color_map.get(int(cls), "C0")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=10,
            alpha=0.18,
            color=c,
            label="train normal" if first_train_label else None,
            zorder=2,
        )
        first_train_label = False

    first_test_label = True
    for cls in np.unique(y_test_normal):
        mask = y_test_normal == cls
        pts = test_normal_2d[mask]
        c = color_map.get(int(cls), "C1")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=22,
            alpha=0.50,
            marker="^",
            color=c,
            label="test normal" if first_test_label else None,
            zorder=3,
        )
        first_test_label = False

    ax.scatter(
        abn_pts_plot[:, 0],
        abn_pts_plot[:, 1],
        s=70,
        alpha=0.9,
        marker="x",
        color="black",
        linewidths=1.4,
        label=f"abnormal (subsampled, n={len(abn_pts_plot)})",
        zorder=5,
    )

    title = f"t-SNE of MotionCLIP embeddings - {split_name}"
    if normal_classes is not None:
        title += f"\nNormal classes: {list(map(int, normal_classes))}"

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        frameon=True,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_clean_pca(
    train_means,
    test_normal_means,
    test_abnormal_means,
    train_counts,
    test_normal_counts,
    test_abnormal_counts,
    train_2d,
    test_normal_2d,
    test_abnormal_2d,
    y_train_normal,
    y_test_normal,
    y_test_abnormal,
    split_name,
    normal_classes,
    output_path,
    annotate=False,
    show_train=True,
    max_abnormal_points=200,
    abnormal_seed=42,
    ellipse_std=2.0,
):
    fig, ax = plt.subplots(figsize=(13, 9))

    train_count_dict = counts_to_dict(train_counts)
    test_normal_count_dict = counts_to_dict(test_normal_counts)
    test_abnormal_count_dict = counts_to_dict(test_abnormal_counts)

    cmap = plt.get_cmap("tab10")
    normal_cls_list = sorted(np.unique(y_test_normal))
    color_map = {cls: cmap(i % 10) for i, cls in enumerate(normal_cls_list)}

    abn_pts_plot, abn_lbl_plot = subsample_points(
        test_abnormal_2d,
        y_test_abnormal,
        max_points=max_abnormal_points,
        seed=abnormal_seed,
    )

    ax.scatter(
        abn_pts_plot[:, 0],
        abn_pts_plot[:, 1],
        s=80,
        marker="x",
        color="black",
        alpha=0.9,
        linewidths=1.5,
        label=f"abnormal samples (subsampled, n={len(abn_pts_plot)})",
        zorder=10,
    )

    if show_train:
        first_train_label = True
        for row in train_means:
            cls = int(row[0])
            x, y = row[1], row[2]
            cls_points = train_2d[y_train_normal == cls]
            c = color_map.get(cls, "C0")

            add_cov_ellipse(
                ax,
                cls_points,
                n_std=ellipse_std,
                edgecolor=c,
                facecolor=c,
                alpha=0.08,
                linewidth=1.2,
                zorder=1,
            )

            ax.scatter(
                cls_points[:, 0],
                cls_points[:, 1],
                s=18,
                marker="o",
                color=c,
                alpha=0.18,
                label="train normal samples" if first_train_label else None,
                zorder=3,
            )
            first_train_label = False

            ax.scatter(
                x,
                y,
                s=180,
                marker="o",
                color=c,
                edgecolor="black",
                linewidth=0.8,
                label=f"train mean cls {cls}",
                zorder=5,
            )

            if annotate:
                ax.text(
                    x,
                    y,
                    f"T{cls}\n(n={train_count_dict[cls]})",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )

    first_test_label = True
    for row in test_normal_means:
        cls = int(row[0])
        x, y = row[1], row[2]
        cls_points = test_normal_2d[y_test_normal == cls]
        c = color_map.get(cls, "C1")

        add_cov_ellipse(
            ax,
            cls_points,
            n_std=ellipse_std,
            edgecolor=c,
            facecolor=c,
            alpha=0.18,
            linewidth=1.5,
            zorder=1,
        )

        ax.scatter(
            cls_points[:, 0],
            cls_points[:, 1],
            s=28,
            marker="^",
            color=c,
            alpha=0.45,
            label="test normal samples" if first_test_label else None,
            zorder=4,
        )
        first_test_label = False

        ax.scatter(
            x,
            y,
            s=240,
            marker="X",
            color=c,
            edgecolor="black",
            linewidth=0.9,
            label=f"test mean cls {cls}",
            zorder=6,
        )

        if annotate:
            ax.text(
                x,
                y,
                f"N{cls}\n(n={test_normal_count_dict[cls]})",
                fontsize=8,
                ha="left",
                va="bottom",
            )

    if annotate:
        for row in test_abnormal_means:
            cls = int(row[0])
            x, y = row[1], row[2]
            ax.scatter(
                x,
                y,
                s=70,
                marker="+",
                color="black",
                linewidths=1.2,
                zorder=5,
            )
            ax.text(
                x,
                y,
                f"A{cls}\n(n={test_abnormal_count_dict[cls]})",
                fontsize=7,
                ha="left",
                va="bottom",
                color="black",
            )

    title = f"PCA of MotionCLIP embeddings - {split_name}"
    if normal_classes is not None:
        title += f"\nNormal classes: {list(map(int, normal_classes))}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        frameon=True,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.77, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_normals_only_pca(
    train_means,
    test_normal_means,
    train_counts,
    test_normal_counts,
    train_2d,
    test_normal_2d,
    y_train_normal,
    y_test_normal,
    split_name,
    normal_classes,
    output_path,
    annotate=False,
    show_train=True,
    ellipse_std=2.0,
):
    fig, ax = plt.subplots(figsize=(11, 8))

    train_count_dict = counts_to_dict(train_counts)
    test_normal_count_dict = counts_to_dict(test_normal_counts)

    cmap = plt.get_cmap("tab10")
    normal_cls_list = sorted(np.unique(y_test_normal))
    color_map = {cls: cmap(i % 10) for i, cls in enumerate(normal_cls_list)}

    if show_train:
        first_train_label = True
        for row in train_means:
            cls = int(row[0])
            x, y = row[1], row[2]
            cls_points = train_2d[y_train_normal == cls]
            c = color_map.get(cls, "C0")

            add_cov_ellipse(
                ax,
                cls_points,
                n_std=ellipse_std,
                edgecolor=c,
                facecolor=c,
                alpha=0.08,
                linewidth=1.2,
                zorder=1,
            )

            ax.scatter(
                cls_points[:, 0],
                cls_points[:, 1],
                s=18,
                marker="o",
                color=c,
                alpha=0.18,
                label="train normal samples" if first_train_label else None,
                zorder=3,
            )
            first_train_label = False

            ax.scatter(
                x,
                y,
                s=180,
                marker="o",
                color=c,
                edgecolor="black",
                linewidth=0.8,
                label=f"train mean cls {cls}",
                zorder=5,
            )

            if annotate:
                ax.text(
                    x,
                    y,
                    f"T{cls}\n(n={train_count_dict[cls]})",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )

    first_test_label = True
    for row in test_normal_means:
        cls = int(row[0])
        x, y = row[1], row[2]
        cls_points = test_normal_2d[y_test_normal == cls]
        c = color_map.get(cls, "C1")

        add_cov_ellipse(
            ax,
            cls_points,
            n_std=ellipse_std,
            edgecolor=c,
            facecolor=c,
            alpha=0.18,
            linewidth=1.5,
            zorder=1,
        )

        ax.scatter(
            cls_points[:, 0],
            cls_points[:, 1],
            s=28,
            marker="^",
            color=c,
            alpha=0.45,
            label="test normal samples" if first_test_label else None,
            zorder=4,
        )
        first_test_label = False

        ax.scatter(
            x,
            y,
            s=240,
            marker="X",
            color=c,
            edgecolor="black",
            linewidth=0.9,
            label=f"test mean cls {cls}",
            zorder=6,
        )

        if annotate:
            ax.text(
                x,
                y,
                f"N{cls}\n(n={test_normal_count_dict[cls]})",
                fontsize=8,
                ha="left",
                va="bottom",
            )

    title = f"PCA of MotionCLIP embeddings (normals only) - {split_name}"
    if normal_classes is not None:
        title += f"\nNormal classes: {list(map(int, normal_classes))}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        frameon=True,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.77, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--fit_on", type=str, default="train", choices=["train", "all"])
    parser.add_argument("--hide_train", action="store_true")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--max_abnormal_points", type=int, default=200)
    parser.add_argument("--abnormal_seed", type=int, default=42)
    parser.add_argument("--ellipse_std", type=float, default=2.0)
    parser.add_argument(
        "--make_normals_only_plot",
        action="store_true",
        help="Also save a second plot with only normal classes.",
    )
    args = parser.parse_args()

    data = load_embeddings(args.embeddings_path)

    pca, train_2d, test_normal_2d, test_abnormal_2d = fit_and_project_pca(
        data["Z_train_normal"],
        data["Z_test_normal"],
        data["Z_test_abnormal"],
        fit_on=args.fit_on,
    )

    train_means, train_counts = compute_class_means(train_2d, data["y_train_normal"])
    test_normal_means, test_normal_counts = compute_class_means(train_2d if False else test_normal_2d, data["y_test_normal"])
    test_abnormal_means, test_abnormal_counts = compute_class_means(test_abnormal_2d, data["y_test_abnormal"])

    plot_clean_pca(
        train_means=train_means,
        test_normal_means=test_normal_means,
        test_abnormal_means=test_abnormal_means,
        train_counts=train_counts,
        test_normal_counts=test_normal_counts,
        test_abnormal_counts=test_abnormal_counts,
        train_2d=train_2d,
        test_normal_2d=test_normal_2d,
        test_abnormal_2d=test_abnormal_2d,
        y_train_normal=data["y_train_normal"],
        y_test_normal=data["y_test_normal"],
        y_test_abnormal=data["y_test_abnormal"],
        split_name=data["split_name"],
        normal_classes=data["normal_classes"],
        output_path=args.output_path,
        annotate=args.annotate,
        show_train=not args.hide_train,
        max_abnormal_points=args.max_abnormal_points,
        abnormal_seed=args.abnormal_seed,
        ellipse_std=args.ellipse_std,
    )

    if args.make_normals_only_plot:
        base, ext = os.path.splitext(args.output_path)
        normals_only_path = f"{base}_normals_only{ext}"
        plot_normals_only_pca(
            train_means=train_means,
            test_normal_means=test_normal_means,
            train_counts=train_counts,
            test_normal_counts=test_normal_counts,
            train_2d=train_2d,
            test_normal_2d=test_normal_2d,
            y_train_normal=data["y_train_normal"],
            y_test_normal=data["y_test_normal"],
            split_name=data["split_name"],
            normal_classes=data["normal_classes"],
            output_path=normals_only_path,
            annotate=args.annotate,
            show_train=not args.hide_train,
            ellipse_std=args.ellipse_std,
        )
        print(f"Saved normals-only PCA plot to: {normals_only_path}")

    gaussian_dict = fit_class_gaussians(
        data["Z_train_normal"],
        data["y_train_normal"],
        reg_eps=1e-6,
    )

    d_train_min, _, _, _ = compute_min_mahalanobis_distances(
        data["Z_train_normal"], gaussian_dict
    )
    d_test_normal_min, _, _, _ = compute_min_mahalanobis_distances(
        data["Z_test_normal"], gaussian_dict
    )
    d_test_abnormal_min, _, _, _ = compute_min_mahalanobis_distances(
        data["Z_test_abnormal"], gaussian_dict
    )

    base, ext = os.path.splitext(args.output_path)
    hist_path = f"{base}_mahal_hist.png"
    cdf_path = f"{base}_mahal_cdf.png"
    umap_path = f"{base}_umap.png"
    tsne_path = f"{base}_tsne.png"

    plot_distance_histogram(
        d_train_min=d_train_min,
        d_test_normal_min=d_test_normal_min,
        d_test_abnormal_min=d_test_abnormal_min,
        split_name=data["split_name"],
        normal_classes=data["normal_classes"],
        output_path=hist_path,
    )

    plot_distance_cdf(
        d_train_min=d_train_min,
        d_test_normal_min=d_test_normal_min,
        d_test_abnormal_min=d_test_abnormal_min,
        split_name=data["split_name"],
        normal_classes=data["normal_classes"],
        output_path=cdf_path,
    )

    plot_umap(
        Z_train_normal=data["Z_train_normal"],
        Z_test_normal=data["Z_test_normal"],
        Z_test_abnormal=data["Z_test_abnormal"],
        y_train_normal=data["y_train_normal"],
        y_test_normal=data["y_test_normal"],
        y_test_abnormal=data["y_test_abnormal"],
        split_name=data["split_name"],
        normal_classes=data["normal_classes"],
        output_path=umap_path,
        max_abnormal_points=args.max_abnormal_points,
        abnormal_seed=args.abnormal_seed,
    )

    plot_tsne(
        Z_train_normal=data["Z_train_normal"],
        Z_test_normal=data["Z_test_normal"],
        Z_test_abnormal=data["Z_test_abnormal"],
        y_train_normal=data["y_train_normal"],
        y_test_normal=data["y_test_normal"],
        y_test_abnormal=data["y_test_abnormal"],
        split_name=data["split_name"],
        normal_classes=data["normal_classes"],
        output_path=tsne_path,
        max_abnormal_points=args.max_abnormal_points,
        abnormal_seed=args.abnormal_seed,
    )

    explained = pca.explained_variance_ratio_
    print(f"Saved PCA plot to: {args.output_path}")
    print(f"Saved min-Mahalanobis histogram to: {hist_path}")
    print(f"Saved min-Mahalanobis CDF to: {cdf_path}")
    print(f"Saved UMAP plot to: {umap_path}")
    print(f"Explained variance ratio: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}")
    print(f"Total shown in 2D: {(explained[0] + explained[1]):.4f}")
    print(f"Saved t-SNE plot to: {tsne_path}")

if __name__ == "__main__":
    main()
