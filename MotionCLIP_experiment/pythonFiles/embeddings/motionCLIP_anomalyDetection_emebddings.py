#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import os
import sys
import math
import argparse
from torch.utils.data import Dataset, DataLoader

from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER


# --------------------------------------------------
# Step 1: Split the input dataset
# --------------------------------------------------
def make_markovitz_split(
    X,
    y,
    normal_classes=(28, 29, 30, 33),
    label_base="auto",
    train_fraction=0.8,
    seed=42,
):
    """
    Markovitz-style Few-vs-Many split:
      - train: held-out subset of normal-class samples only
      - test normal: remaining unseen samples from the same normal classes
      - test abnormal: all samples from non-normal classes
    """
    X = np.asarray(X)
    y = np.asarray(y)
    normal_classes = np.asarray(list(normal_classes))

    if label_base == "auto":
        if y.min() == 0:
            normal_cmp = normal_classes - 1
        else:
            normal_cmp = normal_classes
    elif label_base == 0:
        normal_cmp = normal_classes - 1
    elif label_base == 1:
        normal_cmp = normal_classes
    else:
        raise ValueError("label_base must be 'auto', 0, or 1")

    normal_mask = np.isin(y, normal_cmp)
    abnormal_mask = ~normal_mask

    X_normal = X[normal_mask]
    y_normal = y[normal_mask]

    X_abnormal = X[abnormal_mask]
    y_abnormal = y[abnormal_mask]

    rng = np.random.default_rng(seed)

    train_idx_parts = []
    test_idx_parts = []

    for cls in normal_cmp:
        cls_idx = np.flatnonzero(y_normal == cls)
        rng.shuffle(cls_idx)

        n_cls = len(cls_idx)
        n_train = int(np.floor(train_fraction * n_cls))

        if n_cls < 2:
            raise ValueError(f"Class {cls} has fewer than 2 samples, cannot split train/test.")
        if n_train == 0:
            n_train = 1
        if n_train == n_cls:
            n_train = n_cls - 1

        train_idx_parts.append(cls_idx[:n_train])
        test_idx_parts.append(cls_idx[n_train:])

    train_idx = np.concatenate(train_idx_parts)
    test_idx = np.concatenate(test_idx_parts)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train_normal = X_normal[train_idx]
    y_train_normal = y_normal[train_idx]

    X_test_normal = X_normal[test_idx]
    y_test_normal = y_normal[test_idx]

    X_test_abnormal = X_abnormal
    y_test_abnormal = y_abnormal

    return {
        "X_train_normal": X_train_normal,
        "y_train_normal": y_train_normal,
        "X_test_normal": X_test_normal,
        "y_test_normal": y_test_normal,
        "X_test_abnormal": X_test_abnormal,
        "y_test_abnormal": y_test_abnormal,
        "normal_classes_mapped": normal_cmp,
    }

# --------------------------------------------------
# Step 2: MotionCLIP dataset wrapper
# --------------------------------------------------
class NTURot6dDataset(Dataset):
    def __init__(self, X):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert X.ndim == 4, f"Expected 4D array, got shape {X.shape}"
        assert X.shape[1:] == (60, 25, 6), f"Expected [N, 60, 25, 6], got {X.shape}"
        self.X = X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        pose = self.X[idx]                    # [60, 25, 6]
        pose = np.transpose(pose, (1, 2, 0)) # [25, 6, 60]

        return {
            "x": torch.from_numpy(pose),
            "y": torch.tensor(0, dtype=torch.long),
            "lengths": torch.tensor(60, dtype=torch.long),
        }


def collate_motionclip(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)          # [B, 25, 6, 60]
    y = torch.stack([b["y"] for b in batch], dim=0)          # [B]
    lengths = torch.stack([b["lengths"] for b in batch], 0)  # [B]

    T = x.shape[-1]
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)   # [B, 60]

    return {
        "x": x,
        "y": y,
        "lengths": lengths,
        "mask": mask,
    }


def build_motionclip_encoder(checkpoint_path, device):
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

#    encoder_state = {}
#    for k, v in ckpt.items():
#        if k.startswith("encoder."):
#            encoder_state[k[len("encoder."):]] = v

    # handle both formats
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    encoder_state = {}
    for k, v in ckpt.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys: {unexpected}")
    if missing:
        print("Warning: missing encoder keys:", missing)

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_motionclip_embeddings(encoder, X, batch_size=32, device="cuda"):
    dataset = NTURot6dDataset(X)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_motionclip,
    )

    all_embeddings = []

    for batch in loader:
        batch["x"] = batch["x"].to(device).float()
        batch["y"] = batch["y"].to(device)
        batch["lengths"] = batch["lengths"].to(device)
        batch["mask"] = batch["mask"].to(device)

        out = encoder(batch)
        z = out["mu"]  # [B, 512]
        all_embeddings.append(z.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_path", type=str, default="../Dataset/MotionCLIP_ready/X.npy")
    parser.add_argument("--y_path", type=str, default="../Dataset/MotionCLIP_ready/y.npy")
    parser.add_argument("--motionclip_repo", type=str, default="MotionCLIP")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--label_base", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="motionclip_embeddings.npz")
    parser.add_argument("--normal_classes", type=int, nargs="+", default=[28, 29, 30, 33])
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X = np.load(args.x_path)
    y = np.load(args.y_path)

    splits = make_markovitz_split(
        X,
        y,
        normal_classes=args.normal_classes,
        label_base=args.label_base,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    X_train_normal = splits["X_train_normal"]
    X_test_normal = splits["X_test_normal"]
    X_test_abnormal = splits["X_test_abnormal"]
    mapped_normal_classes = splits["normal_classes_mapped"]

    print("Split shapes:")
    print("X_train_normal:", X_train_normal.shape)
    print("X_test_normal:", X_test_normal.shape)
    print("X_test_abnormal:", X_test_abnormal.shape)
    print("Normal classes:", args.normal_classes)
    print("Mapped normal classes:", mapped_normal_classes)

    if args.motionclip_repo not in sys.path:
        sys.path.append(args.motionclip_repo)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            args.motionclip_repo,
            "exps",
            "paper-model",
            "checkpoint_0100.pth.tar"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_motionclip_encoder(checkpoint_path, device=device)

    Z_train_normal = extract_motionclip_embeddings(
        encoder, X_train_normal, batch_size=args.batch_size, device=device
    )
    Z_test_normal = extract_motionclip_embeddings(
        encoder, X_test_normal, batch_size=args.batch_size, device=device
    )
    Z_test_abnormal = extract_motionclip_embeddings(
        encoder, X_test_abnormal, batch_size=args.batch_size, device=device
    )

    np.savez_compressed(
        args.output_path,
        Z_train_normal=Z_train_normal,
        Z_test_normal=Z_test_normal,
        Z_test_abnormal=Z_test_abnormal,
        y_train_normal=splits["y_train_normal"],
        y_test_normal=splits["y_test_normal"],
        y_test_abnormal=splits["y_test_abnormal"],
        normal_classes=np.array(args.normal_classes),
    )

    print("\nSaved embeddings to:", args.output_path)
    print("Z_train_normal:", Z_train_normal.shape)
    print("Z_test_normal:", Z_test_normal.shape)
    print("Z_test_abnormal:", Z_test_abnormal.shape)


if __name__ == "__main__":
    main()
