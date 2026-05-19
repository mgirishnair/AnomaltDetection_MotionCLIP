#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import os
import sys
import math
import argparse
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER


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
        pose = np.transpose(pose, (1, 2, 0))  # [25, 6, 60]

        return {
            "x": torch.from_numpy(pose),
            "y": torch.tensor(0, dtype=torch.long),
            "lengths": torch.tensor(60, dtype=torch.long),
        }


def collate_motionclip(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)           # [B, 25, 6, 60]
    y = torch.stack([b["y"] for b in batch], dim=0)           # [B]
    lengths = torch.stack([b["lengths"] for b in batch], 0)   # [B]

    T = x.shape[-1]
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)

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

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    encoder_state = {}
    for k, v in ckpt.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v

    if not encoder_state:
        encoder_state = ckpt

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

    if unexpected:
        print("Warning: unexpected encoder keys:", unexpected)
    if missing:
        print("Warning: missing encoder keys:", missing)

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_motionclip_embeddings(encoder, X, batch_size=32, device="cuda"):
    if len(X) == 0:
        return np.empty((0, 512), dtype=np.float32)

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


def build_first60_embedding_split_indices(
    y,
    train80_idx,
    test20_idx,
    normal_classes,
    first60_classes,
):
    y = np.asarray(y)

    train80_idx = np.asarray(train80_idx, dtype=np.int64)
    test20_idx = np.asarray(test20_idx, dtype=np.int64)
    normal_classes = np.asarray(normal_classes, dtype=np.int64)
    first60_classes = np.asarray(first60_classes, dtype=np.int64)

    # Only work with first 60 classes
    train80_first60_mask = np.isin(y[train80_idx], first60_classes)
    test20_first60_mask = np.isin(y[test20_idx], first60_classes)

    # Normal vs abnormal within first 60
    train80_normal_mask = np.isin(y[train80_idx], normal_classes)
    test20_normal_mask = np.isin(y[test20_idx], normal_classes)

    # Used for embeddings
    train_idx = train80_idx[train80_first60_mask & train80_normal_mask]
    test_normal_idx = test20_idx[test20_first60_mask & test20_normal_mask]
    test_abnormal_idx = test20_idx[test20_first60_mask & (~test20_normal_mask)]

    # Ignored
    ignored_train_abnormal_idx = train80_idx[
        train80_first60_mask & (~train80_normal_mask)
    ]

    ignored_train_61_120_idx = train80_idx[~train80_first60_mask]
    ignored_test_61_120_idx = test20_idx[~test20_first60_mask]

    return {
        "train_idx": train_idx,
        "test_normal_idx": test_normal_idx,
        "test_abnormal_idx": test_abnormal_idx,
        "ignored_train_abnormal_idx": ignored_train_abnormal_idx,
        "ignored_train_61_120_idx": ignored_train_61_120_idx,
        "ignored_test_61_120_idx": ignored_test_61_120_idx,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--x_path", type=str, required=True)
    parser.add_argument("--y_path", type=str, required=True)
    parser.add_argument("--split_indices_path", type=str, required=True)

    parser.add_argument("--motionclip_repo", type=str, default="MotionCLIP")
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--split_name", type=str, required=True)
    parser.add_argument("--normal_classes", type=int, nargs="+", required=True)

    # Use 1..60 by default for NTU labels A1-A60.
    # If your labels are 0..119, pass --first60_classes 0 1 ... 59 manually.
    parser.add_argument(
        "--first60_classes",
        type=int,
        nargs="+",
        default=list(range(1, 61)),
    )

    args = parser.parse_args()

    X = np.load(args.x_path)
    y = np.load(args.y_path)

    split_data = np.load(args.split_indices_path, allow_pickle=True)

    train80_idx = split_data["train80_idx"].astype(np.int64)
    test20_idx = split_data["test20_idx"].astype(np.int64)

    normal_classes = np.array(args.normal_classes, dtype=np.int64)
    first60_classes = np.array(args.first60_classes, dtype=np.int64)

    split_indices = build_first60_embedding_split_indices(
        y=y,
        train80_idx=train80_idx,
        test20_idx=test20_idx,
        normal_classes=normal_classes,
        first60_classes=first60_classes,
    )

    train_idx = split_indices["train_idx"]
    test_normal_idx = split_indices["test_normal_idx"]
    test_abnormal_idx = split_indices["test_abnormal_idx"]

    X_train_normal = X[train_idx]
    y_train_normal = y[train_idx]

    X_test_normal = X[test_normal_idx]
    y_test_normal = y[test_normal_idx]

    X_test_abnormal = X[test_abnormal_idx]
    y_test_abnormal = y[test_abnormal_idx]

    print("\nClass counts:")
    for name, yy in [
        ("train_normal", y_train_normal),
        ("test_normal", y_test_normal),
        ("test_abnormal", y_test_abnormal),
    ]:
        vals, counts = np.unique(yy, return_counts=True)
        print(name, dict(zip(vals.tolist(), counts.tolist())))

    assert np.all(np.isin(y_train_normal, normal_classes))
    assert np.all(np.isin(y_test_normal, normal_classes))
    assert not np.any(np.isin(y_test_abnormal, normal_classes))

    assert np.all(np.isin(y_train_normal, first60_classes))
    assert np.all(np.isin(y_test_normal, first60_classes))
    assert np.all(np.isin(y_test_abnormal, first60_classes))

    print("\nSanity check passed: no 61-120 leakage and normal/abnormal split is correct.")
    print("Loaded split file:", args.split_indices_path)
    print("Split name:", args.split_name)
    print("Normal classes:", normal_classes.tolist())
    print("First 60 classes:", first60_classes.tolist())

    print("\nOriginal fixed split indices:")
    print("train80_idx:", train80_idx.shape)
    print("test20_idx:", test20_idx.shape)

    print("\nUsed for embeddings:")
    print("train_idx:", train_idx.shape)
    print("test_normal_idx:", test_normal_idx.shape)
    print("test_abnormal_idx:", test_abnormal_idx.shape)

    print("\nIgnored:")
    print("ignored_train_abnormal_idx:", split_indices["ignored_train_abnormal_idx"].shape)
    print("ignored_train_61_120_idx:", split_indices["ignored_train_61_120_idx"].shape)
    print("ignored_test_61_120_idx:", split_indices["ignored_test_61_120_idx"].shape)

    print("\nSubset shapes:")
    print("X_train_normal:", X_train_normal.shape)
    print("X_test_normal:", X_test_normal.shape)
    print("X_test_abnormal:", X_test_abnormal.shape)

    if args.motionclip_repo not in sys.path:
        sys.path.append(args.motionclip_repo)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_motionclip_encoder(args.checkpoint_path, device=device)

    Z_train_normal = extract_motionclip_embeddings(
        encoder,
        X_train_normal,
        batch_size=args.batch_size,
        device=device,
    )

    Z_test_normal = extract_motionclip_embeddings(
        encoder,
        X_test_normal,
        batch_size=args.batch_size,
        device=device,
    )

    Z_test_abnormal = extract_motionclip_embeddings(
        encoder,
        X_test_abnormal,
        batch_size=args.batch_size,
        device=device,
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    np.savez_compressed(
        args.output_path,
        split_name=np.array(args.split_name),
        normal_classes=normal_classes,
        first60_classes=first60_classes,

        train80_idx=train80_idx,
        test20_idx=test20_idx,

        train_idx=train_idx,
        test_normal_idx=test_normal_idx,
        test_abnormal_idx=test_abnormal_idx,

        ignored_train_abnormal_idx=split_indices["ignored_train_abnormal_idx"],
        ignored_train_61_120_idx=split_indices["ignored_train_61_120_idx"],
        ignored_test_61_120_idx=split_indices["ignored_test_61_120_idx"],

        Z_train_normal=Z_train_normal,
        Z_test_normal=Z_test_normal,
        Z_test_abnormal=Z_test_abnormal,

        y_train_normal=y_train_normal,
        y_test_normal=y_test_normal,
        y_test_abnormal=y_test_abnormal,
    )

    print("\nSaved embeddings to:", args.output_path)
    print("Z_train_normal:", Z_train_normal.shape)
    print("Z_test_normal:", Z_test_normal.shape)
    print("Z_test_abnormal:", Z_test_abnormal.shape)


if __name__ == "__main__":
    main()
