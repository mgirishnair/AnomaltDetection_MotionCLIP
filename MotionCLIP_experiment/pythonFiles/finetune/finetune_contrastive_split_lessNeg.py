import sys
import os
import math
import json
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler

#BASE_DIR = os.path.dirname(__file__)
#motionclip_path = os.path.abspath(os.path.join("..", ".."))
#sys.path.append(motionclip_path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER

# --------------------------------------------------
# Utils
# --------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_summary(summary, save_path):
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)


def save_epoch_metrics_npz(history, save_path):
    epochs = np.array([h["epoch"] for h in history], dtype=np.int64)
    train_loss = np.array([h["train_loss"] for h in history], dtype=np.float32)

    has_val = all("val_loss" in h for h in history)
    if has_val:
        val_loss = np.array([h["val_loss"] for h in history], dtype=np.float32)
    else:
        val_loss = np.array([], dtype=np.float32)

    lr = np.array([h.get("lr", np.nan) for h in history], dtype=np.float32)

    np.savez(
        save_path,
        epoch=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        lr=lr,
    )


def stratified_split_indices(y, val_fraction=0.1, seed=42):
    """
    Class-balanced train/val split for a provided y array.
    Returns indices relative to that y array.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    unique_classes = np.unique(y)

    train_idx = []
    val_idx = []

    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0].copy()
        rng.shuffle(cls_idx)

        n_cls = len(cls_idx)
        n_val_cls = int(round(n_cls * val_fraction))

        if val_fraction > 0.0:
            if n_cls >= 2:
                n_val_cls = max(1, min(n_val_cls, n_cls - 1))
            else:
                n_val_cls = 0
        else:
            n_val_cls = 0

        val_idx.extend(cls_idx[:n_val_cls].tolist())
        train_idx.extend(cls_idx[n_val_cls:].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def stratified_train_test_indices_for_classes(y, normal_classes, train_fraction=0.8, seed=42):
    """
    Build reproducible train/test split using only the given normal classes.
    train_fraction is usually 0.8.
    Returns absolute dataset indices.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    train_idx = []
    test_idx = []

    for cls in normal_classes:
        cls_idx = np.where(y == cls)[0].copy()
        if len(cls_idx) == 0:
            raise ValueError(f"No samples found for class {cls}")

        rng.shuffle(cls_idx)

        n = len(cls_idx)
        if n >= 2:
            n_train = int(round(n * train_fraction))
            n_train = max(1, min(n_train, n - 1))
        else:
            n_train = 1

        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def save_split_artifacts(save_path, split_name, normal_classes, class_to_local, train_idx, test_idx, seed):
    np.savez(
        save_path,
        split_name=np.array(split_name),
        normal_classes=np.array(normal_classes, dtype=np.int64),
        class_to_local_keys=np.array(list(class_to_local.keys()), dtype=np.int64),
        class_to_local_vals=np.array(list(class_to_local.values()), dtype=np.int64),
        train_idx=np.asarray(train_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
        seed=np.array(seed, dtype=np.int64),
    )


def summarize_split(y, indices, name):
    if len(indices) == 0:
        print(f"\n{name} split is empty.")
        return

    y_subset = np.asarray(y[indices], dtype=np.int64)
    counts = defaultdict(int)
    for cls in y_subset:
        counts[int(cls)] += 1

    print(f"\n{name} split class coverage:")
    print(f"num classes: {len(counts)}")
    print(f"min samples/class: {min(counts.values())}")
    print(f"max samples/class: {max(counts.values())}")


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class NTURot6dSplitDataset(Dataset):
    def __init__(self, X, y, indices, class_to_local):
        assert X.ndim == 4, f"Expected 4D array, got shape {X.shape}"
        assert X.shape[1:] == (60, 25, 6), f"Expected [N, 60, 25, 6], got {X.shape}"

        self.X = X
        self.y = np.asarray(y)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.class_to_local = dict(class_to_local)

        ys = self.y[self.indices]
        bad = [int(v) for v in np.unique(ys) if int(v) not in self.class_to_local]
        if bad:
            raise ValueError(f"Found labels not in class_to_local: {bad}")

        self.local_labels = np.array([self.class_to_local[int(lbl)] for lbl in ys], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def get_label(self, idx):
        return int(self.local_labels[idx])

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])

        pose = np.array(self.X[real_idx], dtype=np.float32, copy=True)   # [60, 25, 6]
        pose = np.transpose(pose, (1, 2, 0)).copy()                      # [25, 6, 60]

        label_global = int(self.y[real_idx])
        label_local = int(self.class_to_local[label_global])

        return {
            "x": torch.from_numpy(pose),
            "y": torch.tensor(label_local, dtype=torch.long),
            "lengths": torch.tensor(60, dtype=torch.long),
        }


def collate_motionclip(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)          # [B, 25, 6, 60]
    y = torch.stack([b["y"] for b in batch], dim=0)          # [B]
    lengths = torch.stack([b["lengths"] for b in batch], 0)  # [B]

    T = x.shape[-1]
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "x": x,
        "y": y,
        "lengths": lengths,
        "mask": mask,
    }


# --------------------------------------------------
# Class-aware batch sampler
# --------------------------------------------------
class ClassAwareBatchSampler(BatchSampler):
    """
    Each batch contains:
      n_classes_per_batch classes
      n_samples_per_class samples from each class

    batch_size = n_classes_per_batch * n_samples_per_class
    """
    def __init__(self, dataset, n_classes_per_batch, n_samples_per_class, batches_per_epoch=None):
        self.dataset = dataset
        self.n_classes_per_batch = int(n_classes_per_batch)
        self.n_samples_per_class = int(n_samples_per_class)

        if self.n_classes_per_batch <= 0 or self.n_samples_per_class <= 0:
            raise ValueError("n_classes_per_batch and n_samples_per_class must be positive")

        self.batch_size = self.n_classes_per_batch * self.n_samples_per_class

        class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            class_to_indices[dataset.get_label(idx)].append(idx)

        self.class_to_indices = {c: np.array(v, dtype=np.int64) for c, v in class_to_indices.items()}
        self.classes = np.array(sorted(self.class_to_indices.keys()), dtype=np.int64)

        if len(self.classes) < self.n_classes_per_batch:
            raise ValueError(
                f"Need at least {self.n_classes_per_batch} classes in training set, "
                f"but got only {len(self.classes)}"
            )

        if batches_per_epoch is None:
            self.batches_per_epoch = max(1, len(dataset) // self.batch_size)
        else:
            self.batches_per_epoch = int(batches_per_epoch)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen_classes = np.random.choice(
                self.classes,
                size=self.n_classes_per_batch,
                replace=False
            )

            batch = []
            for c in chosen_classes:
                cls_indices = self.class_to_indices[int(c)]
                replace = len(cls_indices) < self.n_samples_per_class
                sampled = np.random.choice(
                    cls_indices,
                    size=self.n_samples_per_class,
                    replace=replace
                )
                batch.extend(sampled.tolist())

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


# --------------------------------------------------
# MotionCLIP encoder
# --------------------------------------------------
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
    encoder.train()
    return encoder


def freeze_encoder_except_last_layers(encoder, num_trainable_blocks=2):
    for p in encoder.parameters():
        p.requires_grad = False

    unfroze_any = False

    if hasattr(encoder, "seqTransEncoder"):
        seq_encoder = encoder.seqTransEncoder

        if hasattr(seq_encoder, "layers"):
            layers = seq_encoder.layers
            n = min(num_trainable_blocks, len(layers))
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            unfroze_any = True

        if hasattr(seq_encoder, "norm") and seq_encoder.norm is not None:
            for p in seq_encoder.norm.parameters():
                p.requires_grad = True

    if not unfroze_any:
        print(
            "Warning: could not find encoder.seqTransEncoder.layers; "
            "all encoder params remain frozen except any manually unfrozen modules."
        )


class MotionCLIPEncoderOnly(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch):
        out = self.encoder(batch)
        z = out["mu"]  # [B, 512]
        return z


# --------------------------------------------------
# Loss
# --------------------------------------------------
class PositiveAttractionLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)

        sim = torch.matmul(features, features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)

        eye = torch.eye(features.size(0), device=device)
        positive_mask = positive_mask * (1.0 - eye)

        positives_per_row = positive_mask.sum(dim=1)
        valid_rows = positives_per_row > 0

        if valid_rows.sum() == 0:
            return features.new_tensor(0.0)

        mean_pos_sim = (sim * positive_mask).sum(dim=1) / (positives_per_row + 1e-12)

        loss = -mean_pos_sim[valid_rows].mean()
        return loss

class WeightedSupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss with down-weighted negatives.

    neg_weight = 1.0  -> standard supervised contrastive loss
    neg_weight << 1.0 -> mostly positive attraction, small negative repulsion
    neg_weight = 0.0  -> no negative repulsion
    """
    def __init__(self, temperature=0.07, neg_weight=0.05):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if neg_weight < 0:
            raise ValueError("neg_weight must be >= 0")

        self.temperature = float(temperature)
        self.neg_weight = float(neg_weight)

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)

        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)

        eye = torch.eye(features.size(0), device=device)
        non_self_mask = 1.0 - eye

        positive_mask = torch.eq(labels, labels.T).float().to(device)
        positive_mask = positive_mask * non_self_mask

        negative_mask = (1.0 - torch.eq(labels, labels.T).float().to(device)) * non_self_mask

        exp_logits = torch.exp(logits)

        weighted_exp_logits = exp_logits * (
            positive_mask + self.neg_weight * negative_mask
        )

        denom = weighted_exp_logits.sum(dim=1, keepdim=True) + 1e-12
        log_prob = logits - torch.log(denom)

        positives_per_row = positive_mask.sum(dim=1)
        valid_rows = positives_per_row > 0

        if valid_rows.sum() == 0:
            return features.new_tensor(0.0)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positives_per_row + 1e-12)
        loss = -mean_log_prob_pos[valid_rows].mean()
        return loss

# --------------------------------------------------
# Train / Eval
# --------------------------------------------------
def run_one_epoch(
    model,
    loader,
    optimizer,
    device,
    train=True,
    contrastive_temp=0.07,
    contrastive_neg_weight=0.05,
    con_criterion=None,
):
    if train:
        model.train()
    else:
        model.eval()

    if con_criterion is None:
        con_criterion = WeightedSupervisedContrastiveLoss(temperature=contrastive_temp,neg_weight=contrastive_neg_weight)
        #con_criterion = SupervisedContrastiveLoss(temperature=contrastive_temp)
        #con_criterion = PositiveAttractionLoss(temperature=contrastive_temp)

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch["x"] = batch["x"].to(device, non_blocking=True).float()
        batch["y"] = batch["y"].to(device, non_blocking=True)
        batch["lengths"] = batch["lengths"].to(device, non_blocking=True)
        batch["mask"] = batch["mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            z = model(batch)
            loss = con_criterion(z, batch["y"])

            if train:
                loss.backward()
                optimizer.step()

        bs = batch["y"].size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    avg_loss = total_loss / total_samples
    return avg_loss


def save_finetuned_encoder(encoder, save_path):
    state_dict = {f"encoder.{k}": v.cpu() for k, v in encoder.state_dict().items()}
    torch.save(state_dict, save_path)


# --------------------------------------------------
# One split
# --------------------------------------------------
def finetune_one_split(args, X, y, checkpoint_path, device, split_name, normal_classes):
    print("\n" + "=" * 80)
    print(f"Split: {split_name}")
    print(f"Normal classes: {normal_classes}")

    split_dir = os.path.join(args.output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    class_to_local = {cls: i for i, cls in enumerate(sorted(normal_classes))}

    train80_idx, test20_idx = stratified_train_test_indices_for_classes(
        y=y,
        normal_classes=normal_classes,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    if args.val_fraction > 0.0:
        y_train80 = np.asarray(y)[train80_idx]
        rel_train_idx, rel_val_idx = stratified_split_indices(
            y=y_train80,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        train_idx = train80_idx[rel_train_idx]
        val_idx = train80_idx[rel_val_idx]
    else:
        train_idx = train80_idx
        val_idx = np.array([], dtype=np.int64)

    print(f"Train80 total: {len(train80_idx)}")
    print(f"Test20 total : {len(test20_idx)}")
    print(f"Actual train : {len(train_idx)}")
    print(f"Actual val   : {len(val_idx)}")

    summarize_split(y, train_idx, "Train")
    summarize_split(y, val_idx, "Val")
    summarize_split(y, test20_idx, "Held-out Test20")

    save_split_artifacts(
        save_path=os.path.join(split_dir, f"{split_name}_split_indices.npz"),
        split_name=split_name,
        normal_classes=normal_classes,
        class_to_local=class_to_local,
        train_idx=train_idx,
        test_idx=test20_idx,
        seed=args.seed,
    )

    encoder = build_motionclip_encoder(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    freeze_encoder_except_last_layers(
        encoder,
        num_trainable_blocks=args.num_trainable_blocks,
    )

    model = MotionCLIPEncoderOnly(encoder=encoder).to(device)

    train_dataset = NTURot6dSplitDataset(X, y, train_idx, class_to_local)

    if args.use_class_aware_sampler:
        max_classes_available = len(normal_classes)
        n_classes_per_batch = min(args.n_classes_per_batch, max_classes_available)
        expected_batch_size = n_classes_per_batch * args.n_samples_per_class

        if args.batch_size != expected_batch_size:
            raise ValueError(
                f"For split '{split_name}', batch_size must equal "
                f"min(n_classes_per_batch, num_split_classes) * n_samples_per_class = "
                f"{n_classes_per_batch} * {args.n_samples_per_class} = {expected_batch_size}, "
                f"but got batch_size={args.batch_size}"
            )

        train_batch_sampler = ClassAwareBatchSampler(
            dataset=train_dataset,
            n_classes_per_batch=n_classes_per_batch,
            n_samples_per_class=args.n_samples_per_class,
            batches_per_epoch=args.train_batches_per_epoch,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_motionclip,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        print(
            f"Using class-aware sampler: "
            f"{n_classes_per_batch} classes/batch x {args.n_samples_per_class} samples/class "
            f"= batch_size {args.batch_size}"
        )
        print(f"Train batches per epoch: {len(train_loader)}")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_motionclip,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    val_loader = None
    if len(val_idx) > 0:
        val_dataset = NTURot6dSplitDataset(X, y, val_idx, class_to_local)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_motionclip,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    trainable_encoder = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    total_encoder = sum(p.numel() for p in model.encoder.parameters())

    print("\nParameter counts:")
    print(f"Encoder trainable: {trainable_encoder:,} / {total_encoder:,}")

    encoder_trainable_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if len(encoder_trainable_params) == 0:
        raise ValueError("No trainable encoder parameters found.")

    optimizer = torch.optim.AdamW(
        encoder_trainable_params,
        lr=args.lr_encoder,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if val_loader is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )

    con_criterion = WeightedSupervisedContrastiveLoss(temperature=args.contrastive_temp,neg_weight=args.contrastive_neg_weight)
    #con_criterion = SupervisedContrastiveLoss(temperature=args.contrastive_temp)
    #con_criterion = PositiveAttractionLoss(temperature=args.contrastive_temp)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = []

    print("\nStarting fine-tuning...")
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            contrastive_temp=args.contrastive_temp,
            contrastive_neg_weight=args.contrastive_neg_weight,
            con_criterion=con_criterion,
        )

        if val_loader is not None:
            val_loss = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                train=False,
                contrastive_temp=args.contrastive_temp,
                contrastive_neg_weight=args.contrastive_neg_weight,
                con_criterion=con_criterion,
            )

            if scheduler is not None:
                scheduler.step(val_loss)

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(current_lr),
            }
            history.append(epoch_record)

            print(
                f"[{split_name}] Epoch {epoch+1:03d}/{args.epochs} | "
                f"lr={current_lr:.2e} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f}"
            )

            improved = val_loss < (best_val_loss - args.min_delta)

            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.encoder.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(
                        f"\n[{split_name}] Early stopping at epoch {epoch+1}. "
                        f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}"
                    )
                    break
        else:
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "lr": float(current_lr),
            }
            history.append(epoch_record)

            print(
                f"[{split_name}] Epoch {epoch+1:03d}/{args.epochs} | "
                f"lr={current_lr:.2e} | "
                f"train_loss={train_loss:.4f}"
            )

    summary = {
        "split_name": split_name,
        "normal_classes": normal_classes,
        "class_to_local": class_to_local,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction_inside_train80": args.val_fraction,
        "num_split_classes": len(normal_classes),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "best_epoch": best_epoch,
        "best_val_loss": None if best_epoch == -1 else float(best_val_loss),
        "contrastive_temp": args.contrastive_temp,
        "contrastive_neg_weight": args.contrastive_neg_weight,
        "use_class_aware_sampler": args.use_class_aware_sampler,
        "n_classes_per_batch_requested": args.n_classes_per_batch,
        "n_samples_per_class": args.n_samples_per_class,
        "scheduler": {
            "type": "ReduceLROnPlateau" if scheduler is not None else None,
            "factor": args.scheduler_factor,
            "patience": args.scheduler_patience,
            "min_lr": args.scheduler_min_lr,
        },
        "history": history,
    }

    if best_state is not None:
        model.encoder.load_state_dict(best_state)
        print(
            f"\n[{split_name}] Using best validation encoder from epoch {best_epoch} "
            f"(val_loss={best_val_loss:.4f}) before saving."
        )
    else:
        print(f"\n[{split_name}] No validation split used; saving final encoder.")

    checkpoint_name = args.save_checkpoint.format(split=split_name)
    summary_name = args.save_summary.format(split=split_name)
    metrics_name = args.save_metrics_npz.format(split=split_name)

    checkpoint_out = os.path.join(split_dir, checkpoint_name)
    summary_out = os.path.join(split_dir, summary_name)
    metrics_out = os.path.join(split_dir, metrics_name)

    save_finetuned_encoder(model.encoder, checkpoint_out)
    save_training_summary(summary, summary_out)
    save_epoch_metrics_npz(history, metrics_out)

    print(f"[{split_name}] Saved fine-tuned encoder to: {checkpoint_out}")
    print(f"[{split_name}] Saved training summary to : {summary_out}")
    print(f"[{split_name}] Saved epoch metrics to    : {metrics_out}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--x_path", type=str, default="/scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll/X.npy")
    parser.add_argument("--y_path", type=str, default="/scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll/y.npy")
    parser.add_argument("--motionclip_repo", type=str, default="MotionCLIP")
    parser.add_argument("--checkpoint_path", type=str, default=None)

    parser.add_argument("--splits_txt", type=str, default=None)
    parser.add_argument("--split_name", type=str, default=None)
    parser.add_argument("--normal_classes", type=int, nargs="+", default=None)
    parser.add_argument("--train_fraction", type=float, default=0.8)

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_trainable_blocks", type=int, default=2)

    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_encoder_only", action="store_true")
    parser.add_argument("--contrastive_neg_weight", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=0.0)

    parser.add_argument("--output_dir", type=str, default="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/finetune/optuna_outputs")
    parser.add_argument("--save_checkpoint", type=str, default="motionclip_finetuned_{split}.pth")
    parser.add_argument("--save_summary", type=str, default="finetune_summary_{split}.json")
    parser.add_argument("--save_metrics_npz", type=str, default="finetune_metrics_{split}.npz")

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--contrastive_temp", type=float, default=0.07)

    parser.add_argument("--use_class_aware_sampler", action="store_true")
    parser.add_argument("--n_classes_per_batch", type=int, default=3)
    parser.add_argument("--n_samples_per_class", type=int, default=8)
    parser.add_argument("--train_batches_per_epoch", type=int, default=None)

    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--scheduler_factor", type=float, default=0.7)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-7)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

    def resolve_path(path_str):
        if path_str is None:
            return None
        if os.path.isabs(path_str):
            return path_str
        return os.path.abspath(os.path.join(PROJECT_ROOT, path_str))

    args.x_path = resolve_path(args.x_path)
    args.y_path = resolve_path(args.y_path)
    args.motionclip_repo = resolve_path(args.motionclip_repo)
    args.checkpoint_path = resolve_path(args.checkpoint_path) if args.checkpoint_path is not None else None
    args.output_dir = resolve_path(args.output_dir)


    checkpoint_path = args.checkpoint_path
   # checkpoint_path = os.path.abspath(os.path.join("..", "..", checkpoint_path))
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            args.motionclip_repo,
            "exps",
            "paper-model",
            "checkpoint_0100.pth.tar"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = build_motionclip_encoder(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    print(encoder)
    if args.print_encoder_only:
        return

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X = np.load(args.x_path, mmap_mode="r")
    y = np.load(args.y_path, mmap_mode="r")

    print("Loaded dataset:")
    print("X:", X.shape, X.dtype)
    print("y:", y.shape, y.dtype)
    print("num unique labels:", len(np.unique(y)))
    print("min label:", y.min(), "max label:", y.max())

    if args.split_name is not None and args.normal_classes is not None:
        print(f"Using split from command line: {args.split_name} -> {args.normal_classes}")
        selected_splits = [{
            "name": args.split_name,
            "normal_classes": args.normal_classes,
        }]
    else:
        if args.splits_txt is None:
            raise ValueError(
                "Either provide --split_name and --normal_classes, or provide --splits_txt"
            )
        raise ValueError(
            "This script is intended for per-split training. "
            "Pass --split_name and --normal_classes from the shell script."
        )

    for split in selected_splits:
        finetune_one_split(
            args=args,
            X=X,
            y=y,
            checkpoint_path=checkpoint_path,
            device=device,
            split_name=split["name"],
            normal_classes=split["normal_classes"],
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
