import os
import re
import json
import shutil
import argparse
import subprocess
from pathlib import Path

import optuna


def build_trial_params(trial, mode):
    if mode == "contrastive":
        return {
            "lr_encoder": trial.suggest_float("lr_encoder", 3e-6, 1e-4, log=True),
            "contrastive_temp": trial.suggest_float("contrastive_temp", 0.2, 0.5, log=True),
            "n_samples_per_class": trial.suggest_categorical(
                "n_samples_per_class", [4, 6, 8]
            ),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
        }

    if mode == "classcontras":
        return {
            "lr_encoder": trial.suggest_float("lr_encoder", 1e-6, 5e-4, log=True),
            "lr_head": trial.suggest_float("lr_head", 5e-5, 5e-3, log=True),
            "contrastive_temp": trial.suggest_float("contrastive_temp", 0.03, 0.20),
            "contrastive_weight": trial.suggest_float(
                "contrastive_weight", 1e-3, 0.5, log=True
            ),
            "ce_weight": trial.suggest_categorical("ce_weight", [0.25, 0.5, 1.0, 2.0]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

    raise ValueError(f"Unknown mode: {mode}")


def load_best_val_loss(summary_path):
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    best_val_loss = summary.get("best_val_loss", None)
    if best_val_loss is None:
        raise RuntimeError(f"'best_val_loss' missing in summary: {summary_path}")

    return float(best_val_loss)


def run_subprocess(cmd):
    print("\nRunning command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def run_one_trial_for_split(args, params, trial_number):
    """
    Runs exactly one split for one Optuna trial.
    Returns best_val_loss from finetune_summary_{split}.json
    """
    trial_dir = os.path.join(
        args.work_dir,
        args.study_name,
        f"trial_{trial_number:04d}",
    )
    os.makedirs(trial_dir, exist_ok=True)

    if args.mode == "contrastive":
        # batch_size must match n_classes_per_batch * n_samples_per_class
        batch_size = args.n_classes_per_batch * params["n_samples_per_class"]

        cmd = [
            "python",
            args.contrastive_script,
            "--x_path", args.x_path,
            "--y_path", args.y_path,
            "--motionclip_repo", args.motionclip_repo,
            "--checkpoint_path", args.checkpoint_path,
            "--split_name", args.split_name,
            "--train_fraction", str(args.train_fraction),
            "--epochs", str(args.epochs),
            "--batch_size", str(batch_size),
            "--num_trainable_blocks", str(args.num_trainable_blocks_fixed),
            "--lr_encoder", str(params["lr_encoder"]),
            "--weight_decay", str(params["weight_decay"]),
            "--contrastive_temp", str(params["contrastive_temp"]),
            "--contrastive_neg_weight","0.1",
            "--use_class_aware_sampler",
            "--n_classes_per_batch", str(args.n_classes_per_batch),
            "--n_samples_per_class", str(params["n_samples_per_class"]),
            "--val_fraction", str(args.val_fraction),
            "--patience", str(args.patience),
            "--scheduler_patience", str(args.scheduler_patience),
            "--scheduler_factor", str(args.scheduler_factor),
            "--scheduler_min_lr", str(args.scheduler_min_lr),
            "--min_delta", str(args.min_delta),
            "--num_workers", str(args.num_workers),
            "--output_dir", trial_dir,
            "--save_checkpoint", "motionclip_finetuned_{split}.pth",
            "--save_metrics_npz", "finetune_metrics_{split}.npz",
            "--save_summary", "finetune_summary_{split}.json",
            "--seed", str(args.seed),
            "--normal_classes",
        ] + [str(c) for c in args.normal_classes]

    elif args.mode == "classcontras":
        batch_size = args.n_classes_per_batch * args.n_samples_per_class_fixed

        cmd = [
            "python",
            args.classcontras_script,
            "--x_path", args.x_path,
            "--y_path", args.y_path,
            "--motionclip_repo", args.motionclip_repo,
            "--checkpoint_path", args.checkpoint_path,
            "--split_name", args.split_name,
            "--train_fraction", str(args.train_fraction),
            "--epochs", str(args.epochs),
            "--batch_size", str(batch_size),
            "--num_trainable_blocks", str(args.num_trainable_blocks_fixed),
            "--lr_head", str(params["lr_head"]),
            "--lr_encoder", str(params["lr_encoder"]),
            "--weight_decay", str(params["weight_decay"]),
            "--ce_weight", str(params["ce_weight"]),
            "--contrastive_weight", str(params["contrastive_weight"]),
            "--contrastive_temp", str(params["contrastive_temp"]),
            "--use_class_aware_sampler",
            "--n_classes_per_batch", str(args.n_classes_per_batch),
            "--n_samples_per_class", str(args.n_samples_per_class_fixed),
            "--val_fraction", str(args.val_fraction),
            "--patience", str(args.patience),
            "--scheduler_patience", str(args.scheduler_patience),
            "--scheduler_factor", str(args.scheduler_factor),
            "--scheduler_min_lr", str(args.scheduler_min_lr),
            "--min_delta", str(args.min_delta),
            "--num_workers", str(args.num_workers),
            "--output_dir", trial_dir,
            "--save_checkpoint", "motionclip_finetuned_{split}.pth",
            "--save_metrics_npz", "finetune_metrics_{split}.npz",
            "--save_summary", "finetune_summary_{split}.json",
            "--seed", str(args.seed),
            "--normal_classes",
        ] + [str(c) for c in args.normal_classes]

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if args.pin_memory:
        cmd.append("--pin_memory")

    run_subprocess(cmd)

    # Your finetuning scripts create:
    #   output_dir / split_name / finetune_summary_{split}.json
    split_dir = os.path.join(trial_dir, args.split_name)

    summary_path = os.path.join(
        split_dir,
        f"finetune_summary_{args.split_name}.json"
    )
    checkpoint_path = os.path.join(
        split_dir,
        f"motionclip_finetuned_{args.split_name}.pth"
    )

    score = load_best_val_loss(summary_path)

    # Keep summaries/metrics only if you want.
    # Delete checkpoint to avoid filling storage during tuning.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Optional full cleanup of the whole trial folder after reading score.
    # Comment this out if you want to inspect JSON/NPZ later.
    shutil.rmtree(trial_dir, ignore_errors=True)

    return score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["contrastive", "classcontras"], required=True)
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=30)

    parser.add_argument("--split_name", type=str, required=True)
    parser.add_argument("--normal_classes", type=int, nargs="+", required=True)

    parser.add_argument("--contrastive_script", type=str, default="finetune_contrastive_split.py")
    parser.add_argument("--classcontras_script", type=str, default="finetune_classContras_split.py")

    parser.add_argument("--x_path", type=str, required=True)
    parser.add_argument("--y_path", type=str, required=True)
    parser.add_argument("--motionclip_repo", type=str, default="MotionCLIP")
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--work_dir", type=str, default="optuna_runs")

    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=0.0)

    # fixed on purpose
    parser.add_argument("--num_trainable_blocks_fixed", type=int, default=2)
    parser.add_argument("--n_classes_per_batch", type=int, default=3)
    parser.add_argument("--n_samples_per_class_fixed", type=int, default=8)

    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--scheduler_factor", type=float, default=0.7)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-7)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
    )

    def objective(trial):
        params = build_trial_params(trial, args.mode)
        score = run_one_trial_for_split(args=args, params=params, trial_number=trial.number)

        trial.set_user_attr("split_name", args.split_name)
        trial.set_user_attr("normal_classes", args.normal_classes)
        trial.set_user_attr("score", float(score))

        return float(score)

    study.optimize(objective, n_trials=args.n_trials)

    print("\nBest trial:")
    print(f"  split : {args.split_name}")
    print(f"  number: {study.best_trial.number}")
    print(f"  value : {study.best_trial.value}")
    print("  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
