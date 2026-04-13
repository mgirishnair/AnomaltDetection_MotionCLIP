import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path

import optuna


def build_trial_params(trial):
    return {
        "lr_encoder": trial.suggest_float("lr_encoder", 1e-5, 2e-4, log=True),
        "contrastive_temp": trial.suggest_float("contrastive_temp", 0.05, 0.2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 2e-4, 6e-4, log=True),
        "n_classes_per_batch": trial.suggest_categorical("n_classes_per_batch", [4, 6, 8]),
        "n_samples_per_class": trial.suggest_categorical("n_samples_per_class", [4, 6, 8]),
    }


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


def run_one_trial(args, params, trial_number):
    """
    Runs one Optuna trial for the global all-classes fine-tuning script.
    Returns best_val_loss from finetune_summary_global_all_classes.json
    """
    trial_dir = os.path.join(
        args.work_dir,
        args.study_name,
        f"trial_{trial_number:04d}",
    )
    os.makedirs(trial_dir, exist_ok=True)

    run_name = args.run_name

    batch_size = params["n_classes_per_batch"] * params["n_samples_per_class"]

    cmd = [
        "python",
        args.global_contrastive_script,
        "--x_path", args.x_path,
        "--y_path", args.y_path,
        "--motionclip_repo", args.motionclip_repo,
        "--checkpoint_path", args.checkpoint_path,
        "--run_name", run_name,
        "--train_fraction", str(args.train_fraction),
        "--val_fraction", str(args.val_fraction),
        "--epochs", str(args.epochs),
        "--batch_size", str(batch_size),
        "--num_trainable_blocks", str(args.num_trainable_blocks_fixed),
        "--lr_encoder", str(params["lr_encoder"]),
        "--weight_decay", str(params["weight_decay"]),
        "--contrastive_temp", str(params["contrastive_temp"]),
        "--contrastive_neg_weight", "0.1",
        "--use_class_aware_sampler",
        "--n_classes_per_batch", str(params["n_classes_per_batch"]),
        "--n_samples_per_class", str(params["n_samples_per_class"]),
        "--patience", str(args.patience),
        "--scheduler_patience", str(args.scheduler_patience),
        "--scheduler_factor", str(args.scheduler_factor),
        "--scheduler_min_lr", str(args.scheduler_min_lr),
        "--min_delta", str(args.min_delta),
        "--num_workers", str(args.num_workers),
        "--output_dir", trial_dir,
        "--save_checkpoint", "motionclip_finetuned_global_all_classes.pth",
        "--save_metrics_npz", "finetune_metrics_global_all_classes.npz",
        "--save_summary", "finetune_summary_global_all_classes.json",
        "--save_indices_npz", "global_split_indices.npz",
        "--seed", str(args.seed),
    ]

    if args.pin_memory:
        cmd.append("--pin_memory")

    run_subprocess(cmd)

    run_dir = os.path.join(trial_dir, run_name)

    summary_path = os.path.join(
        run_dir,
        "finetune_summary_global_all_classes.json",
    )
    checkpoint_path = os.path.join(
        run_dir,
        "motionclip_finetuned_global_all_classes.pth",
    )

    score = load_best_val_loss(summary_path)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    if args.cleanup_trial_dir:
        shutil.rmtree(trial_dir, ignore_errors=True)

    return score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=30)

    parser.add_argument(
        "--global_contrastive_script",
        type=str,
        default="finetune_contrastive_allData.py",
    )

    parser.add_argument("--x_path", type=str, required=True)
    parser.add_argument("--y_path", type=str, required=True)
    parser.add_argument("--motionclip_repo", type=str, default="MotionCLIP")
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--work_dir", type=str, default="optuna_runs")
    parser.add_argument("--run_name", type=str, default="global_all_classes_train80")

    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=0.0)

    parser.add_argument("--num_trainable_blocks_fixed", type=int, default=2)

    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--scheduler_factor", type=float, default=0.7)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-7)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--cleanup_trial_dir",
        action="store_true",
        help="Delete each trial directory after reading the score.",
    )

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
        params = build_trial_params(trial)
        score = run_one_trial(
            args=args,
            params=params,
            trial_number=trial.number,
        )

        trial.set_user_attr("run_name", args.run_name)
        trial.set_user_attr("score", float(score))

        return float(score)

    study.optimize(objective, n_trials=args.n_trials)

    print("\nBest trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  value : {study.best_trial.value}")
    print("  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
