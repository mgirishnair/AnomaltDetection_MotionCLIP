#!/usr/bin/env python3
"""Optuna wrapper for train_shared_anomaly_direction.py.
Each Optuna trial launches the shared anomaly-direction training script as a
subprocess with a new output directory. The default optimization target is the
minimum validation classification loss saved by the training script. Test
metrics are never used to select hyperparameters.
The wrapper also watches direction_training/epoch_metrics.csv so Optuna can
prune weak trials while they are still running.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import optuna
import pandas as pd
class TrialRunError(RuntimeError):
    """Raised when a training subprocess fails without pruning."""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune shared anomaly-direction MotionCLIP hyperparameters with Optuna."
    )
    # Study configuration
    parser.add_argument("--train_script", required=True, help="Path to train_shared_anomaly_direction.py.")
    parser.add_argument("--study_dir", required=True, help="Directory for the Optuna DB and trial outputs.")
    parser.add_argument("--study_name", default="shared_anomaly_direction_tuning")
    parser.add_argument(
        "--storage",
        default="",
        help="Optuna storage URL. Defaults to SQLite inside --study_dir.",
    )
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument(
        "--objective",
        choices=["val_loss", "val_auroc"],
        default="val_loss",
        help=(
            "Optimization target. val_loss minimizes the best validation classification loss. "
            "val_auroc maximizes best validation AUROC."
        ),
    )
    parser.add_argument(
        "--study_timeout_hours",
        type=float,
        default=0.0,
        help="Total study time limit. 0 disables the limit.",
    )
    parser.add_argument(
        "--trial_timeout_minutes",
        type=float,
        default=0.0,
        help="Per-trial time limit. 0 disables the limit.",
    )
    parser.add_argument("--sampler_seed", type=int, default=2026)
    parser.add_argument("--poll_seconds", type=float, default=15.0)
    parser.add_argument("--pruner_startup_trials", type=int, default=5)
    parser.add_argument("--pruner_warmup_epochs", type=int, default=4)
    parser.add_argument(
        "--no_enqueue_baseline",
        action="store_true",
        help="Do not evaluate the current hand-selected configuration as the first trial.",
    )
    parser.add_argument(
        "--cleanup_nonbest_checkpoints",
        action="store_true",
        help="After tuning, delete checkpoint folders for non-best completed trials.",
    )
    # Fixed training inputs
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--path_col", default="motion_path")
    parser.add_argument("--action_col", default="action_label")
    parser.add_argument("--condition_col", default="condition_label")
    parser.add_argument("--actor_col", default="actor_label")
    parser.add_argument("--label_col", default="is_anomaly")
    parser.add_argument("--motion_key", default="auto")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--njoints", type=int, default=25)
    parser.add_argument("--nfeats", type=int, default=6)
    parser.add_argument("--healthy_condition", default="healthy")
    parser.add_argument("--normal_target", type=int, default=200)
    parser.add_argument("--anomaly_target", type=int, default=200)
    parser.add_argument("--test_fraction", type=float, default=0.20)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument(
        "--order_seed",
        type=int,
        default=None,
        help="If omitted, defaults to --train_seed. Kept fixed across trials for fair comparison.",
    )
    parser.add_argument("--unseen_actions", nargs="*", default=[])
    parser.add_argument("--unseen_actors", nargs="*", default=[])
    parser.add_argument("--unseen_styles", nargs="*", default=[])
    parser.add_argument("--no_seen_healthy_in_unseen_style_test", action="store_true")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--motion_num_layers", type=int, default=8)
    parser.add_argument("--motion_num_heads", type=int, default=4)
    parser.add_argument("--motion_dropout", type=float, default=0.1)
    parser.add_argument("--motion_adapter_layers", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--action_prompt_templates", nargs="+", default=["{action}"])
    parser.add_argument(
        "--direction_init",
        choices=["paired_raw", "random"],
        default="paired_raw",
        help="Usually keep paired_raw. random is mainly for ablation.",
    )
    parser.add_argument("--direction_init_max_pairs", type=int, default=512)
    parser.add_argument(
        "--pairing_mode",
        choices=["same_action_actor_only", "action_actor_or_action", "same_action_only"],
        default="action_actor_or_action",
    )
    parser.add_argument("--pairs_per_epoch", type=int, default=0)
    parser.add_argument("--no_balanced_pair_sampling", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument(
        "--threshold_criterion",
        choices=["balanced_accuracy", "f1", "accuracy"],
        default="balanced_accuracy",
    )
    parser.add_argument("--amp", action="store_true")
    # Search space
    parser.add_argument("--epochs_min", type=int, default=10)
    parser.add_argument("--epochs_max", type=int, default=40)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=2e-3)
    parser.add_argument("--weight_decay_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay_max", type=float, default=1e-2)
    parser.add_argument("--lambda_binary_min", type=float, default=0.3)
    parser.add_argument("--lambda_binary_max", type=float, default=3.0)
    parser.add_argument("--lambda_direction_min", type=float, default=1e-2)
    parser.add_argument("--lambda_direction_max", type=float, default=2.0)
    parser.add_argument("--lambda_action_min", type=float, default=1e-2)
    parser.add_argument("--lambda_action_max", type=float, default=2.0)
    parser.add_argument("--temperature_min", type=float, default=1e-2)
    parser.add_argument("--temperature_max", type=float, default=2e-1)
    parser.add_argument("--adapter_ratio_min", type=float, default=0.03)
    parser.add_argument("--adapter_ratio_max", type=float, default=0.30)
    parser.add_argument("--alpha_init_min", type=float, default=0.03)
    parser.add_argument("--alpha_init_max", type=float, default=0.50)
    parser.add_argument("--weak_pair_weight_min", type=float, default=0.10)
    parser.add_argument("--weak_pair_weight_max", type=float, default=1.00)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument(
        "--max_pairs_per_anomaly_choices",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Categorical choices for --max_pairs_per_anomaly.",
    )
    parser.add_argument(
        "--pairs_per_batch_choices",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Optional categorical choices for --pairs_per_batch. If omitted, the training "
            "script uses batch_size//2 pairs."
        ),
    )
    # Any unchanged arguments not exposed above can be appended after this flag.
    parser.add_argument(
        "--extra_train_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Remaining arguments are passed unchanged to train_shared_anomaly_direction.py.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args
def validate_args(args: argparse.Namespace) -> None:
    train_script = Path(args.train_script)
    if not train_script.is_file():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    if not Path(args.csv_path).is_file():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.project_root).is_dir():
        raise NotADirectoryError(f"Project root not found: {args.project_root}")
    if args.epochs_min > args.epochs_max:
        raise ValueError("Epoch minimum exceeds maximum.")
    if any(b < 2 for b in args.batch_sizes):
        raise ValueError("Every batch size must be at least 2.")
    if args.pairs_per_batch_choices and any(v < 1 for v in args.pairs_per_batch_choices):
        raise ValueError("Every pairs_per_batch choice must be at least 1.")
    if any(v < 1 for v in args.max_pairs_per_anomaly_choices):
        raise ValueError("Every max_pairs_per_anomaly choice must be at least 1.")
    positive_ranges = [
        ("lr", args.lr_min, args.lr_max),
        ("weight_decay", args.weight_decay_min, args.weight_decay_max),
        ("lambda_binary", args.lambda_binary_min, args.lambda_binary_max),
        ("lambda_direction", args.lambda_direction_min, args.lambda_direction_max),
        ("lambda_action", args.lambda_action_min, args.lambda_action_max),
        ("temperature", args.temperature_min, args.temperature_max),
        ("adapter_ratio", args.adapter_ratio_min, args.adapter_ratio_max),
        ("alpha_init", args.alpha_init_min, args.alpha_init_max),
        ("weak_pair_weight", args.weak_pair_weight_min, args.weak_pair_weight_max),
    ]
    for name, low, high in positive_ranges:
        if low <= 0 or high <= 0 or low > high:
            raise ValueError(f"Invalid positive log/float range for {name}: [{low}, {high}]")
    if args.extra_train_args and args.extra_train_args[0] == "--":
        args.extra_train_args = args.extra_train_args[1:]
def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 20.0) -> None:
    """Terminate the trial process and its DataLoader children."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
def tail_text(path: Path, max_lines: int = 100) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except OSError:
        return ""
def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
def tuned_parameters(trial: optuna.Trial, args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "epochs": trial.suggest_int("epochs", args.epochs_min, args.epochs_max),
        "lr": trial.suggest_float("lr", args.lr_min, args.lr_max, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", args.weight_decay_min, args.weight_decay_max, log=True
        ),
        "lambda_binary": trial.suggest_float(
            "lambda_binary", args.lambda_binary_min, args.lambda_binary_max, log=True
        ),
        "lambda_direction": trial.suggest_float(
            "lambda_direction", args.lambda_direction_min, args.lambda_direction_max, log=True
        ),
        "lambda_action": trial.suggest_float(
            "lambda_action", args.lambda_action_min, args.lambda_action_max, log=True
        ),
        "temperature": trial.suggest_float(
            "temperature", args.temperature_min, args.temperature_max, log=True
        ),
        "adapter_ratio": trial.suggest_float(
            "adapter_ratio", args.adapter_ratio_min, args.adapter_ratio_max
        ),
        "alpha_init": trial.suggest_float("alpha_init", args.alpha_init_min, args.alpha_init_max, log=True),
        "weak_pair_weight": trial.suggest_float(
            "weak_pair_weight", args.weak_pair_weight_min, args.weak_pair_weight_max
        ),
        "batch_size": trial.suggest_categorical("batch_size", sorted(set(args.batch_sizes))),
        "max_pairs_per_anomaly": trial.suggest_categorical(
            "max_pairs_per_anomaly", sorted(set(args.max_pairs_per_anomaly_choices))
        ),
    }
    if args.pairs_per_batch_choices:
        params["pairs_per_batch"] = trial.suggest_categorical(
            "pairs_per_batch", sorted(set(args.pairs_per_batch_choices))
        )
    return params
def append_option(command: List[str], option: str, value: Any) -> None:
    command.extend([option, str(value)])
def append_repeated_option(command: List[str], option: str, values: Sequence[Any]) -> None:
    if values:
        command.append(option)
        command.extend(str(v) for v in values)
def build_command(args: argparse.Namespace, trial_dir: Path, params: Dict[str, Any]) -> List[str]:
    order_seed = args.train_seed if args.order_seed is None else args.order_seed
    command = [
        args.python_executable,
        str(Path(args.train_script).resolve()),
        "--csv_path",
        str(Path(args.csv_path).resolve()),
        "--output_dir",
        str(trial_dir.resolve()),
        "--project_root",
        str(Path(args.project_root).resolve()),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--path_col",
        args.path_col,
        "--action_col",
        args.action_col,
        "--condition_col",
        args.condition_col,
        "--actor_col",
        args.actor_col,
        "--label_col",
        args.label_col,
        "--motion_key",
        args.motion_key,
        "--num_frames",
        str(args.num_frames),
        "--njoints",
        str(args.njoints),
        "--nfeats",
        str(args.nfeats),
        "--healthy_condition",
        args.healthy_condition,
        "--normal_target",
        str(args.normal_target),
        "--anomaly_target",
        str(args.anomaly_target),
        "--test_fraction",
        str(args.test_fraction),
        "--val_fraction",
        str(args.val_fraction),
        "--seed",
        str(args.train_seed),
        "--split_seed",
        str(args.split_seed),
        "--train_seed",
        str(args.train_seed),
        "--order_seed",
        str(order_seed),
        "--clip_model",
        args.clip_model,
        "--latent_dim",
        str(args.latent_dim),
        "--ff_size",
        str(args.ff_size),
        "--motion_num_layers",
        str(args.motion_num_layers),
        "--motion_num_heads",
        str(args.motion_num_heads),
        "--motion_dropout",
        str(args.motion_dropout),
        "--direction_init",
        args.direction_init,
        "--direction_init_max_pairs",
        str(args.direction_init_max_pairs),
        "--pairing_mode",
        args.pairing_mode,
        "--pairs_per_epoch",
        str(args.pairs_per_epoch),
        "--num_workers",
        str(args.num_workers),
        "--grad_clip",
        str(args.grad_clip),
        "--early_stopping_patience",
        str(args.early_stopping_patience),
        "--threshold_criterion",
        args.threshold_criterion,
    ]
    append_repeated_option(command, "--unseen_actions", args.unseen_actions)
    append_repeated_option(command, "--unseen_actors", args.unseen_actors)
    append_repeated_option(command, "--unseen_styles", args.unseen_styles)
    append_repeated_option(command, "--action_prompt_templates", args.action_prompt_templates)
    append_repeated_option(command, "--motion_adapter_layers", args.motion_adapter_layers)
    if args.no_seen_healthy_in_unseen_style_test:
        command.append("--no_seen_healthy_in_unseen_style_test")
    if args.no_balanced_pair_sampling:
        command.append("--no_balanced_pair_sampling")
    if args.amp:
        command.append("--amp")
    for name in [
        "epochs",
        "lr",
        "weight_decay",
        "lambda_binary",
        "lambda_direction",
        "lambda_action",
        "temperature",
        "batch_size",
        "adapter_ratio",
        "alpha_init",
        "weak_pair_weight",
        "max_pairs_per_anomaly",
    ]:
        append_option(command, f"--{name}", params[name])
    if "pairs_per_batch" in params:
        append_option(command, "--pairs_per_batch", params["pairs_per_batch"])
    command.extend(args.extra_train_args)
    return command
def monitored_column(args: argparse.Namespace) -> str:
    return "val_auroc" if args.objective == "val_auroc" else "val_loss"
def objective_direction(args: argparse.Namespace) -> str:
    return "maximize" if args.objective == "val_auroc" else "minimize"
def report_new_epochs(
    trial: optuna.Trial,
    csv_path: Path,
    reported_epochs: Set[int],
    args: argparse.Namespace,
) -> None:
    if not csv_path.exists():
        return
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return
    col = monitored_column(args)
    if "epoch" not in frame.columns or col not in frame.columns:
        return
    for _, row in frame.sort_values("epoch").iterrows():
        epoch = int(row["epoch"])
        value = float(row[col])
        if epoch in reported_epochs or not math.isfinite(value):
            continue
        trial.report(value, step=epoch)
        reported_epochs.add(epoch)
def read_objective_value(trial_dir: Path, args: argparse.Namespace) -> Tuple[float, Dict[str, Any]]:
    metrics_path = trial_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if args.objective == "val_auroc":
            value = float(metrics["best_val_auroc"])
        else:
            value = float(metrics["best_val_loss"])
        if math.isfinite(value):
            return value, metrics
    # Fallback if final metrics writing was interrupted after training completed.
    history_csv = trial_dir / "direction_training" / "epoch_metrics.csv"
    if history_csv.exists():
        frame = pd.read_csv(history_csv)
        col = monitored_column(args)
        values = pd.to_numeric(frame.get(col), errors="coerce").dropna()
        if len(values):
            value = float(values.max()) if args.objective == "val_auroc" else float(values.min())
            return value, {}
    raise TrialRunError(f"No finite objective value found in {trial_dir}")
def make_objective(args: argparse.Namespace, trials_root: Path):
    def objective(trial: optuna.Trial) -> float:
        params = tuned_parameters(trial, args)
        trial_dir = trials_root / f"trial_{trial.number:04d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=False)
        command = build_command(args, trial_dir, params)
        write_json(params, trial_dir / "optuna_params.json")
        write_json(command, trial_dir / "command.json")
        trial.set_user_attr("trial_dir", str(trial_dir.resolve()))
        log_path = trial_dir / "train.log"
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("MPLBACKEND", "Agg")
        environment["OPTUNA_TRIAL_NUMBER"] = str(trial.number)
        print(f"[TRIAL {trial.number}] Starting: {trial_dir}", flush=True)
        start_time = time.monotonic()
        reported_epochs: Set[int] = set()
        history_csv = trial_dir / "direction_training" / "epoch_metrics.csv"
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(Path(args.project_root).resolve()),
                env=environment,
                start_new_session=True,
            )
            try:
                while True:
                    report_new_epochs(trial, history_csv, reported_epochs, args)
                    if trial.should_prune():
                        terminate_process_group(process)
                        write_json(
                            {
                                "reason": "Optuna pruner",
                                "reported_epochs": sorted(reported_epochs),
                                "objective": args.objective,
                            },
                            trial_dir / "pruned.json",
                        )
                        raise optuna.TrialPruned(
                            f"Pruned after epoch {max(reported_epochs, default=0)}"
                        )
                    return_code = process.poll()
                    if return_code is not None:
                        break
                    if args.trial_timeout_minutes > 0:
                        elapsed_minutes = (time.monotonic() - start_time) / 60.0
                        if elapsed_minutes > args.trial_timeout_minutes:
                            terminate_process_group(process)
                            raise TrialRunError(
                                f"Trial exceeded {args.trial_timeout_minutes:.1f} minutes."
                            )
                    time.sleep(max(1.0, args.poll_seconds))
            except BaseException:
                terminate_process_group(process)
                raise
        report_new_epochs(trial, history_csv, reported_epochs, args)
        if process.returncode != 0:
            log_tail = tail_text(log_path)
            write_json(
                {"return_code": process.returncode, "log_tail": log_tail},
                trial_dir / "failure.json",
            )
            if "out of memory" in log_tail.lower():
                trial.set_user_attr("failure_reason", "CUDA out of memory")
            else:
                trial.set_user_attr("failure_reason", f"return code {process.returncode}")
            raise TrialRunError(
                f"Training failed for trial {trial.number}. See {log_path}\n{log_tail}"
            )
        objective_value, metrics = read_objective_value(trial_dir, args)
        trial.set_user_attr("objective", args.objective)
        trial.set_user_attr("objective_source", "metrics.best_val_auroc" if args.objective == "val_auroc" else "metrics.best_val_loss")
        if metrics:
            trial.set_user_attr("best_epoch", metrics.get("best_epoch"))
            trial.set_user_attr("best_val_loss", metrics.get("best_val_loss"))
            trial.set_user_attr("best_val_auroc", metrics.get("best_val_auroc"))
            trial.set_user_attr("validation_threshold", metrics.get("validation_threshold"))
            direction = metrics.get("direction", {})
            if isinstance(direction, dict):
                trial.set_user_attr("alpha", direction.get("alpha"))
            combined = metrics.get("strict_metrics", {}).get("test_combined", {})
            if isinstance(combined, dict):
                trial.set_user_attr("test_combined_auroc_not_used_for_selection", combined.get("auroc"))
                trial.set_user_attr("test_combined_auprc_not_used_for_selection", combined.get("auprc"))
        trial.set_user_attr("elapsed_minutes", (time.monotonic() - start_time) / 60.0)
        print(
            f"[TRIAL {trial.number}] objective({args.objective})={objective_value:.6f}",
            flush=True,
        )
        return objective_value
    return objective
def baseline_params(args: argparse.Namespace) -> Dict[str, Any]:
    baseline: Dict[str, Any] = {
        "epochs": 25,
        "lr": 1.09e-4,
        "weight_decay": 1e-4,
        "lambda_binary": 1.0,
        "lambda_direction": 0.5,
        "lambda_action": 0.5,
        "temperature": 0.07,
        "batch_size": 32,
        "adapter_ratio": 0.1,
        "alpha_init": 0.1,
        "weak_pair_weight": 0.5,
        "max_pairs_per_anomaly": 1,
    }
    if args.pairs_per_batch_choices:
        baseline["pairs_per_batch"] = 16
    return baseline
def baseline_is_inside_search_space(params: Dict[str, Any], args: argparse.Namespace) -> bool:
    checks = [
        args.epochs_min <= params["epochs"] <= args.epochs_max,
        args.lr_min <= params["lr"] <= args.lr_max,
        args.weight_decay_min <= params["weight_decay"] <= args.weight_decay_max,
        args.lambda_binary_min <= params["lambda_binary"] <= args.lambda_binary_max,
        args.lambda_direction_min <= params["lambda_direction"] <= args.lambda_direction_max,
        args.lambda_action_min <= params["lambda_action"] <= args.lambda_action_max,
        args.temperature_min <= params["temperature"] <= args.temperature_max,
        args.adapter_ratio_min <= params["adapter_ratio"] <= args.adapter_ratio_max,
        args.alpha_init_min <= params["alpha_init"] <= args.alpha_init_max,
        args.weak_pair_weight_min <= params["weak_pair_weight"] <= args.weak_pair_weight_max,
        params["batch_size"] in args.batch_sizes,
        params["max_pairs_per_anomaly"] in args.max_pairs_per_anomaly_choices,
    ]
    if args.pairs_per_batch_choices:
        checks.append(params.get("pairs_per_batch") in args.pairs_per_batch_choices)
    return all(checks)
def shell_fragment(best_params: Dict[str, Any]) -> str:
    order = [
        "epochs",
        "lr",
        "weight_decay",
        "lambda_binary",
        "lambda_direction",
        "lambda_action",
        "temperature",
        "batch_size",
        "adapter_ratio",
        "alpha_init",
        "weak_pair_weight",
        "max_pairs_per_anomaly",
        "pairs_per_batch",
    ]
    lines: List[str] = []
    for name in order:
        if name in best_params:
            lines.append(f"--{name} {best_params[name]} \\")
    if lines:
        lines[-1] = lines[-1][:-2].rstrip()
    return "\n".join(lines) + "\n"
def save_study_outputs(study: optuna.Study, study_dir: Path, args: argparse.Namespace) -> None:
    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    summary: Dict[str, Any] = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials_total": len(study.trials),
        "n_complete": len(complete_trials),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "n_failed": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
        "objective": args.objective,
        "objective_description": (
            "Best validation AUROC" if args.objective == "val_auroc"
            else "Minimum validation classification loss"
        ),
        "split_seed": args.split_seed,
        "train_seed": args.train_seed,
        "order_seed": args.train_seed if args.order_seed is None else args.order_seed,
        "test_metrics_used_for_selection": False,
    }
    if complete_trials:
        best = study.best_trial
        summary.update(
            {
                "best_trial_number": best.number,
                "best_value": best.value,
                "best_params": best.params,
                "best_trial_dir": best.user_attrs.get("trial_dir"),
                "best_user_attrs": best.user_attrs,
            }
        )
        write_json(best.params, study_dir / "best_params.json")
        write_json(summary, study_dir / "study_summary.json")
        (study_dir / "best_hyperparameters.sh").write_text(
            shell_fragment(best.params), encoding="utf-8"
        )
        if len(complete_trials) >= 3:
            try:
                importances = optuna.importance.get_param_importances(study)
                write_json(importances, study_dir / "parameter_importance.json")
            except Exception as exc:
                write_json({"error": str(exc)}, study_dir / "parameter_importance.json")
    else:
        write_json(summary, study_dir / "study_summary.json")
def cleanup_nonbest_checkpoints(study: optuna.Study) -> None:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        return
    best_number = study.best_trial.number
    for trial in study.trials:
        if trial.number == best_number:
            continue
        trial_dir_text = trial.user_attrs.get("trial_dir")
        if not trial_dir_text:
            continue
        checkpoint_dir = Path(trial_dir_text) / "checkpoints"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
def main() -> None:
    args = parse_args()
    study_dir = Path(args.study_dir).resolve()
    trials_root = study_dir / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{(study_dir / 'optuna.db').resolve()}"
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.pruner_warmup_epochs,
        interval_steps=1,
    )
    direction = objective_direction(args)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    if study.direction.name != direction.upper():
        raise RuntimeError(
            f"This study already exists with direction={study.direction.name}, but "
            f"--objective {args.objective} requires direction={direction.upper()}. "
            "Use a new --study_name/--study_dir or remove the old Optuna database."
        )
    if not args.no_enqueue_baseline and len(study.trials) == 0:
        baseline = baseline_params(args)
        if baseline_is_inside_search_space(baseline, args):
            study.enqueue_trial(baseline, user_attrs={"configuration": "current_shared_direction_baseline"})
        else:
            print("[WARN] Baseline is outside the requested search space and was not enqueued.")
    write_json(vars(args), study_dir / "optuna_args.json")
    timeout_seconds: Optional[float] = (
        args.study_timeout_hours * 3600.0 if args.study_timeout_hours > 0 else None
    )
    study.optimize(
        make_objective(args, trials_root),
        n_trials=args.n_trials,
        timeout=timeout_seconds,
        gc_after_trial=True,
        catch=(TrialRunError,),
        show_progress_bar=False,
    )
    save_study_outputs(study, study_dir, args)
    if args.cleanup_nonbest_checkpoints:
        cleanup_nonbest_checkpoints(study)
    complete = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if complete:
        print(f"[DONE] Best objective ({args.objective}): {study.best_value:.6f}")
        print(f"[DONE] Best trial: {study.best_trial.number}")
        print(json.dumps(study.best_params, indent=2, sort_keys=True))
        print(f"[DONE] Study outputs: {study_dir}")
    else:
        raise RuntimeError("No Optuna trial completed successfully. Check the trial logs.")
if __name__ == "__main__":
    main()
