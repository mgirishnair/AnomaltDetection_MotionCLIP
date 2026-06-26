#!/usr/bin/env python3
"""Optuna wrapper for train_AACLIP.py.
Each Optuna trial launches the existing training script as a subprocess with a
new output directory. The optimization target is the minimum Stage-2 validation classification
loss. Test metrics are never used to select hyperparameters.
The wrapper also watches Stage-2's epoch_metrics.csv so Optuna can prune weak
trials while they are still running.
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
from typing import Any, Dict, Iterable, List, Sequence
import optuna
import pandas as pd
class TrialRunError(RuntimeError):
    """Raised when a training subprocess fails without pruning."""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune AA-MotionCLIP hyperparameters with Optuna."
    )
    # Study configuration
    parser.add_argument("--train_script", required=True, help="Path to train_AACLIP.py.")
    parser.add_argument("--study_dir", required=True, help="Directory for the Optuna DB and trial outputs.")
    parser.add_argument("--study_name", default="aaclip_permo_tuning")
    parser.add_argument(
        "--storage",
        default="",
        help="Optuna storage URL. Defaults to SQLite inside --study_dir.",
    )
    parser.add_argument("--n_trials", type=int, default=30)
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
    parser.add_argument("--healthy_condition", default="healthy")
    parser.add_argument("--normal_target", type=int, default=200)
    parser.add_argument("--anomaly_target", type=int, default=200)
    parser.add_argument("--test_fraction", type=float, default=0.20)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--unseen_actions", nargs="*", default=[])
    parser.add_argument("--unseen_actors", nargs="*", default=[])
    parser.add_argument("--unseen_styles", nargs="*", default=[])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--stage1_templates_per_batch", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable_prompted_unseen_style_eval", action="store_true")
    # Search space
    parser.add_argument("--stage1_epochs_min", type=int, default=3)
    parser.add_argument("--stage1_epochs_max", type=int, default=12)
    parser.add_argument("--stage2_epochs_min", type=int, default=10)
    parser.add_argument("--stage2_epochs_max", type=int, default=40)
    parser.add_argument("--stage1_lr_min", type=float, default=1e-6)
    parser.add_argument("--stage1_lr_max", type=float, default=1e-4)
    parser.add_argument("--stage2_lr_min", type=float, default=1e-5)
    parser.add_argument("--stage2_lr_max", type=float, default=2e-3)
    parser.add_argument("--stage1_weight_decay_min", type=float, default=1e-6)
    parser.add_argument("--stage1_weight_decay_max", type=float, default=1e-2)
    parser.add_argument("--stage2_weight_decay_min", type=float, default=1e-6)
    parser.add_argument("--stage2_weight_decay_max", type=float, default=1e-2)
    parser.add_argument(
        "--share_weight_decay",
        action="store_true",
        help="Tune one shared weight decay instead of separate values for Stage 1 and Stage 2.",
    )
    parser.add_argument("--disentangle_weight_min", type=float, default=1e-3)
    parser.add_argument("--disentangle_weight_max", type=float, default=1.0)
    parser.add_argument("--contrastive_weight_min", type=float, default=1e-3)
    parser.add_argument("--contrastive_weight_max", type=float, default=1.0)
    parser.add_argument("--temperature_min", type=float, default=1e-2)
    parser.add_argument("--temperature_max", type=float, default=2e-1)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 16, 32])
    # Any unchanged arguments not exposed above can be appended after this flag.
    parser.add_argument(
        "--extra_train_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Remaining arguments are passed unchanged to train_AACLIP.py.",
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
    if args.stage1_epochs_min > args.stage1_epochs_max:
        raise ValueError("Stage-1 epoch minimum exceeds maximum.")
    if args.stage2_epochs_min > args.stage2_epochs_max:
        raise ValueError("Stage-2 epoch minimum exceeds maximum.")
    if any(b < 2 for b in args.batch_sizes):
        raise ValueError("Every batch size must be at least 2 for BalancedBinaryBatchSampler.")
    positive_ranges = [
        ("stage1_lr", args.stage1_lr_min, args.stage1_lr_max),
        ("stage2_lr", args.stage2_lr_min, args.stage2_lr_max),
        ("stage1_weight_decay", args.stage1_weight_decay_min, args.stage1_weight_decay_max),
        ("stage2_weight_decay", args.stage2_weight_decay_min, args.stage2_weight_decay_max),
        ("disentangle_weight", args.disentangle_weight_min, args.disentangle_weight_max),
        ("contrastive_weight", args.contrastive_weight_min, args.contrastive_weight_max),
        ("temperature", args.temperature_min, args.temperature_max),
    ]
    for name, low, high in positive_ranges:
        if low <= 0 or high <= 0 or low > high:
            raise ValueError(f"Invalid positive log-scale range for {name}: [{low}, {high}]")
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
def tail_text(path: Path, max_lines: int = 80) -> str:
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
        "stage1_epochs": trial.suggest_int(
            "stage1_epochs", args.stage1_epochs_min, args.stage1_epochs_max
        ),
        "stage2_epochs": trial.suggest_int(
            "stage2_epochs", args.stage2_epochs_min, args.stage2_epochs_max
        ),
        "stage1_lr": trial.suggest_float(
            "stage1_lr", args.stage1_lr_min, args.stage1_lr_max, log=True
        ),
        "stage2_lr": trial.suggest_float(
            "stage2_lr", args.stage2_lr_min, args.stage2_lr_max, log=True
        ),
        "stage1_disentangle_weight": trial.suggest_float(
            "stage1_disentangle_weight",
            args.disentangle_weight_min,
            args.disentangle_weight_max,
            log=True,
        ),
        "stage2_contrastive_weight": trial.suggest_float(
            "stage2_contrastive_weight",
            args.contrastive_weight_min,
            args.contrastive_weight_max,
            log=True,
        ),
        "temperature": trial.suggest_float(
            "temperature", args.temperature_min, args.temperature_max, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", sorted(set(args.batch_sizes))),
    }
    if args.share_weight_decay:
        shared_low = max(args.stage1_weight_decay_min, args.stage2_weight_decay_min)
        shared_high = min(args.stage1_weight_decay_max, args.stage2_weight_decay_max)
        if shared_low > shared_high:
            raise ValueError("Stage-1 and Stage-2 weight-decay ranges do not overlap.")
        shared = trial.suggest_float("weight_decay", shared_low, shared_high, log=True)
        params["stage1_weight_decay"] = shared
        params["stage2_weight_decay"] = shared
    else:
        params["stage1_weight_decay"] = trial.suggest_float(
            "stage1_weight_decay",
            args.stage1_weight_decay_min,
            args.stage1_weight_decay_max,
            log=True,
        )
        params["stage2_weight_decay"] = trial.suggest_float(
            "stage2_weight_decay",
            args.stage2_weight_decay_min,
            args.stage2_weight_decay_max,
            log=True,
        )
    return params
def append_option(command: List[str], option: str, value: Any) -> None:
    command.extend([option, str(value)])
def build_command(
    args: argparse.Namespace,
    trial_dir: Path,
    params: Dict[str, Any],
) -> List[str]:
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
        "--num_workers",
        str(args.num_workers),
        "--stage1_templates_per_batch",
        str(args.stage1_templates_per_batch),
        "--early_stopping_patience",
        str(args.early_stopping_patience),
    ]
    if args.unseen_actions:
        command.append("--unseen_actions")
        command.extend(args.unseen_actions)
    if args.unseen_actors:
        command.append("--unseen_actors")
        command.extend(args.unseen_actors)
    if args.unseen_styles:
        command.append("--unseen_styles")
        command.extend(args.unseen_styles)
    if args.amp:
        command.append("--amp")
    if args.disable_prompted_unseen_style_eval:
        command.append("--disable_prompted_unseen_style_eval")
    for name in [
        "stage1_epochs",
        "stage2_epochs",
        "stage1_lr",
        "stage2_lr",
        "stage1_weight_decay",
        "stage2_weight_decay",
        "stage1_disentangle_weight",
        "stage2_contrastive_weight",
        "temperature",
        "batch_size",
    ]:
        append_option(command, f"--{name}", params[name])
    command.extend(args.extra_train_args)
    return command
def report_new_stage2_epochs(
    trial: optuna.Trial,
    csv_path: Path,
    reported_epochs: set[int],
) -> None:
    if not csv_path.exists():
        return
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return
    if "epoch" not in frame.columns or "val_loss" not in frame.columns:
        return
    for _, row in frame.sort_values("epoch").iterrows():
        epoch = int(row["epoch"])
        value = float(row["val_loss"])
        if epoch in reported_epochs or not math.isfinite(value):
            continue
        trial.report(value, step=epoch)
        reported_epochs.add(epoch)
def read_objective_value(trial_dir: Path) -> tuple[float, Dict[str, Any]]:
    metrics_path = trial_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        value = float(metrics["stage2"]["best_val_loss"])
        if math.isfinite(value):
            return value, metrics
    # Fallback if final metrics writing was interrupted after Stage 2 completed.
    stage2_csv = trial_dir / "stage2_motion" / "epoch_metrics.csv"
    if stage2_csv.exists():
        frame = pd.read_csv(stage2_csv)
        values = pd.to_numeric(frame.get("val_loss"), errors="coerce").dropna()
        if len(values):
            return float(values.min()), {}
    raise TrialRunError(f"No finite Stage-2 validation loss found in {trial_dir}")
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
        reported_epochs: set[int] = set()
        stage2_csv = trial_dir / "stage2_motion" / "epoch_metrics.csv"
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
                    report_new_stage2_epochs(trial, stage2_csv, reported_epochs)
                    if trial.should_prune():
                        terminate_process_group(process)
                        write_json(
                            {
                                "reason": "Optuna pruner",
                                "reported_stage2_epochs": sorted(reported_epochs),
                            },
                            trial_dir / "pruned.json",
                        )
                        raise optuna.TrialPruned(
                            f"Pruned after Stage-2 epoch {max(reported_epochs, default=0)}"
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
        report_new_stage2_epochs(trial, stage2_csv, reported_epochs)
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
        objective_value, metrics = read_objective_value(trial_dir)
        trial.set_user_attr("objective", "stage2.best_val_loss")
        if metrics:
            trial.set_user_attr("stage1_best_epoch", metrics["stage1"].get("best_epoch"))
            trial.set_user_attr("stage1_best_val_loss", metrics["stage1"].get("best_val_loss"))
            trial.set_user_attr("stage1_best_val_auroc", metrics["stage1"].get("best_val_auroc"))
            trial.set_user_attr("stage2_best_epoch", metrics["stage2"].get("best_epoch"))
            trial.set_user_attr("stage2_best_val_auroc", metrics["stage2"].get("best_val_auroc"))
            trial.set_user_attr(
                "validation_threshold", metrics["stage2"].get("validation_threshold")
            )
        trial.set_user_attr("elapsed_minutes", (time.monotonic() - start_time) / 60.0)
        print(
            f"[TRIAL {trial.number}] Stage-2 minimum validation loss={objective_value:.6f}",
            flush=True,
        )
        return objective_value
    return objective
def baseline_params(args: argparse.Namespace) -> Dict[str, Any]:
    baseline: Dict[str, Any] = {
        "stage1_epochs": 5,
        "stage2_epochs": 20,
        "stage1_lr": 1e-5,
        "stage2_lr": 5e-4,
        "stage1_disentangle_weight": 0.1,
        "stage2_contrastive_weight": 0.1,
        "temperature": 0.07,
        "batch_size": 16,
    }
    if args.share_weight_decay:
        baseline["weight_decay"] = 1e-4
    else:
        baseline["stage1_weight_decay"] = 1e-4
        baseline["stage2_weight_decay"] = 1e-4
    return baseline
def baseline_is_inside_search_space(params: Dict[str, Any], args: argparse.Namespace) -> bool:
    checks = [
        args.stage1_epochs_min <= params["stage1_epochs"] <= args.stage1_epochs_max,
        args.stage2_epochs_min <= params["stage2_epochs"] <= args.stage2_epochs_max,
        args.stage1_lr_min <= params["stage1_lr"] <= args.stage1_lr_max,
        args.stage2_lr_min <= params["stage2_lr"] <= args.stage2_lr_max,
        args.disentangle_weight_min
        <= params["stage1_disentangle_weight"]
        <= args.disentangle_weight_max,
        args.contrastive_weight_min
        <= params["stage2_contrastive_weight"]
        <= args.contrastive_weight_max,
        args.temperature_min <= params["temperature"] <= args.temperature_max,
        params["batch_size"] in args.batch_sizes,
    ]
    if args.share_weight_decay:
        checks.append(
            max(args.stage1_weight_decay_min, args.stage2_weight_decay_min)
            <= params["weight_decay"]
            <= min(args.stage1_weight_decay_max, args.stage2_weight_decay_max)
        )
    else:
        checks.extend(
            [
                args.stage1_weight_decay_min
                <= params["stage1_weight_decay"]
                <= args.stage1_weight_decay_max,
                args.stage2_weight_decay_min
                <= params["stage2_weight_decay"]
                <= args.stage2_weight_decay_max,
            ]
        )
    return all(checks)
def shell_fragment(best_params: Dict[str, Any], share_weight_decay: bool) -> str:
    params = dict(best_params)
    if share_weight_decay:
        shared = params.pop("weight_decay")
        params["stage1_weight_decay"] = shared
        params["stage2_weight_decay"] = shared
    order = [
        "stage1_epochs",
        "stage2_epochs",
        "stage1_lr",
        "stage2_lr",
        "stage1_weight_decay",
        "stage2_weight_decay",
        "stage1_disentangle_weight",
        "stage2_contrastive_weight",
        "temperature",
        "batch_size",
    ]
    lines = [f"--{name} {params[name]} \\" for name in order]
    # Python 3.8-compatible: avoid str.removesuffix().
    lines[-1] = f"--{order[-1]} {params[order[-1]]}"
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
        "objective": "Stage-2 minimum validation classification loss",
        "split_seed": args.split_seed,
        "train_seed": args.train_seed,
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
            shell_fragment(best.params, args.share_weight_decay), encoding="utf-8"
        )
        if len(complete_trials) >= 3:
            try:
                importances = optuna.importance.get_param_importances(study)
                write_json(importances, study_dir / "parameter_importance.json")
            except Exception as exc:  # Importance is optional and may fail for sparse studies.
                write_json({"error": str(exc)}, study_dir / "parameter_importance.json")
    else:
        write_json(summary, study_dir / "study_summary.json")
def cleanup_nonbest_checkpoints(study: optuna.Study) -> None:
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
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
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    if study.direction.name != "MINIMIZE":
        raise RuntimeError(
            "This study already exists with direction=MAXIMIZE from the AUROC version. "
            "Use a new --study_name/--study_dir or remove the old Optuna database before "
            "running validation-loss minimization."
        )
    if not args.no_enqueue_baseline and len(study.trials) == 0:
        baseline = baseline_params(args)
        if baseline_is_inside_search_space(baseline, args):
            study.enqueue_trial(baseline, user_attrs={"configuration": "current_baseline"})
        else:
            print("[WARN] Baseline is outside the requested search space and was not enqueued.")
    write_json(vars(args), study_dir / "optuna_args.json")
    timeout_seconds = (
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
        print(f"[DONE] Lowest validation loss: {study.best_value:.6f}")
        print(f"[DONE] Best trial: {study.best_trial.number}")
        print(json.dumps(study.best_params, indent=2, sort_keys=True))
        print(f"[DONE] Study outputs: {study_dir}")
    else:
        raise RuntimeError("No Optuna trial completed successfully. Check the trial logs.")
if __name__ == "__main__":
    main()
##!/usr/bin/env python3
#"""Optuna wrapper for train_AACLIP.py.
#Each Optuna trial launches the existing training script as a subprocess with a
#new output directory. The optimization target is the best Stage-2 validation
#AUROC. Test metrics are never used to select hyperparameters.
#The wrapper also watches Stage-2's epoch_metrics.csv so Optuna can prune weak
#trials while they are still running.
#"""
#from __future__ import annotations
#import argparse
#import json
#import math
#import os
#import shutil
#import signal
#import subprocess
#import sys
#import time
#from pathlib import Path
#from typing import Any, Dict, Iterable, List, Sequence
#import optuna
#import pandas as pd
#class TrialRunError(RuntimeError):
#    """Raised when a training subprocess fails without pruning."""
#def parse_args() -> argparse.Namespace:
#    parser = argparse.ArgumentParser(
#        description="Tune AA-MotionCLIP hyperparameters with Optuna."
#    )
#    # Study configuration
#    parser.add_argument("--train_script", required=True, help="Path to train_AACLIP.py.")
#    parser.add_argument("--study_dir", required=True, help="Directory for the Optuna DB and trial outputs.")
#    parser.add_argument("--study_name", default="aaclip_permo_tuning")
#    parser.add_argument(
#        "--storage",
#        default="",
#        help="Optuna storage URL. Defaults to SQLite inside --study_dir.",
#    )
#    parser.add_argument("--n_trials", type=int, default=30)
#    parser.add_argument(
#        "--study_timeout_hours",
#        type=float,
#        default=0.0,
#        help="Total study time limit. 0 disables the limit.",
#    )
#    parser.add_argument(
#        "--trial_timeout_minutes",
#        type=float,
#        default=0.0,
#        help="Per-trial time limit. 0 disables the limit.",
#    )
#    parser.add_argument("--sampler_seed", type=int, default=2026)
#    parser.add_argument("--poll_seconds", type=float, default=15.0)
#    parser.add_argument("--pruner_startup_trials", type=int, default=5)
#    parser.add_argument("--pruner_warmup_epochs", type=int, default=4)
#    parser.add_argument(
#        "--no_enqueue_baseline",
#        action="store_true",
#        help="Do not evaluate the current hand-selected configuration as the first trial.",
#    )
#    parser.add_argument(
#        "--cleanup_nonbest_checkpoints",
#        action="store_true",
#        help="After tuning, delete checkpoint folders for non-best completed trials.",
#    )
#    # Fixed training inputs
#    parser.add_argument("--csv_path", required=True)
#    parser.add_argument("--project_root", required=True)
#    parser.add_argument("--checkpoint", required=True)
#    parser.add_argument("--python_executable", default=sys.executable)
#    parser.add_argument("--path_col", default="motion_path")
#    parser.add_argument("--action_col", default="action_label")
#    parser.add_argument("--condition_col", default="condition_label")
#    parser.add_argument("--actor_col", default="actor_label")
#    parser.add_argument("--label_col", default="is_anomaly")
#    parser.add_argument("--healthy_condition", default="healthy")
#    parser.add_argument("--normal_target", type=int, default=200)
#    parser.add_argument("--anomaly_target", type=int, default=200)
#    parser.add_argument("--test_fraction", type=float, default=0.20)
#    parser.add_argument("--val_fraction", type=float, default=0.10)
#    parser.add_argument("--split_seed", type=int, default=0)
#    parser.add_argument("--train_seed", type=int, default=42)
#    parser.add_argument("--unseen_actions", nargs="*", default=[])
#    parser.add_argument("--unseen_actors", nargs="*", default=[])
#    parser.add_argument("--unseen_styles", nargs="*", default=[])
#    parser.add_argument("--num_workers", type=int, default=2)
#    parser.add_argument("--stage1_templates_per_batch", type=int, default=1)
#    parser.add_argument("--early_stopping_patience", type=int, default=8)
#    parser.add_argument("--amp", action="store_true")
#    parser.add_argument("--disable_prompted_unseen_style_eval", action="store_true")
#    # Search space
#    parser.add_argument("--stage1_epochs_min", type=int, default=3)
#    parser.add_argument("--stage1_epochs_max", type=int, default=12)
#    parser.add_argument("--stage2_epochs_min", type=int, default=10)
#    parser.add_argument("--stage2_epochs_max", type=int, default=40)
#    parser.add_argument("--stage1_lr_min", type=float, default=1e-6)
#    parser.add_argument("--stage1_lr_max", type=float, default=1e-4)
#    parser.add_argument("--stage2_lr_min", type=float, default=1e-5)
#    parser.add_argument("--stage2_lr_max", type=float, default=2e-3)
#    parser.add_argument("--stage1_weight_decay_min", type=float, default=1e-6)
#    parser.add_argument("--stage1_weight_decay_max", type=float, default=1e-2)
#    parser.add_argument("--stage2_weight_decay_min", type=float, default=1e-6)
#    parser.add_argument("--stage2_weight_decay_max", type=float, default=1e-2)
#    parser.add_argument(
#        "--share_weight_decay",
#        action="store_true",
#        help="Tune one shared weight decay instead of separate values for Stage 1 and Stage 2.",
#    )
#    parser.add_argument("--disentangle_weight_min", type=float, default=1e-3)
#    parser.add_argument("--disentangle_weight_max", type=float, default=1.0)
#    parser.add_argument("--contrastive_weight_min", type=float, default=1e-3)
#    parser.add_argument("--contrastive_weight_max", type=float, default=1.0)
#    parser.add_argument("--temperature_min", type=float, default=1e-2)
#    parser.add_argument("--temperature_max", type=float, default=2e-1)
#    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 16, 32])
#    # Any unchanged arguments not exposed above can be appended after this flag.
#    parser.add_argument(
#        "--extra_train_args",
#        nargs=argparse.REMAINDER,
#        default=[],
#        help="Remaining arguments are passed unchanged to train_AACLIP.py.",
#    )
#    args = parser.parse_args()
#    validate_args(args)
#    return args
#def validate_args(args: argparse.Namespace) -> None:
#    train_script = Path(args.train_script)
#    if not train_script.is_file():
#        raise FileNotFoundError(f"Training script not found: {train_script}")
#    if not Path(args.csv_path).is_file():
#        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
#    if not Path(args.checkpoint).is_file():
#        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
#    if not Path(args.project_root).is_dir():
#        raise NotADirectoryError(f"Project root not found: {args.project_root}")
#    if args.stage1_epochs_min > args.stage1_epochs_max:
#        raise ValueError("Stage-1 epoch minimum exceeds maximum.")
#    if args.stage2_epochs_min > args.stage2_epochs_max:
#        raise ValueError("Stage-2 epoch minimum exceeds maximum.")
#    if any(b < 2 for b in args.batch_sizes):
#        raise ValueError("Every batch size must be at least 2 for BalancedBinaryBatchSampler.")
#    positive_ranges = [
#        ("stage1_lr", args.stage1_lr_min, args.stage1_lr_max),
#        ("stage2_lr", args.stage2_lr_min, args.stage2_lr_max),
#        ("stage1_weight_decay", args.stage1_weight_decay_min, args.stage1_weight_decay_max),
#        ("stage2_weight_decay", args.stage2_weight_decay_min, args.stage2_weight_decay_max),
#        ("disentangle_weight", args.disentangle_weight_min, args.disentangle_weight_max),
#        ("contrastive_weight", args.contrastive_weight_min, args.contrastive_weight_max),
#        ("temperature", args.temperature_min, args.temperature_max),
#    ]
#    for name, low, high in positive_ranges:
#        if low <= 0 or high <= 0 or low > high:
#            raise ValueError(f"Invalid positive log-scale range for {name}: [{low}, {high}]")
#def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 20.0) -> None:
#    """Terminate the trial process and its DataLoader children."""
#    if process.poll() is not None:
#        return
#    try:
#        os.killpg(process.pid, signal.SIGTERM)
#    except ProcessLookupError:
#        return
#    try:
#        process.wait(timeout=grace_seconds)
#        return
#    except subprocess.TimeoutExpired:
#        pass
#    try:
#        os.killpg(process.pid, signal.SIGKILL)
#    except ProcessLookupError:
#        pass
#    process.wait()
#def tail_text(path: Path, max_lines: int = 80) -> str:
#    if not path.exists():
#        return ""
#    try:
#        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
#        return "\n".join(lines[-max_lines:])
#    except OSError:
#        return ""
#def write_json(data: Any, path: Path) -> None:
#    path.parent.mkdir(parents=True, exist_ok=True)
#    with path.open("w", encoding="utf-8") as handle:
#        json.dump(data, handle, indent=2, sort_keys=True)
#def tuned_parameters(trial: optuna.Trial, args: argparse.Namespace) -> Dict[str, Any]:
#    params: Dict[str, Any] = {
#        "stage1_epochs": trial.suggest_int(
#            "stage1_epochs", args.stage1_epochs_min, args.stage1_epochs_max
#        ),
#        "stage2_epochs": trial.suggest_int(
#            "stage2_epochs", args.stage2_epochs_min, args.stage2_epochs_max
#        ),
#        "stage1_lr": trial.suggest_float(
#            "stage1_lr", args.stage1_lr_min, args.stage1_lr_max, log=True
#        ),
#        "stage2_lr": trial.suggest_float(
#            "stage2_lr", args.stage2_lr_min, args.stage2_lr_max, log=True
#        ),
#        "stage1_disentangle_weight": trial.suggest_float(
#            "stage1_disentangle_weight",
#            args.disentangle_weight_min,
#            args.disentangle_weight_max,
#            log=True,
#        ),
#        "stage2_contrastive_weight": trial.suggest_float(
#            "stage2_contrastive_weight",
#            args.contrastive_weight_min,
#            args.contrastive_weight_max,
#            log=True,
#        ),
#        "temperature": trial.suggest_float(
#            "temperature", args.temperature_min, args.temperature_max, log=True
#        ),
#        "batch_size": trial.suggest_categorical("batch_size", sorted(set(args.batch_sizes))),
#    }
#    if args.share_weight_decay:
#        shared_low = max(args.stage1_weight_decay_min, args.stage2_weight_decay_min)
#        shared_high = min(args.stage1_weight_decay_max, args.stage2_weight_decay_max)
#        if shared_low > shared_high:
#            raise ValueError("Stage-1 and Stage-2 weight-decay ranges do not overlap.")
#        shared = trial.suggest_float("weight_decay", shared_low, shared_high, log=True)
#        params["stage1_weight_decay"] = shared
#        params["stage2_weight_decay"] = shared
#    else:
#        params["stage1_weight_decay"] = trial.suggest_float(
#            "stage1_weight_decay",
#            args.stage1_weight_decay_min,
#            args.stage1_weight_decay_max,
#            log=True,
#        )
#        params["stage2_weight_decay"] = trial.suggest_float(
#            "stage2_weight_decay",
#            args.stage2_weight_decay_min,
#            args.stage2_weight_decay_max,
#            log=True,
#        )
#    return params
#def append_option(command: List[str], option: str, value: Any) -> None:
#    command.extend([option, str(value)])
#def build_command(
#    args: argparse.Namespace,
#    trial_dir: Path,
#    params: Dict[str, Any],
#) -> List[str]:
#    command = [
#        args.python_executable,
#        str(Path(args.train_script).resolve()),
#        "--csv_path",
#        str(Path(args.csv_path).resolve()),
#        "--output_dir",
#        str(trial_dir.resolve()),
#        "--project_root",
#        str(Path(args.project_root).resolve()),
#        "--checkpoint",
#        str(Path(args.checkpoint).resolve()),
#        "--path_col",
#        args.path_col,
#        "--action_col",
#        args.action_col,
#        "--condition_col",
#        args.condition_col,
#        "--actor_col",
#        args.actor_col,
#        "--label_col",
#        args.label_col,
#        "--healthy_condition",
#        args.healthy_condition,
#        "--normal_target",
#        str(args.normal_target),
#        "--anomaly_target",
#        str(args.anomaly_target),
#        "--test_fraction",
#        str(args.test_fraction),
#        "--val_fraction",
#        str(args.val_fraction),
#        "--seed",
#        str(args.train_seed),
#        "--split_seed",
#        str(args.split_seed),
#        "--train_seed",
#        str(args.train_seed),
#        "--num_workers",
#        str(args.num_workers),
#        "--stage1_templates_per_batch",
#        str(args.stage1_templates_per_batch),
#        "--early_stopping_patience",
#        str(args.early_stopping_patience),
#    ]
#    if args.unseen_actions:
#        command.append("--unseen_actions")
#        command.extend(args.unseen_actions)
#    if args.unseen_actors:
#        command.append("--unseen_actors")
#        command.extend(args.unseen_actors)
#    if args.unseen_styles:
#        command.append("--unseen_styles")
#        command.extend(args.unseen_styles)
#    if args.amp:
#        command.append("--amp")
#    if args.disable_prompted_unseen_style_eval:
#        command.append("--disable_prompted_unseen_style_eval")
#    for name in [
#        "stage1_epochs",
#        "stage2_epochs",
#        "stage1_lr",
#        "stage2_lr",
#        "stage1_weight_decay",
#        "stage2_weight_decay",
#        "stage1_disentangle_weight",
#        "stage2_contrastive_weight",
#        "temperature",
#        "batch_size",
#    ]:
#        append_option(command, f"--{name}", params[name])
#    command.extend(args.extra_train_args)
#    return command
#def report_new_stage2_epochs(
#    trial: optuna.Trial,
#    csv_path: Path,
#    reported_epochs: set[int],
#) -> None:
#    if not csv_path.exists():
#        return
#    try:
#        frame = pd.read_csv(csv_path)
#    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
#        return
#    if "epoch" not in frame.columns or "val_auroc" not in frame.columns:
#        return
#    for _, row in frame.sort_values("epoch").iterrows():
#        epoch = int(row["epoch"])
#        value = float(row["val_auroc"])
#        if epoch in reported_epochs or not math.isfinite(value):
#            continue
#        trial.report(value, step=epoch)
#        reported_epochs.add(epoch)
#def read_objective_value(trial_dir: Path) -> tuple[float, Dict[str, Any]]:
#    metrics_path = trial_dir / "metrics.json"
#    if metrics_path.exists():
#        with metrics_path.open("r", encoding="utf-8") as handle:
#            metrics = json.load(handle)
#        value = float(metrics["stage2"]["best_val_auroc"])
#        if math.isfinite(value):
#            return value, metrics
#    # Fallback if final metrics writing was interrupted after Stage 2 completed.
#    stage2_csv = trial_dir / "stage2_motion" / "epoch_metrics.csv"
#    if stage2_csv.exists():
#        frame = pd.read_csv(stage2_csv)
#        values = pd.to_numeric(frame.get("val_auroc"), errors="coerce").dropna()
#        if len(values):
#            return float(values.max()), {}
#    raise TrialRunError(f"No finite Stage-2 validation AUROC found in {trial_dir}")
#def make_objective(args: argparse.Namespace, trials_root: Path):
#    def objective(trial: optuna.Trial) -> float:
#        params = tuned_parameters(trial, args)
#        trial_dir = trials_root / f"trial_{trial.number:04d}"
#        if trial_dir.exists():
#            shutil.rmtree(trial_dir)
#        trial_dir.mkdir(parents=True, exist_ok=False)
#        command = build_command(args, trial_dir, params)
#        write_json(params, trial_dir / "optuna_params.json")
#        write_json(command, trial_dir / "command.json")
#        trial.set_user_attr("trial_dir", str(trial_dir.resolve()))
#        log_path = trial_dir / "train.log"
#        environment = os.environ.copy()
#        environment["PYTHONUNBUFFERED"] = "1"
#        environment.setdefault("MPLBACKEND", "Agg")
#        environment["OPTUNA_TRIAL_NUMBER"] = str(trial.number)
#        print(f"[TRIAL {trial.number}] Starting: {trial_dir}", flush=True)
#        start_time = time.monotonic()
#        reported_epochs: set[int] = set()
#        stage2_csv = trial_dir / "stage2_motion" / "epoch_metrics.csv"
#        with log_path.open("w", encoding="utf-8") as log_handle:
#            process = subprocess.Popen(
#                command,
#                stdout=log_handle,
#                stderr=subprocess.STDOUT,
#                cwd=str(Path(args.project_root).resolve()),
#                env=environment,
#                start_new_session=True,
#            )
#            try:
#                while True:
#                    report_new_stage2_epochs(trial, stage2_csv, reported_epochs)
#                    if trial.should_prune():
#                        terminate_process_group(process)
#                        write_json(
#                            {
#                                "reason": "Optuna pruner",
#                                "reported_stage2_epochs": sorted(reported_epochs),
#                            },
#                            trial_dir / "pruned.json",
#                        )
#                        raise optuna.TrialPruned(
#                            f"Pruned after Stage-2 epoch {max(reported_epochs, default=0)}"
#                        )
#                    return_code = process.poll()
#                    if return_code is not None:
#                        break
#                    if args.trial_timeout_minutes > 0:
#                        elapsed_minutes = (time.monotonic() - start_time) / 60.0
#                        if elapsed_minutes > args.trial_timeout_minutes:
#                            terminate_process_group(process)
#                            raise TrialRunError(
#                                f"Trial exceeded {args.trial_timeout_minutes:.1f} minutes."
#                            )
#                    time.sleep(max(1.0, args.poll_seconds))
#            except BaseException:
#                terminate_process_group(process)
#                raise
#        report_new_stage2_epochs(trial, stage2_csv, reported_epochs)
#        if process.returncode != 0:
#            log_tail = tail_text(log_path)
#            write_json(
#                {"return_code": process.returncode, "log_tail": log_tail},
#                trial_dir / "failure.json",
#            )
#            if "out of memory" in log_tail.lower():
#                trial.set_user_attr("failure_reason", "CUDA out of memory")
#            else:
#                trial.set_user_attr("failure_reason", f"return code {process.returncode}")
#            raise TrialRunError(
#                f"Training failed for trial {trial.number}. See {log_path}\n{log_tail}"
#            )
#        objective_value, metrics = read_objective_value(trial_dir)
#        trial.set_user_attr("objective", "stage2.best_val_auroc")
#        if metrics:
#            trial.set_user_attr("stage1_best_epoch", metrics["stage1"].get("best_epoch"))
#            trial.set_user_attr("stage1_best_val_auroc", metrics["stage1"].get("best_val_auroc"))
#            trial.set_user_attr("stage2_best_epoch", metrics["stage2"].get("best_epoch"))
#            trial.set_user_attr(
#                "validation_threshold", metrics["stage2"].get("validation_threshold")
#            )
#        trial.set_user_attr("elapsed_minutes", (time.monotonic() - start_time) / 60.0)
#        print(
#            f"[TRIAL {trial.number}] Stage-2 best validation AUROC={objective_value:.6f}",
#            flush=True,
#        )
#        return objective_value
#    return objective
#def baseline_params(args: argparse.Namespace) -> Dict[str, Any]:
#    baseline: Dict[str, Any] = {
#        "stage1_epochs": 5,
#        "stage2_epochs": 20,
#        "stage1_lr": 1e-5,
#        "stage2_lr": 5e-4,
#        "stage1_disentangle_weight": 0.1,
#        "stage2_contrastive_weight": 0.1,
#        "temperature": 0.07,
#        "batch_size": 16,
#    }
#    if args.share_weight_decay:
#        baseline["weight_decay"] = 1e-4
#    else:
#        baseline["stage1_weight_decay"] = 1e-4
#        baseline["stage2_weight_decay"] = 1e-4
#    return baseline
#def baseline_is_inside_search_space(params: Dict[str, Any], args: argparse.Namespace) -> bool:
#    checks = [
#        args.stage1_epochs_min <= params["stage1_epochs"] <= args.stage1_epochs_max,
#        args.stage2_epochs_min <= params["stage2_epochs"] <= args.stage2_epochs_max,
#        args.stage1_lr_min <= params["stage1_lr"] <= args.stage1_lr_max,
#        args.stage2_lr_min <= params["stage2_lr"] <= args.stage2_lr_max,
#        args.disentangle_weight_min
#        <= params["stage1_disentangle_weight"]
#        <= args.disentangle_weight_max,
#        args.contrastive_weight_min
#        <= params["stage2_contrastive_weight"]
#        <= args.contrastive_weight_max,
#        args.temperature_min <= params["temperature"] <= args.temperature_max,
#        params["batch_size"] in args.batch_sizes,
#    ]
#    if args.share_weight_decay:
#        checks.append(
#            max(args.stage1_weight_decay_min, args.stage2_weight_decay_min)
#            <= params["weight_decay"]
#            <= min(args.stage1_weight_decay_max, args.stage2_weight_decay_max)
#        )
#    else:
#        checks.extend(
#            [
#                args.stage1_weight_decay_min
#                <= params["stage1_weight_decay"]
#                <= args.stage1_weight_decay_max,
#                args.stage2_weight_decay_min
#                <= params["stage2_weight_decay"]
#                <= args.stage2_weight_decay_max,
#            ]
#        )
#    return all(checks)
#def shell_fragment(best_params: Dict[str, Any], share_weight_decay: bool) -> str:
#    params = dict(best_params)
#    if share_weight_decay:
#        shared = params.pop("weight_decay")
#        params["stage1_weight_decay"] = shared
#        params["stage2_weight_decay"] = shared
#    order = [
#        "stage1_epochs",
#        "stage2_epochs",
#        "stage1_lr",
#        "stage2_lr",
#        "stage1_weight_decay",
#        "stage2_weight_decay",
#        "stage1_disentangle_weight",
#        "stage2_contrastive_weight",
#        "temperature",
#        "batch_size",
#    ]
#    lines = [f"--{name} {params[name]} \\" for name in order]
#    lines[-1] = lines[-1].removesuffix(" \\")
#    return "\n".join(lines) + "\n"
#def save_study_outputs(study: optuna.Study, study_dir: Path, args: argparse.Namespace) -> None:
#    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
#    complete_trials = [
#        trial
#        for trial in study.trials
#        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
#    ]
#    summary: Dict[str, Any] = {
#        "study_name": study.study_name,
#        "direction": study.direction.name,
#        "n_trials_total": len(study.trials),
#        "n_complete": len(complete_trials),
#        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
#        "n_failed": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
#        "objective": "Stage-2 best validation AUROC",
#        "split_seed": args.split_seed,
#        "train_seed": args.train_seed,
#    }
#    if complete_trials:
#        best = study.best_trial
#        summary.update(
#            {
#                "best_trial_number": best.number,
#                "best_value": best.value,
#                "best_params": best.params,
#                "best_trial_dir": best.user_attrs.get("trial_dir"),
#                "best_user_attrs": best.user_attrs,
#            }
#        )
#        write_json(best.params, study_dir / "best_params.json")
#        write_json(summary, study_dir / "study_summary.json")
#        (study_dir / "best_hyperparameters.sh").write_text(
#            shell_fragment(best.params, args.share_weight_decay), encoding="utf-8"
#        )
#        if len(complete_trials) >= 3:
#            try:
#                importances = optuna.importance.get_param_importances(study)
#                write_json(importances, study_dir / "parameter_importance.json")
#            except Exception as exc:  # Importance is optional and may fail for sparse studies.
#                write_json({"error": str(exc)}, study_dir / "parameter_importance.json")
#    else:
#        write_json(summary, study_dir / "study_summary.json")
#def cleanup_nonbest_checkpoints(study: optuna.Study) -> None:
#    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
#        return
#    best_number = study.best_trial.number
#    for trial in study.trials:
#        if trial.number == best_number:
#            continue
#        trial_dir_text = trial.user_attrs.get("trial_dir")
#        if not trial_dir_text:
#            continue
#        checkpoint_dir = Path(trial_dir_text) / "checkpoints"
#        if checkpoint_dir.exists():
#            shutil.rmtree(checkpoint_dir)
#def main() -> None:
#    args = parse_args()
#    study_dir = Path(args.study_dir).resolve()
#    trials_root = study_dir / "trials"
#    trials_root.mkdir(parents=True, exist_ok=True)
#    storage = args.storage or f"sqlite:///{(study_dir / 'optuna.db').resolve()}"
#    sampler = optuna.samplers.TPESampler(
#        seed=args.sampler_seed,
#        multivariate=True,
#        group=True,
#    )
#    pruner = optuna.pruners.MedianPruner(
#        n_startup_trials=args.pruner_startup_trials,
#        n_warmup_steps=args.pruner_warmup_epochs,
#        interval_steps=1,
#    )
#    study = optuna.create_study(
#        study_name=args.study_name,
#        storage=storage,
#        direction="maximize",
#        sampler=sampler,
#        pruner=pruner,
#        load_if_exists=True,
#    )
#    if not args.no_enqueue_baseline and len(study.trials) == 0:
#        baseline = baseline_params(args)
#        if baseline_is_inside_search_space(baseline, args):
#            study.enqueue_trial(baseline, user_attrs={"configuration": "current_baseline"})
#        else:
#            print("[WARN] Baseline is outside the requested search space and was not enqueued.")
#    write_json(vars(args), study_dir / "optuna_args.json")
#    timeout_seconds = (
#        args.study_timeout_hours * 3600.0 if args.study_timeout_hours > 0 else None
#    )
#    study.optimize(
#        make_objective(args, trials_root),
#        n_trials=args.n_trials,
#        timeout=timeout_seconds,
#        gc_after_trial=True,
#        catch=(TrialRunError,),
#        show_progress_bar=False,
#    )
#    save_study_outputs(study, study_dir, args)
#    if args.cleanup_nonbest_checkpoints:
#        cleanup_nonbest_checkpoints(study)
#    complete = [
#        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
#    ]
#    if complete:
#        print(f"[DONE] Best validation AUROC: {study.best_value:.6f}")
#        print(f"[DONE] Best trial: {study.best_trial.number}")
#        print(json.dumps(study.best_params, indent=2, sort_keys=True))
#        print(f"[DONE] Study outputs: {study_dir}")
#    else:
#        raise RuntimeError("No Optuna trial completed successfully. Check the trial logs.")
#if __name__ == "__main__":
#    main()
