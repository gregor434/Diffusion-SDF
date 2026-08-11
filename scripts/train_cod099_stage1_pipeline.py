#!/usr/bin/env python3

"""Run the isolated COD-0.99 SDF-head bootstrap, polish, and refinement."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "config/cod/stage1_sdf_head_multiray21_cod099"
POLISH = ROOT / "config/cod/stage1_sdf_head_polish_multiray21_cod099"
REFINEMENT = ROOT / "config/cod/stage1_sdf_head_conv_refine_multiray21_cod099"
QUALITY_LIMITS = {
    "bootstrap": 0.05,
    "polish": 0.04,
}


def train_command(exp_dir, batch_size=16, workers=8, resume=False):
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--exp_dir",
        str(exp_dir),
        "--batch_size",
        str(batch_size),
        "--workers",
        str(workers),
    ]
    if resume:
        command.extend(("--resume", "last"))
    return command


def run(command, dry_run=False):
    print(f"\n$ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def require_checkpoint(path, description, dry_run=False):
    if not dry_run and not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")


def prepare_experiment(path, description, resume, dry_run=False):
    if resume:
        require_checkpoint(
            path / "last.ckpt", f"{description} resume checkpoint", dry_run
        )
        return
    checkpoints = sorted(path.glob("*.ckpt"))
    if not dry_run and checkpoints:
        raise FileExistsError(
            f"{description} already contains checkpoints; pass its resume "
            f"option or remove/archive the old run first: {path}"
        )


def checkpoint_best_validation_loss(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    for state in checkpoint.get("callbacks", {}).values():
        if state.get("monitor") != "val/loss":
            continue
        score = state.get("best_model_score")
        if score is not None:
            return float(score)
    raise ValueError(f"checkpoint has no monitored val/loss score: {path}")


def require_quality(path, stage, dry_run=False):
    require_checkpoint(path, f"{stage} best checkpoint", dry_run)
    if dry_run:
        return
    score = checkpoint_best_validation_loss(path)
    limit = QUALITY_LIMITS[stage]
    if score >= limit:
        raise RuntimeError(
            f"{stage} best val/loss {score:.6f} did not pass the {limit:.3f} "
            "quality gate"
        )
    print(f"{stage} quality gate passed: val/loss={score:.6f}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    for stage in ("bootstrap", "polish", "refinement"):
        parser.add_argument(
            f"--skip-{stage}",
            action="store_true",
            help=f"Use the existing {stage} best checkpoint.",
        )
        parser.add_argument(
            f"--resume-{stage}",
            action="store_true",
            help=f"Resume {stage} from last.ckpt.",
        )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for stage in ("bootstrap", "polish", "refinement"):
        if getattr(args, f"skip_{stage}") and getattr(args, f"resume_{stage}"):
            parser.error(
                f"--skip-{stage} and --resume-{stage} are mutually exclusive"
            )
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def train_stage(path, label, args, resume):
    prepare_experiment(path, label, resume, args.dry_run)
    print(f"\n=== COD-0.99 Stage 1: {label} ===", flush=True)
    run(
        train_command(path, args.batch_size, args.workers, resume=resume),
        args.dry_run,
    )


def main(argv=None):
    args = parse_args(argv)

    if not args.skip_bootstrap:
        train_stage(BOOTSTRAP, "SDF-head bootstrap", args, args.resume_bootstrap)
    require_quality(BOOTSTRAP / "best.ckpt", "bootstrap", args.dry_run)

    if not args.skip_polish:
        train_stage(POLISH, "SDF-head polish", args, args.resume_polish)
    require_quality(POLISH / "best.ckpt", "polish", args.dry_run)

    if args.skip_refinement:
        require_checkpoint(
            REFINEMENT / "best.ckpt", "refinement best checkpoint", args.dry_run
        )
        return
    train_stage(
        REFINEMENT,
        "SDF-head and convolutional refinement",
        args,
        args.resume_refinement,
    )


if __name__ == "__main__":
    main()
