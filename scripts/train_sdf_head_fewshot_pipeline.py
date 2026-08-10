#!/usr/bin/env python3

"""Run shared FPS ShapeNet pretraining and SDF-head-only ABO adaptations."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRETRAIN = ROOT / "config/cod/stage2_transformer_diffusion_shapenetpart_pretrain_fps"
ADAPTATIONS = {
    "finetune": ROOT / "config/cod/stage2_fewshot_sdf_head_pretrained_finetune",
    "pairwise": ROOT / "config/cod/stage2_fewshot_sdf_head_pairwise_adaptation",
}
STAGE1_CHECKPOINT = ROOT / "config/cod/stage1_sdf_head_multiray21/best.ckpt"


def train_command(exp_dir, batch_size, workers, resume=False):
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


def run(command, dry_run):
    print(f"\n$ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def require_checkpoint(path, description, dry_run):
    if not dry_run and not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")


def prepare_experiment(path, description, resume, dry_run):
    if resume:
        require_checkpoint(path / "last.ckpt", f"{description} resume checkpoint", dry_run)
        return
    checkpoints = sorted(path.glob("*.ckpt"))
    if not dry_run and checkpoints:
        raise FileExistsError(
            f"{description} already contains checkpoints; pass the matching "
            f"resume option or move the old run first: {path}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--refinement-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--adaptation",
        choices=("both", *ADAPTATIONS),
        default="both",
        help="Run ordinary fine-tuning, pairwise adaptation, or both in sequence.",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Use the existing shared ShapeNet pretraining best checkpoint.",
    )
    parser.add_argument(
        "--resume-pretrain",
        action="store_true",
        help="Resume ShapeNet pretraining from its last checkpoint.",
    )
    parser.add_argument(
        "--resume-refinements",
        action="store_true",
        help="Resume selected ABO adaptations from their last checkpoints.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.skip_pretrain and args.resume_pretrain:
        raise ValueError("--skip-pretrain and --resume-pretrain are mutually exclusive")

    require_checkpoint(STAGE1_CHECKPOINT, "SDF-head-only Stage-1 checkpoint", args.dry_run)

    if not args.skip_pretrain:
        prepare_experiment(
            PRETRAIN,
            "shared ShapeNet pretraining experiment",
            args.resume_pretrain,
            args.dry_run,
        )
        print("\n=== Shared FPS ShapeNetPart diffusion pretraining ===", flush=True)
        run(
            train_command(
                PRETRAIN,
                args.pretrain_batch_size,
                args.workers,
                resume=args.resume_pretrain,
            ),
            args.dry_run,
        )

    require_checkpoint(
        PRETRAIN / "best.ckpt",
        "shared ShapeNet pretraining best checkpoint",
        args.dry_run,
    )
    selected = ADAPTATIONS if args.adaptation == "both" else {
        args.adaptation: ADAPTATIONS[args.adaptation]
    }
    for name, exp_dir in selected.items():
        prepare_experiment(
            exp_dir,
            f"SDF-head-only {name} adaptation",
            args.resume_refinements,
            args.dry_run,
        )
        print(f"\n=== SDF-head-only ABO adaptation: {name} ===", flush=True)
        run(
            train_command(
                exp_dir,
                args.refinement_batch_size,
                args.workers,
                resume=args.resume_refinements,
            ),
            args.dry_run,
        )


if __name__ == "__main__":
    main()
