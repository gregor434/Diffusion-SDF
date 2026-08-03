#!/usr/bin/env python3

"""Train one COD diffusion variant and launch its reconstruction refinement."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINES = {
    "unconditional": {
        "stage2": ROOT / "config/cod/stage2_transformer_diffusion_multiray21",
        "stage3": ROOT / "config/cod/stage3_diffusion_reconstruction_multiray21",
    },
    "conditional": {
        "stage2": ROOT / "config/cod/stage2_transformer_image_diffusion_multiray21",
        "stage3": ROOT / "config/cod/stage3_image_diffusion_reconstruction_multiray21",
    },
    "joint-refined": {
        "stage2": ROOT / "config/cod/stage2_transformer_diffusion_multiray21_joint_refined",
        "stage3": ROOT / "config/cod/stage3_diffusion_reconstruction_multiray21_joint_refined",
    },
    "encoder-refined-unconditional": {
        "stage2": ROOT / "config/cod/stage2_transformer_diffusion_multiray21_encoder_refined",
        "stage3": ROOT / "config/cod/stage3_diffusion_reconstruction_multiray21_encoder_refined",
    },
    "encoder-refined-conditional": {
        "stage2": ROOT / "config/cod/stage2_transformer_image_diffusion_multiray21_encoder_refined",
        "stage3": ROOT / "config/cod/stage3_image_diffusion_reconstruction_multiray21_encoder_refined",
    },
}


def train_command(exp_dir, batch_size, workers, resume=None):
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
    if resume is not None:
        command.extend(("--resume", resume))
    return command


def diffusion_checkpoint(stage3_dir):
    specs = json.loads((stage3_dir / "specs.json").read_text())
    path = Path(specs["diffusion_ckpt_path"])
    return path if path.is_absolute() else ROOT / path


def run(command, dry_run=False):
    print(f"\n$ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def require_checkpoint(path, description, dry_run):
    if not dry_run and not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")


def require_fresh_directory(path, description, dry_run):
    existing = sorted(path.glob("*.ckpt"))
    if not dry_run and existing:
        raise FileExistsError(
            f"{description} already contains checkpoints; use the matching "
            f"--resume option or choose a clean experiment directory: {path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one stage-two diffusion training and then automatically start "
            "its matching reconstruction-guided stage-three refinement."
        )
    )
    parser.add_argument(
        "--model",
        choices=tuple(PIPELINES),
        required=True,
        help="Select the independent diffusion/refinement pipeline to run.",
    )
    parser.add_argument("--stage2-batch-size", type=int, default=32)
    parser.add_argument("--refinement-batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Use existing stage-two best checkpoints and run only refinement.",
    )
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Train stage two only.",
    )
    parser.add_argument(
        "--resume-stage2",
        action="store_true",
        help="Resume each selected stage-two experiment from last.ckpt.",
    )
    parser.add_argument(
        "--resume-refinement",
        action="store_true",
        help="Resume each selected refinement experiment from last.ckpt.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    name = args.model
    paths = PIPELINES[name]
    if not args.skip_stage2:
        if args.resume_stage2:
            require_checkpoint(
                paths["stage2"] / "last.ckpt",
                f"{name} stage-two resume checkpoint",
                args.dry_run,
            )
        else:
            require_fresh_directory(
                paths["stage2"],
                f"{name} stage-two experiment",
                args.dry_run,
            )
        print(f"\n=== Stage 2: {name} ===", flush=True)
        run(
            train_command(
                paths["stage2"],
                args.stage2_batch_size,
                args.workers,
                resume="last" if args.resume_stage2 else None,
            ),
            dry_run=args.dry_run,
        )

    if args.skip_refinement:
        return

    checkpoint = diffusion_checkpoint(paths["stage3"])
    require_checkpoint(
        checkpoint,
        f"{name} stage-two best checkpoint required for refinement",
        args.dry_run,
    )
    if args.resume_refinement:
        require_checkpoint(
            paths["stage3"] / "last.ckpt",
            f"{name} refinement resume checkpoint",
            args.dry_run,
        )
    else:
        require_fresh_directory(
            paths["stage3"],
            f"{name} refinement experiment",
            args.dry_run,
        )
    print(f"\n=== Reconstruction refinement: {name} ===", flush=True)
    run(
        train_command(
            paths["stage3"],
            args.refinement_batch_size,
            args.workers,
            resume="last" if args.resume_refinement else "finetune",
        ),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
