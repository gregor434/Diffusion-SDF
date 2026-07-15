#!/usr/bin/env python3
"""Inspect PyTorch/PyTorch Lightning checkpoints without constructing models."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch


CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth"}


def format_time(timestamp):
    if timestamp is None:
        return "n/a"
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def format_size(num_bytes):
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024


def find_checkpoints(path, recursive=False):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)

    pattern = "**/*" if recursive else "*"
    return sorted(
        candidate
        for candidate in path.glob(pattern)
        if candidate.is_file() and candidate.suffix in CHECKPOINT_SUFFIXES
    )


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_nearby_specs(path):
    for directory in (path.parent, *path.parents):
        specs_path = directory / "specs.json"
        if specs_path.exists():
            with specs_path.open("r") as handle:
                return specs_path, json.load(handle)
    return None, None


def tensor_count(state_dict):
    if not isinstance(state_dict, dict):
        return None

    tensors = [value for value in state_dict.values() if torch.is_tensor(value)]
    if not tensors:
        return {"keys": len(state_dict), "tensors": 0, "parameters": 0}

    return {
        "keys": len(state_dict),
        "tensors": len(tensors),
        "parameters": sum(tensor.numel() for tensor in tensors),
    }


def summarize_prefixes(state_dict, limit=10):
    if not isinstance(state_dict, dict):
        return []

    counts = {}
    for key in state_dict:
        prefix = key.split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def print_mapping(title, mapping):
    if not mapping:
        return
    print(title)
    width = max(len(key) for key in mapping)
    for key, value in mapping.items():
        print(f"  {key:<{width}} : {value}")


def inspect_checkpoint(path, show_keys=False):
    stat = path.stat()
    created = getattr(stat, "st_birthtime", None)
    checkpoint = load_checkpoint(path)
    specs_path, specs = load_nearby_specs(path)

    print(f"\n{path}")
    print_mapping(
        "file",
        {
            "size": format_size(stat.st_size),
            "modified": format_time(stat.st_mtime),
            "created": format_time(created),
            "metadata_changed": format_time(stat.st_ctime),
        },
    )

    if not isinstance(checkpoint, dict):
        print_mapping("checkpoint", {"type": type(checkpoint).__name__})
        return

    checkpoint_info = {"format": "dict"}
    if "pytorch-lightning_version" in checkpoint:
        checkpoint_info["lightning_version"] = checkpoint["pytorch-lightning_version"]
    if "epoch" in checkpoint:
        checkpoint_info["epoch_index"] = checkpoint["epoch"]
        checkpoint_info["epochs_completed_estimate"] = checkpoint["epoch"] + 1
    if "global_step" in checkpoint:
        checkpoint_info["global_step"] = checkpoint["global_step"]
    if "iters" in checkpoint:
        checkpoint_info["iters"] = checkpoint["iters"]
    if "loss" in checkpoint:
        checkpoint_info["loss"] = checkpoint["loss"]
    print_mapping("checkpoint", checkpoint_info)

    if specs:
        specs_info = {"path": specs_path}
        for key in (
            "training_task",
            "num_epochs",
            "log_freq",
            "batch_size",
            "lr",
            "learning_rate",
        ):
            if key in specs:
                specs_info[key] = specs[key]
        print_mapping("nearby specs", specs_info)

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    counts = tensor_count(state_dict)
    if counts:
        print_mapping(
            "state_dict",
            {
                "keys": counts["keys"],
                "tensors": counts["tensors"],
                "tensor_values": f"{counts['parameters']:,}",
            },
        )
        prefixes = summarize_prefixes(state_dict)
        if prefixes:
            print("state_dict prefixes")
            for prefix, count in prefixes:
                print(f"  {prefix}: {count}")

    optimizer_states = checkpoint.get("optimizer_states")
    if optimizer_states is None and "optimizer_state_dict" in checkpoint:
        optimizer_states = [checkpoint["optimizer_state_dict"]]
    if optimizer_states is not None:
        print_mapping("optimizers", {"count": len(optimizer_states)})

    lr_schedulers = checkpoint.get("lr_schedulers")
    if lr_schedulers is not None:
        print_mapping("lr_schedulers", {"count": len(lr_schedulers)})

    callbacks = checkpoint.get("callbacks")
    if isinstance(callbacks, dict):
        callback_info = {"count": len(callbacks)}
        for state in callbacks.values():
            if not isinstance(state, dict):
                continue
            for key in ("dirpath", "last_model_path", "best_model_path", "best_model_score"):
                if key in state and state[key] not in ("", None):
                    callback_info[key] = state[key]
        print_mapping("callbacks", callback_info)

    if show_keys:
        print("top-level keys")
        for key in sorted(checkpoint):
            print(f"  {key}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch or PyTorch Lightning checkpoint metadata."
    )
    parser.add_argument("paths", nargs="+", help="Checkpoint files or directories.")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="When a path is a directory, search for checkpoints recursively.",
    )
    parser.add_argument(
        "--keys",
        action="store_true",
        help="Print top-level checkpoint keys.",
    )
    args = parser.parse_args()

    checkpoints = []
    for input_path in args.paths:
        checkpoints.extend(find_checkpoints(input_path, recursive=args.recursive))

    if not checkpoints:
        raise SystemExit("No checkpoint files found.")

    for checkpoint in checkpoints:
        inspect_checkpoint(checkpoint, show_keys=args.keys)


if __name__ == "__main__":
    main()
