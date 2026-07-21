#!/usr/bin/env python3

"""Extraction, reconstruction, and generation entry point for COD Diffusion-SDF."""

import argparse
import csv
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from dataloader.modulation_loader import (
    ModulationLoader,
    compute_latent_statistics,
    save_latent_statistics,
)
from dataloader.sdf_loader import SdfLoader
from models import CombinedModel, SdfModel
from utils import evaluate, mesh
from utils.reconstruct import filter_threshold


def checkpoint_path(exp_dir, resume):
    name = (
        f"{resume}.ckpt"
        if resume in {"last", "best"}
        else f"epoch={resume}.ckpt"
    )
    return str(Path(exp_dir) / name)


def load_prefixed(module, path, prefix):
    checkpoint = torch.load(path, map_location="cpu")
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise RuntimeError(f"no parameters with prefix '{prefix}' in {path}")
    module.load_state_dict(state)


def make_sdf_dataset(specs, split_name="TestSplit", condition_surface=False):
    split = json.loads(Path(specs[split_name]).read_text())
    return SdfLoader(
        specs["DataSource"],
        split,
        samples_per_mesh=specs.get("ValidationSamplesPerMesh", specs.get("SampPerMesh", 16000)),
        surface_point_count=specs.get("SurfacePointCount", 2048),
        near_surface_ratio=specs.get("NearSurfaceRatio", 0.7),
        condition_surface=condition_surface,
    )


def make_generation_dataset(specs):
    split = json.loads(Path(specs["TestSplit"]).read_text())
    return ModulationLoader(
        specs["data_path"],
        split_file=split,
        conditioning=specs.get("conditioning"),
        latent_stats_path=specs.get("latent_stats_path"),
    )


def save_modulation(path, object_id, posterior, conditioning=None):
    arrays = {
        "object_id": np.asarray(object_id),
        "posterior_mean": posterior.mean[0].detach().cpu().numpy(),
        "posterior_logvar": posterior.logvar[0].detach().cpu().numpy(),
    }
    for name, value in (conditioning or {}).items():
        if torch.is_tensor(value):
            arrays[f"condition_{name}"] = value[0].detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


@torch.no_grad()
def extract_modulations(specs, args, recon_dir, latent_dir, device):
    split_name = "ModulationSplit" if specs.get("ModulationSplit") else "TestSplit"
    dataset = make_sdf_dataset(specs, split_name=split_name)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    model = CombinedModel.load_from_checkpoint(
        checkpoint_path(args.exp_dir, args.resume), specs=specs
    ).to(device).eval()
    records = []
    metrics_path = recon_dir / "sdf_metrics.csv"
    with metrics_path.open("w", newline="") as metric_file:
        writer = csv.DictWriter(
            metric_file,
            fieldnames=[
                "object_id", "sdf_reconstruction_error", "near_surface_error",
                "uniform_error", "sign_accuracy", "encoding_time",
                "decoding_time", "peak_memory", "chamfer_distance",
                "f_score", "normal_consistency", "mesh_validity",
            ],
        )
        writer.writeheader()
        for batch in tqdm(loader, desc="extracting COD modulations"):
            surface = batch["surface_points"].to(device)
            queries = batch["query_points"].to(device)
            target = batch["query_sdf"].to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            latent, posterior, _ = model.sdf_model.encode_surface(
                surface, sample_posterior=False
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            encoding_time = time.perf_counter() - start
            start = time.perf_counter()
            decoded = model.sdf_model.decode_latent(latent)
            prediction = model.sdf_model.query_sdf(decoded["planes"], queries)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decoding_time = time.perf_counter() - start

            object_id = batch["object_id"][0]
            class_name = batch["class_name"][0]
            output_dir = recon_dir / class_name / object_id
            output_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = output_dir / "reconstruct"
            mesh.create_mesh(
                model.sdf_model,
                decoded["planes"],
                str(mesh_path),
                N=args.recon_resolution,
                max_batch=args.max_batch,
                from_plane_features=True,
            )
            try:
                evaluate.main(surface, str(mesh_path), str(recon_dir / "cd.csv"), f"{class_name}/{object_id}")
            except Exception as error:
                warnings.warn(f"mesh metric failed for {object_id}: {error}")
            mesh_metrics = evaluate.mesh_metrics(
                surface,
                str(mesh_path),
                batch.get("surface_normals"),
                fscore_threshold=float(specs.get("FScoreThreshold", 0.01)),
            )

            if (
                args.modulation_filter_threshold is not None
                and not filter_threshold(
                    str(mesh_path), surface, args.modulation_filter_threshold
                )
            ):
                continue
            modulation_path = latent_dir / class_name / object_id / "modulation.npz"
            save_modulation(
                modulation_path, object_id, posterior, batch.get("conditioning")
            )
            records.append({"latent_path": str(modulation_path)})

            absolute = (prediction - target).abs()
            near = batch["query_is_near"].to(device)
            writer.writerow({
                "object_id": object_id,
                "sdf_reconstruction_error": absolute.mean().item(),
                "near_surface_error": absolute[near].mean().item(),
                "uniform_error": absolute[~near].mean().item(),
                "sign_accuracy": (
                    (prediction >= 0) == (target >= 0)
                ).float().mean().item(),
                "encoding_time": encoding_time,
                "decoding_time": decoding_time,
                "peak_memory": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else 0
                ),
                **mesh_metrics,
            })
            metric_file.flush()

    if not records:
        raise RuntimeError("no modulations were saved")
    train_split = json.loads(Path(specs["TrainSplit"]).read_text())
    train_records = ModulationLoader.build_records(latent_dir, train_split)
    if not train_records:
        raise RuntimeError(
            "no training-set modulations were saved; ensure ModulationSplit "
            "contains every object from TrainSplit"
        )
    mean, std = compute_latent_statistics(train_records)
    save_latent_statistics(latent_dir / "latent_stats.npz", mean, std)


def load_generation_models(specs, args, device):
    if specs["training_task"] == "combined" and args.resume != "finetune":
        model = CombinedModel.load_from_checkpoint(
            checkpoint_path(args.exp_dir, args.resume), specs=specs
        )
        return model.to(device).eval(), model.sdf_model

    model = CombinedModel(specs)
    diffusion_path = (
        specs["diffusion_ckpt_path"]
        if args.resume == "finetune"
        else checkpoint_path(args.exp_dir, args.resume)
    )
    load_prefixed(model.diffusion_model, diffusion_path, "diffusion_model.")
    modulation_checkpoint = torch.load(
        specs["modulation_ckpt_path"], map_location="cpu"
    )
    stage1_specs = modulation_checkpoint.get("hyper_parameters", {}).get(
        "specs", specs
    )
    sdf_model = SdfModel(stage1_specs)
    load_prefixed(sdf_model, specs["modulation_ckpt_path"], "sdf_model.")
    return model.to(device).eval(), sdf_model.to(device).eval()


@torch.no_grad()
def generate(specs, args, recon_dir, device):
    model, sdf_model = load_generation_models(specs, args, device)
    conditional = bool(specs["diffusion_model_specs"].get("cond", False))
    batches = [None]
    if conditional:
        dataset = make_generation_dataset(specs)
        batches = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    metrics_file = (recon_dir / "generated_metrics.csv").open("w", newline="")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "object_id", "sample", "chamfer_distance", "f_score",
            "normal_consistency", "mesh_validity",
        ],
    )
    metrics_writer.writeheader()
    for batch_index, batch in enumerate(tqdm(batches, desc="generating COD latents")):
        conditioning = None
        output_dir = recon_dir
        surface = None
        if batch is not None:
            conditioning = {
                name: value.to(device)
                for name, value in batch["conditioning"].items()
            }
            surface = conditioning.get("point_cloud")
            output_dir = (
                recon_dir / batch["class_name"][0] / batch["object_id"][0]
            )
            output_dir.mkdir(parents=True, exist_ok=True)
        normalized = model.diffusion_model.sample(
            args.num_samples, conditioning=conditioning
        )
        latent = model.denormalize_latent(normalized)
        planes = sdf_model.decode_latent(latent)["planes"]
        for sample_index in range(len(planes)):
            path = output_dir / f"{sample_index}_recon"
            mesh.create_mesh(
                sdf_model,
                planes[sample_index:sample_index + 1],
                str(path),
                N=args.recon_resolution,
                max_batch=args.max_batch,
                from_plane_features=True,
            )
            if surface is not None:
                result = evaluate.mesh_metrics(
                    surface,
                    str(path),
                    batch.get("surface_normals"),
                    fscore_threshold=float(specs.get("FScoreThreshold", 0.01)),
                )
                metrics_writer.writerow({
                    "object_id": batch["object_id"][0],
                    "sample": sample_index,
                    **result,
                })
                metrics_file.flush()
            else:
                metrics_writer.writerow({
                    "object_id": f"unconditional_{batch_index}",
                    "sample": sample_index,
                    "chamfer_distance": float("nan"),
                    "f_score": float("nan"),
                    "normal_consistency": float("nan"),
                    "mesh_validity": evaluate.mesh_validity(str(path)),
                })
                metrics_file.flush()
        if not conditional:
            break
    metrics_file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", "-e", required=True)
    parser.add_argument("--resume", "-r", default="last")
    parser.add_argument("--num_samples", "-n", default=5, type=int)
    parser.add_argument("--recon_resolution", default=256, type=int)
    parser.add_argument("--max_batch", default=2**18, type=int)
    parser.add_argument("--modulation_filter_threshold", default=None, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    specs = json.loads((Path(args.exp_dir) / "specs.json").read_text())
    print(specs["Description"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recon_dir = Path(args.exp_dir) / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)
    if specs["training_task"] == "modulation":
        latent_dir = Path(args.exp_dir) / "modulations"
        latent_dir.mkdir(parents=True, exist_ok=True)
        extract_modulations(specs, args, recon_dir, latent_dir, device)
    elif specs["training_task"] in {"diffusion", "combined"}:
        generate(specs, args, recon_dir, device)
