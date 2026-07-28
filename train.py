#!/usr/bin/env python3

import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    Callback,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning import loggers as pl_loggers

import os
import json, csv
import time
from tqdm.auto import tqdm
from einops import rearrange, reduce
import numpy as np
import trimesh
import warnings

# add paths in model/__init__.py for new models
from models import * 
from utils import mesh, evaluate
from diff_utils.helpers import save_code_to_conf
#from metrics.evaluation_metrics import *#compute_all_metrics
#from metrics import evaluation_metrics

from dataloader.sdf_loader import SdfLoader
from dataloader.modulation_loader import ModulationLoader, ensure_modulation_cache
from dataloader.conditioning import build_conditioning_sources
from dataloader.virtual_dataset import VirtualDataset


def train():
    
    # initialize dataset and loader
    split = json.load(open(specs["TrainSplit"], "r"))
    val_split = None
    if specs.get("ValSplit") is not None:
        val_split = json.load(open(specs["ValSplit"], "r"))

    use_spawn_workers = False
    if specs['training_task'] == 'diffusion':
        cache_path, stats_path = ensure_modulation_cache(
            specs,
            args.exp_dir,
            batch_size=specs.get("modulation_batch_size", min(args.batch_size, 8)),
            workers=specs.get("modulation_workers", args.workers),
        )
        specs["data_path"] = cache_path
        specs["latent_stats_path"] = stats_path
        conditioning_sources = build_conditioning_sources(get_conditioning_specs(specs))
        modulation_variants = max(1, int(specs.get("modulation_variants", 1)))
        train_records = ModulationLoader.build_records(
            specs["data_path"],
            split,
            conditioning_sources,
            modulation_variants=modulation_variants,
        )
        val_records = (
            ModulationLoader.build_records(specs["data_path"], val_split, conditioning_sources)
            if val_split is not None
            else None
        )
        all_records = train_records + (val_records or [])
        use_spawn_workers = prepare_conditioning_sources(conditioning_sources, all_records) and args.workers > 0
        train_dataset = build_dataset(
            split,
            conditioning_sources=conditioning_sources,
            records=train_records,
            sample_posterior_latents=bool(
                specs.get("sample_posterior_latents", False)
            ),
        )
        val_dataset = (
            build_dataset(
                val_split,
                conditioning_sources=conditioning_sources,
                records=val_records,
            )
            if val_split is not None
            else None
        )
    else:
        conditioning_sources = (
            build_conditioning_sources(get_conditioning_specs(specs))
            if specs["training_task"] == "combined"
            else []
        )
        train_dataset = build_dataset(
            split,
            conditioning_sources=conditioning_sources,
            deterministic_surface_sampling=bool(
                specs.get("DeterministicTrainSurfaceSampling", False)
            ),
            sampling_seed=int(specs.get("TrainSurfaceSamplingSeed", 0)),
        )
        val_dataset = (
            build_dataset(
                val_split,
                conditioning_sources=conditioning_sources,
                deterministic_sampling=bool(
                    specs.get("DeterministicValidationSampling", False)
                ),
            )
            if val_split is not None
            else None
        )
        if conditioning_sources:
            all_records = train_dataset.records + (
                val_dataset.records if val_dataset is not None else []
            )
            use_spawn_workers = (
                prepare_conditioning_sources(conditioning_sources, all_records)
                and args.workers > 0
            )

    if args.virtual_train_size is not None:
        train_dataset = VirtualDataset(train_dataset, args.virtual_train_size)

    train_dataloader = build_dataloader(
        train_dataset,
        drop_last=True,
        shuffle=True,
        use_spawn_workers=use_spawn_workers,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            drop_last=False,
            shuffle=False,
            use_spawn_workers=use_spawn_workers,
        )

    # creates a copy of current code / files in the config folder
    save_code_to_conf(args.exp_dir) 
    
    # pytorch lightning callbacks 
    periodic_callback = ModelCheckpoint(
        dirpath=args.exp_dir,
        filename='{epoch}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=specs["log_freq"],
    )
    checkpoint_monitor = specs.get("checkpoint_monitor", "val/loss")
    best_callback = ModelCheckpoint(
        dirpath=args.exp_dir,
        filename='best',
        monitor=checkpoint_monitor,
        mode=specs.get("checkpoint_mode", "min"),
        save_top_k=1,
    ) if val_dataloader is not None else None
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [periodic_callback, lr_monitor]
    if best_callback is not None:
        callbacks.append(best_callback)
    early_stopping_specs = specs.get("early_stopping")
    if early_stopping_specs:
        if val_dataloader is None:
            raise ValueError("early_stopping requires a validation split")
        callbacks.append(EarlyStopping(
            monitor=early_stopping_specs.get("monitor", "val/loss"),
            mode=early_stopping_specs.get("mode", "min"),
            patience=int(early_stopping_specs.get("patience", 100)),
            min_delta=float(early_stopping_specs.get("min_delta", 0.0)),
            verbose=bool(early_stopping_specs.get("verbose", True)),
            check_finite=bool(early_stopping_specs.get("check_finite", True)),
        ))

    model = CombinedModel(specs)

    # note on loading from checkpoint:
    # if resuming from training modulation, diffusion, or end-to-end, just load saved checkpoint 
    # however, if fine-tuning end-to-end after training modulation and diffusion separately, will need to load sdf and diffusion checkpoints separately
    if args.init_from is not None:
        load_weights_only(
            model,
            args.init_from,
            allowed_missing_prefixes=specs.get(
                "init_from_allowed_missing_prefixes", ()
            ),
            excluded_keys=specs.get("init_from_excluded_keys", ()),
        )
        resume = None
    elif args.resume == 'finetune':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modulation = torch.load(specs["modulation_ckpt_path"], map_location="cpu")
            sdf_state = {
                key[len("sdf_model."):]: value
                for key, value in modulation["state_dict"].items()
                if key.startswith("sdf_model.")
            }
            model.sdf_model.load_state_dict(sdf_state)
            diffusion = torch.load(specs["diffusion_ckpt_path"], map_location="cpu")
            diffusion_state = {
                key[len("diffusion_model."):]: value
                for key, value in diffusion["state_dict"].items()
                if key.startswith("diffusion_model.")
            }
            model.diffusion_model.load_state_dict(diffusion_state)
        resume = None
    elif args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
    else:
        resume = None  

    log_every_n_steps = specs.get("log_every_n_steps", 1)

    # precision 16 can be unstable (nan loss); recommend using 32
    trainer = pl.Trainer(accelerator='gpu', devices=-1, precision=32, max_epochs=specs["num_epochs"], callbacks=callbacks, log_every_n_steps=log_every_n_steps,
                        gradient_clip_val=float(specs.get("gradient_clip_val", 0.0)),
                        default_root_dir=os.path.join("tensorboard_logs", args.exp_dir))
    if val_dataloader is not None:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=resume)
    else:
        trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=resume)


def load_weights_only(
    model,
    checkpoint_path,
    allowed_missing_prefixes=(),
    excluded_keys=(),
):
    """Load model parameters without restoring trainer or optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"weights-only initialization requires a Lightning checkpoint "
            f"with a state_dict: {checkpoint_path}"
        )
    allowed_missing_prefixes = tuple(allowed_missing_prefixes)
    excluded_keys = set(excluded_keys)
    if not allowed_missing_prefixes and not excluded_keys:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return
    state_dict = {
        key: value
        for key, value in checkpoint["state_dict"].items()
        if key not in excluded_keys
    }
    result = model.load_state_dict(state_dict, strict=False)
    disallowed_missing = [
        key for key in result.missing_keys
        if key not in excluded_keys
        and not key.startswith(allowed_missing_prefixes)
    ]
    if disallowed_missing or result.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch: "
            f"missing keys={disallowed_missing}, "
            f"unexpected keys={result.unexpected_keys}"
        )


def build_dataloader(dataset, drop_last, shuffle, use_spawn_workers=False):
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "drop_last": drop_last,
        "shuffle": shuffle,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if use_spawn_workers:
        dataloader_kwargs["multiprocessing_context"] = "spawn"
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def build_dataset(
    split,
    conditioning_sources=None,
    records=None,
    sample_posterior_latents=False,
    deterministic_sampling=False,
    deterministic_surface_sampling=False,
    sampling_seed=None,
):
    if specs['training_task'] == 'diffusion':
        return ModulationLoader(
            specs["data_path"],
            split_file=split,
            conditioning=get_conditioning_specs(specs),
            conditioning_sources=conditioning_sources,
            records=records,
            latent_stats_path=specs.get("latent_stats_path"),
            sample_posterior=sample_posterior_latents,
        )

    return SdfLoader(
        specs["DataSource"],
        split,
        samples_per_mesh=specs.get("SampPerMesh", 16000),
        surface_point_count=specs.get("SurfacePointCount", 2048),
        near_surface_ratio=specs.get("NearSurfaceRatio", 0.7),
        modulation_path=specs.get("modulation_path", None),
        condition_surface=bool(
            specs.get("diffusion_model_specs", {}).get("cond", False)
        ),
        deterministic_sampling=deterministic_sampling,
        deterministic_surface_sampling=deterministic_surface_sampling,
        sampling_seed=(
            int(specs.get("ValidationSamplingSeed", 0))
            if sampling_seed is None
            else int(sampling_seed)
        ),
        conditioning_sources=conditioning_sources,
    )


def prepare_conditioning_sources(conditioning_sources, records, force=False):
    prepared_with_cuda = False
    for source in conditioning_sources:
        source.prepare(records, force=force)
        prepared_with_cuda = prepared_with_cuda or getattr(source, "prepared_with_cuda", False)
    return prepared_with_cuda


def get_conditioning_specs(specs):
    conditioning = specs.get("conditioning", None)
    if conditioning is not None:
        single_spec = isinstance(conditioning, dict)
        conditioning_specs = [conditioning] if single_spec else list(conditioning)
        normalized_specs = []
        for spec in conditioning_specs:
            normalized_spec = dict(spec)
            if normalized_spec.get("type") == "image":
                normalized_spec["require_cached"] = True
            normalized_specs.append(normalized_spec)
        return normalized_specs[0] if single_spec else normalized_specs

    if specs.get("pc_path", None) is None:
        return None

    return {
        "type": "point_cloud",
        "path": specs["pc_path"],
        "pc_size": specs.get("total_pc_size", 1024),
    }


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    checkpoint_group = arg_parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume", "-r", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )
    checkpoint_group.add_argument(
        "--init_from",
        default=None,
        help=(
            "initialize model weights from a Lightning checkpoint while "
            "starting a fresh optimizer, learning rate, epoch, and global step"
        ),
    )

    arg_parser.add_argument("--batch_size", "-b", default=32, type=int)
    arg_parser.add_argument( "--workers", "-w", default=8, type=int)
    arg_parser.add_argument(
        "--virtual_train_size",
        default=None,
        type=int,
        help=(
            "Expose the training dataset at this virtual size by cycling through "
            "its records; each access still runs the dataset's sampling logic."
        ),
    )

    args = arg_parser.parse_args()
    specs_path = os.path.join(args.exp_dir, "specs.json")
    specs = json.load(open(specs_path))
    print(specs["Description"])

    train()
