"""Three-stage Diffusion-SDF training harness using COD-VAE latents."""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from models.diffusion import EDMLatentDiffusion
from models.sdf_model import SdfModel


STAGE1_COMPONENTS = {
    "sdf_head_only": {"sdf_network"},
    "cod_decoder_finetune": {"latent_decoder", "triplane_decoder", "sdf_network"},
    "full_cod_finetune": {
        "point_encoder", "variational_block", "latent_decoder",
        "triplane_decoder", "sdf_network",
    },
    "train_from_scratch": {
        "point_encoder", "variational_block", "latent_decoder",
        "triplane_decoder", "sdf_network",
    },
}
STAGE3_COMPONENTS = {
    "diffusion_only": set(),
    "diffusion_and_sdf": {"sdf_network"},
    "diffusion_and_cod_decoder": {"latent_decoder", "triplane_decoder"},
    "full_joint_finetune": {
        "point_encoder", "variational_block", "latent_decoder",
        "triplane_decoder", "sdf_network",
    },
}


class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.task = specs["training_task"]
        if self.task not in {"modulation", "diffusion", "combined"}:
            raise ValueError(f"unknown training_task: {self.task}")

        if self.task in {"modulation", "combined"}:
            self.sdf_model = SdfModel(specs)
        if self.task in {"diffusion", "combined"}:
            self.diffusion_model = EDMLatentDiffusion(
                specs["diffusion_model_specs"], specs["diffusion_specs"]
            )

        self._configure_trainability()
        self._load_latent_statistics()

    def _configure_trainability(self):
        if self.task == "modulation":
            mode = self.specs.get("stage1_mode", "sdf_head_only")
            if mode not in STAGE1_COMPONENTS:
                raise ValueError(f"unknown stage1_mode: {mode}")
            checkpoint_path = self.specs.get("CODVaeSpecs", {}).get("checkpoint_path")
            if mode == "train_from_scratch" and checkpoint_path:
                raise ValueError("train_from_scratch must not specify a COD checkpoint")
            if mode != "train_from_scratch" and not checkpoint_path:
                raise ValueError(f"{mode} requires CODVaeSpecs.checkpoint_path")
            self.sdf_model.set_trainable_components(STAGE1_COMPONENTS[mode])
        elif self.task == "combined":
            mode = self.specs.get("stage3_mode", "diffusion_only")
            if mode not in STAGE3_COMPONENTS:
                raise ValueError(f"unknown stage3_mode: {mode}")
            self.sdf_model.set_trainable_components(STAGE3_COMPONENTS[mode])
            self.diffusion_model.requires_grad_(True)

    def _load_latent_statistics(self):
        dimension = int(
            self.specs.get("diffusion_model_specs", {}).get(
                "latent_dimension",
                self.specs.get("CODVaeSpecs", {}).get("latent_dimension", 32),
            )
        )
        mean = torch.zeros(1, 1, dimension)
        std = torch.ones(1, 1, dimension)
        stats_path = self.specs.get("latent_stats_path")
        if stats_path and Path(stats_path).is_file():
            with np.load(stats_path) as data:
                mean = torch.from_numpy(np.asarray(data["mean"], dtype=np.float32))
                std = torch.from_numpy(np.asarray(data["std"], dtype=np.float32))
        self.register_buffer("latent_mean", mean, persistent=True)
        self.register_buffer("latent_std", std.clamp_min(1e-6), persistent=True)

    def normalize_latent(self, latent):
        return (latent - self.latent_mean) / self.latent_std

    def denormalize_latent(self, latent):
        return latent * self.latent_std + self.latent_mean

    @staticmethod
    def _conditioning(batch):
        conditioning = batch.get("conditioning")
        return conditioning if conditioning else None

    def sdf_reconstruction_loss(self, prediction, target):
        loss_specs = self.specs.get("sdf_loss", {})
        loss_type = loss_specs.get("type", "l1")
        if loss_type == "l1":
            return F.l1_loss(prediction, target)
        if loss_type == "huber":
            return F.smooth_l1_loss(
                prediction, target, beta=float(loss_specs.get("huber_delta", 0.01))
            )
        if loss_type == "truncated_sdf":
            threshold = float(loss_specs.get("truncation", 0.1))
            return F.l1_loss(
                prediction.clamp(-threshold, threshold),
                target.clamp(-threshold, threshold),
            )
        raise ValueError(f"unknown SDF reconstruction loss: {loss_type}")

    @staticmethod
    def kl_loss(posterior):
        if posterior is None:
            return torch.zeros((), device="cpu")
        return posterior.kl().mean()

    @staticmethod
    def cod_auxiliary_loss(encoded, decoded):
        encoded = F.layer_norm(encoded, (encoded.shape[-1],))
        decoded = F.layer_norm(decoded, (decoded.shape[-1],))
        return F.mse_loss(decoded, encoded.detach())

    def stage1_losses(self, batch):
        output = self.sdf_model(
            batch["surface_points"],
            batch["query_points"],
            sample_posterior=bool(self.specs.get("sample_posterior", True)),
        )
        sdf = self.sdf_reconstruction_loss(output["sdf"], batch["query_sdf"])
        kl = self.kl_loss(output["posterior"]).to(sdf.device)
        aux = self.cod_auxiliary_loss(
            output["encoded_features"], output["decoded_latent"]
        )
        weights = self.specs.get("loss_weights", {})
        total = (
            float(weights.get("sdf", 1.0)) * sdf
            + float(weights.get("kl", self.specs.get("kld_weight", 0.0))) * kl
            + float(weights.get("cod_aux", 0.0)) * aux
        )
        return {"loss": total, "sdf": sdf, "kl": kl, "cod_aux": aux}

    def stage2_losses(self, batch):
        loss, clean_estimate, _, _ = self.diffusion_model.training_loss(
            batch["latent"], self._conditioning(batch)
        )
        return {"loss": loss, "diffusion": loss, "clean_latent": clean_estimate}

    def stage3_losses(self, batch):
        direct = self.sdf_model(
            batch["surface_points"],
            batch["query_points"],
            sample_posterior=True,
        )
        direct_sdf = self.sdf_reconstruction_loss(
            direct["sdf"], batch["query_sdf"]
        )
        clean = self.normalize_latent(direct["latent"])
        diffusion_loss, clean_estimate, _, _ = self.diffusion_model.training_loss(
            clean, self._conditioning(batch)
        )
        denoised_latent = self.denormalize_latent(clean_estimate)
        generated_planes = self.sdf_model.decode_latent(denoised_latent)["planes"]
        generated_sdf = self.sdf_model.query_sdf(
            generated_planes, batch["query_points"]
        )
        generated_loss = self.sdf_reconstruction_loss(
            generated_sdf, batch["query_sdf"]
        )
        kl = self.kl_loss(direct["posterior"]).to(direct_sdf.device)
        weights = self.specs.get("loss_weights", {})
        total = (
            float(weights.get("direct", 1.0)) * direct_sdf
            + float(weights.get("diffusion", 1.0)) * diffusion_loss
            + float(weights.get("generated", 1.0)) * generated_loss
            + float(weights.get("kl", 0.0)) * kl
        )
        return {
            "loss": total,
            "sdf_direct": direct_sdf,
            "diffusion": diffusion_loss,
            "sdf_denoised": generated_loss,
            "kl": kl,
        }

    def _losses(self, batch):
        if self.task == "modulation":
            return self.stage1_losses(batch)
        if self.task == "diffusion":
            return self.stage2_losses(batch)
        return self.stage3_losses(batch)

    def training_step(self, batch, batch_idx):
        losses = self._losses(batch)
        for name, value in losses.items():
            if name != "clean_latent":
                self.log(f"train/{name}", value, on_step=True, on_epoch=False)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self._losses(batch)
        for name, value in losses.items():
            if name != "clean_latent":
                self.log(f"val/{name}", value, on_step=False, on_epoch=True)
        return losses["loss"]

    def configure_optimizers(self):
        rates = self.specs.get("learning_rates", {})
        groups = []
        seen = set()

        def add_group(name, module, fallback):
            parameters = [
                parameter for parameter in module.parameters()
                if parameter.requires_grad and id(parameter) not in seen
            ]
            if not parameters:
                return
            seen.update(map(id, parameters))
            groups.append({
                "params": parameters,
                "lr": float(rates.get(name, fallback)),
                "name": name,
            })

        if self.task in {"modulation", "combined"}:
            for name, module in self.sdf_model.component_modules().items():
                add_group(name, module, self.specs.get("sdf_lr", 1e-4))
        if self.task in {"diffusion", "combined"}:
            add_group("diffusion", self.diffusion_model, self.specs.get("diff_lr", 1e-5))
        if not groups:
            raise ValueError("the selected training mode has no trainable parameters")
        return torch.optim.AdamW(
            groups, weight_decay=float(self.specs.get("weight_decay", 0.0))
        )
