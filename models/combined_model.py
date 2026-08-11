"""Three-stage Diffusion-SDF training harness using COD-VAE latents."""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from models.diffusion import EDMLatentDiffusion, latent_set_distance_matrix
from models.sdf_model import SdfModel


STAGE1_COMPONENTS = {
    "sdf_head_only": {"sdf_network"},
    "sdf_head_conv_refine": {"conv_refine", "sdf_network"},
    "encoder_finetune": {"point_encoder", "variational_block"},
    "triplane_sdf_finetune": {"triplane_decoder", "sdf_network"},
    "cod_decoder_finetune": {"latent_decoder", "triplane_decoder", "sdf_network"},
    "joint_refinement": {
        "variational_block", "latent_decoder", "triplane_decoder", "sdf_network",
    },
    "full_cod_finetune": {
        "point_encoder", "variational_block", "latent_decoder",
        "triplane_decoder", "sdf_network",
    },
    "train_from_scratch": {
        "point_encoder", "variational_block", "latent_decoder",
        "triplane_decoder", "sdf_network",
    },
    "learned_query_adaptation": {
        "compact_queries", "compact_token_attention",
    },
    "learned_query_encoder_refinement": {
        "compact_queries", "compact_token_attention",
        "point_encoder_backbone", "variational_block",
    },
    "learned_query_vae_finetune": {
        "compact_queries", "compact_token_attention",
        "point_encoder_backbone", "variational_block",
        "latent_decoder", "triplane_decoder", "sdf_network",
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


def validate_training_specs(specs):
    """Reject component and uncertainty combinations that silently do nothing."""
    task = specs.get("training_task")
    if task == "modulation":
        mode = specs.get("stage1_mode", "sdf_head_only")
        active = set(STAGE1_COMPONENTS.get(mode, ()))
        if mode.startswith("learned_query_") and not bool(
            specs.get("CODVaeSpecs", {}).get("use_learnable_positions", False)
        ):
            raise ValueError(
                f"{mode} requires CODVaeSpecs.use_learnable_positions=true"
            )
    elif task == "diffusion":
        active = {"diffusion"}
    elif task == "combined":
        mode = specs.get("stage3_mode", "diffusion_only")
        active = set(STAGE3_COMPONENTS.get(mode, ())) | {"diffusion"}
    else:
        return
    if bool(
        specs.get("diffusion_model_specs", {}).get(
            "use_learnable_slot_embeddings", False
        )
    ):
        active.add("diffusion_slot_embeddings")
    decoder_specs = specs.get("CODVaeSpecs", {}).get("decoder_params", {})
    if (
        "triplane_decoder" in active
        and bool(decoder_specs.get("use_conv_refine", False))
    ):
        active.add("conv_refine")
    unused_rates = set(specs.get("learning_rates", {})).difference(active)
    if unused_rates:
        raise ValueError(
            "learning rates configured for frozen components: "
            f"{sorted(unused_rates)}"
        )

    cod_specs = specs.get("CODVaeSpecs", {})
    uncertainty_mode = cod_specs.get("uncertainty_mode")
    if uncertainty_mode is None and cod_specs.get("disable_uncertainty_pruning", True):
        uncertainty_mode = "full_sdf"
    if uncertainty_mode not in {
        None, "full_sdf", "occupancy_pruning", "disabled",
    }:
        raise ValueError(f"unknown uncertainty mode: {uncertainty_mode}")
    uncertainty_weight = float(
        specs.get("loss_weights", {}).get("uncertainty", 0.0)
    )
    if uncertainty_mode == "disabled" and uncertainty_weight > 0:
        raise ValueError(
            "uncertainty_mode='disabled' requires a zero uncertainty loss weight"
        )


class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        validate_training_specs(specs)
        self.save_hyperparameters({"specs": specs})
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

        self._configure_few_shot_adaptation()

        self._configure_trainability()
        self._load_latent_statistics()

    def _configure_few_shot_adaptation(self):
        """Create the frozen source prior without registering it in checkpoints."""
        self.lambda_pairwise = float(self.specs.get("lambda_pairwise", 0.0))
        if self.lambda_pairwise < 0:
            raise ValueError("lambda_pairwise must be non-negative")
        self.pairwise_distance = self.specs.get(
            "pairwise_distance", "sliced_wasserstein"
        )
        if self.pairwise_distance not in {"sliced_wasserstein", "mmd"}:
            raise ValueError(
                f"unknown pairwise_distance: {self.pairwise_distance}"
            )
        self.mmd_bandwidth = float(self.specs.get("pairwise_mmd_bandwidth", 1.0))
        projection_count = int(self.specs.get("pairwise_num_projections", 32))
        if projection_count <= 0:
            raise ValueError("pairwise_num_projections must be positive")
        dimension = int(
            self.specs.get("diffusion_model_specs", {}).get("latent_dimension", 32)
        )
        generator = torch.Generator().manual_seed(
            int(self.specs.get("pairwise_projection_seed", 0))
        )
        projections = torch.randn(projection_count, dimension, generator=generator)
        self.register_buffer(
            "pairwise_projections", F.normalize(projections, dim=-1), persistent=False
        )
        self._training_latent_bank = []

        source = None
        source_path = self.specs.get("source_diffusion_checkpoint")
        if self.lambda_pairwise > 0 and self.task != "diffusion":
            raise ValueError("pairwise adaptation is currently supported for stage 2")
        if self.lambda_pairwise > 0 and not source_path:
            raise ValueError(
                "a positive lambda_pairwise requires source_diffusion_checkpoint"
            )
        if source_path:
            if self.task != "diffusion":
                raise ValueError("source_diffusion_checkpoint is only valid for stage 2")
            source = EDMLatentDiffusion(
                self.specs["diffusion_model_specs"], self.specs["diffusion_specs"]
            )
            self._load_source_diffusion(source, source_path)
            source.requires_grad_(False).eval()
        # Deliberately bypass nn.Module registration: the immutable source is
        # reconstructed from its configured checkpoint and is not duplicated in
        # every target checkpoint.
        object.__setattr__(self, "source_diffusion_model", source)

    @staticmethod
    def _load_source_diffusion(source, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(
                "source_diffusion_checkpoint must be a Lightning checkpoint "
                "with a state_dict"
            )
        state = checkpoint["state_dict"]
        prefix = "diffusion_model."
        selected = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not selected and all(
            key.startswith(("model.",)) or key in {
                "latent_mean", "latent_std", "pairwise_projections"
            }
            for key in state
        ):
            selected = {key: value for key, value in state.items() if key.startswith("model.")}
        if not selected:
            raise ValueError(
                "source checkpoint does not contain diffusion_model parameters"
            )
        source.load_state_dict(selected, strict=True)

    def latent_set_distances(self, left, right=None):
        return latent_set_distance_matrix(
            left,
            right,
            metric=self.pairwise_distance,
            mmd_bandwidth=self.mmd_bandwidth,
            projections=self.pairwise_projections,
        )

    def pairwise_preservation_loss(self, target, source):
        if target.shape[0] < 2:
            return target.new_zeros(())
        target_distances = self.latent_set_distances(target)
        with torch.no_grad():
            source_distances = self.latent_set_distances(source)
        indices = torch.triu_indices(
            target.shape[0], target.shape[0], offset=1, device=target.device
        )
        return F.mse_loss(
            target_distances[indices[0], indices[1]],
            source_distances[indices[0], indices[1]],
        )

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

    @staticmethod
    def latent_set_consistency_loss(first, second):
        """Squared symmetric Chamfer distance between latent token sets."""
        distances = (
            first[:, :, None, :] - second[:, None, :, :]
        ).square().mean(dim=-1)
        return (
            distances.min(dim=-1).values.mean()
            + distances.min(dim=-2).values.mean()
        )

    def stage1_geometry_losses(
        self, planes, batch, *, compute_surface_zero, compute_eikonal,
        compute_normal,
    ):
        """SDF-specific surface constraints evaluated on a small point subset."""
        count = min(
            int(self.specs.get("GeometryRegularizationSamples", 512)),
            batch["surface_points"].shape[1],
        )
        difference_step = float(
            self.specs.get("GeometryFiniteDifferenceStep", 0.01)
        )
        surface = batch["surface_points"][:, :count].detach().clone()
        surface = surface.clamp(-1.0 + difference_step, 0.999 - difference_step)
        zero = planes.new_zeros(())
        surface_zero = zero
        if compute_surface_zero:
            surface_zero = self.sdf_model.query_sdf(planes, surface).abs().mean()

        gradient = None
        if compute_eikonal or compute_normal:
            derivatives = []
            for axis in range(3):
                offset = torch.zeros_like(surface)
                offset[..., axis] = difference_step
                positive = self.sdf_model.query_sdf(planes, surface + offset)
                negative = self.sdf_model.query_sdf(planes, surface - offset)
                derivatives.append(
                    (positive - negative) / (2.0 * difference_step)
                )
            gradient = torch.stack(derivatives, dim=-1)

        eikonal = zero
        if compute_eikonal:
            eikonal = (gradient.norm(dim=-1) - 1.0).square().mean()
        normal = zero
        if compute_normal and batch.get("surface_normals") is not None:
            target_normals = batch["surface_normals"][:, :count].to(gradient)
            cosine = F.cosine_similarity(gradient, target_normals, dim=-1)
            normal = (1.0 - cosine.abs()).mean()
        return surface_zero, eikonal, normal

    def stage1_losses(self, batch):
        weights = self.specs.get("loss_weights", {})
        needs_initial_sdf = float(weights.get("sdf_initial", 0.0)) > 0
        needs_uncertainty = float(weights.get("uncertainty", 0.0)) > 0
        sample_posterior = bool(self.specs.get("sample_posterior", True))
        if not self.training:
            sample_posterior = bool(
                self.specs.get(
                    "validation_sample_posterior",
                    sample_posterior,
                )
            )
        output = self.sdf_model(
            batch.get("encoder_surface_points", batch["surface_points"]),
            batch["query_points"],
            sample_posterior=sample_posterior,
            return_initial_sdf=needs_initial_sdf or needs_uncertainty,
            return_query_uncertainty=needs_uncertainty,
        )
        sdf = self.sdf_reconstruction_loss(output["sdf"], batch["query_sdf"])
        initial_sdf = sdf.new_zeros(())
        uncertainty = sdf.new_zeros(())
        if needs_initial_sdf:
            initial_sdf = self.sdf_reconstruction_loss(
                output["initial_sdf"], batch["query_sdf"]
            )
        if needs_uncertainty:
            uncertainty_scale = float(
                self.specs.get("uncertainty_target_scale", 0.1)
            )
            uncertainty_target = (
                (output["initial_sdf"] - batch["query_sdf"]).abs()
                / uncertainty_scale
            ).clamp(0.0, 1.0).detach()
            uncertainty = F.mse_loss(
                output["query_uncertainty"], uncertainty_target
            )
        kl = sdf.new_zeros(())
        if float(weights.get("kl", self.specs.get("kld_weight", 0.0))) > 0:
            kl = self.kl_loss(output["posterior"]).to(sdf.device)
        aux = sdf.new_zeros(())
        if float(weights.get("cod_aux", 0.0)) > 0:
            aux = self.cod_auxiliary_loss(
                output["encoded_features"], output["decoded_latent"]
            )
        latent_consistency = sdf.new_zeros(())
        if float(weights.get("latent_consistency", 0.0)) > 0:
            paired_surface = batch.get(
                "paired_encoder_surface_points",
                batch.get("paired_surface_points"),
            )
            if paired_surface is None:
                raise ValueError(
                    "a positive latent_consistency weight requires "
                    "PairedSurfaceSampling"
                )
            _, paired_posterior, _ = self.sdf_model.encode_surface(
                paired_surface,
                sample_posterior=False,
            )
            if output["posterior"] is None or paired_posterior is None:
                raise ValueError(
                    "latent_consistency requires variational encoder posteriors"
                )
            latent_consistency = self.latent_set_consistency_loss(
                output["posterior"].mean,
                paired_posterior.mean,
            )
        surface_zero = sdf.new_zeros(())
        eikonal = sdf.new_zeros(())
        normal = sdf.new_zeros(())
        geometry_enabled = {
            name: float(weights.get(name, 0.0)) > 0
            for name in ("surface_zero", "eikonal", "normal")
        }
        needs_geometry = any(geometry_enabled.values())
        validate_geometry = (
            not self.training
            and bool(self.specs.get("ValidateGeometryRegularization", False))
        )
        if needs_geometry and (
            (self.training and torch.is_grad_enabled()) or validate_geometry
        ):
            surface_zero, eikonal, normal = self.stage1_geometry_losses(
                output["planes"], batch,
                compute_surface_zero=geometry_enabled["surface_zero"],
                compute_eikonal=geometry_enabled["eikonal"],
                compute_normal=geometry_enabled["normal"],
            )
        total = (
            float(weights.get("sdf", 1.0)) * sdf
            + float(weights.get("sdf_initial", 0.0)) * initial_sdf
            + float(weights.get("uncertainty", 0.0)) * uncertainty
            + float(weights.get("kl", self.specs.get("kld_weight", 0.0))) * kl
            + float(weights.get("cod_aux", 0.0)) * aux
            + float(weights.get("latent_consistency", 0.0))
            * latent_consistency
            + float(weights.get("surface_zero", 0.0)) * surface_zero
            + float(weights.get("eikonal", 0.0)) * eikonal
            + float(weights.get("normal", 0.0)) * normal
        )
        posterior = output["posterior"]
        posterior_std = posterior.std.detach().mean() if posterior is not None else sdf.new_zeros(())
        active_dimensions = (
            (posterior.mean.detach().std(dim=(0, 1)) > 0.01).float().sum()
            if posterior is not None
            else sdf.new_zeros(())
        )
        return {
            "loss": total,
            "sdf": sdf,
            "sdf_initial": initial_sdf,
            "uncertainty": uncertainty,
            "surface_zero": surface_zero,
            "eikonal": eikonal,
            "normal": normal,
            "kl": kl,
            "cod_aux": aux,
            "latent_consistency": latent_consistency,
            "posterior_std": posterior_std,
            "active_dimensions": active_dimensions,
        }

    def stage2_losses(self, batch, noise=None, sigma=None):
        diffusion_loss, clean_estimate, noisy, sigma = self.diffusion_model.training_loss(
            batch["latent"], self._conditioning(batch), noise=noise, sigma=sigma
        )
        pairwise = diffusion_loss.new_zeros(())
        if self.lambda_pairwise > 0:
            source = self.source_diffusion_model
            source.to(noisy.device).eval()
            with torch.no_grad():
                source_estimate = source(noisy, sigma, self._conditioning(batch))
            pairwise = self.pairwise_preservation_loss(
                clean_estimate, source_estimate
            )
        total = diffusion_loss + self.lambda_pairwise * pairwise
        return {
            "loss": total,
            "diffusion": diffusion_loss,
            "pairwise": pairwise,
            "clean_latent": clean_estimate,
        }

    def deterministic_validation_noise(self, clean, batch_idx):
        seed = int(self.specs.get("validation_noise_seed", 0)) + int(batch_idx)
        generator = torch.Generator(device=clean.device).manual_seed(seed)
        rnd = torch.randn(
            clean.shape[0], device=clean.device, generator=generator
        )
        sigma = (
            rnd * self.diffusion_model.p_std + self.diffusion_model.p_mean
        ).exp()
        noise = torch.randn(
            clean.shape,
            dtype=clean.dtype,
            device=clean.device,
            generator=generator,
        )
        return noise, sigma

    def stage3_losses(self, batch, deterministic_noise_batch_idx=None):
        weights = self.specs.get("loss_weights", {})
        direct_weight = float(weights.get("direct", 1.0))
        generated_weight = float(weights.get("generated", 1.0))
        sample_posterior = bool(self.specs.get("sample_posterior", True))
        if not self.training:
            sample_posterior = bool(
                self.specs.get(
                    "validation_sample_posterior",
                    sample_posterior,
                )
            )

        if direct_weight > 0:
            direct = self.sdf_model(
                batch["surface_points"],
                batch["query_points"],
                sample_posterior=sample_posterior,
            )
            direct_sdf = self.sdf_reconstruction_loss(
                direct["sdf"], batch["query_sdf"]
            )
            latent = direct["latent"]
            posterior = direct["posterior"]
        else:
            latent, posterior, _ = self.sdf_model.encode_surface(
                batch["surface_points"],
                sample_posterior=sample_posterior,
            )
            direct_sdf = latent.new_zeros(())

        clean = self.normalize_latent(latent)
        noise = sigma = None
        if deterministic_noise_batch_idx is not None:
            noise, sigma = self.deterministic_validation_noise(
                clean, deterministic_noise_batch_idx
            )
        diffusion_loss, clean_estimate, _, _ = self.diffusion_model.training_loss(
            clean,
            self._conditioning(batch),
            noise=noise,
            sigma=sigma,
        )
        if generated_weight > 0:
            denoised_latent = self.denormalize_latent(clean_estimate)
            generated_planes = self.sdf_model.decode_latent(
                denoised_latent
            )["planes"]
            generated_sdf = self.sdf_model.query_sdf(
                generated_planes, batch["query_points"]
            )
            generated_loss = self.sdf_reconstruction_loss(
                generated_sdf, batch["query_sdf"]
            )
        else:
            generated_loss = direct_sdf.new_zeros(())
        kl = self.kl_loss(posterior).to(direct_sdf.device)
        total = (
            direct_weight * direct_sdf
            + float(weights.get("diffusion", 1.0)) * diffusion_loss
            + generated_weight * generated_loss
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
        if self.task == "diffusion":
            self._update_training_latent_bank(batch["latent"])
        batch_size = (
            batch["latent"].shape[0]
            if self.task == "diffusion"
            else batch["surface_points"].shape[0]
        )
        for name, value in losses.items():
            if name != "clean_latent":
                self.log(
                    f"train/{name}", value, on_step=True, on_epoch=False,
                    batch_size=batch_size,
                )
        return losses["loss"]

    def _update_training_latent_bank(self, latent):
        monitoring = self.specs.get("few_shot_monitoring", {})
        if not bool(monitoring.get("enabled", False)):
            return
        capacity = max(1, int(monitoring.get("train_bank_size", 256)))
        self._training_latent_bank.append(latent.detach().cpu())
        total = sum(value.shape[0] for value in self._training_latent_bank)
        while total > capacity and self._training_latent_bank:
            removed = self._training_latent_bank.pop(0)
            total -= removed.shape[0]

    def on_validation_epoch_end(self):
        if self.task != "diffusion" or not self._training_latent_bank:
            return
        monitoring = self.specs.get("few_shot_monitoring", {})
        if not bool(monitoring.get("enabled", False)):
            return
        frequency = max(1, int(monitoring.get("every_n_epochs", 1)))
        if (self.current_epoch + 1) % frequency:
            return
        count = max(2, int(monitoring.get("num_samples", 16)))
        steps = int(
            monitoring.get("sampling_steps", self.diffusion_model.sampling_steps)
        )
        device = next(self.diffusion_model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(
            int(monitoring.get("seed", 0)) + int(self.current_epoch)
        )
        noise = torch.randn(
            count,
            self.diffusion_model.latent_tokens,
            self.diffusion_model.latent_dimension,
            device=device,
            generator=generator,
        )
        conditioning = None
        if self.diffusion_model.model.conditional:
            return
        with torch.no_grad():
            generated = self.diffusion_model.sample(
                count, conditioning=conditioning, noise=noise, num_steps=steps
            )
            training = torch.cat(self._training_latent_bank, dim=0).to(device)
            nearest = self.latent_set_distances(generated, training).min(dim=1).values
            diversity_matrix = self.latent_set_distances(generated)
            indices = torch.triu_indices(count, count, offset=1, device=device)
            diversity = diversity_matrix[indices[0], indices[1]].mean()
        self.log(
            "val/generated_to_training_nn", nearest.mean(), on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/generation_diversity", diversity, on_epoch=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        if self.task == "diffusion":
            noise, sigma = self.deterministic_validation_noise(
                batch["latent"], batch_idx
            )
            losses = self.stage2_losses(batch, noise=noise, sigma=sigma)
        elif self.task == "combined":
            losses = self.stage3_losses(
                batch,
                deterministic_noise_batch_idx=batch_idx,
            )
        else:
            losses = self._losses(batch)
        batch_size = (
            batch["latent"].shape[0]
            if self.task == "diffusion"
            else batch["surface_points"].shape[0]
        )
        for name, value in losses.items():
            if name != "clean_latent":
                self.log(
                    f"val/{name}", value, on_step=False, on_epoch=True,
                    batch_size=batch_size,
                )
        return losses["loss"]

    def on_after_backward(self):
        squared_norm = None
        for parameter in self.parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().square().sum()
            squared_norm = value if squared_norm is None else squared_norm + value
        if squared_norm is not None:
            self.log("train/gradient_norm", squared_norm.sqrt(), on_step=True)

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
            decoder = self.sdf_model.cod_vae.autoencoder.decoder
            if "conv_refine" in rates and decoder.conv_refine is not None:
                add_group(
                    "conv_refine",
                    decoder.conv_refine,
                    rates.get("triplane_decoder", self.specs.get("sdf_lr", 1e-4)),
                )
            selected = getattr(
                self.sdf_model, "_trainable_component_names", set()
            )
            for name, parameters in self.sdf_model.component_parameters().items():
                if name not in selected:
                    continue
                add_group(
                    name,
                    nn.ParameterList(parameters),
                    self.specs.get("sdf_lr", 1e-4),
                )
        if self.task in {"diffusion", "combined"}:
            slot_embedding = self.diffusion_model.model.slot_embedding
            if slot_embedding is not None:
                add_group(
                    "diffusion_slot_embeddings",
                    nn.ParameterList([slot_embedding]),
                    rates.get(
                        "diffusion",
                        self.specs.get("diff_lr", 1e-5),
                    ),
                )
            add_group("diffusion", self.diffusion_model, self.specs.get("diff_lr", 1e-5))
        if not groups:
            raise ValueError("the selected training mode has no trainable parameters")
        optimizer = torch.optim.AdamW(
            groups, weight_decay=float(self.specs.get("weight_decay", 0.0))
        )
        scheduler_specs = self.specs.get("lr_scheduler")
        if not scheduler_specs:
            return optimizer
        scheduler_type = scheduler_specs.get("type", "cosine")
        if scheduler_type != "cosine":
            raise ValueError(f"unknown lr_scheduler type: {scheduler_type}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                int(scheduler_specs.get("epochs", self.specs["num_epochs"])),
            ),
            eta_min=float(scheduler_specs.get("min_lr", 0.0)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
