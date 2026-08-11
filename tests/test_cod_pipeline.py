import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from dataloader.sdf_loader import SdfLoader
from models.combined_model import (
    STAGE1_COMPONENTS,
    CombinedModel,
    validate_training_specs,
)
from models.cod_vae.checkpoint import load_cod_checkpoint
from models.diffusion import CODLatentTransformer, latent_set_distance_matrix
from models.sdf_model import SdfModel


def tiny_specs():
    return {
        "CODVaeSpecs": {
            "latent_tokens": 2,
            "latent_dimension": 3,
            "embed_dimension": 16,
            "triplane_dimension": 4,
            "latent_decoder_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
            "encoder_params": {
                "num_patches": 4,
                "num_blocks": 1,
                "num_layers_per_block": 1,
                "num_heads": 4,
                "dropout": 0.0,
            },
            "decoder_params": {
                "output_resolution": 8,
                "output_patch_size": 4,
                "num_layers": 1,
                "num_init_layers": 1,
                "num_heads": 4,
                "keep_ratio": 1.0,
                "num_merged_tokens": -1,
                "dropout": 0.0,
            },
        },
        "SdfModelSpecs": {"hidden_dim": 16, "feature_dim": 4},
    }


class CODPipelineTests(unittest.TestCase):
    def test_latent_set_distances_are_permutation_invariant(self):
        first = torch.randn(3, 5, 4)
        second = torch.randn(2, 5, 4)
        projections = torch.randn(12, 4)
        token_permutation = torch.tensor([3, 0, 4, 1, 2])
        for metric in ("sliced_wasserstein", "mmd"):
            with self.subTest(metric=metric):
                expected = latent_set_distance_matrix(
                    first,
                    second,
                    metric=metric,
                    projections=projections,
                )
                actual = latent_set_distance_matrix(
                    first[:, token_permutation],
                    second[:, token_permutation.flip(0)],
                    metric=metric,
                    projections=projections,
                )
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_pairwise_adaptation_uses_frozen_source_and_shared_noise(self):
        specs = {
            "training_task": "diffusion",
            "diffusion_specs": {
                "sigma_data": 1.0,
                "P_mean": -1.2,
                "P_std": 1.2,
                "sampling_steps": 2,
            },
            "diffusion_model_specs": {
                "latent_tokens": 3,
                "latent_dimension": 4,
                "width": 16,
                "depth": 1,
                "heads": 4,
                "dropout": 0.0,
            },
            "lambda_pairwise": 2.0,
            "pairwise_distance": "sliced_wasserstein",
            "pairwise_num_projections": 8,
        }
        source = CombinedModel({**specs, "lambda_pairwise": 0.0})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.ckpt"
            torch.save({"state_dict": source.state_dict()}, path)
            adapted = CombinedModel({
                **specs,
                "source_diffusion_checkpoint": str(path),
            })

        self.assertFalse(any(
            parameter.requires_grad
            for parameter in adapted.source_diffusion_model.parameters()
        ))
        self.assertFalse(any(
            key.startswith("source_diffusion_model")
            for key in adapted.state_dict()
        ))
        clean = torch.randn(4, 3, 4)
        noise = torch.randn_like(clean)
        sigma = torch.rand(4) + 0.1
        initial = adapted.stage2_losses(
            {"latent": clean}, noise=noise, sigma=sigma
        )
        self.assertAlmostEqual(initial["pairwise"].item(), 0.0, places=6)
        initial["loss"].backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in adapted.diffusion_model.parameters()
        ))
        adapted.zero_grad(set_to_none=True)
        with torch.no_grad():
            weight = adapted.diffusion_model.model.output_projection.weight
            weight.copy_(
                torch.linspace(-0.5, 0.5, weight.numel()).reshape_as(weight)
            )
        changed = adapted.stage2_losses(
            {"latent": clean}, noise=noise, sigma=sigma
        )
        self.assertGreater(changed["pairwise"].item(), 0.0)
        torch.testing.assert_close(
            changed["loss"],
            changed["diffusion"] + 2.0 * changed["pairwise"],
        )

    def test_diffusion_slot_embeddings_break_token_permutation_equivariance(self):
        plain = CODLatentTransformer(
            latent_tokens=3,
            latent_dimension=4,
            width=16,
            depth=1,
            heads=4,
        ).eval()
        slot_aware = CODLatentTransformer(
            latent_tokens=3,
            latent_dimension=4,
            width=16,
            depth=1,
            heads=4,
            use_learnable_slot_embeddings=True,
        ).eval()
        nn.init.normal_(plain.output_projection.weight)
        nn.init.normal_(slot_aware.output_projection.weight)
        latent = torch.randn(2, 3, 4)
        noise = torch.randn(2)
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.argsort(permutation)

        plain_permuted = plain(latent[:, permutation], noise)[:, inverse]
        torch.testing.assert_close(plain(latent, noise), plain_permuted)
        slot_permuted = slot_aware(latent[:, permutation], noise)[:, inverse]
        self.assertFalse(torch.allclose(slot_aware(latent, noise), slot_permuted))
        self.assertEqual(slot_aware.slot_embedding.shape, (1, 3, 16))

    def test_all_repository_configs_are_semantically_consistent(self):
        repository = Path(__file__).resolve().parents[1]
        for path in (repository / "config").glob("**/specs.json"):
            with self.subTest(path=path):
                validate_training_specs(json.loads(path.read_text()))

    def test_cod_sdf_forward_preserves_native_shapes(self):
        model = SdfModel(tiny_specs()).eval()
        output = model(
            torch.rand(2, 8, 3) * 1.8 - 0.9,
            torch.rand(2, 5, 3) * 1.8 - 0.9,
            sample_posterior=False,
        )
        self.assertEqual(output["latent"].shape, torch.Size([2, 2, 3]))
        self.assertEqual(output["planes"].shape, torch.Size([2, 3, 4, 8, 8]))
        self.assertEqual(output["sdf"].shape, torch.Size([2, 5]))
        self.assertEqual(output["posterior"].mean.shape, torch.Size([2, 2, 3]))

    def test_learned_queries_replace_only_compact_fps(self):
        specs = tiny_specs()
        specs["CODVaeSpecs"]["use_learnable_positions"] = True
        model = SdfModel(specs)
        points = torch.rand(2, 8, 3) * 1.8 - 0.9
        from models.cod_vae import pointops

        with mock.patch.object(pointops, "fps", wraps=pointops.fps) as fps:
            encoded = model.cod_vae.encode_embed(points)

        self.assertEqual(fps.call_count, 1)
        self.assertEqual(fps.call_args.args[1], 4)
        self.assertEqual(encoded.shape, torch.Size([2, 2, 16]))
        encoded.sum().backward()
        self.assertIsNotNone(model.cod_vae.autoencoder.latent_pos.grad)

    def test_learned_query_adaptation_trains_only_full_interaction_path(self):
        source = SdfModel(tiny_specs())
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "cod.pt"
            torch.save(
                {
                    "state_dict": {
                        f"model.{name}": value.clone()
                        for name, value in source.cod_vae.state_dict().items()
                    }
                },
                checkpoint_path,
            )
            specs = tiny_specs()
            specs["CODVaeSpecs"].update(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "use_learnable_positions": True,
                    "checkpoint_allowed_missing_prefixes": (
                        "autoencoder.latent_pos",
                    ),
                }
            )
            specs.update(
                {
                    "training_task": "modulation",
                    "stage1_mode": "learned_query_adaptation",
                    "learning_rates": {
                        "compact_queries": 1e-3,
                        "compact_token_attention": 1e-5,
                    },
                }
            )
            model = CombinedModel(specs).train()

        groups = model.sdf_model.component_parameters()
        trainable = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        expected = {
            id(parameter)
            for name in ("compact_queries", "compact_token_attention")
            for parameter in groups[name]
        }
        self.assertEqual(trainable, expected)
        for name in (
            "point_encoder_backbone", "variational_block", "latent_decoder",
            "triplane_decoder", "sdf_network",
        ):
            self.assertTrue(all(not parameter.requires_grad for parameter in groups[name]))

        optimizer = model.configure_optimizers()
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["compact_queries", "compact_token_attention"],
        )

    def test_learned_query_modes_require_learned_positions(self):
        specs = tiny_specs()
        specs.update(
            {
                "training_task": "modulation",
                "stage1_mode": "learned_query_adaptation",
            }
        )
        with self.assertRaisesRegex(ValueError, "use_learnable_positions"):
            validate_training_specs(specs)

    def test_all_learned_query_stages_select_exact_parameter_unions(self):
        specs = tiny_specs()
        specs["CODVaeSpecs"]["use_learnable_positions"] = True
        model = SdfModel(specs)
        groups = model.component_parameters()

        for mode in (
            "learned_query_adaptation",
            "learned_query_encoder_refinement",
            "learned_query_vae_finetune",
        ):
            with self.subTest(mode=mode):
                selected = STAGE1_COMPONENTS[mode]
                model.set_trainable_components(selected)
                expected = {
                    id(parameter)
                    for name in selected
                    for parameter in groups[name]
                }
                actual = {
                    id(parameter)
                    for parameter in model.parameters()
                    if parameter.requires_grad
                }
                self.assertEqual(actual, expected)
                sdf_trainable = mode == "learned_query_vae_finetune"
                self.assertEqual(
                    any(
                        parameter.requires_grad
                        for parameter in groups["sdf_network"]
                    ),
                    sdf_trainable,
                )

    def test_sdf_head_conv_refine_mode_freezes_the_rest_of_cod(self):
        specs = tiny_specs()
        specs["CODVaeSpecs"]["decoder_params"]["use_conv_refine"] = True
        model = SdfModel(specs)
        groups = model.component_parameters()

        selected = STAGE1_COMPONENTS["sdf_head_conv_refine"]
        model.set_trainable_components(selected)

        expected = {
            id(parameter)
            for name in selected
            for parameter in groups[name]
        }
        actual = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(actual, expected)
        self.assertTrue(expected)
        for name in (
            "point_encoder", "variational_block", "latent_decoder"
        ):
            self.assertTrue(
                all(not parameter.requires_grad for parameter in groups[name]),
                name,
            )
        conv_ids = {id(parameter) for parameter in groups["conv_refine"]}
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in groups["triplane_decoder"]
                if id(parameter) not in conv_ids
            )
        )

    def test_stage_two_validation_noise_and_cosine_schedule(self):
        specs = {
            "training_task": "diffusion",
            "diffusion_specs": {
                "sigma_data": 1.0,
                "P_mean": -1.2,
                "P_std": 1.2,
                "sampling_steps": 2,
            },
            "diffusion_model_specs": {
                "latent_tokens": 2,
                "latent_dimension": 3,
                "width": 16,
                "depth": 1,
                "heads": 4,
                "cond": False,
            },
            "learning_rates": {"diffusion": 2e-5},
            "lr_scheduler": {
                "type": "cosine",
                "min_lr": 1e-6,
                "epochs": 10,
            },
            "validation_noise_seed": 17,
            "num_epochs": 10,
        }
        model = CombinedModel(specs)
        clean = torch.randn(3, 2, 3)
        first_noise, first_sigma = model.deterministic_validation_noise(clean, 2)
        second_noise, second_sigma = model.deterministic_validation_noise(clean, 2)
        other_noise, other_sigma = model.deterministic_validation_noise(clean, 3)
        torch.testing.assert_close(first_noise, second_noise, rtol=0, atol=0)
        torch.testing.assert_close(first_sigma, second_sigma, rtol=0, atol=0)
        self.assertFalse(torch.equal(first_noise, other_noise))
        self.assertFalse(torch.equal(first_sigma, other_sigma))

        configured = model.configure_optimizers()
        optimizer = configured["optimizer"]
        scheduler = configured["lr_scheduler"]["scheduler"]
        self.assertEqual(configured["lr_scheduler"]["interval"], "epoch")
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-5)
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)

    def test_slot_embeddings_have_a_separate_learning_rate(self):
        specs = {
            "training_task": "diffusion",
            "diffusion_specs": {"sampling_steps": 2},
            "diffusion_model_specs": {
                "latent_tokens": 2,
                "latent_dimension": 3,
                "width": 16,
                "depth": 1,
                "heads": 4,
                "use_learnable_slot_embeddings": True,
            },
            "learning_rates": {
                "diffusion": 1e-5,
                "diffusion_slot_embeddings": 1e-4,
            },
        }
        model = CombinedModel(specs)
        optimizer = model.configure_optimizers()
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["diffusion_slot_embeddings", "diffusion"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [1e-4, 1e-5],
        )
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 1)
        self.assertIs(
            optimizer.param_groups[0]["params"][0],
            model.diffusion_model.model.slot_embedding,
        )

    def test_stage_three_reconstruction_refines_only_diffusion(self):
        specs = tiny_specs()
        specs.update({
            "training_task": "combined",
            "stage3_mode": "diffusion_only",
            "sample_posterior": False,
            "validation_sample_posterior": False,
            "validation_noise_seed": 11,
            "diffusion_specs": {
                "sigma_data": 1.0,
                "P_mean": -1.2,
                "P_std": 1.2,
                "sampling_steps": 2,
            },
            "diffusion_model_specs": {
                "latent_tokens": 2,
                "latent_dimension": 3,
                "width": 16,
                "depth": 1,
                "heads": 4,
                "dropout": 0.0,
                "cond": True,
                "condition_dim": 8,
                "condition_encoders": [
                    {"type": "image", "clip_feature_dim": 12}
                ],
            },
            "loss_weights": {
                "direct": 0.0,
                "diffusion": 0.1,
                "generated": 1.0,
                "kl": 0.0,
            },
            "learning_rates": {"diffusion": 5e-6},
        })
        model = CombinedModel(specs).train()
        batch = {
            "surface_points": torch.rand(2, 8, 3) * 1.8 - 0.9,
            "query_points": torch.rand(2, 6, 3) * 1.8 - 0.9,
            "query_sdf": torch.randn(2, 6) * 0.05,
            "conditioning": {"image": torch.randn(2, 1, 12)},
        }
        losses = model.stage3_losses(batch)
        losses["loss"].backward()

        self.assertEqual(losses["sdf_direct"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["sdf_denoised"]))
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.diffusion_model.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.sdf_model.parameters()
            )
        )

    def test_official_solver_checkpoint_prefix_loads_strictly(self):
        specs = tiny_specs()
        specs["CODVaeSpecs"]["decoder_params"]["num_merged_tokens"] = 2
        original = SdfModel(specs).cod_vae
        checkpoint = {
            "state_dict": {
                f"model.{name}": value.clone()
                for name, value in original.state_dict().items()
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.pt"
            torch.save(checkpoint, path)
            restored = SdfModel(specs).cod_vae
            result = load_cod_checkpoint(restored, path, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for expected, actual in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_conv_refiner_starts_as_an_exact_identity_and_loads_explicitly(self):
        source_specs = tiny_specs()
        source = SdfModel(source_specs).cod_vae
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.pt"
            torch.save(
                {
                    "state_dict": {
                        f"model.{name}": value.clone()
                        for name, value in source.state_dict().items()
                    }
                },
                path,
            )
            target_specs = tiny_specs()
            target_specs["CODVaeSpecs"]["decoder_params"]["use_conv_refine"] = True
            target = SdfModel(target_specs).cod_vae
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_cod_checkpoint(target, path, strict=True)
            result = load_cod_checkpoint(
                target,
                path,
                strict=True,
                allowed_missing_prefixes=(
                    "autoencoder.decoder.conv_refine.",
                ),
            )

        self.assertTrue(result.missing_keys)
        self.assertTrue(
            all(
                key.startswith("autoencoder.decoder.conv_refine.")
                for key in result.missing_keys
            )
        )
        decoder = target.autoencoder.decoder
        plane = torch.randn(2, decoder.query_dim, 8, 8)
        torch.testing.assert_close(
            decoder.conv_refine(plane),
            torch.zeros_like(plane),
            rtol=0,
            atol=0,
        )

    def test_disabled_mode_bypasses_uncertainty_pruning_and_merging(self):
        specs = tiny_specs()
        specs["CODVaeSpecs"]["uncertainty_mode"] = "disabled"
        specs["CODVaeSpecs"]["decoder_params"]["num_merged_tokens"] = 2
        model = SdfModel(specs).eval()
        decoder = model.cod_vae.autoencoder.decoder
        self.assertEqual(decoder.uncertainty_mode, "disabled")

        def fail_if_called(*args, **kwargs):
            raise AssertionError("disabled uncertainty decoding invoked a bypassed module")

        decoder.uncertainty_out.forward = fail_if_called
        decoder._select_by_uncertainty = fail_if_called
        decoder.merging_module.forward = fail_if_called
        projected = {}
        hooks = [
            decoder.init_out.register_forward_hook(
                lambda module, inputs, output: projected.update(initial=output)
            ),
            decoder.decoder_out.register_forward_hook(
                lambda module, inputs, output: projected.update(residual=output)
            ),
        ]
        with torch.no_grad():
            decoded = model.decode_latent(torch.randn(1, 2, 3))
        for hook in hooks:
            hook.remove()
        self.assertTrue(torch.isfinite(decoded["planes"]).all())
        self.assertEqual(decoded["planes"].shape, torch.Size([1, 3, 4, 8, 8]))
        self.assertIsNone(decoded["uncertainty"])
        expected = decoder.patches_to_planes(
            projected["initial"] + projected["residual"]
        )
        torch.testing.assert_close(decoded["planes"], expected)
        torch.testing.assert_close(
            decoded["initial_planes"],
            decoder.patches_to_planes(projected["initial"]),
        )

    def test_decoder_finetune_losses_preserve_encoder_and_posterior(self):
        source_specs = tiny_specs()
        source_specs["CODVaeSpecs"]["decoder_params"]["num_merged_tokens"] = 2
        source = SdfModel(source_specs)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "cod.pt"
            torch.save({
                "state_dict": {
                    f"model.{name}": value.clone()
                    for name, value in source.cod_vae.state_dict().items()
                }
            }, checkpoint_path)
            specs = tiny_specs()
            specs.update({
                "training_task": "modulation",
                "stage1_mode": "cod_decoder_finetune",
                "sample_posterior": True,
                "sdf_loss": {"type": "truncated_sdf", "truncation": 0.1},
                "loss_weights": {
                    "sdf": 1.0,
                    "sdf_initial": 0.0,
                    "cod_aux": 1.0,
                    "uncertainty": 0.0,
                    "surface_zero": 0.0,
                    "eikonal": 0.0,
                    "normal": 0.0,
                    "kl": 0.0,
                },
                "learning_rates": {
                    "latent_decoder": 1e-5,
                    "triplane_decoder": 1e-5,
                    "sdf_network": 1e-4,
                },
            })
            specs["CODVaeSpecs"]["uncertainty_mode"] = "disabled"
            specs["CODVaeSpecs"]["decoder_params"]["num_merged_tokens"] = 2
            specs["CODVaeSpecs"]["checkpoint_path"] = str(checkpoint_path)
            model = CombinedModel(specs).train()

            decoder = model.sdf_model.cod_vae.autoencoder.decoder
            checkpoint_only_modules = [decoder.uncertainty_out, decoder.merging_module]
            checkpoint_only_before = {
                id(parameter): parameter.detach().clone()
                for module in checkpoint_only_modules
                for parameter in module.parameters()
            }

            def fail_if_called(*args, **kwargs):
                raise AssertionError("a zero-weight auxiliary path was evaluated")

            model.sdf_model.query_uncertainty = fail_if_called
            model.stage1_geometry_losses = fail_if_called
            query_sdf = model.sdf_model.query_sdf
            query_count = 0

            def counted_query(*args, **kwargs):
                nonlocal query_count
                query_count += 1
                return query_sdf(*args, **kwargs)

            model.sdf_model.query_sdf = counted_query

            frozen_before = {
                name: value.detach().clone()
                for component in ("point_encoder", "variational_block")
                for name, value in model.sdf_model.component_modules()[component].named_parameters(
                    prefix=component
                )
            }
            surface = torch.rand(2, 8, 3) * 1.8 - 0.9
            normals = torch.nn.functional.normalize(torch.randn_like(surface), dim=-1)
            batch = {
                "surface_points": surface,
                "surface_normals": normals,
                "query_points": torch.rand(2, 6, 3) * 1.8 - 0.9,
                "query_sdf": torch.randn(2, 6) * 0.05,
            }
            losses = model.stage1_losses(batch)
            self.assertEqual(query_count, 1)
            expected = {
                "sdf", "sdf_initial", "cod_aux", "uncertainty",
                "surface_zero", "eikonal", "normal", "kl",
            }
            self.assertTrue(expected.issubset(losses))
            self.assertTrue(all(torch.isfinite(losses[name]) for name in expected))
            losses["loss"].backward()
            optimizer = model.configure_optimizers()
            optimized = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            self.assertTrue(
                all(
                    id(parameter) not in optimized and not parameter.requires_grad
                    for module in checkpoint_only_modules
                    for parameter in module.parameters()
                )
            )
            optimizer.step()

            for component in ("point_encoder", "variational_block"):
                module = model.sdf_model.component_modules()[component]
                self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
                for name, value in module.named_parameters(prefix=component):
                    torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)
            for component in ("latent_decoder", "triplane_decoder", "sdf_network"):
                module = model.sdf_model.component_modules()[component]
                self.assertTrue(any(parameter.grad is not None for parameter in module.parameters()))
            for module in checkpoint_only_modules:
                for parameter in module.parameters():
                    torch.testing.assert_close(
                        parameter, checkpoint_only_before[id(parameter)], rtol=0, atol=0
                    )

    def test_triplane_sdf_finetune_freezes_the_complete_latent_path(self):
        source_specs = tiny_specs()
        source_specs["CODVaeSpecs"]["decoder_params"]["use_conv_refine"] = True
        source = SdfModel(source_specs)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "cod.pt"
            torch.save(
                {
                    "state_dict": {
                        f"model.{name}": value.clone()
                        for name, value in source.cod_vae.state_dict().items()
                    }
                },
                checkpoint_path,
            )
            specs = tiny_specs()
            specs["CODVaeSpecs"]["decoder_params"]["use_conv_refine"] = True
            specs.update(
                {
                    "training_task": "modulation",
                    "stage1_mode": "triplane_sdf_finetune",
                    "sample_posterior": False,
                    "loss_weights": {"sdf": 1.0},
                    "learning_rates": {
                        "conv_refine": 1e-4,
                        "triplane_decoder": 5e-6,
                        "sdf_network": 2e-5,
                    },
                }
            )
            specs["CODVaeSpecs"]["checkpoint_path"] = str(checkpoint_path)
            model = CombinedModel(specs).train()

        batch = {
            "surface_points": torch.rand(2, 8, 3) * 1.8 - 0.9,
            "query_points": torch.rand(2, 6, 3) * 1.8 - 0.9,
            "query_sdf": torch.randn(2, 6) * 0.05,
        }
        losses = model.stage1_losses(batch)
        losses["loss"].backward()
        components = model.sdf_model.component_modules()
        for name in ("point_encoder", "variational_block", "latent_decoder"):
            self.assertTrue(
                all(
                    not parameter.requires_grad and parameter.grad is None
                    for parameter in components[name].parameters()
                ),
                name,
            )
        for name in ("triplane_decoder", "sdf_network"):
            self.assertTrue(
                any(
                    parameter.requires_grad and parameter.grad is not None
                    for parameter in components[name].parameters()
                ),
                name,
            )
        configured = model.configure_optimizers()
        optimizer = (
            configured["optimizer"]
            if isinstance(configured, dict)
            else configured
        )
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["conv_refine", "triplane_decoder", "sdf_network"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [1e-4, 5e-6, 2e-5],
        )
        conv_parameters = {
            id(parameter)
            for parameter in model.sdf_model.cod_vae.autoencoder.decoder.conv_refine.parameters()
        }
        self.assertEqual(
            {id(parameter) for parameter in optimizer.param_groups[0]["params"]},
            conv_parameters,
        )
        self.assertTrue(
            conv_parameters.isdisjoint(
                id(parameter)
                for parameter in optimizer.param_groups[1]["params"]
            )
        )

    def test_encoder_finetune_updates_only_encoder_and_variational_block(self):
        source_specs = tiny_specs()
        source = SdfModel(source_specs)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "cod.pt"
            torch.save(
                {
                    "state_dict": {
                        f"model.{name}": value.clone()
                        for name, value in source.cod_vae.state_dict().items()
                    }
                },
                checkpoint_path,
            )
            specs = tiny_specs()
            specs.update(
                {
                    "training_task": "modulation",
                    "stage1_mode": "encoder_finetune",
                    "sample_posterior": True,
                    "loss_weights": {"sdf": 1.0, "kl": 1e-4},
                    "learning_rates": {
                        "point_encoder": 2e-6,
                        "variational_block": 1e-5,
                    },
                }
            )
            specs["CODVaeSpecs"]["checkpoint_path"] = str(checkpoint_path)
            model = CombinedModel(specs).train()

        batch = {
            "surface_points": torch.rand(2, 8, 3) * 1.8 - 0.9,
            "query_points": torch.rand(2, 6, 3) * 1.8 - 0.9,
            "query_sdf": torch.randn(2, 6) * 0.05,
        }
        losses = model.stage1_losses(batch)
        losses["loss"].backward()
        components = model.sdf_model.component_modules()
        for name in ("point_encoder", "variational_block"):
            self.assertTrue(
                any(
                    parameter.requires_grad and parameter.grad is not None
                    for parameter in components[name].parameters()
                ),
                name,
            )
        for name in ("latent_decoder", "triplane_decoder", "sdf_network"):
            self.assertTrue(
                all(
                    not parameter.requires_grad and parameter.grad is None
                    for parameter in components[name].parameters()
                ),
                name,
            )
        configured = model.configure_optimizers()
        optimizer = (
            configured["optimizer"]
            if isinstance(configured, dict)
            else configured
        )
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["point_encoder", "variational_block"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [2e-6, 1e-5],
        )

    def test_joint_refinement_freezes_only_the_point_encoder_backbone(self):
        source_specs = tiny_specs()
        source_specs["CODVaeSpecs"]["decoder_params"]["use_conv_refine"] = True
        source = SdfModel(source_specs)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "cod.pt"
            torch.save(
                {
                    "state_dict": {
                        f"model.{name}": value.clone()
                        for name, value in source.cod_vae.state_dict().items()
                    }
                },
                checkpoint_path,
            )
            specs = tiny_specs()
            specs["CODVaeSpecs"]["decoder_params"]["use_conv_refine"] = True
            specs["CODVaeSpecs"]["checkpoint_path"] = str(checkpoint_path)
            specs.update(
                {
                    "training_task": "modulation",
                    "stage1_mode": "joint_refinement",
                    "learning_rates": {
                        "variational_block": 1e-5,
                        "latent_decoder": 1e-5,
                        "triplane_decoder": 1e-5,
                        "conv_refine": 2e-4,
                        "sdf_network": 1e-4,
                    },
                }
            )
            model = CombinedModel(specs)

        components = model.sdf_model.component_modules()
        self.assertTrue(
            all(not parameter.requires_grad for parameter in components["point_encoder"].parameters())
        )
        for name in (
            "variational_block", "latent_decoder", "triplane_decoder", "sdf_network",
        ):
            self.assertTrue(
                any(parameter.requires_grad for parameter in components[name].parameters()),
                name,
            )
        configured = model.configure_optimizers()
        optimizer = configured["optimizer"] if isinstance(configured, dict) else configured
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            [
                "conv_refine", "variational_block", "latent_decoder",
                "triplane_decoder", "sdf_network",
            ],
        )

    def test_disabled_uncertainty_rejects_positive_loss_weight(self):
        specs = tiny_specs()
        specs.update({
            "training_task": "modulation",
            "stage1_mode": "train_from_scratch",
            "loss_weights": {"uncertainty": 0.1},
        })
        specs["CODVaeSpecs"]["uncertainty_mode"] = "disabled"
        with self.assertRaisesRegex(ValueError, "zero uncertainty loss weight"):
            validate_training_specs(specs)

    def test_stage_one_geometry_regularization_can_be_validated(self):
        specs = tiny_specs()
        specs.update({
            "training_task": "modulation",
            "stage1_mode": "train_from_scratch",
            "sample_posterior": False,
            "ValidateGeometryRegularization": True,
            "GeometryRegularizationSamples": 4,
            "loss_weights": {
                "sdf": 1.0,
                "surface_zero": 1.0,
                "eikonal": 0.01,
            },
        })
        model = CombinedModel(specs).eval()
        batch = {
            "surface_points": torch.rand(2, 8, 3) * 1.8 - 0.9,
            "query_points": torch.rand(2, 6, 3) * 1.8 - 0.9,
            "query_sdf": torch.randn(2, 6) * 0.05,
        }
        with torch.no_grad():
            losses = model.stage1_losses(batch)
        self.assertGreater(losses["surface_zero"].item(), 0.0)
        self.assertGreater(losses["eikonal"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_latent_consistency_matches_unordered_token_sets(self):
        first = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]]
        )
        permuted = first[:, [2, 0, 1]]
        shifted = first + 1.0

        torch.testing.assert_close(
            CombinedModel.latent_set_consistency_loss(first, permuted),
            torch.zeros(()),
        )
        self.assertGreater(
            CombinedModel.latent_set_consistency_loss(first, shifted).item(),
            0.0,
        )

    def test_latent_consistency_updates_encoder_from_paired_surfaces(self):
        source = SdfModel(tiny_specs())
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "cod.pt"
            torch.save(
                {
                    "state_dict": {
                        f"model.{name}": value.clone()
                        for name, value in source.cod_vae.state_dict().items()
                    }
                },
                checkpoint_path,
            )
            specs = tiny_specs()
            specs.update(
                {
                    "training_task": "modulation",
                    "stage1_mode": "encoder_finetune",
                    "sample_posterior": True,
                    "loss_weights": {
                        "sdf": 0.0,
                        "latent_consistency": 1.0,
                    },
                }
            )
            specs["CODVaeSpecs"]["checkpoint_path"] = str(checkpoint_path)
            model = CombinedModel(specs).train()
        batch = {
            "surface_points": torch.rand(2, 8, 3) * 1.8 - 0.9,
            "paired_surface_points": torch.rand(2, 8, 3) * 1.8 - 0.9,
            "query_points": torch.rand(2, 6, 3) * 1.8 - 0.9,
            "query_sdf": torch.randn(2, 6) * 0.05,
        }

        losses = model.stage1_losses(batch)
        self.assertGreater(losses["latent_consistency"].item(), 0.0)
        losses["loss"].backward()

        components = model.sdf_model.component_modules()
        for name in ("point_encoder", "variational_block"):
            self.assertTrue(
                any(parameter.grad is not None for parameter in components[name].parameters()),
                name,
            )
        for name in ("latent_decoder", "triplane_decoder", "sdf_network"):
            self.assertTrue(
                all(parameter.grad is None for parameter in components[name].parameters()),
                name,
            )

    def test_sdf_loader_separates_cod_surface_and_query_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "abo" / "ABO" / "item" / "cod_sdf.npz"
            path.parent.mkdir(parents=True)
            np.savez(
                path,
                surface_points=np.random.randn(12, 3).astype(np.float32),
                near_surface_query_points=np.random.randn(10, 3).astype(np.float32),
                near_surface_sdf=np.linspace(-1, 1, 10).astype(np.float32),
                uniform_query_points=np.random.randn(8, 3).astype(np.float32),
                uniform_sdf=np.linspace(-1, 1, 8).astype(np.float32),
                normalization_center=np.zeros(3, np.float32),
                normalization_scale=np.asarray(1, np.float32),
            )
            dataset = SdfLoader(
                tmpdir,
                {"abo": {"ABO": ["item"]}},
                samples_per_mesh=10,
                surface_point_count=8,
                near_surface_ratio=0.6,
                deterministic_sampling=True,
                sampling_seed=17,
            )
            item = dataset[0]
            repeated = dataset[0]
        self.assertEqual(item["surface_points"].shape, torch.Size([8, 3]))
        self.assertEqual(item["query_points"].shape, torch.Size([10, 3]))
        self.assertEqual(item["query_sdf"].shape, torch.Size([10]))
        self.assertEqual(item["query_is_near"].sum().item(), 6)
        self.assertEqual(item["object_id"], "item")
        torch.testing.assert_close(
            item["surface_points"], repeated["surface_points"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            item["query_points"], repeated["query_points"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            item["query_sdf"], repeated["query_sdf"], rtol=0, atol=0
        )

    def test_sdf_loader_can_fix_surface_without_fixing_query_supervision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "abo" / "ABO" / "item" / "cod_sdf.npz"
            path.parent.mkdir(parents=True)
            surface = np.arange(90, dtype=np.float32).reshape(30, 3)
            near = np.arange(180, dtype=np.float32).reshape(60, 3)
            uniform = np.arange(180, 360, dtype=np.float32).reshape(60, 3)
            np.savez(
                path,
                surface_points=surface,
                near_surface_query_points=near,
                near_surface_sdf=np.arange(60, dtype=np.float32),
                uniform_query_points=uniform,
                uniform_sdf=np.arange(60, 120, dtype=np.float32),
            )
            dataset = SdfLoader(
                tmpdir,
                {"abo": {"ABO": ["item"]}},
                samples_per_mesh=20,
                surface_point_count=12,
                near_surface_ratio=0.5,
                deterministic_surface_sampling=True,
                sampling_seed=23,
            )
            first = dataset[0]
            second = dataset[0]

        torch.testing.assert_close(
            first["surface_points"],
            second["surface_points"],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(first["query_points"], second["query_points"]))
        self.assertFalse(torch.equal(first["query_sdf"], second["query_sdf"]))

    def test_sdf_loader_returns_reproducible_independent_surface_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "abo" / "ABO" / "item" / "cod_sdf.npz"
            path.parent.mkdir(parents=True)
            surface = np.arange(180, dtype=np.float32).reshape(60, 3)
            np.savez(
                path,
                surface_points=surface,
                near_surface_query_points=surface[:30],
                near_surface_sdf=np.arange(30, dtype=np.float32),
                uniform_query_points=surface[30:],
                uniform_sdf=np.arange(30, 60, dtype=np.float32),
            )
            dataset = SdfLoader(
                tmpdir,
                {"abo": {"ABO": ["item"]}},
                samples_per_mesh=20,
                surface_point_count=12,
                paired_surface_sampling=True,
                deterministic_sampling=True,
                sampling_seed=29,
            )
            first = dataset[0]
            repeated = dataset[0]

        self.assertEqual(first["paired_surface_points"].shape, torch.Size([12, 3]))
        self.assertFalse(
            torch.equal(first["surface_points"], first["paired_surface_points"])
        )
        torch.testing.assert_close(
            first["surface_points"], repeated["surface_points"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            first["paired_surface_points"],
            repeated["paired_surface_points"],
            rtol=0,
            atol=0,
        )

    def test_sdf_loader_supports_stage_three_conditioning_sources(self):
        class StubImageSource:
            name = "image"

            def exists(self, record):
                return record["instance_name"] == "item"

            def load(self, record):
                return torch.ones(1, 12)

            def resolve(self, record):
                return f"{record['instance_name']}.png"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "abo" / "ABO" / "item" / "cod_sdf.npz"
            path.parent.mkdir(parents=True)
            np.savez(
                path,
                surface_points=np.random.randn(12, 3).astype(np.float32),
                near_surface_query_points=np.random.randn(10, 3).astype(np.float32),
                near_surface_sdf=np.linspace(-1, 1, 10).astype(np.float32),
                uniform_query_points=np.random.randn(8, 3).astype(np.float32),
                uniform_sdf=np.linspace(-1, 1, 8).astype(np.float32),
            )
            dataset = SdfLoader(
                tmpdir,
                {"abo": {"ABO": ["item"]}},
                samples_per_mesh=10,
                surface_point_count=8,
                conditioning_sources=[StubImageSource()],
            )
            item = dataset[0]

        self.assertEqual(
            item["conditioning"]["image"].shape,
            torch.Size([1, 12]),
        )
        self.assertEqual(item["conditioning_paths"]["image"], "item.png")


if __name__ == "__main__":
    unittest.main()
