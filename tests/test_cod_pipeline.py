import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch

from dataloader.sdf_loader import SdfLoader
from models.combined_model import CombinedModel, validate_training_specs
from models.cod_vae.checkpoint import load_cod_checkpoint
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
            )
            item = dataset[0]
        self.assertEqual(item["surface_points"].shape, torch.Size([8, 3]))
        self.assertEqual(item["query_points"].shape, torch.Size([10, 3]))
        self.assertEqual(item["query_sdf"].shape, torch.Size([10]))
        self.assertEqual(item["query_is_near"].sum().item(), 6)
        self.assertEqual(item["object_id"], "item")


if __name__ == "__main__":
    unittest.main()
