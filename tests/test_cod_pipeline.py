import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataloader.sdf_loader import SdfLoader
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
        original = SdfModel(tiny_specs()).cod_vae
        checkpoint = {
            "state_dict": {
                f"model.{name}": value.clone()
                for name, value in original.state_dict().items()
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.pt"
            torch.save(checkpoint, path)
            restored = SdfModel(tiny_specs()).cod_vae
            result = load_cod_checkpoint(restored, path, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for expected, actual in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected)

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
