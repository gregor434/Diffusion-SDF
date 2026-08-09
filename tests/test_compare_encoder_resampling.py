import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.compare_encoder_resampling import (
    checkpoint_metrics,
    compare_metrics,
    evaluate_checkpoint,
    flatten_split,
    sample_object,
)
from models.sdf_model import SdfModel


class EncoderResamplingComparisonTests(unittest.TestCase):
    def test_tiny_lightning_checkpoint_runs_end_to_end(self):
        specs = {
            "CODVaeSpecs": {
                "latent_tokens": 2,
                "latent_dimension": 3,
                "embed_dimension": 16,
                "triplane_dimension": 4,
                "latent_decoder_layers": 1,
                "num_heads": 4,
                "dropout": 0.0,
                "use_learnable_positions": True,
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
            "NearSurfaceRatio": 0.5,
        }
        model = SdfModel(specs)
        rng = np.random.RandomState(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "dataset" / "class" / "object" / "cod_sdf.npz"
            data_path.parent.mkdir(parents=True)
            np.savez(
                data_path,
                surface_points=rng.randn(20, 3).astype(np.float32),
                near_surface_query_points=rng.randn(20, 3).astype(np.float32),
                near_surface_sdf=rng.randn(20).astype(np.float32),
                uniform_query_points=rng.randn(20, 3).astype(np.float32),
                uniform_sdf=rng.randn(20).astype(np.float32),
            )
            checkpoint_path = root / "model.ckpt"
            torch.save(
                {
                    "state_dict": {
                        f"sdf_model.{name}": value.clone()
                        for name, value in model.state_dict().items()
                    },
                    "hyper_parameters": {"specs": specs},
                    "epoch": 3,
                    "global_step": 7,
                },
                checkpoint_path,
            )
            records = flatten_split(
                {"dataset": {"class": ["object"]}}, root
            )
            args = SimpleNamespace(
                seed=11,
                surface_samples=2,
                surface_points=8,
                query_points=6,
                variant_batch_size=1,
            )
            result = evaluate_checkpoint(
                checkpoint_path,
                records,
                args,
                torch.device("cpu"),
                near_surface_ratio=0.5,
            )

        self.assertEqual(result["latents"].shape, (1, 2, 2, 3))
        self.assertEqual(result["predictions"].shape, (1, 2, 6))
        self.assertEqual(result["targets"].shape, (1, 6))
        self.assertEqual(result["metadata"]["epoch_index"], 3)

    def test_object_sampling_is_reproducible_and_varies_only_surface_variant(self):
        rng = np.random.RandomState(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cod_sdf.npz"
            np.savez(
                path,
                surface_points=rng.randn(40, 3).astype(np.float32),
                near_surface_query_points=rng.randn(30, 3).astype(np.float32),
                near_surface_sdf=rng.randn(30).astype(np.float32),
                uniform_query_points=rng.randn(30, 3).astype(np.float32),
                uniform_sdf=rng.randn(30).astype(np.float32),
            )
            args = SimpleNamespace(
                seed=17,
                surface_samples=3,
                surface_points=8,
                query_points=10,
            )
            first = sample_object(path, 2, args, 0.6)
            second = sample_object(path, 2, args, 0.6)

        for expected, actual in zip(first, second):
            np.testing.assert_array_equal(expected, actual)
        self.assertEqual(first[0].shape, (3, 8, 3))
        self.assertEqual(first[1].shape, (10, 3))
        self.assertEqual(first[2].shape, (10,))
        self.assertFalse(np.array_equal(first[0][0], first[0][1]))

    def test_metrics_identify_a_more_stable_candidate(self):
        targets = np.zeros((2, 4), dtype=np.float32)
        baseline_latents = np.array(
            [
                [[[0.0, 0.0], [1.0, 1.0]], [[0.5, 0.0], [1.5, 1.0]]],
                [[[2.0, 2.0], [3.0, 3.0]], [[2.5, 2.0], [3.5, 3.0]]],
            ],
            dtype=np.float32,
        )
        candidate_latents = baseline_latents.copy()
        candidate_latents[:, 1] = (
            candidate_latents[:, 0]
            + 0.1 * (candidate_latents[:, 1] - candidate_latents[:, 0])
        )
        baseline_predictions = np.array(
            [
                [[-1.0, -0.5, 0.5, 1.0], [1.0, -0.25, 0.25, 1.0]],
                [[-1.0, -0.5, 0.5, 1.0], [1.0, -0.25, 0.25, 1.0]],
            ],
            dtype=np.float32,
        )
        candidate_predictions = baseline_predictions.copy()
        candidate_predictions[:, 1] = (
            candidate_predictions[:, 0]
            + 0.1 * (
                candidate_predictions[:, 1] - candidate_predictions[:, 0]
            )
        )

        baseline = checkpoint_metrics(
            {
                "latents": baseline_latents,
                "predictions": baseline_predictions,
                "targets": targets,
            }
        )
        candidate = checkpoint_metrics(
            {
                "latents": candidate_latents,
                "predictions": candidate_predictions,
                "targets": targets,
            }
        )
        comparison = compare_metrics(baseline, candidate)

        self.assertTrue(
            comparison["latent_aligned_mse"]["candidate_improves_median"]
        )
        self.assertTrue(
            comparison["sdf_pairwise_mae"]["candidate_improves_median"]
        )
        self.assertTrue(
            comparison["sdf_sign_flip_fraction"]["candidate_improves_median"]
        )
        self.assertTrue(comparison["candidate_improves_all_stability_medians"])


if __name__ == "__main__":
    unittest.main()
