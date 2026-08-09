import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dataloader.modulation_loader import (
    ModulationLoader,
    SurfacePointLoader,
    compute_latent_statistics,
    ensure_modulation_cache,
)
from models.sdf_model import SdfModel


def tiny_stage1_specs(data_source):
    return {
        "DataSource": str(data_source),
        "SurfacePointCount": 8,
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


class ModulationCacheTests(unittest.TestCase):
    def test_surface_cache_sampling_can_be_reproduced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "abo" / "ABO" / "item" / "cod_sdf.npz"
            path.parent.mkdir(parents=True)
            np.savez(
                path,
                surface_points=np.arange(90, dtype=np.float32).reshape(30, 3),
            )
            records = [{
                "dataset": "abo",
                "class_name": "ABO",
                "instance_name": "item",
                "variant_index": 0,
                "latent_path": str(root / "cache" / "modulation.npz"),
            }]
            first = SurfacePointLoader(
                root,
                records,
                surface_point_count=12,
                deterministic_sampling=True,
                sampling_seed=31,
            )[0]["surface_points"]
            second = SurfacePointLoader(
                root,
                records,
                surface_point_count=12,
                deterministic_sampling=True,
                sampling_seed=31,
            )[0]["surface_points"]

        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_caches_union_of_splits_and_reuses_complete_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_source = root / "data"
            for index, object_id in enumerate(("train", "val", "test")):
                path = data_source / "abo" / "ABO" / object_id / "cod_sdf.npz"
                path.parent.mkdir(parents=True)
                rng = np.random.default_rng(index)
                np.savez(
                    path,
                    surface_points=rng.normal(size=(12, 3)).astype(np.float32),
                )

            split_paths = {}
            for name, object_ids in {
                "TrainSplit": ["train"],
                "ValSplit": ["val"],
                "TestSplit": ["val", "test"],
            }.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps({"abo": {"ABO": object_ids}}))
                split_paths[name] = str(path)

            stage1_specs = tiny_stage1_specs(data_source)
            stage1 = SdfModel(stage1_specs)
            checkpoint_path = root / "stage1.ckpt"
            torch.save(
                {
                    "state_dict": {
                        f"sdf_model.{name}": value
                        for name, value in stage1.state_dict().items()
                    },
                    "hyper_parameters": {"specs": stage1_specs},
                },
                checkpoint_path,
            )
            del stage1

            exp_dir = root / "stage2"
            specs = {
                **split_paths,
                "modulation_ckpt_path": str(checkpoint_path),
                "modulation_variants": 3,
                "sample_posterior_latents": True,
            }
            cache_path, stats_path = ensure_modulation_cache(
                specs, exp_dir, batch_size=2, workers=0, device=torch.device("cpu")
            )

            cache_path = Path(cache_path)
            self.assertEqual(cache_path, exp_dir / "modulations")
            for object_id in ("train", "val", "test"):
                self.assertTrue(
                    (cache_path / "ABO" / object_id / "modulation.npz").is_file()
                )
            for variant_index in (1, 2):
                self.assertTrue(
                    (
                        cache_path
                        / "ABO"
                        / "train"
                        / f"modulation_{variant_index:03d}.npz"
                    ).is_file()
                )
                self.assertFalse(
                    (
                        cache_path
                        / "ABO"
                        / "val"
                        / f"modulation_{variant_index:03d}.npz"
                    ).exists()
                )

            train_records = ModulationLoader.build_records(
                cache_path,
                {"abo": {"ABO": ["train"]}},
                modulation_variants=3,
            )
            self.assertEqual(len(train_records), 3)
            expected_mean, expected_std = compute_latent_statistics(
                train_records, include_posterior_variance=True
            )
            with np.load(stats_path) as statistics:
                np.testing.assert_allclose(statistics["mean"], expected_mean)
                np.testing.assert_allclose(statistics["std"], expected_std)

            dataset = ModulationLoader(
                cache_path,
                records=train_records,
                latent_stats_path=stats_path,
                sample_posterior=True,
            )
            torch.manual_seed(0)
            first = dataset[0]["latent"]
            torch.manual_seed(1)
            second = dataset[0]["latent"]
            self.assertFalse(torch.equal(first, second))

            # A continuation experiment can reuse this exact cache instead of
            # silently creating a new set of randomly sampled modulations.
            continuation_specs = {
                **specs,
                "modulation_cache_path": str(cache_path),
            }
            continuation_cache, continuation_stats = ensure_modulation_cache(
                continuation_specs,
                root / "stage2_continuation",
                batch_size=2,
                workers=0,
                device=torch.device("cpu"),
            )
            self.assertEqual(Path(continuation_cache), cache_path)
            self.assertEqual(Path(continuation_stats), Path(stats_path))
            self.assertFalse((root / "stage2_continuation" / "modulations").exists())

            # A complete cache must not reload stage one on subsequent starts.
            checkpoint_path.unlink()
            ensure_modulation_cache(
                specs, exp_dir, batch_size=2, workers=0, device=torch.device("cpu")
            )

    def test_quality_filter_selects_variants_drops_objects_and_reuses_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_source = root / "data"
            for index, object_id in enumerate(("partial", "rejected", "validation")):
                path = data_source / "abo" / "ABO" / object_id / "cod_sdf.npz"
                path.parent.mkdir(parents=True)
                rng = np.random.default_rng(index)
                np.savez(
                    path,
                    surface_points=rng.normal(size=(12, 3)).astype(np.float32),
                )

            train_split = {"abo": {"ABO": ["partial", "rejected"]}}
            val_split = {"abo": {"ABO": ["validation"]}}
            train_path = root / "train.json"
            val_path = root / "val.json"
            train_path.write_text(json.dumps(train_split))
            val_path.write_text(json.dumps(val_split))

            stage1_specs = tiny_stage1_specs(data_source)
            stage1 = SdfModel(stage1_specs)
            checkpoint_path = root / "stage1.ckpt"
            torch.save(
                {
                    "state_dict": {
                        f"sdf_model.{name}": value
                        for name, value in stage1.state_dict().items()
                    },
                    "hyper_parameters": {"specs": stage1_specs},
                },
                checkpoint_path,
            )
            del stage1

            specs = {
                "TrainSplit": str(train_path),
                "ValSplit": str(val_path),
                "modulation_ckpt_path": str(checkpoint_path),
                "modulation_variants": 2,
                "modulation_filter_threshold": 0.005,
            }

            def fake_score(_model, record, *_args):
                if record["instance_name"] == "partial" and record["variant_index"] == 0:
                    return {"chamfer_distance": 0.005}
                if record["instance_name"] == "rejected" and record["variant_index"] == 0:
                    return {"chamfer_distance": None}
                return {"chamfer_distance": 0.006}

            with mock.patch(
                "dataloader.modulation_loader._score_modulation_reconstruction",
                side_effect=fake_score,
            ) as scorer:
                cache_path, stats_path = ensure_modulation_cache(
                    specs,
                    root / "stage2",
                    batch_size=2,
                    workers=0,
                    device=torch.device("cpu"),
                )
            self.assertEqual(scorer.call_count, 4)

            train_records = ModulationLoader.build_records(
                cache_path,
                train_split,
                modulation_variants=2,
                quality_threshold=0.005,
            )
            self.assertEqual(
                [(record["instance_name"], record["variant_index"]) for record in train_records],
                [("partial", 0)],
            )
            self.assertTrue(
                (Path(cache_path) / "ABO" / "rejected" / "modulation_001.npz").is_file()
            )

            val_records = ModulationLoader.build_records(cache_path, val_split)
            self.assertEqual(len(val_records), 1)
            expected_mean, expected_std = compute_latent_statistics(train_records)
            with np.load(stats_path) as statistics:
                np.testing.assert_allclose(statistics["mean"], expected_mean)
                np.testing.assert_allclose(statistics["std"], expected_std)

            manifest = json.loads(
                (Path(cache_path) / "latent_quality.json").read_text()
            )
            self.assertEqual(len(manifest["scores"]), 4)

            checkpoint_path.unlink()
            with mock.patch(
                "dataloader.modulation_loader._score_modulation_reconstruction"
            ) as scorer:
                ensure_modulation_cache(
                    specs,
                    root / "stage2",
                    batch_size=2,
                    workers=0,
                    device=torch.device("cpu"),
                )
            scorer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
