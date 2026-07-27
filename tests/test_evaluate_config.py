import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import evaluate_config


class CODEvaluationTests(unittest.TestCase):
    def _write_configuration(self, root, data_source, task="modulation"):
        split = {"abo": {"ABO": ["item"]}}
        split_path = root / "test_split.json"
        split_path.write_text(json.dumps(split))
        specs = {
            "Description": "test COD configuration",
            "DataSource": str(data_source),
            "TestSplit": str(split_path),
            "training_task": task,
            "CODVaeSpecs": {
                "latent_tokens": 2,
                "latent_dimension": 3,
            },
        }
        (root / "specs.json").write_text(json.dumps(specs))
        return specs

    def test_loads_surface_points_from_cod_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_source = root / "datasets"
            record_dir = data_source / "abo" / "ABO" / "item"
            record_dir.mkdir(parents=True)
            surface = np.arange(30, dtype=np.float32).reshape(10, 3)
            np.savez(
                record_dir / "cod_sdf.npz",
                surface_points=surface,
                uniform_query_points=np.zeros((4, 3), dtype=np.float32),
                uniform_sdf=np.ones(4, dtype=np.float32),
            )
            self._write_configuration(root, data_source)

            context = evaluate_config.resolve_configuration(root)
            result = evaluate_config.load_reference_points(
                context["records"][0],
                context["data_source"],
                evaluate_config.EvaluationOptions(num_points=6),
            )

            self.assertIsNone(result["error"])
            self.assertEqual(result["points"].shape, (6, 3))
            self.assertEqual(Path(result["path"]).name, "cod_sdf.npz")

    def test_validates_native_cod_modulation_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_source = root / "datasets"
            data_source.mkdir()
            self._write_configuration(root, data_source)
            modulation_dir = root / "modulations" / "ABO" / "item"
            modulation_dir.mkdir(parents=True)
            np.savez(
                modulation_dir / "modulation.npz",
                object_id=np.asarray("item"),
                posterior_mean=np.zeros((2, 3), dtype=np.float32),
                posterior_logvar=np.zeros((2, 3), dtype=np.float32),
            )
            np.savez(
                root / "modulations" / "latent_stats.npz",
                mean=np.zeros((1, 1, 3), dtype=np.float32),
                std=np.ones((1, 1, 3), dtype=np.float32),
            )

            result = evaluate_config.validate_modulations(
                evaluate_config.resolve_configuration(root)
            )

            self.assertEqual(result["expected_shape"], [2, 3])
            self.assertEqual(result["valid"], 1)
            self.assertEqual(result["observed_shapes"], {"2x3": 1})
            self.assertTrue(result["latent_statistics"]["valid"])

    def test_rejects_flattened_legacy_latent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_source = root / "datasets"
            data_source.mkdir()
            self._write_configuration(root, data_source)
            modulation_dir = root / "modulations" / "ABO" / "item"
            modulation_dir.mkdir(parents=True)
            np.savez(
                modulation_dir / "modulation.npz",
                object_id=np.asarray("item"),
                posterior_mean=np.zeros(6, dtype=np.float32),
                posterior_logvar=np.zeros(6, dtype=np.float32),
            )

            result = evaluate_config.validate_modulations(
                evaluate_config.resolve_configuration(root)
            )

            self.assertEqual(result["valid"], 0)
            self.assertIn("posterior shape", result["invalid"]["ABO/item"])

    def test_stage_three_validates_modulation_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_source = root / "datasets"
            data_source.mkdir()
            specs = self._write_configuration(root, data_source, task="combined")
            specs["diffusion_model_specs"] = {
                "latent_tokens": 2,
                "latent_dimension": 3,
                "cond": False,
            }
            specs["modulation_path"] = str(root / "cached_modulations")
            (root / "specs.json").write_text(json.dumps(specs))
            modulation_dir = root / "cached_modulations" / "ABO" / "item"
            modulation_dir.mkdir(parents=True)
            np.savez(
                modulation_dir / "modulation.npz",
                object_id=np.asarray("item"),
                posterior_mean=np.zeros((2, 3), dtype=np.float32),
                posterior_logvar=np.zeros((2, 3), dtype=np.float32),
            )

            result = evaluate_config.validate_modulations(
                evaluate_config.resolve_configuration(root)
            )

            self.assertEqual(result["path"], str((root / "cached_modulations")))
            self.assertEqual(result["valid"], 1)

    def test_stage_two_resolves_data_source_from_stage_one_specs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_source = root / "datasets"
            data_source.mkdir()
            stage_one = root / "stage_one"
            stage_one.mkdir()
            self._write_configuration(stage_one, data_source)

            stage_two = root / "stage_two"
            stage_two.mkdir()
            split_path = stage_one / "test_split.json"
            specs = {
                "TestSplit": str(split_path),
                "training_task": "diffusion",
                "modulation_ckpt_path": str(stage_one / "last.ckpt"),
                "diffusion_model_specs": {
                    "latent_tokens": 2,
                    "latent_dimension": 3,
                    "cond": False,
                },
                "CODVaeSpecs": {
                    "latent_tokens": 2,
                    "latent_dimension": 3,
                },
            }
            (stage_two / "specs.json").write_text(json.dumps(specs))

            context = evaluate_config.resolve_configuration(stage_two)

            self.assertEqual(context["data_source"], data_source.resolve())

    def test_discovers_unconditional_diffusion_samples_in_flat_recon_dir(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recon = root / "recon"
            recon.mkdir()
            (recon / "0_recon.ply").touch()
            (recon / "1_recon.ply").touch()
            context = {
                "config_dir": root,
                "task": "diffusion",
                "conditional": False,
                "records": [
                    {
                        "key": "ABO/item",
                        "class_name": "ABO",
                        "instance_name": "item",
                    }
                ],
            }

            artifacts = evaluate_config._discover_artifacts(context)

            self.assertEqual([item["sample"] for item in artifacts], ["0", "1"])
            self.assertTrue(all(item["record"] is None for item in artifacts))

    def test_discovers_image_conditioned_samples_by_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recon = root / "recon" / "ABO" / "item"
            recon.mkdir(parents=True)
            (recon / "0_recon.ply").touch()
            (recon / "1_recon.ply").touch()
            record = {
                "key": "ABO/item",
                "class_name": "ABO",
                "instance_name": "item",
            }
            context = {
                "config_dir": root,
                "task": "diffusion",
                "conditional": True,
                "records": [record],
            }

            artifacts = evaluate_config._discover_artifacts(context)

            self.assertEqual([item["sample"] for item in artifacts], ["0", "1"])
            self.assertTrue(all(item["key"] == "ABO/item" for item in artifacts))
            self.assertTrue(all(item["record"] == record for item in artifacts))

    def test_jsd_grid_covers_full_cod_cube(self):
        grid = evaluate_config._grid_coordinates(3, 1.0)

        self.assertEqual(grid.shape, (27, 3))
        self.assertTrue(np.any(np.all(grid == np.asarray([1.0, 1.0, 1.0]), axis=1)))

    def test_chamfer_is_sum_of_unsquared_directional_l2_means(self):
        sample = np.asarray([[0.0, 0.0, 0.0]])
        reference = np.asarray([[2.0, 0.0, 0.0]])

        metrics = evaluate_config.paired_surface_metrics(
            sample, reference, thresholds=(1.0,)
        )

        self.assertEqual(metrics["cd_sample_to_reference"], 2.0)
        self.assertEqual(metrics["cd_reference_to_sample"], 2.0)
        self.assertEqual(metrics["chamfer"], 4.0)
        self.assertNotEqual(metrics["chamfer"], 8.0)

    def test_pairwise_and_distribution_use_unsquared_chamfer(self):
        sample = np.asarray([[0.0, 0.0, 0.0]])
        reference = np.asarray([[2.0, 0.0, 0.0]])

        pairwise = evaluate_config.pairwise_chamfer([sample], [reference])
        distribution = evaluate_config.distribution_metrics(
            [sample],
            [reference],
            evaluate_config.EvaluationOptions(jsd_resolution=2),
            domain=2.0,
        )

        self.assertEqual(evaluate_config.chamfer_distance(sample, reference), 4.0)
        np.testing.assert_array_equal(pairwise, np.asarray([[4.0]]))
        self.assertEqual(distribution["mmd_cd"], 4.0)

    def test_num_points_defaults_to_ten_thousand(self):
        self.assertEqual(evaluate_config.DEFAULT_NUM_POINTS, 10000)
        self.assertEqual(evaluate_config.EvaluationOptions().num_points, 10000)
        self.assertEqual(
            evaluate_config.parse_args(["configuration"]).num_points,
            10000,
        )


if __name__ == "__main__":
    unittest.main()
