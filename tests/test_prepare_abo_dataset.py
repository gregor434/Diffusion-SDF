import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import trimesh

import scripts.prepare_abo_dataset as preprocessing
from scripts.prepare_abo_dataset import (
    REPAIR_MANIFOLDPLUS,
    RepairConfig,
    build_repair_config,
    build_split_sets,
    compute_split_counts,
    default_metadata_out,
    load_repaired_mesh,
    normalize_mesh_with_transform,
    repair_mesh_with_manifoldplus,
    save_cod_sdf,
)


class CODPreprocessingTests(unittest.TestCase):
    def test_preprocessing_metadata_is_stored_with_dataset(self):
        args = SimpleNamespace(
            datasets_root=Path("datasets"),
            dataset_key="abo",
        )
        self.assertEqual(
            default_metadata_out(args),
            Path("datasets/abo/preprocessing_metadata.json"),
        )

    def test_normalization_is_isotropic_and_in_cod_range(self):
        mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
        mesh.apply_translation((3.0, -2.0, 7.0))
        normalized, center, scale = normalize_mesh_with_transform(mesh)

        np.testing.assert_allclose(normalized.bounds.mean(axis=0), 0.0, atol=1e-6)
        maximum = float(np.abs(np.asarray(normalized.vertices)).max())
        self.assertAlmostEqual(maximum, 0.999, places=6)
        transformed = (mesh.vertices - center) * scale
        np.testing.assert_allclose(transformed, normalized.vertices, atol=1e-6)
        self.assertEqual(np.asarray(scale).shape, ())

    def test_cod_npz_has_separate_surface_and_supervision_arrays(self):
        arrays = {
            "surface_points": np.zeros((8, 3), np.float32),
            "near_surface_query_points": np.ones((6, 3), np.float32),
            "near_surface_sdf": np.ones(6, np.float32),
            "uniform_query_points": np.full((4, 3), 0.5, np.float32),
            "uniform_sdf": np.full(4, 0.5, np.float32),
            "normalization_center": np.arange(3, dtype=np.float32),
            "normalization_scale": np.asarray(2.0, np.float32),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cod_sdf.npz"
            save_cod_sdf(path, arrays)
            with np.load(path) as restored:
                self.assertEqual(set(restored.files), set(arrays))
                for name, expected in arrays.items():
                    np.testing.assert_array_equal(restored[name], expected)
            self.assertFalse(path.with_suffix(".npz.tmp").exists())

    def test_supervision_queries_stay_in_cod_sampling_range(self):
        surface = np.full((16, 3), 0.999, dtype=np.float32)
        normals = np.zeros_like(surface)
        with mock.patch(
            "scripts.prepare_abo_dataset.sample_surface",
            return_value=(surface, normals),
        ), mock.patch(
            "scripts.prepare_abo_dataset.compute_signed_distances",
            side_effect=lambda scene, points, batch_size, sign_method: np.zeros(
                (len(points), 1), dtype=np.float32
            ),
        ):
            arrays = preprocessing.sample_cod_supervision(
                mesh=mock.Mock(),
                scene=mock.Mock(),
                surface_point_count=len(surface),
                near_surface_stds=(0.1, 0.01),
                uniform_point_count=32,
                batch_size=32,
                rng=np.random.default_rng(0),
            )

        for name in ("near_surface_query_points", "uniform_query_points"):
            self.assertGreaterEqual(float(arrays[name].min()), -1.0)
            self.assertLessEqual(
                float(arrays[name].max()), float(np.float32(0.999))
            )

    def test_compute_split_counts_preserves_validation(self):
        self.assertEqual(compute_split_counts(10, 0.8), (8, 2))
        self.assertEqual(compute_split_counts(3, 0.8), (2, 1))

    def test_balanced_aggregate_splits(self):
        all_splits, per_type = build_split_sets(
            {"CHAIR": ["a", "b", "c"], "TABLE": ["d", "e", "f", "g"]},
            train_ratio=0.8,
            seed=7,
        )
        self.assertEqual(len(all_splits["all"]), 6)
        self.assertEqual(set(per_type), {"CHAIR", "TABLE"})

    def test_build_repair_config_resolves_workspace_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            binary = Path(tmpdir) / "ManifoldPlus"
            binary.write_text("#!/bin/sh\n")
            args = SimpleNamespace(
                repair_method=REPAIR_MANIFOLDPLUS,
                manifoldplus_bin=None,
                manifoldplus_depth=8,
                repaired_mesh_dir=None,
                datasets_root=Path(tmpdir) / "datasets",
                force_repair=False,
            )
            with mock.patch.dict(
                "scripts.prepare_abo_dataset.os.environ",
                {"MANIFOLDPLUS_BIN": str(binary)},
            ):
                config = build_repair_config(args)
            self.assertEqual(config.manifoldplus_bin, binary)
            self.assertEqual(
                config.repaired_mesh_dir,
                args.datasets_root / "repaired_meshes_cod_0999",
            )

    def test_manifold_repair_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            binary = root / "ManifoldPlus"
            binary.write_text("#!/bin/sh\n")
            output = root / "proxy.obj"
            mesh = trimesh.creation.box()
            config = RepairConfig(
                method=REPAIR_MANIFOLDPLUS,
                manifoldplus_bin=binary,
                manifoldplus_depth=8,
                repaired_mesh_dir=root,
            )

            def fake_run(command, capture_output, text):
                mesh.export(output)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch(
                "scripts.prepare_abo_dataset.subprocess.run", side_effect=fake_run
            ) as run:
                first = repair_mesh_with_manifoldplus(mesh, output, config)
                second = repair_mesh_with_manifoldplus(mesh, output, config)
            self.assertFalse(first.used_cache)
            self.assertTrue(second.used_cache)
            self.assertEqual(run.call_count, 1)
            self.assertTrue(load_repaired_mesh(output).is_watertight)

    def test_skip_existing_keeps_repair_cache_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            record = root / "abo" / "ABO" / "sample" / "cod_sdf.npz"
            record.parent.mkdir(parents=True)
            record.touch()
            repaired_root = root / "repaired_meshes_cod_0999"
            proxy = repaired_root / "abo" / "ABO" / "sample.obj"
            proxy.parent.mkdir(parents=True)
            proxy.touch()
            config = RepairConfig(
                method=REPAIR_MANIFOLDPLUS,
                manifoldplus_bin=root / "ManifoldPlus",
                manifoldplus_depth=8,
                repaired_mesh_dir=repaired_root,
            )

            output, repair_info = preprocessing.process_model(
                mesh_path=root / "sample.glb",
                datasets_root=root,
                dataset_key="abo",
                class_name="ABO",
                surface_point_count=8,
                near_surface_stds=(0.005, 0.0005),
                uniform_point_count=8,
                batch_size=8,
                rng=np.random.default_rng(0),
                skip_existing=True,
                repair_config=config,
            )

            self.assertEqual(output, record)
            self.assertTrue(repair_info["skipped_existing"])
            self.assertEqual(repair_info["method"], REPAIR_MANIFOLDPLUS)
            self.assertEqual(repair_info["repaired_mesh_path"], str(proxy))
            self.assertTrue(repair_info["cache_hit"])

    @unittest.skipIf(preprocessing.o3d is None, "Open3D runtime unavailable")
    def test_signed_distance_scales_with_normalized_coordinates(self):
        normalized, _, scale = normalize_mesh_with_transform(
            trimesh.creation.box(extents=(2, 2, 2))
        )
        scene = preprocessing.make_raycast_scene(normalized)
        point = np.asarray([[1.5, 0, 0]], dtype=np.float32)
        distance = preprocessing.compute_signed_distances(
            scene, point, batch_size=1, sign_method="occupancy"
        )
        self.assertGreater(distance.item(), 0)
        self.assertGreater(scale, 0)


if __name__ == "__main__":
    unittest.main()
