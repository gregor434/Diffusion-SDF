import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh

from scripts.prepare_abo_dataset import (
    REPAIR_MANIFOLDPLUS,
    RepairConfig,
    build_repair_config,
    build_split_sets,
    compute_signed_distances,
    compute_grid_sdf,
    compute_split_counts,
    load_repaired_mesh,
    make_raycast_scene,
    model_ids_from_manifest,
    normalize_mesh,
    repair_mesh_with_manifoldplus,
    repaired_mesh_output_path,
    save_csv,
    sample_near_surface,
    validate_repair_config,
    write_split_manifests,
)


class PrepareAboDatasetSplitTests(unittest.TestCase):
    def test_normalize_mesh_sets_tight_bounding_box_diagonal_to_one(self) -> None:
        mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
        mesh.apply_translation((3.0, -2.0, 7.0))

        normalized = normalize_mesh(mesh)

        extent = normalized.bounds[1] - normalized.bounds[0]
        np.testing.assert_allclose(normalized.bounds.mean(axis=0), 0.0, atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(extent)), 1.0, places=6)
        self.assertLessEqual(float(np.abs(normalized.bounds).max()), 0.5)

    def test_near_surface_sampling_matches_paper_row_layout(self) -> None:
        mesh = normalize_mesh(trimesh.creation.box(extents=(1.0, 2.0, 3.0)))
        scene = make_raycast_scene(mesh)

        rows = sample_near_surface(
            mesh=mesh,
            scene=scene,
            surface_point_count=20,
            near_surface_stds=(0.005, 0.0005),
            batch_size=16,
            rng=np.random.default_rng(4),
        )

        self.assertEqual(rows.shape, (60, 4))
        self.assertEqual(int(np.count_nonzero(rows[:, 3] == 0.0)), 20)
        self.assertTrue(np.isfinite(rows).all())

    def test_grid_is_regular_and_covers_cube_boundaries(self) -> None:
        mesh = normalize_mesh(trimesh.creation.box())
        rows = compute_grid_sdf(make_raycast_scene(mesh), resolution=3, batch_size=8)

        self.assertEqual(rows.shape, (27, 4))
        for axis in range(3):
            np.testing.assert_array_equal(np.unique(rows[:, axis]), [-1.0, 0.0, 1.0])

    def test_occupancy_signed_distance_is_available_for_watertight_meshes(self) -> None:
        mesh = normalize_mesh(trimesh.creation.box())
        scene = make_raycast_scene(mesh)
        points = np.array([[0.0, 0.0, 0.0], [0.75, 0.75, 0.75]], dtype=np.float32)

        sdf = compute_signed_distances(scene, points, batch_size=2, sign_method="occupancy")

        self.assertLess(float(sdf[0, 0]), 0.0)
        self.assertGreater(float(sdf[1, 0]), 0.0)

    def test_model_ids_from_manifest_reads_nested_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "split.json"
            manifest.write_text('{"abo": {"ABO": ["one", "two"]}}', encoding="utf-8")
            self.assertEqual(model_ids_from_manifest(manifest), {"one", "two"})

    def test_save_csv_preserves_fine_nonzero_distances(self) -> None:
        rows = np.array([[0.0, 0.0, 0.0, 1e-9]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sdf_data.csv"
            save_csv(path, rows)
            restored = np.loadtxt(path, delimiter=",")
            self.assertGreater(float(restored[3]), 0.0)
            self.assertFalse(path.with_suffix(".csv.tmp").exists())

    def test_validate_repair_config_requires_manifoldplus_binary(self) -> None:
        with self.assertRaises(ValueError):
            validate_repair_config(RepairConfig(method=REPAIR_MANIFOLDPLUS))

    def test_build_repair_config_resolves_manifoldplus_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            binary = Path(tmpdir) / "ManifoldPlus"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            args = mock.Mock(
                repair_method=REPAIR_MANIFOLDPLUS,
                manifoldplus_bin=None,
                manifoldplus_depth=8,
                repaired_mesh_dir=None,
                datasets_root=Path("datasets"),
                force_repair=False,
            )

            with mock.patch.dict("scripts.prepare_abo_dataset.os.environ", {"MANIFOLDPLUS_BIN": str(binary)}):
                config = build_repair_config(args)

            self.assertEqual(config.manifoldplus_bin, binary)
            self.assertEqual(config.repaired_mesh_dir, Path("datasets/repaired_meshes"))

    def test_repaired_mesh_output_path_uses_dataset_layout(self) -> None:
        self.assertEqual(
            repaired_mesh_output_path(Path("repaired"), "abo", "ABO", "chair_0"),
            Path("repaired/abo/ABO/chair_0.obj"),
        )

    def test_load_repaired_mesh_rejects_non_watertight_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plane.obj"
            trimesh.creation.box().submesh([[0]], append=True).export(path)

            with self.assertRaises(ValueError):
                load_repaired_mesh(path)

    def test_load_repaired_mesh_preserves_coincident_watertight_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "touching_boxes.obj"
            first = trimesh.creation.box()
            second = trimesh.creation.box()
            second.apply_translation((1.0, 1.0, 0.0))
            trimesh.util.concatenate((first, second)).export(path)

            repaired = load_repaired_mesh(path)

            self.assertTrue(repaired.is_watertight)
            self.assertEqual(len(repaired.vertices), len(first.vertices) + len(second.vertices))

    def test_repair_mesh_with_manifoldplus_invokes_external_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            binary = tmp / "ManifoldPlus"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            output_path = tmp / "proxy.obj"
            mesh = normalize_mesh(trimesh.creation.box())
            config = RepairConfig(
                method=REPAIR_MANIFOLDPLUS,
                manifoldplus_bin=binary,
                manifoldplus_depth=9,
                repaired_mesh_dir=tmp / "repaired",
            )

            def fake_run(command, capture_output, text):
                self.assertEqual(command[0], str(binary))
                self.assertIn("--input", command)
                self.assertEqual(command[command.index("--output") + 1], str(output_path))
                self.assertEqual(command[command.index("--depth") + 1], "9")
                mesh.export(output_path)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("scripts.prepare_abo_dataset.subprocess.run", side_effect=fake_run) as run:
                repaired = repair_mesh_with_manifoldplus(mesh, output_path, config)

            self.assertTrue(repaired.is_watertight)
            self.assertEqual(run.call_count, 1)

            with mock.patch("scripts.prepare_abo_dataset.subprocess.run") as cached_run:
                cached = repair_mesh_with_manifoldplus(mesh, output_path, config)

            self.assertTrue(cached.is_watertight)
            cached_run.assert_not_called()

    def test_compute_split_counts_preserves_validation_split(self) -> None:
        self.assertEqual(compute_split_counts(10, 0.8), (8, 2))
        self.assertEqual(compute_split_counts(4, 0.8), (3, 1))
        self.assertEqual(compute_split_counts(3, 0.8), (2, 1))

    def test_build_split_sets_balances_aggregate_classes(self) -> None:
        type_to_ids = {
            "chair": [f"chair_{idx}" for idx in range(6)],
            "lamp": [f"lamp_{idx}" for idx in range(4)],
            "table": [f"table_{idx}" for idx in range(5)],
        }

        all_splits, per_type_splits = build_split_sets(type_to_ids, 0.8, seed=7)

        self.assertEqual(len(all_splits["all"]), 12)
        self.assertEqual(len(all_splits["train"]), 9)
        self.assertEqual(len(all_splits["val"]), 3)

        for split_name, expected_per_class in (("train", 3), ("val", 1)):
            counts = {}
            for model_id in all_splits[split_name]:
                class_name = model_id.split("_", 1)[0]
                counts[class_name] = counts.get(class_name, 0) + 1
            self.assertEqual(counts, {"chair": expected_per_class, "lamp": expected_per_class, "table": expected_per_class})

        self.assertEqual(len(per_type_splits["chair"]["train"]), 4)
        self.assertEqual(len(per_type_splits["chair"]["val"]), 2)
        self.assertTrue(set(per_type_splits["chair"]["train"]).isdisjoint(per_type_splits["chair"]["val"]))
        self.assertEqual(
            sorted(
                per_type_splits["chair"]["train"]
                + per_type_splits["chair"]["val"]
            ),
            per_type_splits["chair"]["all"],
        )

    def test_build_split_sets_is_deterministic(self) -> None:
        type_to_ids = {
            "chair": [f"chair_{idx}" for idx in range(6)],
            "lamp": [f"lamp_{idx}" for idx in range(4)],
        }

        first = build_split_sets(type_to_ids, 0.8, seed=3)
        second = build_split_sets(type_to_ids, 0.8, seed=3)

        self.assertEqual(first, second)

    def test_build_split_sets_rejects_singleton_classes(self) -> None:
        type_to_ids = {
            "chair": [f"chair_{idx}" for idx in range(6)],
            "lamp": ["lamp_0"],
        }

        with self.assertRaises(ValueError):
            build_split_sets(type_to_ids, 0.8, seed=0)

    def test_write_split_manifests_records_nested_paths(self) -> None:
        all_splits = {
            "all": ["chair_0", "lamp_0"],
            "train": ["chair_0"],
            "val": ["lamp_0"],
        }
        per_type_splits = {
            "chair": {"all": ["chair_0"], "train": ["chair_0"], "val": []},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            split_paths = write_split_manifests(
                splits_dir=Path(tmpdir),
                split_prefix="abo",
                dataset_key="abo",
                class_name="ABO",
                all_splits=all_splits,
                per_type_splits=per_type_splits,
            )

            train_manifest = Path(split_paths["all"]["train"])
            self.assertTrue(train_manifest.is_file())
            self.assertEqual(
                json.loads(train_manifest.read_text(encoding="utf-8")),
                {"abo": {"ABO": ["chair_0"]}},
            )
            self.assertIn("chair", split_paths["by_product_type"])
            self.assertIn("val", split_paths["by_product_type"]["chair"])


if __name__ == "__main__":
    unittest.main()
