import tempfile
import threading
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import trimesh

import scripts.prepare_abo_dataset as preprocessing
from scripts.prepare_abo_dataset import (
    REPAIR_MANIFOLDPLUS,
    RepairConfig,
    RepairFidelityConfig,
    RepairResult,
    build_repair_config,
    build_split_sets,
    compute_split_counts,
    default_metadata_out,
    load_repaired_mesh,
    mesh_summary,
    normalize_mesh_with_transform,
    repaired_mesh_fidelity,
    process_model_isolated,
    repair_mesh_with_manifoldplus,
    save_cod_sdf,
    write_split_manifests,
)


class CODPreprocessingTests(unittest.TestCase):
    @staticmethod
    def accepted_fidelity():
        direction = {
            "mean": 0.001,
            "p95": 0.002,
            "p99": 0.003,
            "max": 0.004,
            "outlier_fraction": 0.0,
        }
        return {
            "accepted": True,
            "sample_count": 32,
            "distance_threshold": 0.02,
            "max_p95_distance": 0.02,
            "max_outlier_fraction": 0.05,
            "original_to_repaired": dict(direction),
            "repaired_to_original": dict(direction),
        }

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

    def test_normalization_extent_uses_actual_off_center_proxy_bounds(self):
        mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
        mesh.apply_translation((13.0, -7.0, 2.5))
        normalized, center, scale = normalize_mesh_with_transform(
            mesh, extent=0.99
        )

        np.testing.assert_allclose(center, [13.0, -7.0, 2.5], atol=1e-6)
        np.testing.assert_allclose(normalized.bounds.mean(axis=0), 0.0, atol=1e-6)
        self.assertAlmostEqual(
            float(np.abs(np.asarray(normalized.vertices)).max()), 0.99, places=6
        )
        np.testing.assert_allclose(
            normalized.vertices, (mesh.vertices - center) * scale, atol=1e-6
        )

    def test_accepted_proxy_mode_inherits_exact_splits_without_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            proxy_dir = root / "proxies"
            proxy_dir.mkdir()
            split_members = {
                "all": ["accepted_b", "accepted_a"],
                "train": ["accepted_b"],
                "val": ["accepted_a"],
            }
            manifests = {}
            for name, members in split_members.items():
                path = root / f"source_{name}.json"
                path.write_text(
                    json.dumps({"abo_multiray21": {"ABO": members}}),
                    encoding="utf-8",
                )
                manifests[name] = path
            fidelity = self.accepted_fidelity()
            source_metadata = root / "source_metadata.json"
            source_metadata.write_text(
                json.dumps({
                    "products": {
                        model_id: {
                            "training_eligible": True,
                            "preprocessing_repair": {"fidelity": fidelity},
                        }
                        for model_id in split_members["all"]
                    }
                }),
                encoding="utf-8",
            )
            for model_id in split_members["all"]:
                (proxy_dir / f"{model_id}.obj").write_text(
                    f"authoritative proxy {model_id}\n", encoding="utf-8"
                )
            args = SimpleNamespace(
                normalization_extent=0.99,
                source_all_manifest=manifests["all"],
                source_train_manifest=manifests["train"],
                source_val_manifest=manifests["val"],
                source_preprocessing_metadata=source_metadata,
                accepted_proxy_dir=proxy_dir,
                datasets_root=root / "output",
                dataset_key="abo_cod099",
                class_name="ABO",
                split_prefix="abo_fullchairs_multiray21_cod099_CHAIR",
                metadata_out=None,
                surface_point_count=8,
                near_surface_stds=(0.005, 0.0005),
                uniform_point_count=8,
                batch_size=8,
                seed=3,
                skip_existing=False,
            )

            with mock.patch.object(
                preprocessing,
                "process_accepted_proxy",
                return_value={
                    "normalization_center": [1.0, 2.0, 3.0],
                    "normalization_scale": 0.5,
                },
            ) as process_proxy, mock.patch.object(
                preprocessing, "repair_mesh_with_manifoldplus"
            ) as repair, mock.patch.object(
                preprocessing, "repaired_mesh_fidelity"
            ) as evaluate_fidelity:
                result = preprocessing.prepare_accepted_proxies(args)

            self.assertEqual(process_proxy.call_count, 2)
            repair.assert_not_called()
            evaluate_fidelity.assert_not_called()
            for name, expected in split_members.items():
                generated = json.loads(
                    (root / "output" / "splits" / f"{args.split_prefix}_{name}.json")
                    .read_text(encoding="utf-8")
                )
                self.assertEqual(generated["abo_cod099"]["ABO"], expected)
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["fidelity_status"], "inherited")
            self.assertFalse(metadata["filtering_performed"])
            self.assertEqual(metadata["canonical_extent"], 0.99)
            self.assertEqual(metadata["counts"], {"all": 2, "train": 1, "val": 1})
            for product in metadata["products"].values():
                self.assertEqual(product["inherited_fidelity_result"], fidelity)
                self.assertEqual(len(product["source_proxy_sha256"]), 64)

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

    def test_mesh_summary_does_not_split_and_copy_components(self):
        mesh = trimesh.util.concatenate([
            trimesh.creation.box(),
            trimesh.creation.box(transform=trimesh.transformations.translation_matrix([3, 0, 0])),
        ])
        with mock.patch.object(
            trimesh.Trimesh,
            "split",
            side_effect=AssertionError("mesh.split must not be called"),
        ):
            self.assertEqual(mesh_summary(mesh)["components"], 2)

    def test_isolated_worker_reports_failure_without_killing_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(RuntimeError, "missing.glb"):
                process_model_isolated(
                    mesh_path=root / "missing.glb",
                    datasets_root=root,
                    dataset_key="abo",
                    class_name="ABO",
                    surface_point_count=8,
                    near_surface_stds=(0.005, 0.0005),
                    uniform_point_count=8,
                    batch_size=4,
                    seed=0,
                    skip_existing=False,
                    repair_config=RepairConfig(),
                )

    def test_model_jobs_can_run_concurrently_and_keep_results_attributed(self):
        barrier = threading.Barrier(2)

        def process(model_id):
            barrier.wait(timeout=2)
            return f"processed-{model_id}"

        results = list(
            preprocessing.iter_model_results(["first", "second"], 2, process)
        )

        self.assertEqual(
            {model_id: result for model_id, result, error in results},
            {"first": "processed-first", "second": "processed-second"},
        )
        self.assertTrue(all(error is None for _, _, error in results))

    def test_model_job_errors_are_returned_with_the_model_id(self):
        def process(model_id):
            if model_id == "broken":
                raise RuntimeError("expected failure")
            return model_id

        results = {
            model_id: (result, error)
            for model_id, result, error in preprocessing.iter_model_results(
                ["working", "broken"], 2, process
            )
        }

        self.assertEqual(results["working"], ("working", None))
        self.assertIsInstance(results["broken"][1], RuntimeError)

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

    @unittest.skipIf(preprocessing.o3d is None, "Open3D runtime unavailable")
    def test_multi_ray_sign_uses_majority_with_bounded_ray_buffers(self):
        class ArrayResult:
            def __init__(self, values):
                self.values = values

            def numpy(self):
                return self.values

        class FakeScene:
            def __init__(self):
                self.intersection_calls = 0
                self.largest_ray_batch = 0

            def compute_distance(self, points):
                return ArrayResult(np.ones(len(points), dtype=np.float32))

            def count_intersections(self, rays):
                values = rays.numpy()
                self.largest_ray_batch = max(
                    self.largest_ray_batch, len(values)
                )
                direction_index = (
                    self.intersection_calls % preprocessing.SIGN_RAY_COUNT
                )
                self.intersection_calls += 1
                counts = np.zeros(len(values), dtype=np.uint32)
                # Negative-x queries receive an inside majority. Positive-x
                # queries have one erroneous odd ray but remain outside.
                counts[
                    (values[:, 0] < 0) & (direction_index <= 10)
                ] = 1
                counts[
                    (values[:, 0] > 0) & (direction_index == 0)
                ] = 1
                return ArrayResult(counts)

        scene = FakeScene()
        points = np.asarray(
            [
                [-0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [-0.25, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.75, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        sdf = preprocessing.compute_signed_distances(
            scene, points, batch_size=2, sign_method="occupancy"
        ).reshape(-1)

        np.testing.assert_array_equal(
            np.sign(sdf), np.asarray([-1, 1, -1, 1, 1])
        )
        self.assertEqual(
            scene.intersection_calls,
            preprocessing.SIGN_RAY_COUNT * 3,
        )
        self.assertLessEqual(scene.largest_ray_batch, 2)

    def test_compute_split_counts_preserves_validation(self):
        self.assertEqual(compute_split_counts(10, 0.8), (8, 2))
        self.assertEqual(compute_split_counts(3, 0.8), (2, 1))

    def test_identical_repaired_geometry_is_deduplicated_before_splitting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = {
                model_id: root / f"{model_id}.obj"
                for model_id in ("a", "b", "c", "d")
            }
            paths["a"].write_bytes(b"same repaired geometry")
            paths["b"].write_bytes(b"same repaired geometry")
            # Same byte length exercises the hash check rather than the
            # inexpensive unique-size path.
            paths["c"].write_bytes(b"other repair geometry!")
            paths["d"].write_bytes(b"unique")

            retained, groups = preprocessing.deduplicate_geometry_files(
                set(paths), paths
            )
            splits = preprocessing.split_model_ids(
                sorted(retained),
                train_ratio=0.67,
                rng=np.random.default_rng(0),
            )

        self.assertEqual(retained, {"a", "c", "d"})
        self.assertEqual(groups, {"a": ["a", "b"]})
        self.assertNotIn("b", splits["all"])
        self.assertTrue(
            set(splits["train"]).isdisjoint(splits["val"])
        )

    def test_balanced_aggregate_splits(self):
        all_splits, per_type = build_split_sets(
            {"CHAIR": ["a", "b", "c"], "TABLE": ["d", "e", "f", "g"]},
            train_ratio=0.8,
            seed=7,
        )
        self.assertEqual(len(all_splits["all"]), 6)
        self.assertEqual(set(per_type), {"CHAIR", "TABLE"})

    def test_can_write_only_per_type_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = write_split_manifests(
                splits_dir=root,
                split_prefix="abo_fullchairs",
                dataset_key="abo",
                class_name="ABO",
                all_splits={"all": ["a", "b"], "train": ["a"], "val": ["b"]},
                per_type_splits={
                    "CHAIR": {"all": ["a", "b"], "train": ["a"], "val": ["b"]}
                },
                write_aggregate=False,
            )

            self.assertEqual(paths["all"], {})
            self.assertEqual(
                {path.name for path in root.glob("*.json")},
                {
                    "abo_fullchairs_CHAIR_all.json",
                    "abo_fullchairs_CHAIR_train.json",
                    "abo_fullchairs_CHAIR_val.json",
                },
            )

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
                generated = Path(command[command.index("--output") + 1])
                mesh.export(generated)
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

    def test_fidelity_cache_is_invalidated_when_sampling_threshold_changes(self):
        fidelity = self.accepted_fidelity()
        matching = RepairFidelityConfig(
            sample_count=32, distance_threshold=0.02
        )
        changed = RepairFidelityConfig(
            sample_count=32, distance_threshold=0.04,
            max_p95_distance=0.04,
        )
        self.assertTrue(
            preprocessing.repair_fidelity_passes(fidelity, matching)
        )
        self.assertFalse(
            preprocessing.repair_fidelity_passes(fidelity, changed)
        )

    def test_regeneration_reuses_compatible_repair_fidelity_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "sample.glb"
            original = trimesh.creation.box()
            original.export(source)
            repaired = original.copy()
            repaired_root = root / "repaired"
            proxy = repaired_root / "abo" / "ABO" / "sample.obj"
            preprocessing.save_repair_fidelity(
                preprocessing.repair_fidelity_output_path(proxy),
                self.accepted_fidelity(),
            )
            config = RepairConfig(
                method=REPAIR_MANIFOLDPLUS,
                manifoldplus_bin=root / "ManifoldPlus",
                repaired_mesh_dir=repaired_root,
            )
            arrays = {
                "surface_points": np.zeros((8, 3), np.float32),
                "surface_normals": np.zeros((8, 3), np.float32),
                "near_surface_query_points": np.zeros((16, 3), np.float32),
                "near_surface_sdf": np.zeros(16, np.float32),
                "uniform_query_points": np.zeros((8, 3), np.float32),
                "uniform_sdf": np.zeros(8, np.float32),
            }

            with mock.patch(
                "scripts.prepare_abo_dataset.repair_mesh_with_manifoldplus",
                return_value=RepairResult(repaired, used_cache=True),
            ), mock.patch(
                "scripts.prepare_abo_dataset.repaired_mesh_fidelity"
            ) as validate, mock.patch(
                "scripts.prepare_abo_dataset.make_raycast_scene",
                return_value=mock.Mock(),
            ) as make_scene, mock.patch(
                "scripts.prepare_abo_dataset.sample_cod_supervision",
                return_value=arrays,
            ):
                output, info = preprocessing.process_model(
                    mesh_path=source,
                    datasets_root=root,
                    dataset_key="abo",
                    class_name="ABO",
                    surface_point_count=8,
                    near_surface_stds=(0.005, 0.0005),
                    uniform_point_count=8,
                    batch_size=8,
                    rng=np.random.default_rng(0),
                    skip_existing=False,
                    repair_config=config,
                    use_repaired_surface=True,
                    fidelity_config=RepairFidelityConfig(sample_count=32),
                    reuse_repair_fidelity=True,
                )

            self.assertIsNotNone(output)
            validate.assert_not_called()
            self.assertEqual(make_scene.call_count, 1)
            self.assertTrue(info["fidelity"]["accepted"])

    def test_new_repaired_proxy_is_always_validated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "sample.glb"
            original = trimesh.creation.box()
            original.export(source)
            repaired = original.copy()
            repaired_root = root / "repaired"
            proxy = repaired_root / "abo" / "ABO" / "sample.obj"
            preprocessing.save_repair_fidelity(
                preprocessing.repair_fidelity_output_path(proxy),
                self.accepted_fidelity(),
            )
            config = RepairConfig(
                method=REPAIR_MANIFOLDPLUS,
                manifoldplus_bin=root / "ManifoldPlus",
                repaired_mesh_dir=repaired_root,
                force_repair=True,
            )
            arrays = {
                "surface_points": np.zeros((8, 3), np.float32),
                "surface_normals": np.zeros((8, 3), np.float32),
                "near_surface_query_points": np.zeros((16, 3), np.float32),
                "near_surface_sdf": np.zeros(16, np.float32),
                "uniform_query_points": np.zeros((8, 3), np.float32),
                "uniform_sdf": np.zeros(8, np.float32),
            }

            with mock.patch(
                "scripts.prepare_abo_dataset.repair_mesh_with_manifoldplus",
                return_value=RepairResult(repaired, used_cache=False),
            ), mock.patch(
                "scripts.prepare_abo_dataset.repaired_mesh_fidelity",
                return_value=self.accepted_fidelity(),
            ) as validate, mock.patch(
                "scripts.prepare_abo_dataset.make_raycast_scene",
                return_value=mock.Mock(),
            ), mock.patch(
                "scripts.prepare_abo_dataset.sample_cod_supervision",
                return_value=arrays,
            ):
                preprocessing.process_model(
                    mesh_path=source,
                    datasets_root=root,
                    dataset_key="abo",
                    class_name="ABO",
                    surface_point_count=8,
                    near_surface_stds=(0.005, 0.0005),
                    uniform_point_count=8,
                    batch_size=8,
                    rng=np.random.default_rng(0),
                    skip_existing=False,
                    repair_config=config,
                    use_repaired_surface=True,
                    fidelity_config=RepairFidelityConfig(sample_count=32),
                    reuse_repair_fidelity=True,
                )

            validate.assert_called_once()

    @unittest.skipIf(preprocessing.o3d is None, "Open3D runtime unavailable")
    def test_repaired_mesh_fidelity_accepts_matching_mesh_and_rejects_drift(self):
        original = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        matching = original.copy()
        drifted = original.copy()
        drifted.apply_translation((0.2, 0.0, 0.0))
        config = RepairFidelityConfig(sample_count=4000)
        rng = np.random.default_rng(0)
        original_scene = preprocessing.make_raycast_scene(original)

        matching_result = repaired_mesh_fidelity(
            original,
            matching,
            original_scene,
            preprocessing.make_raycast_scene(matching),
            config,
            batch_size=1000,
            rng=rng,
        )
        drifted_result = repaired_mesh_fidelity(
            original,
            drifted,
            original_scene,
            preprocessing.make_raycast_scene(drifted),
            config,
            batch_size=1000,
            rng=rng,
        )

        self.assertTrue(matching_result["accepted"])
        self.assertFalse(drifted_result["accepted"])
        self.assertGreater(
            drifted_result["original_to_repaired"]["p95"], 0.02
        )

    def test_repaired_mesh_is_used_for_surface_and_sdf_supervision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "sample.glb"
            trimesh.creation.box().export(source)
            repaired = trimesh.creation.icosphere(radius=0.4)
            repaired_root = root / "repaired"
            config = RepairConfig(
                method=REPAIR_MANIFOLDPLUS,
                manifoldplus_bin=root / "ManifoldPlus",
                repaired_mesh_dir=repaired_root,
            )
            arrays = {
                "surface_points": np.zeros((8, 3), np.float32),
                "surface_normals": np.zeros((8, 3), np.float32),
                "near_surface_query_points": np.zeros((16, 3), np.float32),
                "near_surface_sdf": np.zeros(16, np.float32),
                "uniform_query_points": np.zeros((8, 3), np.float32),
                "uniform_sdf": np.zeros(8, np.float32),
            }
            sampled_meshes = []

            def fake_repair(mesh, output_path, repair_config):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                repaired.export(output_path)
                return RepairResult(repaired, used_cache=False)

            def fake_sample(mesh, **kwargs):
                sampled_meshes.append(mesh)
                return dict(arrays)

            with mock.patch(
                "scripts.prepare_abo_dataset.repair_mesh_with_manifoldplus",
                side_effect=fake_repair,
            ), mock.patch(
                "scripts.prepare_abo_dataset.repaired_mesh_fidelity",
                return_value=self.accepted_fidelity(),
            ), mock.patch(
                "scripts.prepare_abo_dataset.make_raycast_scene",
                return_value=mock.Mock(),
            ), mock.patch(
                "scripts.prepare_abo_dataset.sample_cod_supervision",
                side_effect=fake_sample,
            ):
                output, info = preprocessing.process_model(
                    mesh_path=source,
                    datasets_root=root,
                    dataset_key="abo",
                    class_name="ABO",
                    surface_point_count=8,
                    near_surface_stds=(0.005, 0.0005),
                    uniform_point_count=8,
                    batch_size=8,
                    rng=np.random.default_rng(0),
                    skip_existing=False,
                    repair_config=config,
                    use_repaired_surface=True,
                    fidelity_config=RepairFidelityConfig(sample_count=32),
                )

            self.assertIsNotNone(output)
            self.assertIs(sampled_meshes[0], repaired)
            self.assertEqual(info["surface_source"], "repaired_mesh")
            with np.load(output) as data:
                self.assertEqual(str(data["surface_source"].item()), "repaired_mesh")
                self.assertTrue(bool(data["repair_fidelity_accepted"].item()))

    def test_rejected_repair_does_not_write_training_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "sample.glb"
            trimesh.creation.box().export(source)
            repaired = trimesh.creation.icosphere(radius=0.4)
            rejected = self.accepted_fidelity()
            rejected["accepted"] = False
            rejected["original_to_repaired"]["p95"] = 0.2
            rejected["original_to_repaired"]["outlier_fraction"] = 0.4
            config = RepairConfig(
                method=REPAIR_MANIFOLDPLUS,
                manifoldplus_bin=root / "ManifoldPlus",
                repaired_mesh_dir=root / "repaired",
            )

            def fake_repair(mesh, output_path, repair_config):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                repaired.export(output_path)
                return RepairResult(repaired, used_cache=False)

            with mock.patch(
                "scripts.prepare_abo_dataset.repair_mesh_with_manifoldplus",
                side_effect=fake_repair,
            ), mock.patch(
                "scripts.prepare_abo_dataset.repaired_mesh_fidelity",
                return_value=rejected,
            ), mock.patch(
                "scripts.prepare_abo_dataset.make_raycast_scene",
                return_value=mock.Mock(),
            ):
                output, info = preprocessing.process_model(
                    mesh_path=source,
                    datasets_root=root,
                    dataset_key="abo",
                    class_name="ABO",
                    surface_point_count=8,
                    near_surface_stds=(0.005, 0.0005),
                    uniform_point_count=8,
                    batch_size=8,
                    rng=np.random.default_rng(0),
                    skip_existing=False,
                    repair_config=config,
                    use_repaired_surface=True,
                    fidelity_config=RepairFidelityConfig(sample_count=32),
                )

            self.assertIsNone(output)
            self.assertFalse(info["fidelity"]["accepted"])
            self.assertFalse(
                preprocessing.object_output_paths(root, "abo", "ABO", "sample").exists()
            )

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

    def test_output_dataset_key_can_differ_from_repair_cache_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            record = root / "abo_multiray21" / "ABO" / "sample" / "cod_sdf.npz"
            record.parent.mkdir(parents=True)
            record.touch()
            repaired_root = root / "repaired_meshes_cod_0999"
            proxy = repaired_root / "abo" / "ABO" / "sample.obj"
            proxy.parent.mkdir(parents=True)
            proxy.touch()
            config = RepairConfig(
                method=REPAIR_MANIFOLDPLUS,
                manifoldplus_bin=root / "ManifoldPlus",
                repaired_mesh_dir=repaired_root,
            )

            output, repair_info = preprocessing.process_model(
                mesh_path=root / "sample.glb",
                datasets_root=root,
                dataset_key="abo_multiray21",
                repair_cache_dataset_key="abo",
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
