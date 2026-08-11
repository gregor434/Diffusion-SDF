import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.prepare_shapenetpart_chairs import (
    normalize_points_abo,
    partition_model_ids,
    prepare_dataset,
    resolve_source_root,
)


class ShapeNetPartPreparationTests(unittest.TestCase):
    def test_partition_uses_every_object_without_test_split(self):
        first = partition_model_ids((f"chair_{index}" for index in range(10)), 0.8, 7)
        second = partition_model_ids((f"chair_{index}" for index in range(10)), 0.8, 7)

        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "val", "all"})
        self.assertEqual(len(first["train"]), 8)
        self.assertEqual(len(first["val"]), 2)
        self.assertFalse(set(first["train"]) & set(first["val"]))
        self.assertEqual(set(first["train"]) | set(first["val"]), set(first["all"]))

    def test_point_normalization_matches_abo_convention(self):
        points = np.asarray(
            [[1.0, -2.0, 5.0], [5.0, 2.0, 7.0], [3.0, 0.0, 6.0]],
            dtype=np.float32,
        )
        normalized, center, scale = normalize_points_abo(points)

        np.testing.assert_allclose(
            (normalized.min(axis=0) + normalized.max(axis=0)) / 2,
            0.0,
            atol=1e-6,
        )
        self.assertAlmostEqual(float(np.abs(normalized).max()), 0.999, places=6)
        np.testing.assert_allclose(normalized, (points - center) * scale)

    def test_point_normalization_supports_explicit_cod099_extent(self):
        points = np.asarray([[-4.0, 2.0, 8.0], [2.0, 6.0, 10.0]], np.float32)
        normalized, _, _ = normalize_points_abo(points, extent=0.99)
        self.assertAlmostEqual(float(np.abs(normalized).max()), 0.99, places=6)

    def test_converts_xyz_only_and_writes_pipeline_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extracted = (
                root
                / "input"
                / "shapenetcore_partanno_segmentation_benchmark_v0_normal"
            )
            category = extracted / "03001627"
            split_dir = extracted / "train_test_split"
            category.mkdir(parents=True)
            split_dir.mkdir()

            model_ids = ("chair_train", "chair_val", "chair_test")
            rows = np.asarray(
                [
                    [1, 2, 3, 10, 11, 12, 4],
                    [3, 6, 9, 20, 21, 22, 5],
                    [2, 4, 6, 30, 31, 32, 6],
                ],
                dtype=np.float32,
            )
            for model_id in model_ids:
                np.savetxt(category / f"{model_id}.txt", rows)

            split_values = {
                "train": ["shape_data/03001627/chair_train"],
                "val": ["shape_data/03001627/chair_val"],
                "test": ["shape_data/03001627/chair_test"],
            }
            for split_name, values in split_values.items():
                (split_dir / f"shuffled_{split_name}_file_list.json").write_text(
                    json.dumps(values), encoding="utf-8"
                )

            output_root = root / "output"
            args = SimpleNamespace(
                source_dir=root / "input",
                datasets_root=output_root,
                dataset_key="shapenetpart",
                class_name="CHAIR",
                category_id="03001627",
                split_prefix="shapenetpart_CHAIR",
                train_ratio=2 / 3,
                split_seed=0,
                workers=2,
                limit=None,
                skip_existing=False,
            )
            summary = prepare_dataset(args)

            self.assertEqual(resolve_source_root(args.source_dir, args.category_id), extracted)
            self.assertEqual(summary["written"], 3)
            record_path = (
                output_root
                / "shapenetpart"
                / "CHAIR"
                / "chair_train"
                / "cod_sdf.npz"
            )
            with np.load(record_path) as record:
                self.assertEqual(record["surface_points"].shape, (3, 3))
                self.assertNotIn("surface_normals", record.files)
                self.assertNotIn("uniform_sdf", record.files)
                self.assertAlmostEqual(
                    float(np.abs(record["surface_points"]).max()), 0.999, places=6
                )

            train_manifest = json.loads(
                (
                    output_root
                    / "splits"
                    / "shapenetpart_CHAIR_train.json"
                ).read_text(encoding="utf-8")
            )
            train_ids = train_manifest["shapenetpart"]["CHAIR"]
            val_manifest = json.loads(
                (
                    output_root / "splits" / "shapenetpart_CHAIR_val.json"
                ).read_text(encoding="utf-8")
            )
            val_ids = val_manifest["shapenetpart"]["CHAIR"]
            self.assertEqual(len(train_ids), 2)
            self.assertEqual(len(val_ids), 1)
            self.assertFalse(set(train_ids) & set(val_ids))
            self.assertEqual(set(train_ids) | set(val_ids), set(model_ids))
            self.assertFalse(
                (output_root / "splits" / "shapenetpart_CHAIR_test.json").exists()
            )
            self.assertEqual(set(summary["splits"]["all"]), set(model_ids))


if __name__ == "__main__":
    unittest.main()
