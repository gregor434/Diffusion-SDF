import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.prepare_shapenetpart_chairs import (
    normalize_points_abo,
    prepare_dataset,
    resolve_source_root,
)


class ShapeNetPartPreparationTests(unittest.TestCase):
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
            self.assertEqual(
                train_manifest,
                {"shapenetpart": {"CHAIR": ["chair_train"]}},
            )
            self.assertEqual(set(summary["splits"]["all"]), set(model_ids))


if __name__ == "__main__":
    unittest.main()
