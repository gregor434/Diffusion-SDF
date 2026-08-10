import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.prepare_shapenetcore_chairs import (
    prepare_dataset,
    resolve_category_dir,
    sample_surface_area_weighted,
)
from scripts.prepare_shapenetpart_chairs import normalize_points_abo


OBJ = """\
v -2.0 -1.0 0.0
v 2.0 -1.0 0.0
v 2.0 3.0 0.0
v -2.0 3.0 0.0
f 1 2 3
f 1 3 4
"""


class ShapeNetCorePreparationTests(unittest.TestCase):
    def test_mesh_bound_normalization_precedes_sampling(self):
        vertices = np.asarray(
            [[-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [2.0, 3.0, 0.0]],
            dtype=np.float32,
        )
        normalized, center, scale = normalize_points_abo(vertices)
        points = sample_surface_area_weighted(
            normalized,
            np.asarray([[0, 1, 2]]),
            32,
            np.random.default_rng(7),
        )

        np.testing.assert_allclose(center, [0.0, 1.0, 0.0])
        self.assertAlmostEqual(float(scale), 0.999 / 2.0, places=6)
        self.assertAlmostEqual(float(np.abs(normalized).max()), 0.999, places=6)
        self.assertEqual(points.shape, (32, 3))
        self.assertTrue(np.isfinite(points).all())

    def test_converts_nested_objs_deterministically_without_requiring_watertightness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            category = root / "raw" / "03001627"
            model_ids = ("chair_a", "chair_b", "chair_c")
            for model_id in model_ids:
                mesh_dir = category / model_id / "models"
                mesh_dir.mkdir(parents=True)
                (mesh_dir / "model_normalized.obj").write_text(OBJ, encoding="utf-8")

            output_root = root / "output"
            args = SimpleNamespace(
                source_dir=root / "raw",
                datasets_root=output_root,
                dataset_key="shapenetcore",
                class_name="CHAIR",
                category_id="03001627",
                split_prefix="shapenetcore_CHAIR",
                surface_point_count=64,
                sampling_seed=11,
                train_ratio=2 / 3,
                split_seed=3,
                workers=2,
                limit=None,
                skip_existing=False,
                require_watertight=False,
                continue_on_error=False,
            )
            summary = prepare_dataset(args)

            self.assertEqual(resolve_category_dir(args.source_dir, args.category_id), category)
            self.assertEqual(resolve_category_dir(category, args.category_id), category)
            self.assertEqual(summary["written"], 3)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(set(summary["splits"]["all"]), set(model_ids))

            record_path = (
                output_root / "shapenetcore" / "CHAIR" / "chair_a" / "cod_sdf.npz"
            )
            with np.load(record_path) as record:
                first_points = record["surface_points"].copy()
                self.assertEqual(first_points.shape, (64, 3))
                self.assertFalse(bool(record["source_is_watertight"]))
                self.assertNotIn("uniform_sdf", record.files)
                self.assertNotIn("near_surface_sdf", record.files)

            prepare_dataset(args)
            with np.load(record_path) as record:
                np.testing.assert_array_equal(record["surface_points"], first_points)

            self.assertEqual(len(summary["splits"]["train"]), 2)
            self.assertEqual(len(summary["splits"]["val"]), 1)


if __name__ == "__main__":
    unittest.main()
