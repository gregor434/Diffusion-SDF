import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from utils.reconstruct import filter_threshold, reconstruction_chamfer


class ReconstructionFilterTests(unittest.TestCase):
    def test_seeded_mesh_chamfer_is_finite_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "box.ply"
            mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
            mesh.export(path)
            state = np.random.get_state()
            try:
                np.random.seed(7)
                reference, _ = trimesh.sample.sample_surface(mesh, 256)
            finally:
                np.random.set_state(state)

            first = reconstruction_chamfer(path, reference, seed=19)
            second = reconstruction_chamfer(path, reference, seed=19)
            self.assertTrue(np.isfinite(first))
            self.assertEqual(first, second)
            self.assertTrue(filter_threshold(path, reference, first, seed=19))

    def test_invalid_mesh_fails_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing"
            self.assertEqual(
                reconstruction_chamfer(missing, np.zeros((8, 3), dtype=np.float32)),
                float("inf"),
            )
            self.assertFalse(
                filter_threshold(
                    missing,
                    np.zeros((8, 3), dtype=np.float32),
                    threshold=0.005,
                )
            )


if __name__ == "__main__":
    unittest.main()
