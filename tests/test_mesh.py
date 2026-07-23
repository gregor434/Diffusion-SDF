import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh

from utils import mesh


class MeshExtractionTests(unittest.TestCase):
    def test_boundary_crossing_surface_is_closed_by_exterior_padding(self):
        resolution = 32
        voxel_size = 2.0 / (resolution - 1)
        axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
        coordinates = np.stack(
            np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
        )
        center = np.asarray([0.8, 0.0, 0.0], dtype=np.float32)
        sdf = np.linalg.norm(coordinates - center, axis=-1) - 0.5

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "boundary-crossing.ply"
            mesh.convert_sdf_samples_to_ply(
                torch.from_numpy(sdf),
                voxel_grid_origin=[-1.0, -1.0, -1.0],
                voxel_size=voxel_size,
                ply_filename_out=str(output_path),
            )
            extracted = trimesh.load(output_path, process=False)

        self.assertTrue(extracted.is_watertight)
        self.assertGreater(extracted.bounds[1, 0], 1.0)
        self.assertLessEqual(extracted.bounds[1, 0], 1.0 + voxel_size)


if __name__ == "__main__":
    unittest.main()
