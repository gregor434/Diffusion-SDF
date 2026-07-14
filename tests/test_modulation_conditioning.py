import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataloader.modulation_loader import ModulationLoader


class ModulationConditioningTests(unittest.TestCase):
    def test_loads_image_conditioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_latent(root)
            image_dir = root / "images" / "item0"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (320, 240), color=(128, 64, 32)).save(image_dir / "main.jpg")

            dataset = ModulationLoader(
                str(root / "mods"),
                split_file=self.split(),
                conditioning={"type": "image", "path": str(root / "images")},
            )

            item = dataset[0]
            self.assertEqual(item["latent"].shape, torch.Size([4]))
            self.assertEqual(item["conditioning"]["image"].shape, torch.Size([3, 224, 224]))

    def test_loads_point_cloud_conditioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_latent(root)
            pc_dir = root / "pc" / "abo" / "ABO" / "item0"
            pc_dir.mkdir(parents=True)
            points = np.zeros((8, 4), dtype=np.float32)
            points[:, :3] = np.random.randn(8, 3).astype(np.float32)
            np.savetxt(pc_dir / "sdf_data.csv", points, delimiter=",")

            dataset = ModulationLoader(
                str(root / "mods"),
                split_file=self.split(),
                conditioning={"type": "point_cloud", "path": str(root / "pc"), "pc_size": 4},
            )

            item = dataset[0]
            self.assertEqual(item["point_cloud"].shape, torch.Size([4, 3]))
            self.assertEqual(item["conditioning"]["point_cloud"].shape, torch.Size([4, 3]))

    @staticmethod
    def split():
        return {"abo": {"ABO": ["item0"]}}

    @staticmethod
    def write_latent(root):
        latent_dir = root / "mods" / "ABO" / "item0"
        latent_dir.mkdir(parents=True)
        np.savetxt(latent_dir / "latent.txt", np.arange(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
