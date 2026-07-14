import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from dataloader.conditioning import ImageConditioning
from dataloader.modulation_loader import ModulationLoader


class StubClipModel:
    def __init__(self):
        self.calls = 0

    def encode_image(self, image):
        self.calls += 1
        return torch.arange(512, dtype=torch.float32).view(1, 512)


class ModulationConditioningTests(unittest.TestCase):
    def test_loads_cached_clip_image_conditioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_latent(root)
            image_dir = root / "images" / "item0"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (320, 240), color=(128, 64, 32)).save(image_dir / "main.jpg")
            clip_model = StubClipModel()

            with patch(
                "dataloader.conditioning.load_clip_model",
                return_value=(clip_model, self.stub_preprocess, "cpu"),
            ):
                dataset = ModulationLoader(
                    str(root / "mods"),
                    split_file=self.split(),
                    conditioning={"type": "image", "path": str(root / "images")},
                )

                item = dataset[0]
                self.assertEqual(item["latent"].shape, torch.Size([4]))
                self.assertEqual(item["conditioning"]["image"].shape, torch.Size([1, 512]))
                self.assertEqual(clip_model.calls, 1)

                cached_item = dataset[0]
                self.assertEqual(cached_item["conditioning"]["image"].shape, torch.Size([1, 512]))
                self.assertEqual(clip_model.calls, 1)

            cache_files = list((root / "images" / ".clip_cache").glob("**/*.pt"))
            self.assertEqual(len(cache_files), 1)

    def test_clip_cache_key_differs_by_model_name(self):
        source_a = ImageConditioning("/tmp/images", clip_model="ViT-B/32")
        source_b = ImageConditioning("/tmp/images", clip_model="RN50")

        self.assertNotEqual(
            source_a.resolve_cache_path(self.record()),
            source_b.resolve_cache_path(self.record()),
        )

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
    def record():
        return {"dataset": "abo", "class_name": "ABO", "instance_name": "item0"}

    @staticmethod
    def stub_preprocess(image):
        return torch.zeros(3, 224, 224)

    @staticmethod
    def write_latent(root):
        latent_dir = root / "mods" / "ABO" / "item0"
        latent_dir.mkdir(parents=True)
        np.savetxt(latent_dir / "latent.txt", np.arange(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
