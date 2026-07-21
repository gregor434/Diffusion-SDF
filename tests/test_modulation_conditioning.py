import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import dataloader.conditioning as conditioning
from dataloader.conditioning import ImageConditioning
from dataloader.modulation_loader import ModulationLoader


class StubClipModel:
    def __init__(self):
        self.calls = 0

    def encode_image(self, image):
        self.calls += 1
        return torch.arange(512, dtype=torch.float32).view(1, 512)


class ModulationConditioningTests(unittest.TestCase):
    def tearDown(self):
        conditioning._CLIP_CACHE.clear()

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
                self.assertEqual(item["latent"].shape, torch.Size([2, 3]))
                self.assertEqual(item["conditioning"]["image"].shape, torch.Size([1, 512]))
                self.assertEqual(
                    item["conditioning_paths"]["image"], str(image_dir / "main.jpg")
                )
                self.assertEqual(clip_model.calls, 1)

                cached_item = dataset[0]
                self.assertEqual(cached_item["conditioning"]["image"].shape, torch.Size([1, 512]))
                self.assertEqual(clip_model.calls, 1)

            cache_files = list((root / "images" / ".clip_cache").glob("**/*.pt"))
            self.assertEqual(len(cache_files), 1)

    def test_clip_model_auto_uses_cuda_when_available(self):
        load_devices = []

        class FakeClipModel:
            def eval(self):
                return self

            def parameters(self):
                return []

        def fake_load(model_name, device):
            load_devices.append(device)
            return FakeClipModel(), self.stub_preprocess

        fake_clip = types.SimpleNamespace(load=fake_load)
        with patch.dict(sys.modules, {"clip": fake_clip}):
            with patch("torch.cuda.is_available", return_value=True):
                conditioning.load_clip_model("ViT-B/32")

        self.assertEqual(load_devices, ["cuda"])

    def test_require_cached_image_conditioning_raises_on_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images" / "item0"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (320, 240), color=(128, 64, 32)).save(image_dir / "main.jpg")

            source = ImageConditioning(str(root / "images"), require_cached=True)

            with self.assertRaisesRegex(FileNotFoundError, "conditioning preparation"):
                source.load(self.record())

    def test_modulation_loader_validates_required_image_cache_on_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_latent(root)
            image_dir = root / "images" / "item0"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (320, 240), color=(128, 64, 32)).save(image_dir / "main.jpg")

            with self.assertRaisesRegex(FileNotFoundError, "conditioning preparation"):
                ModulationLoader(
                    str(root / "mods"),
                    split_file=self.split(),
                    conditioning={
                        "type": "image",
                        "path": str(root / "images"),
                        "require_cached": True,
                    },
                )

    def test_clip_cache_key_differs_by_model_name(self):
        source_a = ImageConditioning("tmp/images", clip_model="ViT-B/32")
        source_b = ImageConditioning("tmp/images", clip_model="RN50")

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
            points = np.random.randn(8, 3).astype(np.float32)
            np.savez(pc_dir / "cod_sdf.npz", surface_points=points)

            dataset = ModulationLoader(
                str(root / "mods"),
                split_file=self.split(),
                conditioning={"type": "point_cloud", "path": str(root / "pc"), "pc_size": 4},
            )

            item = dataset[0]
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
        np.savez(
            latent_dir / "modulation.npz",
            object_id=np.asarray("item0"),
            posterior_mean=np.arange(6, dtype=np.float32).reshape(2, 3),
        )


if __name__ == "__main__":
    unittest.main()
