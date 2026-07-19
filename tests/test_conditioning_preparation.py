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
import train
from dataloader.conditioning import ImageConditioning, PointCloudConditioning
from dataloader.modulation_loader import ModulationLoader


class ConditioningPreparationTests(unittest.TestCase):
    def tearDown(self):
        conditioning._CLIP_CACHE.clear()

    def test_warm_image_cache_does_not_load_clip_during_prepare_or_getitem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_latent(root)
            self.write_image(root)
            source = ImageConditioning(str(root / "images"), require_cached=True)
            expected = torch.arange(512, dtype=torch.float32).view(1, 512)
            cache_path = Path(source.resolve_cache_path(self.record()))
            cache_path.parent.mkdir(parents=True)
            torch.save(expected, cache_path)

            with patch("dataloader.conditioning.load_clip_model") as load_clip_model:
                source.prepare([self.record()])
                dataset = ModulationLoader(
                    str(root / "mods"),
                    split_file=self.split(),
                    conditioning_sources=[source],
                )
                item = dataset[0]

            load_clip_model.assert_not_called()
            torch.testing.assert_close(item["conditioning"]["image"], expected)

    def test_cold_image_cache_prepares_then_getitem_reads_cached_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_latent(root)
            self.write_image(root)
            clip_model = StubClipModel()
            source = ImageConditioning(str(root / "images"), require_cached=True)
            records = ModulationLoader.build_records(str(root / "mods"), self.split(), [source])

            with patch(
                "dataloader.conditioning.load_clip_model",
                return_value=(clip_model, self.stub_preprocess, "cpu"),
            ) as load_clip_model:
                source.prepare(records)
                self.assertEqual(clip_model.calls, 1)
                load_clip_model.reset_mock()

                dataset = ModulationLoader(
                    str(root / "mods"),
                    split_file=self.split(),
                    conditioning_sources=[source],
                    records=records,
                )
                item = dataset[0]

            load_clip_model.assert_not_called()
            self.assertEqual(item["conditioning"]["image"].shape, torch.Size([1, 512]))
            self.assertEqual(clip_model.calls, 1)

    def test_cuda_image_cache_preparation_releases_clip_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_image(root)
            source = ImageConditioning(str(root / "images"), clip_device="auto")

            fake_clip = types.SimpleNamespace(
                load=lambda model_name, device: (FakeClipModel(), self.stub_preprocess)
            )

            with patch.dict(sys.modules, {"clip": fake_clip}):
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.empty_cache") as empty_cache:
                        with patch.object(
                            source,
                            "encode_image",
                            return_value=torch.arange(512, dtype=torch.float32).view(1, 512),
                        ):
                            self.assertTrue(source.prepare([self.record()]))

            self.assertNotIn((source.clip_model, "cuda"), conditioning._CLIP_CACHE)
            empty_cache.assert_called_once()

    def test_point_cloud_prepare_is_noop_and_loading_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_latent(root)
            pc_dir = root / "pc" / "abo" / "ABO" / "item0"
            pc_dir.mkdir(parents=True)
            points = np.random.randn(8, 3).astype(np.float32)
            np.savez(pc_dir / "cod_sdf.npz", surface_points=points)

            source = PointCloudConditioning(str(root / "pc"), pc_size=4)
            self.assertFalse(source.prepare([self.record()]))

            dataset = ModulationLoader(
                str(root / "mods"),
                split_file=self.split(),
                conditioning_sources=[source],
            )
            item = dataset[0]

            self.assertEqual(item["conditioning"]["point_cloud"].shape, torch.Size([4, 3]))

    def test_train_dataloader_uses_spawn_only_after_cuda_preparation_with_workers(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(4))
        old_args = getattr(train, "args", None)
        train.args = types.SimpleNamespace(batch_size=2, workers=2)
        try:
            dataloader = train.build_dataloader(
                dataset,
                drop_last=False,
                shuffle=False,
                use_spawn_workers=True,
            )
            self.assertEqual(dataloader.multiprocessing_context.get_start_method(), "spawn")

            dataloader = train.build_dataloader(
                dataset,
                drop_last=False,
                shuffle=False,
                use_spawn_workers=False,
            )
            self.assertIsNone(dataloader.multiprocessing_context)
        finally:
            if old_args is None:
                delattr(train, "args")
            else:
                train.args = old_args

    def test_train_prepare_conditioning_reports_cuda_backed_preparation(self):
        source = StubConditioningSource(prepared_with_cuda=True)

        self.assertTrue(train.prepare_conditioning_sources([source], [self.record()]))
        self.assertEqual(source.prepare_calls, [([self.record()], False)])

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
    def write_image(root):
        image_dir = root / "images" / "item0"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (320, 240), color=(128, 64, 32)).save(image_dir / "main.jpg")

    @staticmethod
    def write_latent(root):
        latent_dir = root / "mods" / "ABO" / "item0"
        latent_dir.mkdir(parents=True)
        np.savez(
            latent_dir / "modulation.npz",
            object_id=np.asarray("item0"),
            posterior_mean=np.arange(6, dtype=np.float32).reshape(2, 3),
        )


class StubClipModel:
    def __init__(self):
        self.calls = 0

    def encode_image(self, image):
        self.calls += 1
        return torch.arange(512, dtype=torch.float32).view(1, 512)


class FakeClipModel:
    def eval(self):
        return self

    def parameters(self):
        return []


class StubConditioningSource:
    def __init__(self, prepared_with_cuda):
        self.prepared_with_cuda = prepared_with_cuda
        self.prepare_calls = []

    def prepare(self, records, force=False):
        self.prepare_calls.append((list(records), force))


if __name__ == "__main__":
    unittest.main()
