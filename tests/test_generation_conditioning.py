import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import test


class _GenerationDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "latent": torch.zeros(2, 3),
            "dataset": "abo",
            "class_name": "ABO",
            "object_id": "item0",
            "conditioning": {"image": torch.ones(1, 512)},
        }


class _Diffusion:
    def __init__(self):
        self.conditioning = None

    def sample(self, batch_size, conditioning=None):
        self.conditioning = conditioning
        return torch.zeros(batch_size, 2, 3)


class _Model:
    def __init__(self):
        self.diffusion_model = _Diffusion()

    @staticmethod
    def denormalize_latent(latent):
        return latent


class _SdfModel:
    @staticmethod
    def decode_latent(latent):
        return {"planes": torch.zeros(len(latent), 1)}


class ImageConditionedGenerationTests(unittest.TestCase):
    def test_named_and_epoch_checkpoint_paths(self):
        self.assertEqual(test.checkpoint_path("experiment", "last"), "experiment/last.ckpt")
        self.assertEqual(test.checkpoint_path("experiment", "best"), "experiment/best.ckpt")
        self.assertEqual(test.checkpoint_path("experiment", "99"), "experiment/epoch=99.ckpt")

    def test_generation_passes_image_features_to_diffusion(self):
        model = _Model()
        specs = {"diffusion_model_specs": {"cond": True}}
        args = type("Args", (), {
            "num_samples": 2,
            "recon_resolution": 16,
            "max_batch": 32,
        })()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("test.load_generation_models", return_value=(model, _SdfModel())):
                with patch("test.make_generation_dataset", return_value=_GenerationDataset()):
                    with patch("test.mesh.create_mesh") as create_mesh:
                        with patch("test.evaluate.mesh_validity", return_value=1.0):
                            test.generate(
                                specs, args, Path(tmpdir), torch.device("cpu")
                            )

        self.assertEqual(
            model.diffusion_model.conditioning["image"].shape,
            torch.Size([1, 1, 512]),
        )
        self.assertEqual(create_mesh.call_count, 2)


if __name__ == "__main__":
    unittest.main()
