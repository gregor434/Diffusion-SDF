import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TEST_MODULE_SPEC = importlib.util.spec_from_file_location(
    "diffusion_sdf_test_entrypoint",
    _REPOSITORY_ROOT / "test.py",
)
test = importlib.util.module_from_spec(_TEST_MODULE_SPEC)
sys.path.insert(0, str(_REPOSITORY_ROOT))
try:
    _TEST_MODULE_SPEC.loader.exec_module(test)
finally:
    sys.path.pop(0)


class _GenerationDataset(torch.utils.data.Dataset):
    image_path = None

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "latent": torch.zeros(2, 3),
            "dataset": "abo",
            "class_name": "ABO",
            "object_id": "item0",
            "conditioning": {"image": torch.ones(1, 512)},
            "conditioning_paths": {"image": str(self.image_path)},
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


class _CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Linear(2, 2)
        self.register_buffer("latent_mean", torch.zeros(1, 1, 2))
        self.register_buffer("latent_std", torch.ones(1, 1, 2))


class ImageConditionedGenerationTests(unittest.TestCase):
    def test_named_and_epoch_checkpoint_paths(self):
        self.assertEqual(test.checkpoint_path("experiment", "last"), "experiment/last.ckpt")
        self.assertEqual(test.checkpoint_path("experiment", "best"), "experiment/best.ckpt")
        self.assertEqual(
            test.checkpoint_path("experiment", "best-v2"),
            "experiment/best-v2.ckpt",
        )
        self.assertEqual(
            test.checkpoint_path("experiment", "best-v2.ckpt"),
            "experiment/best-v2.ckpt",
        )
        self.assertEqual(test.checkpoint_path("experiment", "99"), "experiment/epoch=99.ckpt")
        self.assertEqual(
            test.checkpoint_path("experiment", "1499-v1"),
            "experiment/epoch=1499-v1.ckpt",
        )

    def test_diffusion_checkpoint_restores_embedded_latent_statistics(self):
        model = _CheckpointModel()
        expected_mean = torch.tensor([[[-0.5, 0.25]]])
        expected_std = torch.tensor([[[0.8, 1.2]]])
        checkpoint = {
            "state_dict": {
                "diffusion_model.weight": torch.full_like(
                    model.diffusion_model.weight, 2.0
                ),
                "diffusion_model.bias": torch.full_like(
                    model.diffusion_model.bias, 3.0
                ),
                "latent_mean": expected_mean,
                "latent_std": expected_std,
            }
        }

        with patch.object(test.torch, "load", return_value=checkpoint):
            test.load_diffusion_checkpoint(model, "stage2.ckpt")

        torch.testing.assert_close(model.latent_mean, expected_mean)
        torch.testing.assert_close(model.latent_std, expected_std)
        torch.testing.assert_close(
            model.diffusion_model.weight,
            torch.full_like(model.diffusion_model.weight, 2.0),
        )
        torch.testing.assert_close(
            model.diffusion_model.bias,
            torch.full_like(model.diffusion_model.bias, 3.0),
        )

    def test_generation_passes_image_features_to_diffusion(self):
        model = _Model()
        specs = {"diffusion_model_specs": {"cond": True}}
        args = type("Args", (), {
            "num_samples": 2,
            "recon_resolution": 16,
            "max_batch": 32,
        })()

        with tempfile.TemporaryDirectory() as tmpdir:
            _GenerationDataset.image_path = Path(tmpdir) / "source.jpg"
            Image.new("RGB", (8, 8), color=(10, 20, 30)).save(
                _GenerationDataset.image_path
            )
            with patch.object(
                test, "load_generation_models", return_value=(model, _SdfModel())
            ):
                with patch.object(
                    test, "make_generation_dataset", return_value=_GenerationDataset()
                ):
                    with patch.object(test.mesh, "create_mesh") as create_mesh:
                        with patch.object(test.evaluate, "mesh_validity", return_value=1.0):
                            test.generate(
                                specs, args, Path(tmpdir), torch.device("cpu")
                            )

            copied_image = Path(tmpdir) / "ABO" / "item0" / "input_image.jpg"
            self.assertTrue(copied_image.is_file())
            self.assertEqual(
                copied_image.read_bytes(),
                _GenerationDataset.image_path.read_bytes(),
            )

        self.assertEqual(
            model.diffusion_model.conditioning["image"].shape,
            torch.Size([1, 1, 512]),
        )
        self.assertEqual(create_mesh.call_count, 2)


if __name__ == "__main__":
    unittest.main()
