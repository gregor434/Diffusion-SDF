import unittest

import torch

from models.archs.condition_encoders import ConditionEncoderSet
from models.combined_model import CombinedModel
from models.diffusion import CODLatentTransformer, EDMLatentDiffusion


class ConditionEncoderSetTests(unittest.TestCase):
    def test_point_cloud_condition_returns_point_tokens(self):
        encoder = ConditionEncoderSet(condition_dim=6, condition_encoders=[{"type": "point_cloud"}])
        self.assertEqual(
            encoder({"point_cloud": torch.randn(2, 5, 3)}).shape,
            torch.Size([2, 5, 6]),
        )

    def test_multiple_modalities_concatenate_tokens(self):
        encoder = ConditionEncoderSet(
            condition_dim=5,
            condition_encoders=[{"type": "point_cloud"}, {"type": "image"}],
        )
        tokens = encoder({
            "point_cloud": torch.randn(2, 3, 3),
            "image": torch.randn(2, 1, 512),
        })
        self.assertEqual(tokens.shape, torch.Size([2, 4, 5]))


class CODDiffusionSmokeTests(unittest.TestCase):
    def test_transformer_preserves_native_cod_shape(self):
        model = CODLatentTransformer(
            latent_dimension=3, width=16, depth=1, heads=4, cond=False
        )
        output = model(torch.randn(2, 4, 3), torch.randn(2))
        self.assertEqual(output.shape, torch.Size([2, 4, 3]))

    def test_conditional_transformer_accepts_image_tokens(self):
        model = CODLatentTransformer(
            latent_dimension=3,
            width=16,
            depth=1,
            heads=4,
            cond=True,
            condition_dim=8,
            condition_encoders=[{"type": "image"}],
        )
        output = model(
            torch.randn(2, 4, 3),
            torch.randn(2),
            {"image": torch.randn(2, 1, 512)},
        )
        self.assertEqual(output.shape, torch.Size([2, 4, 3]))

    def test_edm_loss_and_sampler_preserve_token_shape(self):
        diffusion = EDMLatentDiffusion(
            {
                "latent_tokens": 4,
                "latent_dimension": 3,
                "width": 16,
                "depth": 1,
                "heads": 4,
            },
            {"sampling_steps": 2, "sigma_max": 1.0},
        )
        loss, estimate, _, _ = diffusion.training_loss(torch.randn(2, 4, 3))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(estimate.shape, torch.Size([2, 4, 3]))
        self.assertEqual(diffusion.sample(2).shape, torch.Size([2, 4, 3]))

    def test_combined_stage_two_uses_cached_cod_tokens(self):
        specs = {
            "training_task": "diffusion",
            "diffusion_specs": {"sampling_steps": 2},
            "diffusion_model_specs": {
                "latent_tokens": 4,
                "latent_dimension": 3,
                "width": 16,
                "depth": 1,
                "heads": 4,
                "cond": True,
                "condition_dim": 8,
                "condition_encoders": [{"type": "image"}],
            },
        }
        model = CombinedModel(specs)
        losses = model.stage2_losses({
            "latent": torch.randn(2, 4, 3),
            "conditioning": {"image": torch.randn(2, 1, 512)},
        })
        self.assertTrue(torch.isfinite(losses["loss"]))


if __name__ == "__main__":
    unittest.main()
