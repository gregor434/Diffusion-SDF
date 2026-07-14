import unittest

import torch

from models.archs.condition_encoders import ConditionEncoderSet
from models.archs.diffusion_arch import DiffusionNet
from models.combined_model import CombinedModel


class ConditionEncoderSetTests(unittest.TestCase):
    def test_point_cloud_condition_returns_point_tokens(self):
        encoder = ConditionEncoderSet(
            condition_dim=6,
            condition_encoders=[{"type": "point_cloud", "plane_resolution": 8, "unet": False}],
        )

        tokens = encoder({"point_cloud": torch.randn(2, 5, 3)})

        self.assertEqual(tokens.shape, torch.Size([2, 5, 6]))

    def test_image_condition_projects_clip_token(self):
        encoder = ConditionEncoderSet(
            condition_dim=7,
            condition_encoders=[{"type": "image"}],
        )

        tokens = encoder({"image": torch.randn(2, 1, 512)})

        self.assertEqual(tokens.shape, torch.Size([2, 1, 7]))

    def test_image_condition_uses_identity_when_dims_match(self):
        encoder = ConditionEncoderSet(
            condition_dim=512,
            condition_encoders=[{"type": "image"}],
        )

        image_features = torch.randn(2, 1, 512)
        tokens = encoder({"image": image_features})

        self.assertEqual(tokens.shape, torch.Size([2, 1, 512]))
        self.assertTrue(torch.equal(tokens, image_features))

    def test_multiple_modalities_concatenate_tokens(self):
        encoder = ConditionEncoderSet(
            condition_dim=5,
            condition_encoders=[
                {"type": "point_cloud", "plane_resolution": 8, "unet": False},
                {"type": "image"},
            ],
        )

        tokens = encoder({
            "point_cloud": torch.randn(2, 3, 3),
            "image": torch.randn(2, 1, 512),
        })

        self.assertEqual(tokens.shape, torch.Size([2, 4, 5]))


class DiffusionConditioningSmokeTests(unittest.TestCase):
    def test_unconditional_diffusion_net_accepts_latent_only(self):
        model = self.make_net(cond=False)

        out = model(torch.randn(2, 16), torch.tensor([0, 1]))

        self.assertEqual(out.shape, torch.Size([2, 16]))

    def test_point_cloud_conditional_diffusion_net_accepts_legacy_tuple(self):
        model = self.make_net(
            cond=True,
            condition_encoders=[{"type": "point_cloud", "plane_resolution": 8, "unet": False}],
        )

        out = model((torch.randn(2, 16), torch.randn(2, 4, 3)), torch.tensor([0, 1]))

        self.assertEqual(out.shape, torch.Size([2, 16]))

    def test_image_conditional_diffusion_net_accepts_conditioning_dict(self):
        model = self.make_net(
            cond=True,
            condition_encoders=[{"type": "image"}],
        )

        out = model(
            (torch.randn(2, 16), {"image": torch.randn(2, 1, 512)}),
            torch.tensor([0, 1]),
        )

        self.assertEqual(out.shape, torch.Size([2, 16]))

    def test_diffusion_training_step_accepts_image_conditioning(self):
        specs = {
            "training_task": "diffusion",
            "diff_lr": 1e-4,
            "diffusion_specs": {
                "timesteps": 4,
                "objective": "pred_x0",
                "loss_type": "l2",
                "perturb_pc": None,
                "sample_pc_size": 4,
            },
            "diffusion_model_specs": {
                "dim": 16,
                "dim_in_out": 16,
                "depth": 0,
                "num_timesteps": 4,
                "cond": True,
                "cross_attn": True,
                "cond_dropout": False,
                "condition_dim": 8,
                "dim_head": 4,
                "heads": 2,
                "condition_encoders": [{"type": "image"}],
            },
        }
        model = CombinedModel(specs)
        batch = {
            "latent": torch.randn(2, 16),
            "conditioning": {"image": torch.randn(2, 1, 512)},
        }

        loss = model.train_diffusion(batch)

        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    @staticmethod
    def make_net(cond, condition_encoders=None):
        return DiffusionNet(
            dim=16,
            dim_in_out=16,
            depth=0,
            num_timesteps=4,
            cond=cond,
            cross_attn=cond,
            cond_dropout=False,
            condition_dim=8,
            dim_head=4,
            heads=2,
            condition_encoders=condition_encoders,
        )


if __name__ == "__main__":
    unittest.main()
