import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from train import load_weights_only


class WeightsOnlyInitializationTests(unittest.TestCase):
    def test_loads_state_dict_without_reusing_optimizer_learning_rate(self):
        source = nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "source.ckpt"
            torch.save(
                {
                    "state_dict": source.state_dict(),
                    "optimizer_states": [source_optimizer.state_dict()],
                    "epoch": 123,
                    "global_step": 456,
                },
                checkpoint_path,
            )

            target = nn.Linear(3, 2)
            load_weights_only(target, checkpoint_path)
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-4)

        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            torch.testing.assert_close(source_parameter, target_parameter)
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 1e-4)

    def test_rejects_checkpoint_without_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "raw.pt"
            torch.save({"weight": torch.ones(1)}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "state_dict"):
                load_weights_only(nn.Linear(1, 1), checkpoint_path)

    def test_only_explicit_new_parameters_may_be_missing(self):
        class Source(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(3, 2)

        class Target(Source):
            def __init__(self):
                super().__init__()
                self.refiner = nn.Linear(2, 2)

        source = Source()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "source.ckpt"
            torch.save({"state_dict": source.state_dict()}, checkpoint_path)
            target = Target()
            load_weights_only(
                target,
                checkpoint_path,
                allowed_missing_prefixes=("refiner.",),
            )
            torch.testing.assert_close(target.base.weight, source.base.weight)
            torch.testing.assert_close(target.base.bias, source.base.bias)

            with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
                load_weights_only(
                    Target(),
                    checkpoint_path,
                    allowed_missing_prefixes=("unrelated.",),
                )


if __name__ == "__main__":
    unittest.main()
