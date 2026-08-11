import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from scripts import train_cod099_stage1_pipeline as pipeline


class Cod099Stage1PipelineTests(unittest.TestCase):
    def test_default_command_uses_requested_batch_size_and_workers(self):
        args = pipeline.parse_args([])
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.workers, 8)
        command = pipeline.train_command(Path("experiment"))
        self.assertEqual(command[command.index("--batch_size") + 1], "16")
        self.assertEqual(command[command.index("--workers") + 1], "8")
        self.assertNotIn("--resume", command)

    def test_resume_command_uses_last_checkpoint(self):
        command = pipeline.train_command(
            Path("experiment"), batch_size=4, workers=2, resume=True
        )
        self.assertEqual(command[-2:], ["--resume", "last"])

    def test_fresh_stage_rejects_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "best.ckpt").touch()
            with self.assertRaisesRegex(FileExistsError, "already contains"):
                pipeline.prepare_experiment(path, "bootstrap", False)

    def test_quality_gate_reads_lightning_best_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.ckpt"
            torch.save(
                {
                    "callbacks": {
                        "best": {
                            "monitor": "val/loss",
                            "best_model_score": torch.tensor(0.045),
                        }
                    }
                },
                path,
            )
            pipeline.require_quality(path, "bootstrap")
            with self.assertRaisesRegex(RuntimeError, "quality gate"):
                pipeline.require_quality(path, "polish")

    def test_dry_run_prints_all_stages_in_order(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pipeline.main(["--dry-run"])
        text = output.getvalue()
        bootstrap = text.index("SDF-head bootstrap")
        polish = text.index("SDF-head polish")
        refinement = text.index("SDF-head and convolutional refinement")
        self.assertLess(bootstrap, polish)
        self.assertLess(polish, refinement)
        self.assertEqual(text.count("--batch_size 16"), 3)
        self.assertEqual(text.count("--workers 8"), 3)


if __name__ == "__main__":
    unittest.main()
