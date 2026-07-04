import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_abo_dataset import (
    build_split_sets,
    compute_split_counts,
    write_split_manifests,
)


class PrepareAboDatasetSplitTests(unittest.TestCase):
    def test_compute_split_counts_preserves_validation_split(self) -> None:
        self.assertEqual(compute_split_counts(10, 0.8), (8, 2))
        self.assertEqual(compute_split_counts(4, 0.8), (3, 1))
        self.assertEqual(compute_split_counts(3, 0.8), (2, 1))

    def test_build_split_sets_balances_aggregate_classes(self) -> None:
        type_to_ids = {
            "chair": [f"chair_{idx}" for idx in range(6)],
            "lamp": [f"lamp_{idx}" for idx in range(4)],
            "table": [f"table_{idx}" for idx in range(5)],
        }

        all_splits, per_type_splits = build_split_sets(type_to_ids, 0.8, seed=7)

        self.assertEqual(len(all_splits["all"]), 12)
        self.assertEqual(len(all_splits["train"]), 9)
        self.assertEqual(len(all_splits["val"]), 3)

        for split_name, expected_per_class in (("train", 3), ("val", 1)):
            counts = {}
            for model_id in all_splits[split_name]:
                class_name = model_id.split("_", 1)[0]
                counts[class_name] = counts.get(class_name, 0) + 1
            self.assertEqual(counts, {"chair": expected_per_class, "lamp": expected_per_class, "table": expected_per_class})

        self.assertEqual(len(per_type_splits["chair"]["train"]), 4)
        self.assertEqual(len(per_type_splits["chair"]["val"]), 2)
        self.assertTrue(set(per_type_splits["chair"]["train"]).isdisjoint(per_type_splits["chair"]["val"]))
        self.assertEqual(
            sorted(
                per_type_splits["chair"]["train"]
                + per_type_splits["chair"]["val"]
            ),
            per_type_splits["chair"]["all"],
        )

    def test_build_split_sets_is_deterministic(self) -> None:
        type_to_ids = {
            "chair": [f"chair_{idx}" for idx in range(6)],
            "lamp": [f"lamp_{idx}" for idx in range(4)],
        }

        first = build_split_sets(type_to_ids, 0.8, seed=3)
        second = build_split_sets(type_to_ids, 0.8, seed=3)

        self.assertEqual(first, second)

    def test_build_split_sets_rejects_singleton_classes(self) -> None:
        type_to_ids = {
            "chair": [f"chair_{idx}" for idx in range(6)],
            "lamp": ["lamp_0"],
        }

        with self.assertRaises(ValueError):
            build_split_sets(type_to_ids, 0.8, seed=0)

    def test_write_split_manifests_records_nested_paths(self) -> None:
        all_splits = {
            "all": ["chair_0", "lamp_0"],
            "train": ["chair_0"],
            "val": ["lamp_0"],
        }
        per_type_splits = {
            "chair": {"all": ["chair_0"], "train": ["chair_0"], "val": []},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            split_paths = write_split_manifests(
                splits_dir=Path(tmpdir),
                split_prefix="abo",
                dataset_key="abo",
                class_name="ABO",
                all_splits=all_splits,
                per_type_splits=per_type_splits,
            )

            train_manifest = Path(split_paths["all"]["train"])
            self.assertTrue(train_manifest.is_file())
            self.assertEqual(
                json.loads(train_manifest.read_text(encoding="utf-8")),
                {"abo": {"ABO": ["chair_0"]}},
            )
            self.assertIn("chair", split_paths["by_product_type"])
            self.assertIn("val", split_paths["by_product_type"]["chair"])


if __name__ == "__main__":
    unittest.main()
