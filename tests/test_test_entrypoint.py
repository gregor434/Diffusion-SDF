import unittest

from dataloader.modulation_loader import extraction_split_name


class TestExtractionSplitSelection(unittest.TestCase):
    def test_defaults_to_modulation_split_when_configured(self):
        specs = {"TestSplit": "test.json", "ModulationSplit": "all.json"}

        self.assertEqual(extraction_split_name(specs), "ModulationSplit")

    def test_test_split_only_overrides_modulation_split(self):
        specs = {"TestSplit": "test.json", "ModulationSplit": "all.json"}

        self.assertEqual(
            extraction_split_name(specs, test_split_only=True),
            "TestSplit",
        )

    def test_defaults_to_test_split_without_modulation_split(self):
        self.assertEqual(
            extraction_split_name({"TestSplit": "test.json"}),
            "TestSplit",
        )


if __name__ == "__main__":
    unittest.main()
