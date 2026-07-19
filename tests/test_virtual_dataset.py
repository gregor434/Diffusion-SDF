import unittest

from torch.utils.data import Dataset

from dataloader.virtual_dataset import VirtualDataset


class RecordingDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.calls = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.calls.append(index)
        return {"index": index, "access": len(self.calls)}


class VirtualDatasetTests(unittest.TestCase):
    def test_reports_requested_size_and_cycles_indices(self):
        base = RecordingDataset(2)
        dataset = VirtualDataset(base, 5)

        items = [dataset[index] for index in range(len(dataset))]

        self.assertEqual(len(dataset), 5)
        self.assertEqual([item["index"] for item in items], [0, 1, 0, 1, 0])

    def test_repeated_indices_are_fetched_again(self):
        base = RecordingDataset(1)
        dataset = VirtualDataset(base, 2)

        first = dataset[0]
        second = dataset[1]

        self.assertEqual(base.calls, [0, 0])
        self.assertNotEqual(first["access"], second["access"])

    def test_rejects_non_positive_size(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            VirtualDataset(RecordingDataset(1), 0)

    def test_rejects_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "empty dataset"):
            VirtualDataset(RecordingDataset(0), 100)


if __name__ == "__main__":
    unittest.main()
