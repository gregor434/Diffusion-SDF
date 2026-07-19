"""Dataset view with a configurable virtual length."""

from torch.utils.data import Dataset


class VirtualDataset(Dataset):
    """Cycle through a dataset without caching its returned samples."""

    def __init__(self, dataset, size):
        self.dataset = dataset
        self.size = int(size)
        if self.size <= 0:
            raise ValueError("virtual dataset size must be positive")
        if len(self.dataset) == 0:
            raise ValueError("cannot create a virtual view of an empty dataset")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]
