from torch.utils.data import DataLoader


def load_whole_dataset(dataset):
    """
    Loads the entire dataset into memory.

    Args:
        dataset (Dataset): The dataset to be loaded.

    Returns:
        Tensor: A tensor containing all the data from the dataset.
    """

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return next(iter(dataloader))
