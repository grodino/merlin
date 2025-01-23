from torch.utils.data import DataLoader
from torchvision import transforms


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


def make_transformation(
    input_shape, dataset_meanstd: tuple[tuple[float], tuple[float]] | None = None
):
    """
    Creates a transformation pipeline for image data.

    Parameters:
    input_shape (tuple): The shape of the input images.
    dataset_meanstd (Optional[Tuple[Tuple[float], Tuple[float]]]): A tuple containing the mean and standard deviation
                                                                for normalization. If None, default values are used.

    Returns:
    torchvision.transforms.Compose: A composed transformation pipeline.
    """
    if dataset_meanstd is None:
        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalization = transforms.Normalize(
            mean=dataset_meanstd[0], std=dataset_meanstd[1]
        )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(input_shape),
            normalization,
        ]
    )
    return transform
