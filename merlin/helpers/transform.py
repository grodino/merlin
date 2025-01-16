from typing import Optional, Tuple
from torchvision import transforms


def make_transformation(input_shape, dataset_meanstd: Optional[Tuple[Tuple[float], Tuple[float]]]=None):
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
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    else:
        normalization = transforms.Normalize(mean=dataset_meanstd[0], std=dataset_meanstd[1])
    transform = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor(),
        normalization,
    ])
    return transform