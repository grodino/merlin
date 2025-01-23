from torchvision import transforms
from .resnet import ResNet18
from .lenet import LeNet


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


MODEL_ARCHITECTURE_FACTORY = {"resnet18": ResNet18, "lenet": LeNet}
MODEL_INPUT_TRANSFORMATION_FACTORY = {
    "resnet18": lambda meanstd=None: make_transformation((224, 224), meanstd),
    "lenet": lambda meanstd=None: make_transformation((32, 32), meanstd),
}
