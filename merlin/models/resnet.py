from torch import nn
from torchvision.models import resnet18
import torchvision.models

def make_resnet18(num_classes: int, **model_kwargs) -> torchvision.models.resnet.ResNet:
    """
    Creates a ResNet-18 model with a modified fully connected layer to match the number of classes.

    Args:
        num_classes (int): The number of classes for the output layer.
        **model_kwargs: Additional keyword arguments to pass to the ResNet-18 model constructor.

    Returns:
        torchvision.models.resnet.ResNet: A ResNet-18 model with the specified number of output classes.
    """
    model = resnet18(**model_kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model