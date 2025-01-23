from torch import nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    """
    A custom implementation of the ResNet-18 architecture for classification tasks.

    Args:
        num_classes (int): The number of output classes for the classification task.
        **model_kwargs: Additional keyword arguments to pass to the resnet18 model.
    """

    def __init__(self, num_classes: int, **model_kwargs):
        super().__init__()
        self.model = resnet18(**model_kwargs)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
