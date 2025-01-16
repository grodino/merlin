from .resnet import ResNet18
from .lenet import LeNet

MODEL_ARCHITECTURE_FACTORY = {
    "resnet18": ResNet18,
    "lenet": LeNet
}