from .resnet import ResNet18
from .lenet import LeNet

from merlin.helpers.transform import make_transformation

MODEL_ARCHITECTURE_FACTORY = {
    "resnet18": ResNet18,
    "lenet": LeNet
}

MODEL_INPUT_TRANSFORMATION_FACTORY = {
    "resnet18": lambda meanstd=None: make_transformation((224, 224), meanstd),
    "lenet": lambda meanstd=None: make_transformation((32, 32), meanstd)
}