import io

import torch

from torchvision import transforms
from PIL import Image


def transform_images_to_tensors(X_train):
    transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
    ])

    def process_image(image_dict):
        # Convert bytes to an image and apply the transform
        image = Image.open(io.BytesIO(image_dict["bytes"]))
        return transform(image)
    
    return torch.stack([process_image(image) for image in X_train])
