#!/usr/bin/env python3
import json
import os
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import typer

from merlin.datasets import CelebADataset
from merlin.models.torch import (
    MODEL_ARCHITECTURE_FACTORY,
    MODEL_INPUT_TRANSFORMATION_FACTORY,
)


app = typer.Typer()


def eval_accuracy(model, eval_loader, criterion, device):
    """
    Evaluate the accuracy and loss of a given model on a provided evaluation data loader.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        eval_loader (torch.utils.data.DataLoader): DataLoader containing the evaluation dataset.
        criterion (torch.nn.Module): Loss function used to compute the evaluation loss.
    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy of the model on the evaluation dataset.
            - eval_loss (float): The average loss of the model on the evaluation dataset.
    """
    correct, total = 0, 0
    eval_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(eval_loader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y[0].to(device)
            pred = model(x.to(device)).argmax(dim=1)
            correct += (pred == y.to(device)).sum().item()
            total += y.size(0)

            loss = criterion(model(x), y).item()
            eval_loss += loss * x.size(0)
    eval_loss = eval_loss / total
    accuracy = correct / total
    return accuracy, eval_loss


def train_model(
    model_name: str,
    model,
    optimizer,
    criterion,
    train_loader,
    validation_loader,
    device,
    num_epochs=2,
    feature="Smiling",
) -> Tuple[float, float]:
    """
    Trains a given model using the specified optimizer and loss criterion.

    Args:
        model_name: The name of the model being trained.
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        criterion (torch.nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        validation_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        num_epochs (int, optional): Number of epochs to train the model. Default is 10.

    Returns:
        None
    """
    best_model_save_dir = os.path.join("data", "models")
    if not os.path.exists(best_model_save_dir):
        os.makedirs(best_model_save_dir)
    best_model_save_path = os.path.join(
        best_model_save_dir, "%s_celeba_%s.pth" % (model_name, feature)
    )

    best_acc = -1
    best_loss = -1
    for epoch in range(num_epochs):
        model.train()
        step = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x = x.to(device)
            y = y[0].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            step += 1
            if step % 500 == 0:
                val_acc, val_loss = eval_accuracy(
                    model, validation_loader, criterion, device
                )
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_loss = val_loss
                    if best_model_save_path is not None:
                        print(
                            f"Saving best model with accuracy: {best_acc} (Validation Loss: {val_loss})"
                        )
                        torch.save(model.state_dict(), best_model_save_path)
                print(
                    f"Step {step+1}: Validation Accuracy: {val_acc}, Validation Loss: {val_loss}"
                )

    return best_acc, best_loss


def optimal_device() -> torch.device:
    """
    Determines the optimal device for PyTorch operations.

    Returns:
        torch.device: The optimal device for computation. It returns a CUDA device if available,
                      otherwise it checks for an MPS (Metal Performance Shaders) device, and if neither
                      are available, it defaults to the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_dataset(model_name: str, feature: str = "Smiling"):
    meanstd = None
    transformation_factory = MODEL_INPUT_TRANSFORMATION_FACTORY[model_name]
    transformation = transformation_factory(meanstd)

    train_dataset = CelebADataset(
        split="train", target_columns=[feature], transform=transformation
    )
    val_dataset = CelebADataset(
        split="val", target_columns=[feature], transform=transformation
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=6,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=6,
    )

    return train_dataloader, val_dataloader


def load_training_status() -> Dict:
    """
    Loads the training status from the training status file.

    Returns:
        Dict: A dictionary containing the training status.
    """
    training_status_file = os.path.join("data", "training_status.json")
    if os.path.exists(training_status_file):
        with open(training_status_file, "r") as f:
            training_status = json.load(f)
    else:
        training_status = {}
    return training_status


def save_training_status(training_status: Dict):
    """
    Saves the training status to the training status file.

    Args:
        training_status (Dict): A dictionary containing the training status.

    Returns:
        None
    """
    training_status_file = os.path.join("data", "training_status.json")
    with open(training_status_file, "w") as f:
        json.dump(training_status, f)


@app.command()
def train(model_name: str, train_all_features: bool = False):
    assert model_name in [
        "lenet",
        "resnet18",
    ], "Model name must be either 'lenet' or 'resnet18'"
    device = optimal_device()
    criterion = torch.nn.CrossEntropyLoss()
    training_status: Dict = load_training_status()
    if model_name not in training_status:
        training_status[model_name] = {}

    training_status_model = training_status[model_name]

    features_to_train = ["Smiling"]
    if train_all_features:
        features_to_train = CelebADataset.TRAINING_TARGETS

    for feature in features_to_train:
        if feature in training_status_model:
            print(
                f"Model {model_name} for feature: {feature} already trained. Skipping..."
            )
            continue

        print(f"Training model for feature: {feature}")
        train_loader, validation_loader = load_dataset(model_name, feature=feature)

        architecture_factory = MODEL_ARCHITECTURE_FACTORY[model_name]
        model = architecture_factory(num_classes=2)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        val_acc, val_loss = train_model(
            model_name,
            model,
            optimizer,
            criterion,
            train_loader,
            validation_loader,
            device,
            feature=feature,
        )
        training_status_model[feature] = {"val_acc": val_acc, "val_loss": val_loss}
        save_training_status(training_status)


@app.command()
def eval(model_name: str, model_path: str):
    assert model_name in [
        "lenet",
        "resnet18",
    ], "Model name must be either 'lenet' or 'resnet18'"

    _, validation_loader = load_dataset(model_name)

    device = optimal_device()
    architecture_factory = MODEL_ARCHITECTURE_FACTORY[model_name]
    model = architecture_factory(num_classes=2)
    state_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    val_acc, val_loss = eval_accuracy(model, validation_loader, criterion, device)
    print(f"Validation Accuracy: {val_acc}, Validation Loss: {val_loss}")


if __name__ == "__main__":
    app()
