#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm

from merlin.datasets import CelebADataset
from merlin.helpers.dataset import load_whole_dataset
from merlin.models.torch import MODEL_ARCHITECTURE_FACTORY, MODEL_INPUT_TRANSFORMATION_FACTORY


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
    model.to(device)
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


def train_model(model, optimizer, criterion, train_loader, validation_loader, device, num_epochs=10):
    """
    Trains a given model using the specified optimizer and loss criterion.

    Args:
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
    best_model_save_path = os.path.join(best_model_save_dir, "lenet_celeba.pth")

    best_acc = -1
    model.to(device)
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
            if step % 200 == 0:
                val_acc, val_loss = eval_accuracy(model, validation_loader, criterion, device)
                if val_acc > best_acc:
                    best_acc = val_acc
                    if best_model_save_path is not None:
                        model.eval()
                        print(f"Saving best model with accuracy: {best_acc} (Validation Loss: {val_loss})")
                        torch.save(model.state_dict(), best_model_save_path)
                print(f"Step {step+1}: Validation Accuracy: {val_acc}, Validation Loss: {val_loss}")        

        val_acc, val_loss = eval_accuracy(model, validation_loader, criterion, device)
        print(f"Epoch {epoch+1}: Validation Accuracy: {val_acc}, Validation Loss: {val_loss}")


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


def load_dataset():
    transformation = transforms.Compose([transforms.ToTensor()])

    meanstd = None
    transformation_factory = MODEL_INPUT_TRANSFORMATION_FACTORY["lenet"]
    transformation = transformation_factory(meanstd)
    label_col = "Smiling"
    
    train_dataset = CelebADataset(split="train", target_columns=[label_col], transform=transformation)
    val_dataset = CelebADataset(split="val", target_columns=[label_col], transform=transformation)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False
    )

    return train_dataloader, val_dataloader


def main():
    train_loader, validation_loader = load_dataset()

    device = optimal_device()
    architecture_factory = MODEL_ARCHITECTURE_FACTORY["lenet"]
    model = architecture_factory(num_classes=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_model(model, optimizer, criterion, train_loader, validation_loader, device)

if __name__ == "__main__":
    main()
