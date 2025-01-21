#!/usr/bin/env python3
import torch
from torchvision import transforms

from merlin.datasets import CelebADataset
from merlin.helpers.dataset import load_whole_dataset
from merlin.models.torch import MODEL_INPUT_TRANSFORMATION_FACTORY


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
        for x, y, _ in tqdm(eval_loader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)
            pred = model(x.to(device)).argmax(dim=1)
            correct += (pred == y.to(device)).sum().item()
            total += y.size(0)
            
            loss = criterion(model(x), y).item()
            eval_loss += loss * x.size(0)
    eval_loss = eval_loss / total
    accuracy = correct / total
    return accuracy, eval_loss


def train_model(model, optimizer, criterion, train_loader, validation_loader, device, num_epochs=10, best_model_save_path=None):
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
    best_acc = -1
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):    
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            wandb.log({"Training Loss": loss.item()})
        val_acc, val_loss = eval_accuracy(model, validation_loader, criterion, device)
        if val_acc > best_acc:
            best_acc = val_acc
            if best_model_save_path is not None:
                model.eval()
                print(f"Saving best model with accuracy: {best_acc} (Validation Loss: {val_loss})")
                torch.save(model.state_dict(), best_model_save_path)
        wandb.log({"Validation Accuracy": val_acc, "Validation Loss": val_loss})
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
    celeba = CelebADataset(
        target_columns=[label_col], transform=transformation
    )

    return load_whole_dataset(celeba)


def main():
    features, labels = load_dataset()
    print(features.size())

    device = optimal_device()

if __name__ == "__main__":
    main()
