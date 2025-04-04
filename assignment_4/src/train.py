"""
Module: train.py

This module provides functions to train and evaluate a video classification model.
It includes the main training loop that tracks loss and accuracy over epochs,
updates the best performing model based on validation accuracy, and provides helper
functions to compute the learning rate, batch loss, and epoch loss.

Functions:
    - train: Runs the training and validation loops over a specified number of epochs.
    - get_learning_rate: Retrieves the current learning rate from an optimizer.
    - batch_correct_preds: Computes the number of correct predictions in a mini-batch.
    - get_batch_loss: Computes the loss for a mini-batch and performs backpropagation.
    - get_epoch_loss: Computes the average loss and accuracy over an epoch.
"""

import os
import copy
import csv
from pathlib import Path
from tqdm import tqdm
import torch

from utils import get_project_root_path

def train(dataloaders,
          model,
          criterion,
          optimizer,
          scheduler,
          device,
          optim_model_wts_dir,
          n_epochs=30):
    """
    Train and validate the model over a given number of epochs.
    
    This function performs the training loop for a video classification model.
    It iterates through the specified number of epochs, updates model weights
    using backpropagation, and evaluates model performance on a validation set.
    The model with the best validation accuracy is saved to disk.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        model (torch.nn.Module): The video classification model to be trained.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler, 
            which adjusts the learning rate based on validation loss.
        device (torch.device): Device (CPU or GPU) on which to perform training.
        optim_model_wts_dir (str): Directory to save the best model weights.
        n_epochs (int, optional): Number of training epochs. Default is 30.

    Returns:
        tuple: (model, loss_hist, acc_hist)
            - model (torch.nn.Module): The trained model loaded with the best validation weights.
            - loss_hist (dict): Dictionary containing lists of training and validation losses 
                for each epoch.
            - acc_hist (dict): Dictionary containing lists of training and validation 
                accuracies for each epoch.
    """
    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    root_path = get_project_root_path()

    txt_log_file = Path(root_path, "output", "training_terminal_output.txt")
    csv_log_file = Path(root_path, "output", "training_logs.csv")
    with open(csv_log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Accuracy"])

    for epoch in range(n_epochs):
        current_lr = get_learning_rate(optimizer)
        log_message(txt_log_file,
                    f'Epoch {epoch+1}/{n_epochs}; Current learning rate {current_lr}')
        print(f'Epoch {epoch+1}/{n_epochs}; Current learning rate {current_lr}\n')

        # Training phase
        model.train()
        train_loss, train_accuracy = get_epoch_loss(model,
                                                    criterion,
                                                    dataloaders['train'],
                                                    device,
                                                    optimizer)
        loss_hist['train'].append(train_loss)
        acc_hist['train'].append(train_accuracy)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = get_epoch_loss(model, criterion, dataloaders['val'], device)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            best_model_name = 'best_model_wts.pt'
            best_model_path = os.path.join(optim_model_wts_dir, best_model_name)
            torch.save(best_model_wts, best_model_path)
            print(f'Best model weights are updated at epoch {epoch+1}!')
        loss_hist['val'].append(val_loss)
        acc_hist['val'].append(val_accuracy)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        if current_lr != get_learning_rate(optimizer):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        log_to_csv(csv_log_file, epoch, train_loss, val_loss, 100*val_accuracy)
        log_message(txt_log_file, f"train loss: {train_loss:.6f}, "
                                    f"val loss: {val_loss:.6f}, "
                                    f"accuracy: {100*val_accuracy:.2f}")
        log_message(txt_log_file, "-"*60)

        print(f"train loss: {train_loss:.6f}, "
              f"val loss: {val_loss:.6f}, "
              f"accuracy: {100*val_accuracy:.2f}"
            )
        print("-" * 60)
        print()

    # Load the best model weights before returning the model
    model.load_state_dict(best_model_wts)
    return model, loss_hist, acc_hist

def get_learning_rate(optimizer):
    """
    Retrieve the current learning rate from the optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to get the learning rate.
    
    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def batch_correct_preds(output, target):
    """
    Compute the number of correct predictions for a mini-batch.
    
    Args:
        output (torch.Tensor): Model outputs (logits) with shape (batch_size, num_classes).
        target (torch.Tensor): True labels with shape (batch_size).
    
    Returns:
        int: Number of correct predictions in the mini-batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct_preds = pred.eq(target.view_as(pred)).sum().item()
    return correct_preds

def get_batch_loss(criterion, output, target, optimizer=None):
    """
    Compute the loss for a mini-batch and perform backpropagation (if optimizer is provided).
    
    Args:
        criterion (callable): Loss function.
        output (torch.Tensor): Model outputs for the mini-batch.
        target (torch.Tensor): True labels for the mini-batch.
        optimizer (torch.optim.Optimizer, optional): Optimizer to update model weights. 
                                                        If None, no backpropagation is performed.
    
    Returns:
        tuple: (loss_value, n_batch_correct_preds)
            - loss_value (float): Loss value for the mini-batch.
            - n_batch_correct_preds (int): Number of correct predictions in the mini-batch.
    """
    loss = criterion(output, target)
    with torch.no_grad():
        n_batch_correct_preds = batch_correct_preds(output, target)
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), n_batch_correct_preds

def get_epoch_loss(model, criterion, dataloader, device, optimizer=None):
    """
    Compute the average loss and overall accuracy for an epoch.

    Iterates over the entire DataLoader, computes loss and accuracy for each mini-batch,
    and aggregates the results over the epoch.

    Args:
        model (torch.nn.Module): The video classification model.
        criterion (callable): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device (CPU or GPU) on which to perform computations.
        optimizer (torch.optim.Optimizer, optional): If provided, used to update
                                                        model weights during training.

    Returns:
        tuple: (loss, accuracy)
            - loss (float): Average loss over the epoch.
            - accuracy (float): Overall accuracy over the epoch.
    """
    running_loss, running_total_correct_preds = 0.0, 0.0
    len_dataset = len(dataloader.dataset)

    for x_batch, y_batch in tqdm(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        batch_loss, n_batch_correct_preds = get_batch_loss(criterion, output, y_batch, optimizer)

        running_loss += batch_loss
        running_total_correct_preds += n_batch_correct_preds

    loss = running_loss / float(len_dataset)
    accuracy = running_total_correct_preds / float(len_dataset)
    return loss, accuracy


def log_to_csv(log_file, epoch, train_loss, val_loss, accuracy):
    """
    Saves out loss and accuracy values at current epoch to a CSV file.

    Args:
        log_file (str or Path): path to CSV file.
        epoch (int): Current epoch
        train_loss (float): Current train loss
        val_loss (float): Current val loss
        accuracy (float): Current accuracy
    """
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss, accuracy])


def log_message(log_file, message):
    """
    Saves out message to txt log file.

    Args:
        log_file (str or Path):  Path to CSV file.
        message (str): Message to save to txt log file.
    """
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
