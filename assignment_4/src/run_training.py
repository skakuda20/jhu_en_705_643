"""
Module: trainer.py

This module provides an end-to-end training pipeline for a video classification model.
It parses command-line arguments to configure dataset paths, model parameters, and training
hyperparameters. The module performs the following steps:

1. Loads the video dataset from a specified directory.
2. Splits the dataset into training, validation, and test sets using stratified sampling.
3. Applies data augmentation transforms and creates PyTorch Datasets and DataLoaders using a custom
   function `compose_dataloaders`.
4. Initializes a video classification model (e.g., LRCN) with specified hyperparameters.
5. Sets up the loss function, optimizer, and learning rate scheduler.
6. Executes the training procedure and saves the best model weights.

Usage:
    Run the script from the command line with the required arguments. For example:
    
        python trainer.py -fd /path/to/frames -nc 51 -bs 8
        
Arguments include:
    -fd/--frame_dir: Directory where video frames are stored (required).
    -trs/--train_size: Proportion of data for training (default: 0.7).
    -tss/--test_size: Proportion of data for testing (default: 0.1).
    -fpv/--fr_per_vid: Number of frames per video to use (default: 16).
    -nc/--n_classes: Number of classes for classification (required).
    -mt/--model_type: Type of model to use, e.g., 'lrcn' (default: 'lrcn').
    -cnn/--cnn_backbone: CNN backbone for feature extraction (default: 'resnet34').
    -p/--pretrained: Whether to use a pretrained CNN backbone (default: True).
    -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default: 100).
    -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default: 1).
    -bs/--batch_size: Mini-batch size (required).
    -d/--dropout: Dropout rate for regularization (default: 0.1).
    -lr/--learning_rate: Learning rate for training (default: 3e-5).
    -ne/--n_epochs: Number of training epochs (default: 30).
"""

import os
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from video_datasets import VideoDataset, load_dataset, dataset_split
from utils import transform_stats, compose_data_transforms, compose_dataloaders
from models import LRCN
from train import train

def args_parser():
    """
    Parse command-line arguments for configuring the video classification training pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    
    Required Arguments:
        -fd/--frame_dir: Directory for storing video frames.
        -nc/--n_classes: Number of classes for the classification task.
        -bs/--batch_size: Mini-batch size.
    
    Optional Arguments (with defaults):
        -trs/--train_size: Proportion of the dataset used for training (default: 0.7).
        -tss/--test_size: Proportion of the dataset used for testing (default: 0.1).
        -fpv/--fr_per_vid: Number of frames per video (default: 16).
        -mt/--model_type: Model type, e.g., 'lrcn' (default: 'lrcn').
        -cnn/--cnn_backbone: CNN backbone for feature extraction (default: 'resnet34').
        -p/--pretrained: Use pretrained CNN backbone (default: True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default: 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default: 1).
        -d/--dropout: Dropout rate (default: 0.1).
        -lr/--learning_rate: Learning rate for training (default: 3e-5).
        -ne/--n_epochs: Number of training epochs (default: 30).
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames', required=True)
    parser.add_argument('-trs', '--train_size', type=float, default=0.7, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int, help='Number of classes for the classification task', required=True)

    parser.add_argument('-mt', '--model_type', help='3D CNN or LRCN', default='lrcn')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34', help='2D CNN backbone - options: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('-p', '--pretrained', help='Use pretrained 2D CNN backbone', default=True)
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')

    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='Learning rate for the model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')

    args = parser.parse_args()
    return args

def trainer(args):
    """
    Execute the training pipeline for video classification.

    This function performs the following steps:
      1. Loads the video dataset from the specified frame directory.
      2. Splits the dataset into training, validation, and test sets.
      3. Loads image transformation statistics and composes data augmentation transforms.
      4. Creates PyTorch Datasets for training, validation, and test splits.
      5. Constructs DataLoaders using a custom function `compose_dataloaders`.
      6. Initializes the LRCN model with specified hyperparameters.
      7. Sets up the loss function, optimizer, and learning rate scheduler.
      8. Executes the training loop and saves the best model weights.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Dataset parameters
    frame_dir = args.frame_dir
    tr_size = args.train_size
    ts_size = args.test_size
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone

    # Training parameters
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load dataset and generate train/validation/test splits
    vid_dataset, label_dict = load_dataset(frame_dir)
    tr_split, val_split, ts_split = dataset_split(vid_dataset, tr_size, ts_size)

    # Load image transformation statistics and compose data augmentation transforms
    h, w, mean, std = transform_stats(model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    # Create PyTorch Datasets for each split
    tr_dataset = VideoDataset(tr_split, fr_per_vid, tr_transforms)
    val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms)
    ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)

    # Compose DataLoaders for training, validation, and test using a custom function
    dataloaders = compose_dataloaders(tr_dataset, val_dataset, ts_dataset, batch_size, model_type)

    # Initialize the LRCN model with the specified parameters
    model = LRCN(hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout,
                 n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone)
    model = model.to(device)

    # Define the loss function, optimizer, and learning rate scheduler
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=1)
    os.makedirs("./models", exist_ok=True)
    optim_model_dir = './models'

    # Execute the main training procedure and update the model
    model, loss_hist, acc_hist = train(dataloaders, model, loss_func, opt, lr_scheduler, device, optim_model_dir, n_epochs)

if __name__ == "__main__":
    args = args_parser()
    trainer(args)
