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
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='directory for storing video frames', required=True)
    parser.add_argument('-trs', '--train_size', type=float, default=0.7, help='train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.1, help='test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='test set size',)
    parser.add_argument('-nc', '--n_classes', type=int, help='number of classes for the classification task', required=True)

    parser.add_argument('-mt', '--model_type', help='3D CNN or LRCN', default='lrcn')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34', help='2D CNN backbone - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('-p', '--pretrained', help='use pretrained 2D CNN backbone', default=True)
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='number of RNN/LSTM layers')

    parser.add_argument('-bs', '--batch_size', type=int, help='mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='learning rate for the model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='number of training epochs')

    args = parser.parse_args()

    return args

def trainer(args):
    # dataset parameters
    frame_dir = args.frame_dir
    tr_size = args.train_size
    ts_size = args.test_size
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # model parameters
    model = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone

    # training parameters
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load dataset and generate train/validation/test splits
    vid_dataset, label_dict = load_dataset(frame_dir)
    tr_split, val_split, ts_split = dataset_split(vid_dataset, tr_size, ts_size)

    # load statistics for data augmentation
    h, w, mean, std = transform_stats(model)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    # create datasets and dataloaders for train/validation/test splits
    tr_dataset = VideoDataset(tr_split, fr_per_vid, tr_transforms)
    val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms)
    ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
    dataloaders = compose_dataloaders(tr_dataset, val_dataset, ts_dataset, batch_size, model)

    # initialize model
    model = LRCN(hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout,
             n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone)
    model = model.to(device)

    # create loss function, optimizer, learning rate scheduler for training
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=1)
    os.makedirs("./models", exist_ok=True)
    optim_model_dir = './models'

    # main training procedure
    model, loss_hist, acc_hist = train(dataloaders, model, loss_func, opt, lr_scheduler, device, optim_model_dir, n_epochs)

if __name__=="__main__":
    args = args_parser()
    trainer(args)
