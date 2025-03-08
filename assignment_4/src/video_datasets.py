import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from PIL import Image

import os
import glob
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# the dataset class accompanies outputs returned by dataset_split function
class VideoDataset(Dataset):
    def __init__(self, vid_dataset, fr_per_vid, transforms=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        fr_paths = glob.glob(self.dataset[idx][0]+'/*.jpg')
        fr_paths = fr_paths[:self.fpv]
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]
        fr_label = self.dataset[idx][1]
        fr_imgs_trans = [self.transforms(fr_img) for fr_img in fr_imgs] if self.transforms else fr_imgs

        if len(fr_imgs_trans)>0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)

        return fr_imgs_trans, fr_label

# load full frame dataset into a dictionary
# key is the full path to a frame and value is the corresponding video class label
def load_dataset(frame_dir):
    label_dict = {vid_cat : idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))}
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(sorted(os.listdir(frame_dir))):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict

# use outputs of load_dataset and return train/validation/test splits
def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):
    vid_paths = np.array([vid_path for vid_path in vid_dataset.keys()])
    vid_labels = np.array([vid_label for vid_label in vid_dataset.values()])
    print('Splitting train/validation/test datasets....')

    # test split
    ts_spliter = StratifiedShuffleSplit(n_splits=1, test_size=ts_ratio, random_state=seed)
    for tr_val_idx, ts_idx in ts_spliter.split(vid_paths, vid_labels):
        ts_paths, ts_labels = vid_paths[ts_idx], vid_labels[ts_idx]
        tr_val_paths, tr_val_labels = vid_paths[tr_val_idx], vid_labels[tr_val_idx]
    ts_dataset = [(ts_path, ts_label) for ts_path, ts_label in zip(ts_paths, ts_labels)]

    # train/validation split
    val_ratio = 1 - tr_ratio - ts_ratio
    val_wt = val_ratio/(tr_ratio+val_ratio)
    val_spliter = StratifiedShuffleSplit(n_splits=1, test_size=val_wt, random_state=seed)
    for tr_idx, val_idx in val_spliter.split(tr_val_paths, tr_val_labels):
        tr_paths, tr_labels = tr_val_paths[tr_idx], tr_val_labels[tr_idx]
        val_paths, val_labels = tr_val_paths[val_idx], tr_val_labels[val_idx]
    tr_dataset = [(tr_path, tr_label) for tr_path, tr_label in zip(tr_paths, tr_labels)]
    val_dataset = [(val_path, val_label) for val_path, val_label in zip(val_paths, val_labels)]

    return tr_dataset, val_dataset, ts_dataset

def collate_fn_r3d_18(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor,labels_tensor

def collate_fn_rnn(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor,labels_tensor
