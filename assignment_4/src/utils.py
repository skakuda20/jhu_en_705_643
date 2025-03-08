import os
import cv2
import numpy as np

from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from video_datasets import collate_fn_r3d_18, collate_fn_rnn

# uniform sampling frames for each video
def get_frames(vid, n_frames=1):
    frames = []
    v_cap = cv2.VideoCapture(vid)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)    # uniform sampling

    for idx in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if idx in frame_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)             # convert frame tensor to RGB format
            frames.append(frame)
    v_cap.release()
    return frames, v_len

# given sampled array of frames, physically save frame as JPG images
def store_frames(frames, store_path):
    for idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)                 # convert frame tensor back to BGR format for saving
        path_to_frame = os.path.join(store_path, "frame{}.jpg".format(idx))
        cv2.imwrite(path_to_frame, frame)

# given model type: 3dcnn or lrcn
# return height, width, mean, standard deviation
def transform_stats(model='lrcn'):
    if model=='lrcn':
        h, w =224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model=='3dcnn':
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError('model_type arg is undefined....')

    return h, w, mean, std

# given height, width, mean, std
# return train/val/test transforms for train/val/test PyTorch datasets
def compose_data_transforms(height, width, mean, std):
    train_transforms = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std), ])
    val_test_transforms = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std), ])
    return train_transforms, val_test_transforms

def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn'):
    if model == "lrcn":
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_rnn)
        val_dl = DataLoader(val_dataset, batch_size=2*batch_size,
                            shuffle=False, collate_fn=collate_fn_rnn)
    else:
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_r3d_18)
        val_dl = DataLoader(val_dataset, batch_size=2*batch_size,
                            shuffle=False, collate_fn=collate_fn_r3d_18)
    dataloaders = {'train':train_dl, 'val':val_dl,}

    return dataloaders

def test_dloaders(test_dataset, batch_size, model='lrcn'):
    if model == "lrcn":
        test_dl = DataLoader(test_dataset, batch_size=2*batch_size,
                             shuffle=False, collate_fn=collate_fn_rnn)
    else:
        test_dl = DataLoader(test_dataset, batch_size=2*batch_size,
                             shuffle=False, collate_fn=collate_fn_r3d_18)
    dataloaders = {'test':test_dl}

    return dataloaders
