# Video Action Recognition with UCF50 using LRCN

This project implements a video action recgonition pipeline
that uses a Long-term Recurrent Convolutional Network (LRCN)
model arcitecture. The model usies a ResNet backbone pretrained
on ImageNet for feature extraction and learns temporal dynamics
through several recurrent layers. The project includes end-to-end
pipeline scripts for data processing, training and testing.

---

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Training the Model](#training-the-model)
- [Testing and Evaluation](#testing-and-evaluation)
- [Customization and Hyperparameters](#customization-and-hyperparameters)
---

## Dataset Preparation

### Step 0: Download and Unzip the Dataset

1. **Download Dataset:**
    This project uses the UCF50 dataset which can be
    downloaded from [here](https://www.crcv.ucf.edu/data/UCF50.rar)
    This dataset contains 50 different video action classes that
    have been sourced from YouTube.
2. **Unzip and Organize:**
    Unzip the downloaded dataset and place the extracted folder in
    the `/data/` directory. The folder structure should follow as:

        - UCF50
            - Baseball Pitch
            - Basketball
            ... ... ...
            - YoYo
    
    Each subdirectory represents a different action class.

### Step 1: Preprocess the Dataset
The video files in the UCF50 dataset need to be preprocessed
into a format that the LRCN model can ingest. This can be done
by running `python3 preprocess.py`. The script will automatically
look for the UCF50 directory in the `/data/` directory. 

---

## Environment Setup

### Step 2: Install Dependencies

1. **Python Version:**
This project requires Python 3.7 or higher.

2. **Dependencies:**
Install the required python packages by running 
`pip install -r requirements.txt`

---

## Training the Model

### Step 3: Configure Training Parameters
Model training is controled by the bash script `train.sh`, which
calls the training model. The script will expect to find the 
preprocessed data in the `/data/preprocessed/` directory.

### Step 4: Run the Training Script
Execute the traing script with: `bash train.sh`

During training the script will:
- **Load the frame dataset.**
- **Split the dataset** into training, validation, and test sets using stratified sampling.
- **Apply data augmentation** techniques (resizing, random flips, affine transformations).
- **Create custom PyTorch Datasets and DataLoaders.**
- **Initialize the LRCN model** using a specified ResNet backbone.
- **Set up the loss function, optimizer, and learning rate scheduler.**
- **Run the training loop** while tracking loss and accuracy, saving the best model weights.
- **Save out training metrics to txt and csv files.** The script will automatically save out
    training metrics to the `/output/` directory.

---

## Testing and Evaluation

### Step 5: Configure Testing Parameters
Model testing is controlled by the `test.sh` bash script. Ensure
that the `--ckpt` argument points to the correct best model
weights.

### Step 6: Run the Testing Script:
Execute the traing script with: `bash test.sh`

During testing the script will:
    - **Load the dataset splits** (previously saved during training).
    - **Create a DataLoader for the test set.**
    - **Load the trained model checkpoint.**
    - **Evaluate the model** on the test data by computing overall accuracy, generating classification reports, and optionally producing confusion matrices.
    - **Plot Heatmap of Classification Metrics**

### Step 7: Visualize Results
To gain a deeper understanding of the model performance, run:
`python3 visualize_training_logs.py`. This will save out plots
of the loss and accuracy statistics from the training step
to the `/output/` directory. These values can be compared to 
the output from the testing script to measure model preformance. 


## Customization and Hyperparameters
Several parameters can be modified to experiment with different
settings:

### Data Parameters

- `--frame_dir`: Path to your preprocessed frames.
- `--fr_per_vid`: Number of frames to sample per video.

### Model Parameters

- `--model_type`: Choose between `'lrcn'` (default) or other supported models.
- `--cnn_backbone`: Options include `resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152`.
- `--rnn_hidden_size` and `--rnn_n_layers`: Configure the LSTM network.

### Training Parameters

- `--batch_size`, `--learning_rate`, `--n_epochs`, and `--dropout` control the training dynamics.
- `--train_size` and `--test_size` determine dataset splits.


### Step 8: Explore
Now you have a fully functional and trained Video Action 
Recognition pipeline! Try tweaking different training
parameters or test out different datasets.

---