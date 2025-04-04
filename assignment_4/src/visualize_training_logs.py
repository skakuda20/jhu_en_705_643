"""
Module: visualize_training_logs.py

This module contains the visualize_logs function, which is used
to plot the loss and accuracy values of the model from training.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_project_root_path


def visualize_logs(display=False):
    """
    Visualizes the loss and accuracy of the model over training.

    Args:
        display (bool): If true, displays plot to screen, else saves the figure.
    """

    root_path = get_project_root_path()
    log_file = Path(root_path, "output", "training_logs.csv")
    df = pd.read_csv(log_file)

    # Plot Training and Validation Loss
    plt.figure(figsize=(8, 5))
    plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", marker="o")
    plt.plot(
        df["Epoch"],
        df["Val Loss"],
        label="Validation Loss",
        marker="o",
        linestyle="dashed",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    if display:
        plt.show()
    else:
        plt.savefig("train_val_plot.png")

    # Plot accuracy over time
    plt.figure(figsize=(8, 5))
    plt.plot(df["Epoch"], df["Accuracy"], label="Accuracy", marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)

    if display:
        plt.show()
    else:
        plt.savefig("accuracy_plot.png")


if __name__ == "__main__":
    visualize_logs()
