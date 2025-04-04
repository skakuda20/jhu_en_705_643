"""
Module: preprocess.py

This module processes the raw video data into frame sequences that the model is able to digest.
Uses function from utils to sample and save the processed frames.
"""

from pathlib import Path

from utils import get_frames, store_frames, get_project_root_path


def process_raw_video_data():
    """
    Expects to find the UCF50 dataset in the data directory. Preprocess each video by
    sampling frames and saving the sampled frames into the processed directory that
    mirrors the structure of the UCF50 dataset.
    """
    root_path = get_project_root_path()

    raw_data_path = Path(root_path, "data", "UCF50")

    processed_data_path = Path(root_path, "data", "processed")
    processed_data_path.mkdir(exist_ok=True, parents=True)

    # Iterate through all the directories in the raw data path
    for subdir in raw_data_path.iterdir():
        if subdir.is_dir():
            action = subdir.name
            index = 0
            for file in subdir.iterdir():
                if file.is_file():
                    frames, v_len = get_frames(file, 16)

                    if v_len > 0:
                        save_path = Path(processed_data_path, action, str(index))
                        save_path.mkdir(exist_ok=True, parents=True)
                        store_frames(frames, save_path)
                        index += 1


if __name__ == "__main__":
    process_raw_video_data()
