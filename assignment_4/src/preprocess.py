import argparse
import os
from tqdm import tqdm
from utils import get_frames, store_frames

# display the number of videos in each category
def basic_vid_stats(dataset_dir):
    for cat in os.listdir(dataset_dir):
        cat_vids = os.path.join(dataset_dir, cat)
        print('The {} class contains {} videos.'.format(cat, len(os.listdir(cat_vids))))
    print()
    
# split and stores frames of each video 
"""
Original Video Structure
-HMDB51
    - Cat1
        - Cat1_vid1.avi
        - Cat1_vid2.avi
        ....
        - Cat1_vidM.avi
    - Cat2
    ....
    - CatN

Frame Storage Structure
- HMDB51
    - Cat1
        - Cat1_vid1
            - Cat1_vid1_fr1.jpg
            - Cat1_vid1_fr2.jpg
            ....
            - Cat1_vid1_frK.jpg
        - Cat1_vid2
        ....
        - Cat1_vidM
    - Cat2
    ....
    - CatN
"""
def create_frame_dataset(dataset_dir, frame_dir, ext='.avi', n_frames=1):
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir, exist_ok = False)
    for vid_cat_dir in os.listdir(dataset_dir):
        print('Extracting and storing frames for {} class....'.format(vid_cat_dir))
        for vid in tqdm( os.listdir(os.path.join(dataset_dir, vid_cat_dir)) ):
            if not vid.endswith(ext):
                continue
            vid_path = os.path.join(dataset_dir, vid_cat_dir, vid)
            frames, _ = get_frames(vid_path, n_frames)
            vid_fname = vid.split('.')[0]
            fr_dest_path = os.path.join(frame_dir, vid_cat_dir, vid_fname)
            os.makedirs(fr_dest_path, exist_ok=True)
            store_frames(frames, fr_dest_path)

def main():
    parser = argparse.ArgumentParser(description='Video Dataset Preprocessing')
    parser.add_argument('-dd', '--dataset_dir', help='path of original video dataset', required=True)
    parser.add_argument('-fd', '--frame_dir', help='path to store frames extracted from original video dataset', required=True)
    parser.add_argument('-nf', '--n_frames', type=int, default=1, help='number of frames to be extracted from each video', required=True)
    args = parser.parse_args()
    
    # display basic dataset facts  
    basic_vid_stats(args.dataset_dir)
    
    # random sample and store frames for each video
    create_frame_dataset(args.dataset_dir, args.frame_dir, n_frames=args.n_frames)
    
if __name__=="__main__":
    main()