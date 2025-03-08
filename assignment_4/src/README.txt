Step 0: Download and Unzip Dataset.
        Download HMDB51 dataset from https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads.
        Under "DOWNLOAD/VIDEO DATABASE" section, choose the HMDB51 link (2GB, 7000 clips, 51 action classes). Unzip the rar files and save the dataset as follows:

        - HMDB51
            - Action_Class1
            - Action_Class2
            ... ... ... ...
            - Action_Class51

Step 1: Preprocess the Video Dataset and Extract Frames from Each Video.
        Edit and run the bash script file preprocessing.sh to preprocess the extracted video dataset. In particular,

        --dataset_dir: the path where you saved the extracted dataset in Step 0.
        --frame_dir: the path where you will be saving the extracted frames from each video clip.
        --n_frames: a hyper-parameter which specifies how many frames you will sample randomly from each video. Feel free to use a different number.

Step 2: Run Training.
        Run the bash script file train.sh. You must change the "--frame_dir" argument to match where you store the extracted frames from Step 1. You may also play with other parameters to see how they affect the experiment results.

Step 3: Run Testing.
        Run the bash script file test.sh. You must change the "--ckpt" argument to match where you store the best model from step 2. 
