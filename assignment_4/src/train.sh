#!/bin/bash
python run.py \
    --frame_dir /home/kakudas/dev/jhu_en_705_643/assignment_4/data/processed \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --n_epochs 25 \
    --mode 'train'