#!/bin/bash
python run.py \
    --ckpt /home/kakudas/dev/jhu_en_705_643/assignment_4/src/models/best_model_wts.pt \
    --frame_dir /home/kakudas/dev/jhu_en_705_643/assignment_4/data/processed \
    --model_type lrcn \
    --n_classes 50 \
    --model_type lrcn \
    --batch_size 4 \
    --mode eval
