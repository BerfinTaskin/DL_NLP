#!/bin/bash

# === Configuration ===
LOG_NAME="baseline"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

# === Command ===
python improvements_sts.py \
    --epochs 10 \
    --option finetune \
    --use_gpu \
    --seed 142 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1

# python /user/henrich1/u12041/repos/nlp_stuff/DL_NLP/multitask_classifier.py --option finetune --task sts --seed 5 --use_gpu