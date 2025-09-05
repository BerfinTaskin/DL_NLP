#!/bin/bash

# === Configuration ===
LOG_NAME="baseline"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

# === Command ===
python improvements_sts.py \
    --task sts \
    --epochs 10 \
    --option finetune \
    --use_gpu \
    # >"$LOG_DIR/${LOG_NAME}.log" 2>&1
