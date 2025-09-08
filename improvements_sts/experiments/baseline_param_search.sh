#!/bin/bash
#SBATCH -p jupyter          # partition to submit job to
#SBATCH -G V100:1           # specify which gpu to request and how many e.g. V100:2
#SBATCH -t 24:00:00         # time limit in hours:minutes:seconds
#SBATCH --output=/dev/null  # you can delete this line. Then it will automatically log stdout to current_work_dir/output/<jobid>.out
#SBATCH --error=/dev/null   # you can delete this line. Then it will automatically log stderr to current_work_dir/output/<jobid>.out

# optionally put "#SBATCH -C inet as slurm option" and comment out the below lines for internet access on compute node (e.g. for downloading/uploading stuff)
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"


############################### === Configuration ===
LOG_NAME="baseline_search1"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-4 \
    --batch_size 64 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search2"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-4 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search3"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-5 \
    --batch_size 64 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search4"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-5 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search5"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-6 \
    --batch_size 64 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search6"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-6 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search7"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-5 \
    --lr_head 1e-4 \
    --batch_size 64 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search8"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-5 \
    --lr_head 1e-4 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search9"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-5 \
    --lr_head 1e-5 \
    --batch_size 64 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search10"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-5 \
    --lr_head 1e-5 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search11"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-5 \
    --lr_head 1e-6 \
    --batch_size 64 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search12"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-5 \
    --lr_head 1e-6 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search13"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-6 \
    --lr_head 1e-4 \
    --batch_size 64 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search14"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-6 \
    --lr_head 1e-4 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search15"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-6 \
    --lr_head 1e-5 \
    --batch_size 64 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search16"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-6 \
    --lr_head 1e-5 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search17"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-6 \
    --lr_head 1e-6 \
    --batch_size 64 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1


############################### === Configuration ===
LOG_NAME="baseline_search18"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 12 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-6 \
    --lr_head 1e-6 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1
