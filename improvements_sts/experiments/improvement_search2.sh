#!/bin/bash
#SBATCH -p grete:interactive
#SBATCH -G 1g.20gb # 1g.20gb, 2g.10gb
#SBATCH -t 0-12:00:00
#SBATCH -A nib00034
#SBATCH -C inet
#SBATCH -o /user/henrich1/u12041/output/job-%J.out

# optionally put "#SBATCH -C inet as slurm option" and comment out the below lines for internet access on compute node (e.g. for downloading/uploading stuff)
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

echo "Activating conda..."
source /user/henrich1/u12041/scratch/scratch_rzg/conda/etc/profile.d/conda.sh
conda activate dnlp
cd /user/henrich1/u12041/repos/nlp_stuff/DL_NLP_jonathan

############################### built upon config 2 from baseline_param_search1.sh
############################### === Configuration ===
LOG_NAME="baseline1_2_lora1"   # <-- change this to your desired name
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
    --use_lora \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_target_modules query key value \
    --lora_dropout 0.0 \
    --lora_bias none \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1

############################### === Configuration ===
LOG_NAME="baseline1_2_lora2"   # <-- change this to your desired name
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
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules query key value \
    --lora_dropout 0.05 \
    --lora_bias none \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1

############################### === Configuration ===
LOG_NAME="baseline1_2_lora3"   # <-- change this to your desired name
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
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_target_modules query key value \
    --lora_dropout 0.1 \
    --lora_bias none \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1

############################### === Configuration ===
LOG_NAME="baseline1_2_lora4"   # <-- change this to your desired name
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
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules query key value \
    --lora_dropout 0.1 \
    --lora_bias none \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1

############################### === Configuration ===
LOG_NAME="baseline1_2_lora5"   # <-- change this to your desired name
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
    --use_lora \
    --lora_r 16 \
    --lora_alpha 48 \
    --lora_target_modules query key value \
    --lora_dropout 0.1 \
    --lora_bias none \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1
