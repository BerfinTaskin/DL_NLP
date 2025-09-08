#!/bin/bash
#SBATCH -p grete:shared
##SBATCH -G 1g.20gb # 1g.20gb, 2g.10gb
#SBATCH -G A100:1
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
cd /user/henrich1/u12041/repos/nlp_stuff/DL_NLP

############################### built upon config 2 from baseline_param_search1.sh
############################### === Configuration ===
LOG_NAME="baseline1_2_plus_head2"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 8 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 2 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-4 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1

############################### built upon config 2 from baseline_param_search1.sh
############################### === Configuration ===
LOG_NAME="baseline1_2_plus_embeddingmean"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 8 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style mean \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-4 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1

############################### built upon config 2 from baseline_param_search1.sh
############################### === Configuration ===
LOG_NAME="baseline1_2_plus_lossnegpearson"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 8 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function neg_pearson \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-4 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1

############################### built upon config 2 from baseline_param_search1.sh
############################### === Configuration ===
LOG_NAME="baseline1_2_plus_dataaug"   # <-- change this to your desired name
LOG_DIR="improvements_sts/logs"
mkdir -p "$LOG_DIR"

python improvements_sts.py \
    --epochs 8 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.25 \
    --head_style 1 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-4 \
    --batch_size 128 \
    >"$LOG_DIR/${LOG_NAME}.log" 2>&1
