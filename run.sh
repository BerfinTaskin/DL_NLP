#!/bin/bash

#SBATCH --job-name=etpc_data_aug
#SBATCH -t 0-02:00                     # Estimated time (adjust as needed)
#SBATCH -p grete                       # Partition
#SBATCH -G A100:1                      # Request 1 GPU
#SBATCH --mem-per-gpu=8G               # Memory per GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=u17468@stud.uni-goettingen.de
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

# --- Script commands start here ---

# Ensure logs directory exists
mkdir -p ./slurm_files

# Activate conda environment
source activate dnlp
pip install --no-input sentencepiece==0.1.99 sacremoses
echo "Running ETPC Data Augmentation"
echo "Working directory: $PWD"
echo "Node: ${SLURM_NODELIST}"

# Print Python environment info
python --version
python -m torch.utils.collect_env 2> /dev/null

# Navigate to project folder
cd /user/shrinath.madde/u17468/DL_NLP

# Run augmentation script
python -u data_agu.py \
    --input ./data/etpc-paraphrase-train.csv \
    --output ./data/etpc-paraphrase-train.augmented.csv \
    --bt_lang de \
    --bt_frac_pos 0.6 \
    --bt_frac_neg 0.5 \
    --neg_frac 0.7 \
    --batch_size 16 \
    --seed 13

echo "Data augmentation finished!"
