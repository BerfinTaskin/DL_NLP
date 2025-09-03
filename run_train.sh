#!/bin/bash
#SBATCH --job-name=train-multitask_classifier
#SBATCH -t 02:00:00                  # increased time so all tasks can finish
#SBATCH -p grete
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=TODO@stud.uni-goettingen.de
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

# Activate your environment
source activate dnlp

# Debug info
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

python --version
python -m torch.utils.collect_env 2> /dev/null

module load git
echo -e "\nCurrent Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest Commit: $(git rev-parse --short HEAD)"
echo -e "Uncommitted Changes: $(git status --porcelain | wc -l)\n"

# ----------------------
# Run SST
#python -u multitask_classifier.py \
#    --use_gpu --local_files_only --option finetune \
#    --task sst --hidden_dropout_prob 0.1 \
#    --approach "Baseline-SST"

# Run STS
#python -u multitask_classifier.py \
#    --use_gpu --local_files_only --option finetune \
#    --task sts --hidden_dropout_prob 0.1 \
#    --approach "Baseline-STS"

# Run QQP
python -u multitask_classifier.py \
    --use_gpu --local_files_only --option finetune \
    --task qqp --hidden_dropout_prob 0.1 \
    --approach "Baseline-QQP"

# ----------------------
echo "All tasks finished!"
