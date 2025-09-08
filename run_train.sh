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


# Create the slurm_files directory if it doesn't exist
mkdir -p ./slurm_files
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

python -u multitask_classifier.py \
  --use_gpu --option finetune \
  --task sts --hidden_dropout_prob 0.1 \
  --sts_beta 1.0 --early_stop_patience 3 \
  --use_cosine_schedule --warmup_ratio 0.1 \
  --k_bins 3 --bin_seed 2024 \
  --approach "STS-kbins5-meanpool-smoothl1-cosine"


python -u multitask_classifier.py \
  --use_gpu --option finetune \
  --task sst --hidden_dropout_prob 0.1 \
  --label_smoothing 0.1 \
  --use_cosine_schedule --warmup_ratio 0.1 \
  --k_bins 3 --bin_seed 2024 \
  --approach "SST-kbins5-meanpool-cosine-ls01"



# ----------------------
echo "All tasks finished!"
