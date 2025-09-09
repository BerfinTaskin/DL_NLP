#!/bin/bash

#SBATCH --job-name=bart_gpu_finetune
#SBATCH -t 0-04:00                     # Estimated time for your job (adjust as needed)
#SBATCH -p grete                     # Use the 'grete' partition
#SBATCH -G A100:1                    # Request 1 A100 GPU
#SBATCH --mem-per-gpu=8G             # Setting 8GB memory per GPU. Adjust if OOM errors occur.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=u17468@stud.uni-goettingen.de   # IMPORTANT: Replace with your actual email address
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

# --- Script commands start here ---

# Create the slurm_files directory if it doesn't exist
mkdir -p ./slurm_files

# NO module load cuda/12.1 here! It's not needed if PyTorch already finds CUDA.

# Activate your conda environment
source activate dnlp

# Print out some info for debugging purposes
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null

# Print out some git info.
module load git # Load git module for git commands
echo -e "\nCurrent Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest Commit: $(git rev-parse --short HEAD)"
echo -e "Uncommitted Changes: $(git status --porcelain | wc -l)\n"
module unload git # Unload git module if no longer needed

# Navigate to your script's directory
cd /user/shrinath.madde/u17468/DL_NLP

# Run your Python script (Removed --local_files_only)
python -u bart_detection.py \
    --use_gpu \
    --num_epochs 30 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --early_stopping_patience 30 \
    --approach "Warmup + cosine/linear decay reduces early overfitting." \
    --job_id "${SLURM_JOB_ID}"

    
echo "GPU job finished!"
