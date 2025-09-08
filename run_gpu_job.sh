#!/bin/bash
#SBATCH --job-name=bart_gpu_finetune
#SBATCH -t 0-04:00
#SBATCH -p grete
#SBATCH -G A100:3                 # ⬅️ get 3 GPUs in one job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24        # e.g., 8 CPU cores per training x 3
#SBATCH --mem-per-gpu=16G
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=u17468@stud.uni-goettingen.de

mkdir -p ./slurm_files
source activate dnlp
cd /user/shrinath.madde/u17468/DL_NLP

# keep each worker from grabbing all CPU threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

run() {
  local gpu_id="$1"; shift
  CUDA_VISIBLE_DEVICES="$gpu_id" python -u bart_detection.py "$@" \
    --use_gpu --num_epochs 30 --early_stopping_patience 4 \
    --job_id "${SLURM_JOB_ID}_gpu${gpu_id}" \
    > "slurm_files/train_gpu${gpu_id}_$(date +%H%M%S).out" 2>&1 &
}

# launch 3 trainings on GPUs 0,1,2 (adjust hyperparams per run)
run 0 --batch_size 128 --learning_rate 2e-3 --approach baseline_batch128_3
#run 1 --batch_size 128 --learning_rate 2e-4 --approach baseline_batch128_4
#run 2 --batch_size 128 --learning_rate 2e-5 --approach baseline_batch128_5

# add more, still one per GPU:
# run 0/1/2 again ONLY after the first set finishes, or switch to 6 GPUs.

wait
echo "All parallel GPU runs finished."
