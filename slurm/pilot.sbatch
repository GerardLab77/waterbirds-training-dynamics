#!/bin/bash
#SBATCH --job-name=waterbirds-pilot
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=waterbirds-pilot-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Anaconda3/2025.06-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate "${WB_CONDA_ENV:-$PWD/.conda-a100}"
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SLURM_EXPORT_ENV=ALL
date -u
nvidia-smi
srun --export=ALL --ntasks=1 --cpus-per-task=4 --cpu-bind=cores \
  python -u train.py --config configs/pilot.json --seed 42 \
    --data-root "${WB_DATA_ROOT:-$PWD/data}" \
    --weights "${WB_WEIGHTS:-$PWD/weights/resnet18-f37072fd.pth}" \
    --output "${WB_PILOT_OUTPUT:-runs/pilot-$SLURM_JOB_ID}" --resume
date -u
