#!/bin/bash
# Submit from the project root: mkdir -p logs; sbatch scripts/training/submit_prepare_sft.sh
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=amd
#SBATCH --time=60:00:00
#SBATCH --job-name=qwen3-sft-prepare
#SBATCH --output=logs/qwen3-sft-prepare-%j.out
set -euo pipefail

SCRIPT_PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "${SCRIPT_PROJECT_ROOT}"
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-lex_legal_rl_post_training_env}"
set -u
# The Conda C++ runtime must precede cluster compiler libraries (SQLite/ICU).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="$(mktemp -d /tmp/sft-${SLURM_JOB_ID:-manual}-XXXXXX)"
echo "Preparing SFT data on $(hostname); project=${SCRIPT_PROJECT_ROOT}"
srun --cpu-bind=none python scripts/training/train_sft.py \
    --prepare-only \
    --model-path "${MODEL_PATH:-${HOME}/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Legislation-3B}" \
    --prepared-dir "${PREPARED_DIR:-${HOME}/scratch/legal_data_process/datasets/SFT-processed/LawInstruct-General-Balanced-Qwen3}" \
    "$@"
