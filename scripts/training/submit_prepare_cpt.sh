#!/bin/bash
# Submit from the project root before the GPU training job:
#   mkdir -p logs
#   sbatch scripts/training/submit_prepare_cpt.sh

#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --partition=amd
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen3-cpt-prepare
#SBATCH --output=logs/qwen3-cpt-prepare-%j.out

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
    SCRIPT_PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    # Slurm executes a copied script from its spool directory, so BASH_SOURCE[0]
    # does not point at the repository inside a batch job.
    SCRIPT_PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_PROJECT_ROOT}" && pwd)"
PREPARE_SCRIPT="${SCRIPT_PROJECT_ROOT}/scripts/training/prepare_cpt_dataset.py"
MODEL_PATH="${MODEL_PATH:-${HOME}/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base}"
DATA_ROOT="${DATA_ROOT:-${HOME}/scratch/legal_data_process/datasets}"
DATASET_NAME="${DATASET_NAME:-CPT-Proportional-5B}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/${DATASET_NAME}}"
TOKENIZED_DIR="${TOKENIZED_DIR:-${DATA_ROOT}/CPT-tokenized/${DATASET_NAME}-Qwen3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lex_legal_rl_post_training_env}"
REPLAY_REGEX="${REPLAY_REGEX:-(replay|general|fineweb|wikipedia|openwebtext|dolma|slimpajama)}"

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "${SCRIPT_PROJECT_ROOT}"
mkdir -p "${TOKENIZED_DIR}" "${SCRIPT_PROJECT_ROOT}/logs"

echo "Started: $(date --iso-8601=seconds)"
echo "Dataset: ${DATASET_DIR}"
echo "Tokenizer: ${MODEL_PATH}"
echo "Output: ${TOKENIZED_DIR}"

srun --unbuffered python "${PREPARE_SCRIPT}" \
    --input-dir "${DATASET_DIR}" \
    --manifest "${DATASET_DIR}/manifest.json" \
    --tokenizer-path "${MODEL_PATH}" \
    --output-dir "${TOKENIZED_DIR}" \
    --replay-regex "${REPLAY_REGEX}" \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"
