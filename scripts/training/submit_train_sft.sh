#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=160G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=i7_h200
#SBATCH --gres=gpu:4
#SBATCH --time=60:00:00
#SBATCH --job-name=qwen3-8b-sft
#SBATCH --output=logs/qwen3-8b-sft-%j.out
set -euo pipefail

SCRIPT_PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "${SCRIPT_PROJECT_ROOT}"
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-lex_legal_rl_post_training_env}"
set -u
module load "${GCC_MODULE_NAME:-gcc/11.5.0}"
module load "${CUDA_MODULE_NAME:-cuda/12.6.3}"
export CUDA_HOME="${CUDA_TOOLKIT_ROOT:-/iridisfs/ixsoftware/cuda/12.6.3}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
unset _PYTHON_SYSCONFIGDATA_NAME FLASH_ATTENTION_FORCE_BUILD
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MODEL_PATH="${MODEL_PATH:-${HOME}/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Legislation-3B}"
MODEL_RUN_NAME="${MODEL_RUN_NAME:-$(basename "${MODEL_PATH}")-SFT}"
PREPARED_DIR="${PREPARED_DIR:-${HOME}/scratch/legal_data_process/datasets/SFT-processed/LawInstruct-General-1.2M-Qwen3}"
echo "SFT job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "Model: ${MODEL_PATH}; data: ${PREPARED_DIR}; run: ${MODEL_RUN_NAME}"
python - <<'PY'
import os, torch
assert torch.cuda.is_available(), 'Allocated GPU not visible'
assert torch.cuda.device_count() == int(os.environ['GPUS_PER_NODE']), 'GPU count mismatch'
assert torch.cuda.is_bf16_supported(), 'BF16 required'
print(torch.__version__, torch.cuda.get_device_name(0))
PY
srun --cpu-bind=none torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" \
    scripts/training/train_sft.py \
    --model-path "${MODEL_PATH}" \
    --prepared-dir "${PREPARED_DIR}" \
    --output-dir "${HOME}/scratch/llms_model/lexlegal_llms/sft-model/${MODEL_RUN_NAME}" \
    --checkpoint-dir "${HOME}/scratch/llms_model/lexlegal_llms/checkpoints/${MODEL_RUN_NAME}" \
    "$@"




# cd "${HOME}/scratch/dips_project/lex-legal-post-training"
#  mkdir -p logs
#  PROJECT_ROOT="$(pwd)"
#
#  PREP_JOB_ID="$(
#      sbatch --parsable \
#          --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
#          scripts/training/submit_prepare_sft.sh
#  )"
#  PREP_JOB_ID="${PREP_JOB_ID%%;*}"
#
#  SFT_JOB_ID="$(
#      sbatch --parsable \
#          --dependency="afterok:${PREP_JOB_ID}" \
#          --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
#          scripts/training/submit_train_sft.sh \
#          --attn-implementation flash_attention_2
#  )"
#
#  echo "Preparation: ${PREP_JOB_ID}"
#  echo "SFT training: ${SFT_JOB_ID}"


# -------------------------
# If SFT data preparation has already completed, reuse it and submit training directly:
#
#  cd "${HOME}/scratch/dips_project/lex-legal-post-training"
#
#  PROJECT_ROOT="$(pwd)"
#  MODEL_PATH="${HOME}/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Proportional-5B-FA2"
#  MODEL_RUN_NAME="Qwen-Lex-Legal-CPT-using-Proportional-5B-FA2-SFT"
#
#  SFT_JOB_ID="$(
#      sbatch --parsable \
#          --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},MODEL_PATH=${MODEL_PATH},MODEL_RUN_NAME=${MODEL_RUN_NAME}" \
#          scripts/training/submit_train_sft.sh \
#          --attn-implementation flash_attention_2
#  )"
#
#  echo "Proportional-5B-FA2 SFT job: ${SFT_JOB_ID}"
#
#  If preparation is still running, add this option before the script path:
#
#  --dependency="afterok:${PREP_JOB_ID}" \
#
#  The CPT model must have finished saving before SFT starts.
#
#  Final model:
#
#  ~/scratch/llms_model/lexlegal_llms/sft-model/Qwen-Lex-Legal-CPT-using-Proportional-5B-FA2-SFT
#
#  Checkpoints:
#
#  ~/scratch/llms_model/lexlegal_llms/checkpoints/Qwen-Lex-Legal-CPT-using-Proportional-5B-FA2-SFT
