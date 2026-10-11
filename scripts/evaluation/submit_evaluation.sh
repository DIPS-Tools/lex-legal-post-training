#!/bin/bash
# Submit from the project root after creating logs:
#   mkdir -p logs
#   sbatch scripts/evaluation/submit_evaluation.sh base
#   sbatch scripts/evaluation/submit_evaluation.sh cpt
# Arguments after base/cpt are forwarded to the Python evaluator.
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=i7_h200
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --job-name=eval-qwen-legal
#SBATCH --output=logs/qwen3-legal-eval-%j.out

set -euo pipefail

if (($# < 1)); then
    echo "Usage: sbatch $0 {base|cpt} [evaluation arguments ...]" >&2
    exit 2
fi

EVALUATION_TARGET="$1"
shift

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lex_legal_rl_post_training_env}"
CUDA_MODULE_NAME="${CUDA_MODULE_NAME:-cuda/12.6.3}"
GCC_MODULE_NAME="${GCC_MODULE_NAME:-gcc/11.5.0}"
CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/iridisfs/ixsoftware/cuda/12.6.3}"

if [[ -n "${PROJECT_ROOT:-}" ]]; then
    SCRIPT_PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && \
     [[ -f "${SLURM_SUBMIT_DIR}/src/evaluation/runner.py" ]]; then
    SCRIPT_PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    SCRIPT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi
SCRIPT_PROJECT_ROOT="$(cd -- "${SCRIPT_PROJECT_ROOT}" && pwd)"

EVALUATION_ROOT="${EVALUATION_ROOT:-${HOME}/scratch/legal_data_process/datasets/evaluation}"
LEGALBENCH_OFFICIAL_ROOT="${LEGALBENCH_OFFICIAL_ROOT:-${EVALUATION_ROOT}/legalbench_official}"
case "${EVALUATION_TARGET}" in
    base)
        EVALUATION_SCRIPT="${SCRIPT_PROJECT_ROOT}/scripts/evaluation/evaluation_base_model.py"
        MODEL_PATH="${MODEL_PATH:-${HOME}/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base}"
        MODEL_LABEL="${MODEL_LABEL:-Qwen3-8B-Base}"
        OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_PROJECT_ROOT}/results/evaluation/Qwen3-8B-Base}"
        ;;
    cpt)
        EVALUATION_SCRIPT="${SCRIPT_PROJECT_ROOT}/scripts/evaluation/evaluation_cpted_model.py"
        MODEL_PATH="${MODEL_PATH:-${HOME}/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Legislation-3B}"
        MODEL_LABEL="${MODEL_LABEL:-Qwen3-8B-CPT-Legislation-3B}"
        OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_PROJECT_ROOT}/results/evaluation/Qwen3-8B-CPT-Legislation-3B}"
        ;;
    *)
        echo "Unknown evaluation target: ${EVALUATION_TARGET}; expected base or cpt" >&2
        exit 2
        ;;
esac

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export ATTN_IMPLEMENTATION
extra_args=("$@")
for ((index=0; index<${#extra_args[@]}; index++)); do
    case "${extra_args[index]}" in
        --model-path)
            ((index + 1 < ${#extra_args[@]})) || { echo "--model-path requires a value" >&2; exit 2; }
            MODEL_PATH="${extra_args[index+1]}"
            ;;
        --model-path=*) MODEL_PATH="${extra_args[index]#*=}" ;;
        --model-label)
            ((index + 1 < ${#extra_args[@]})) || { echo "--model-label requires a value" >&2; exit 2; }
            MODEL_LABEL="${extra_args[index+1]}"
            ;;
        --model-label=*) MODEL_LABEL="${extra_args[index]#*=}" ;;
        --output-dir)
            ((index + 1 < ${#extra_args[@]})) || { echo "--output-dir requires a value" >&2; exit 2; }
            OUTPUT_DIR="${extra_args[index+1]}"
            ;;
        --output-dir=*) OUTPUT_DIR="${extra_args[index]#*=}" ;;
        --evaluation-root)
            ((index + 1 < ${#extra_args[@]})) || { echo "--evaluation-root requires a value" >&2; exit 2; }
            EVALUATION_ROOT="${extra_args[index+1]}"
            ;;
        --evaluation-root=*) EVALUATION_ROOT="${extra_args[index]#*=}" ;;
        --attn-implementation)
            ((index + 1 < ${#extra_args[@]})) || { echo "--attn-implementation requires a value" >&2; exit 2; }
            ATTN_IMPLEMENTATION="${extra_args[index+1]}"
            ;;
        --attn-implementation=*) ATTN_IMPLEMENTATION="${extra_args[index]#*=}" ;;
    esac
done

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
module load "${GCC_MODULE_NAME}"
module load "${CUDA_MODULE_NAME}"

export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
# GCC 11's module runtime is too old for some Conda libraries (notably ICU 78,
# imported through Python's sqlite3/NLTK stack).  Keep the compiler module for
# CUDA/FlashAttention compatibility, but prefer the Conda environment's newer,
# backward-compatible C++ runtime when launching Python.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/lex-legal-eval-${SLURM_JOB_ID:-manual}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

cd "${SCRIPT_PROJECT_ROOT}"
mkdir -p "${HF_HOME}" "${SCRIPT_PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

echo "Slurm job ID: ${SLURM_JOB_ID:-not-submitted}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project root: ${SCRIPT_PROJECT_ROOT}"
echo "Evaluation target: ${EVALUATION_TARGET}"
echo "Model: ${MODEL_PATH}"
echo "Model label: ${MODEL_LABEL}"
echo "Evaluation data: ${EVALUATION_ROOT}"
echo "Official LegalBench repository: ${LEGALBENCH_OFFICIAL_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Attention implementation: ${ATTN_IMPLEMENTATION}"
printf 'Additional evaluation arguments:'
printf ' %q' "$@"
printf '\n'

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected Conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV:-unset}" >&2
    exit 1
fi
required_files=(
    "${EVALUATION_SCRIPT}"
    "${SCRIPT_PROJECT_ROOT}/src/evaluation/runner.py"
    "${MODEL_PATH}/config.json"
    "${EVALUATION_ROOT}/legalbench_instruct/data/train-00000-of-00001.jsonl"
    "${EVALUATION_ROOT}/legalbench/task_metadata.json"
    "${LEGALBENCH_OFFICIAL_ROOT}/evaluation.py"
    "${LEGALBENCH_OFFICIAL_ROOT}/tasks/abercrombie/base_prompt.txt"
    "${EVALUATION_ROOT}/mmlu/international_law/test-00000-of-00001.jsonl"
    "${EVALUATION_ROOT}/mmlu/professional_law/test-00000-of-00001.jsonl"
    "${EVALUATION_ROOT}/mmlu/jurisprudence/test-00000-of-00001.jsonl"
)
for required_file in "${required_files[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file is missing: ${required_file}" >&2
        exit 1
    fi
done

python --version
python - <<'PY'
import os
import torch
import transformers
import sqlite3
import nltk
import pyarrow
import sklearn

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("PyArrow:", pyarrow.__version__)
print("scikit-learn:", sklearn.__version__)
print("NLTK:", nltk.__version__)
assert torch.cuda.is_available(), "Allocated GPU is not visible"
assert torch.cuda.device_count() == 1, f"Expected one GPU, found {torch.cuda.device_count()}"
assert torch.cuda.is_bf16_supported(), "GPU does not support BF16"
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch CUDA:", torch.version.cuda)
if os.environ.get("ATTN_IMPLEMENTATION") == "flash_attention_2":
    import flash_attn
    print("FlashAttention:", flash_attn.__version__)
PY
nvidia-smi

srun --unbuffered python "${EVALUATION_SCRIPT}" \
    --model-path "${MODEL_PATH}" \
    --model-label "${MODEL_LABEL}" \
    --evaluation-root "${EVALUATION_ROOT}" \
    --legalbench-official-root "${LEGALBENCH_OFFICIAL_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    "$@"

test -s "${OUTPUT_DIR}/evaluation_summary.json" || {
    echo "Evaluation finished without producing evaluation_summary.json" >&2
    exit 1
}

echo "Finished: $(date --iso-8601=seconds)"
echo "Summary: ${OUTPUT_DIR}/evaluation_summary.json"
