#!/bin/bash
# Submit from the project root:
#   mkdir -p logs
#   sbatch scripts/training/submit_train_cpt.sh
#
# Select another corpus/output name:
#   DATASET_NAME=CPT-V1-EU-1B MODEL_RUN_NAME=Qwen-Lex-Legal-CPT-using-EU-1B \
#     sbatch scripts/training/submit_train_cpt.sh
#
# Extra arguments are forwarded to train_cpt.py after the defaults. Example:
#   sbatch scripts/training/submit_train_cpt.sh --max-steps 20 --no-eval

#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=160G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=i7_h200
#SBATCH --gres=gpu:4
#SBATCH --time=60:00:00
#SBATCH --job-name=qwen3-8b-cpt
#SBATCH --output=logs/qwen3-8b-cpt-%j.out

set -euo pipefail


CUDA_MODULE_NAME="${CUDA_MODULE_NAME:-cuda/12.6}"
CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/iridisfs/ixsoftware/cuda/12.6.3}"

# Slurm batch shells must initialize the same CUDA development toolkit used by
# the cu126 PyTorch build. This also makes nvcc available for compiled CUDA
# extensions such as FlashAttention.
module load "${CUDA_MODULE_NAME}"
export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"


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
TRAINING_SCRIPT="${SCRIPT_PROJECT_ROOT}/scripts/training/train_cpt.py"
PREPARE_SCRIPT="${SCRIPT_PROJECT_ROOT}/scripts/training/prepare_cpt_dataset.py"
MODEL_PATH="${MODEL_PATH:-${HOME}/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base}"
DATA_ROOT="${DATA_ROOT:-${HOME}/scratch/legal_data_process/datasets}"
DATASET_NAME="${DATASET_NAME:-CPT-Proportional-5B}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/${DATASET_NAME}}"
TOKENIZED_DIR="${TOKENIZED_DIR:-${DATA_ROOT}/CPT-tokenized/${DATASET_NAME}-Qwen3}"
MODEL_RUN_NAME="${MODEL_RUN_NAME:-Qwen-Lex-Legal-CPT-using-Proportional-5B}"
FINAL_MODEL_DIR="${FINAL_MODEL_DIR:-${HOME}/scratch/llms_model/lexlegal_llms/cpt-model/${MODEL_RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${HOME}/scratch/llms_model/lexlegal_llms/checkpoints/${MODEL_RUN_NAME}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lex_legal_rl_post_training_env}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-4096}"
PREPARE_DATA="${PREPARE_DATA:-0}"
STAGE_TOKENIZED_DATA="${STAGE_TOKENIZED_DATA:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
REPLAY_REGEX="${REPLAY_REGEX:-(replay|general|fineweb|wikipedia|openwebtext|dolma|slimpajama)}"

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/lex-legal-cpt-${SLURM_JOB_ID:-manual}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "${SCRIPT_PROJECT_ROOT}"
mkdir -p "${HF_HOME}" "${FINAL_MODEL_DIR}" "${CHECKPOINT_DIR}" "${TOKENIZED_DIR}" "${SCRIPT_PROJECT_ROOT}/logs"

echo "Slurm job ID: ${SLURM_JOB_ID:-not-submitted}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project root: ${SCRIPT_PROJECT_ROOT}"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unset}"
echo "Base model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_DIR}"
echo "Tokenized dataset: ${TOKENIZED_DIR}"
echo "Final model: ${FINAL_MODEL_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Extra training arguments: $*"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected Conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV:-unset}" >&2
    exit 1
fi
for required_file in "${TRAINING_SCRIPT}" "${PREPARE_SCRIPT}" "${MODEL_PATH}/config.json" "${DATASET_DIR}/manifest.json"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file is missing: ${required_file}" >&2
        exit 1
    fi
done
if find "${DATASET_DIR}" -maxdepth 1 -name '*.jsonl' -print -quit | grep -q .; then
    :
else
    echo "No final JSONL shards found directly under ${DATASET_DIR}" >&2
    echo "The pipeline will not train from .work/preprocessed." >&2
    exit 1
fi

python --version
python -c "import torch, transformers, accelerate, deepspeed, numpy; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('Accelerate:', accelerate.__version__); print('DeepSpeed:', deepspeed.__version__); print('NumPy:', numpy.__version__); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; assert torch.cuda.is_bf16_supported(), 'GPU lacks BF16 support'; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA:', torch.version.cuda)"
nvidia-smi

if [[ ! -f "${TOKENIZED_DIR}/metadata.json" ]]; then
    if [[ "${PREPARE_DATA}" != "1" ]]; then
        echo "Tokenized data is missing and PREPARE_DATA=${PREPARE_DATA}: ${TOKENIZED_DIR}/metadata.json" >&2
        exit 1
    fi
    srun --unbuffered python "${PREPARE_SCRIPT}" \
        --input-dir "${DATASET_DIR}" \
        --manifest "${DATASET_DIR}/manifest.json" \
        --tokenizer-path "${MODEL_PATH}" \
        --output-dir "${TOKENIZED_DIR}" \
        --replay-regex "${REPLAY_REGEX}"
fi

TRAIN_TOKENIZED_DIR="${TOKENIZED_DIR}"
if [[ "${STAGE_TOKENIZED_DATA}" == "1" ]]; then
    LOCAL_TOKENIZED_DIR="${JOB_TEMP_DIR}/tokenized/${DATASET_NAME}-Qwen3"
    mkdir -p "${LOCAL_TOKENIZED_DIR}"
    cp "${TOKENIZED_DIR}"/*.bin "${TOKENIZED_DIR}"/*.json "${LOCAL_TOKENIZED_DIR}/"
    TRAIN_TOKENIZED_DIR="${LOCAL_TOKENIZED_DIR}"
    echo "Staged tokenized dataset to node-local storage: ${TRAIN_TOKENIZED_DIR}"
fi

srun --unbuffered torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${GPUS_PER_NODE}" \
    "${TRAINING_SCRIPT}" \
    --project-root "${SCRIPT_PROJECT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --tokenized-dir "${TRAIN_TOKENIZED_DIR}" \
    --output-dir "${FINAL_MODEL_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --deepspeed "${SCRIPT_PROJECT_ROOT}/configs/training/deepspeed_zero3.json" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 1e-5 \
    --weight-decay 0.1 \
    --warmup-ratio 0.01 \
    --eval-steps 250 \
    --save-steps 250 \
    --save-total-limit 2 \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"
echo "Final model: ${FINAL_MODEL_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
