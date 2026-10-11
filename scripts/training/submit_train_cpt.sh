#!/bin/bash
# Submit from the project root (create logs BEFORE sbatch):
# mkdir -p logs
# sbatch scripts/training/submit_train_cpt.sh --attn-implementation flash_attention_2
# Extra arguments follow defaults and are forwarded to train_cpt.py.
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
CUDA_MODULE_NAME="${CUDA_MODULE_NAME:-cuda/12.6.3}"
GCC_MODULE_NAME="${GCC_MODULE_NAME:-gcc/11.5.0}"
CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/iridisfs/ixsoftware/cuda/12.6.3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lex_legal_rl_post_training_env}"

if [[ -n "${PROJECT_ROOT:-}" ]]; then
    SCRIPT_PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
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
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-4096}"
PREPARE_DATA="${PREPARE_DATA:-0}"
STAGE_TOKENIZED_DATA="${STAGE_TOKENIZED_DATA:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
REPLAY_REGEX="${REPLAY_REGEX:-(replay|general|fineweb|wikipedia|openwebtext|dolma|slimpajama)}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
# Match the final CLI override when deciding whether to run the FA2 check.
extra_args=("$@")
for ((i=0; i<${#extra_args[@]}; i++)); do
    case "${extra_args[i]}" in
        --attn-implementation)
            if ((i+1 >= ${#extra_args[@]})); then
                echo '--attn-implementation requires a value' >&2
                exit 1
            fi
            ATTN_IMPLEMENTATION="${extra_args[i+1]}"
            ;;
        --attn-implementation=*) ATTN_IMPLEMENTATION="${extra_args[i]#*=}" ;;
    esac
done

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
module load "${GCC_MODULE_NAME}"
module load "${CUDA_MODULE_NAME}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="${CXX}"
export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_NO_COMPILER_WRAPPER=1
# Build-only overrides from the installation must not leak into training.
unset _PYTHON_SYSCONFIGDATA_NAME FLASH_ATTENTION_FORCE_BUILD

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
export ATTN_IMPLEMENTATION GPUS_PER_NODE

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
echo "Attention implementation: ${ATTN_IMPLEMENTATION}"
printf 'Extra training arguments:'
printf ' %q' "$@"
printf '\n'

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
shopt -s nullglob
jsonl_shards=("${DATASET_DIR}"/*.jsonl)
if ((${#jsonl_shards[@]} == 0)); then
    echo "No final JSONL shards found directly under ${DATASET_DIR}" >&2
    echo 'The pipeline will not train from .work/preprocessed.' >&2
    exit 1
fi
shopt -u nullglob

python --version
python - <<'PY'
import os
import torch, transformers, accelerate, deepspeed, numpy
for name, package in [('PyTorch', torch), ('Transformers', transformers),
                      ('Accelerate', accelerate), ('DeepSpeed', deepspeed), ('NumPy', numpy)]:
    print(f'{name}: {package.__version__}')
assert torch.cuda.is_available(), 'Allocated GPU is not visible'
expected = int(os.environ['GPUS_PER_NODE'])
assert torch.cuda.device_count() == expected, f'Expected {expected} GPUs, found {torch.cuda.device_count()}'
assert torch.cuda.is_bf16_supported(), 'GPU lacks BF16 support'
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA:', torch.version.cuda)
if os.environ['ATTN_IMPLEMENTATION'] == 'flash_attention_2':
    import flash_attn
    import flash_attn_2_cuda
    from flash_attn import flash_attn_func
    print('FlashAttention:', flash_attn.__version__)
    q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    out = flash_attn_func(q, q, q, causal=True)
    out.float().sum().backward()
    torch.cuda.synchronize()
    print('FlashAttention BF16 forward/backward passed')
PY
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
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    "$@"
echo "Finished: $(date --iso-8601=seconds)"
echo "Final model: ${FINAL_MODEL_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"

# mkdir -p logs
#sbatch scripts/training/submit_train_cpt.sh \
#    --attn-implementation flash_attention_2

#   sbatch scripts/training/submit_train_cpt.sh \
#        --attn-implementation sdpa \
#        --resume-from-checkpoint auto

# when occuring bug, re-running:
#DATASET_NAME=CPT-Proportional-5B
#MODEL_RUN_NAME=Qwen-Lex-Legal-CPT-using-Proportional-5B
#sbatch --exclude=pink7001       scripts/training/submit_train_cpt.sh
#--attn-implementation sdpa       --resume-from-checkpoint
#"${HOME}/scratch/llms_model/lexlegal_llms/checkpoints/Qwen-Lex-Legal-CPT-using-Proportional-5B/checkpoint-13000"
#--save-total-limit 1


#
# srun       --jobid=1785801   \
# --overlap       \
# --nodes=1      \
# --ntasks=1       \
# --cpus-per-task=1  \
# --pty bash
#  watch -n 2 nvidia-smi

#  ### 1. Prepare/tokenize CPT-V2-Legislation-3B
#
#  cd ~/scratch/dips_project/lex-legal-post-training
#  mkdir -p logs
#
#  DATASET_NAME=CPT-V2-Legislation-3B \
#  sbatch scripts/training/submit_prepare_cpt.sh
#
#  This will read:
#
#  ~/scratch/legal_data_process/datasets/CPT-V2-Legislation-3B
#
#  and write tokenized data to:
#
#  ~/scratch/legal_data_process/datasets/CPT-tokenized/CPT-V2-Legislation-3B-Qwen3
#
#  Monitor it:
#
#  squeue -u "${USER}"
#  tail -f logs/qwen3-cpt-prepare-<JOB_ID>.out
#
#  Wait until the preparation job finishes successfully. Confirm:
#
#  test -f \
#    ~/scratch/legal_data_process/datasets/CPT-tokenized/CPT-V2-Legislation-3B-Qwen3/
#    metadata.json \
#    && echo "Tokenized dataset is ready"
#
#  ### 2. Train with FlashAttention
#
#  DATASET_NAME=CPT-V2-Legislation-3B \
#  MODEL_RUN_NAME=Qwen-Lex-Legal-CPT-using-Legislation-3B \
#  sbatch scripts/training/submit_train_cpt.sh \
#      --attn-implementation flash_attention_2
#
#  The outputs will be:
#
#  Final model:
#  ~/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Legislation-3B
#
#  Checkpoints:
#  ~/scratch/llms_model/lexlegal_llms/checkpoints/Qwen-Lex-Legal-CPT-using-Legislation-3B
#
#  The training launcher will test the installed FlashAttention package with a small BF16
#  forward/backward operation before loading the model.
#
#  If the training job later stops after producing checkpoints, resume with the same
#  environment variables:
#
#  DATASET_NAME=CPT-V2-Legislation-3B \
#  MODEL_RUN_NAME=Qwen-Lex-Legal-CPT-using-Legislation-3B \
#  sbatch scripts/training/submit_train_cpt.sh \
#      --attn-implementation flash_attention_2 \
#      --resume-from-checkpoint auto
#
#  Do not run the training command until the preparation job has successfully produced
#  metadata.json.

#  Submit both jobs together using a Slurm afterok dependency:
#
#     cd ~/scratch/dips_project/lex-legal-post-training
#     mkdir -p logs
#
#     PREP_JOB_RESULT="$(
#         sbatch --parsable \
#             --export=ALL,DATASET_NAME=CPT-V2-Legislation-3B \
#             scripts/training/submit_prepare_cpt.sh
#     )"
#     PREP_JOB_ID="${PREP_JOB_RESULT%%;*}"
#
#     TRAIN_JOB_RESULT="$(
#         sbatch --parsable \
#             --dependency="afterok:${PREP_JOB_ID}" \
#             --export=ALL,DATASET_NAME=CPT-V2-Legislation-3B,MODEL_RUN_NAME=Qwen-Lex-Legal-
#             CPT-using-Legislation-3B \
#             scripts/training/submit_train_cpt.sh \
#             --attn-implementation flash_attention_2
#     )"
#     TRAIN_JOB_ID="${TRAIN_JOB_RESULT%%;*}"
#
#     echo "Preparation job: ${PREP_JOB_ID}"
#     echo "Training job:    ${TRAIN_JOB_ID}"
#     echo "Training dependency: afterok:${PREP_JOB_ID}"
#
#     The training job will remain pending until the preparation job finishes successfully. If
#     preparation fails, training will not start.
#
#     Check both jobs:
#
#     squeue -j "${PREP_JOB_ID},${TRAIN_JOB_ID}" \
#         -o "%.18i %.24j %.10T %.20R"
#
#     Monitor preparation:
#
#     tail -f "logs/qwen3-cpt-prepare-${PREP_JOB_ID}.out"
#
#     After preparation succeeds and training starts:
#
#     tail -f "logs/qwen3-8b-cpt-${TRAIN_JOB_ID}.out"
#
#     Outputs will be:
#
#     Tokenized dataset:
#     ~/scratch/legal_data_process/datasets/CPT-tokenized/CPT-V2-Legislation-3B-Qwen3
#
#     Final model:
#     ~/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Legislation-3B
#
#     Checkpoints:
#     ~/scratch/llms_model/lexlegal_llms/checkpoints/Qwen-Lex-Legal-CPT-using-Legislation-3B
#
#     done 11:39 PM