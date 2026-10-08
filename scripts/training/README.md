# Qwen3 continued pretraining

This pipeline performs ordinary causal-language-model training on raw documents. It
does not apply a chat template and does not create user/assistant turns.

## Files

- `prepare_cpt_dataset.py` validates final JSONL shards, assigns a stable split, adds
  EOS after each document, and writes flat `uint32` token files.
- `train_cpt.py` exposes fixed-length memory-mapped blocks and runs full-parameter
  Qwen3 causal-LM training with resumable Hugging Face/DeepSpeed checkpoints.
- `submit_prepare_cpt.sh` runs the expensive validation/tokenization pass on a CPU job.
- `submit_train_cpt.sh` optionally stages prepared tokens on node storage and launches
  distributed training.
- `../evaluation/evaluate_cpt_perplexity.py` evaluates combined, legal, and replay
  held-out token streams.

## Default run

From the project root, create the Slurm log directory once. Prepare the token stream,
then submit training after the preparation job succeeds:

```bash
mkdir -p logs
sbatch scripts/training/submit_prepare_cpt.sh
sbatch scripts/training/submit_train_cpt.sh
```

Defaults:

- base model: `$HOME/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base`
- corpus: `$HOME/scratch/legal_data_process/datasets/CPT-Proportional-5B`
- final model: `$HOME/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Proportional-5B`
- sequence length: 4096
- hardware request: four H200 GPUs from the `i7_h200` partition

The preparation code refuses to use `.work/preprocessed`; final JSONL shards and
`manifest.json` must be directly inside the selected corpus directory.
The preparer classifies the replay validation subset from record metadata; override
`--replay-regex` when your manifest uses different replay corpus names.

## Select another corpus

Set the corpus and run names together so outputs cannot be confused:

```bash
DATASET_NAME=CPT-V1-EU-1B sbatch scripts/training/submit_prepare_cpt.sh

DATASET_NAME=CPT-V1-EU-1B \
MODEL_RUN_NAME=Qwen-Lex-Legal-CPT-using-EU-1B \
sbatch scripts/training/submit_train_cpt.sh
```

The same pattern works for `CPT-V2-Legislation-3B`. Paths can be overridden with
`DATASET_DIR`, `TOKENIZED_DIR`, `MODEL_PATH`, `FINAL_MODEL_DIR`, and
`CHECKPOINT_DIR`.

## Smoke test and resumption

```bash
PREPARE_DATA=0 sbatch scripts/training/submit_train_cpt.sh --max-steps 20 --no-eval
```

Resume the most recent complete Trainer checkpoint with:

```bash
PREPARE_DATA=0 sbatch scripts/training/submit_train_cpt.sh \
  --resume-from-checkpoint auto
```

Do not add `--overwrite-final-output` unless intentionally replacing an already
completed final model. Existing resumable checkpoints also require an explicit
resume argument.

## Hardware adjustment

The checked-in ZeRO-3 configuration assumes multiple large GPUs and does not use CPU
offload. If the Slurm GPU request is changed, set `GPUS_PER_NODE` to the same count
and reconsider gradient accumulation so the desired global token batch is retained:

```text
global tokens/update = sequence length * per-GPU batch * GPU count * accumulation
```

For the defaults this is `4096 * 1 * 4 * 16 = 262,144` tokens per optimizer update.
