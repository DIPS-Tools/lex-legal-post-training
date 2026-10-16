# Legal SFT: preparation, training, and resumption

## Scope

`scripts/training/train_sft.py` now provides two modes:

1. `--prepare-only`: stream the five local instruction datasets, filter, sample, format and tokenize them on CPU.
2. Training (default): load the completed prepared dataset and fine-tune a local base/CPT model using assistant-only loss.

The implementation lives in `src/training/sft_data.py` and `src/training/sft_trainer.py`.

Default input model:

```text
/home/mh1f25/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Legislation-3B
```

Default final model:

```text
/home/mh1f25/scratch/llms_model/lexlegal_llms/sft-model/Qwen-Lex-Legal-CPT-using-Legislation-3B-SFT
```

Checkpoints use a separate `checkpoints/Qwen-Lex-Legal-CPT-using-Legislation-3B-SFT` directory. Existing output models are protected: a nonempty final directory is rejected.

## Relationship to the papers

[SaulLM-7B Section 3.2](https://arxiv.org/html/2403.03883v1#S3.SS2) uses SlimOrca, MetaMathQA, UltraChat and Glaive Code Assistant v2 for its general instruction pool, reporting 600K curated general instructions. Its legal instructions are synthetic conversations.

Here, public English LawInstruct supplies the legal pool. The default **600K legal + 600K general examples** follows this project's `legal_data_process/docs/SFT_data_generation.md` plan. The equal legal/general mixture, learning rate and epoch settings below are experimental choices; they are not claimed to reproduce SaulLM exactly. Record this distinction in the paper.

## Data preparation behavior

Input root: `~/scratch/legal_data_process/datasets/SFT-datasets`.

| Source | Accepted input | Conversion |
|---|---|---|
| LawInstruct | `data/*-train-*.jsonl.xz` | User = instruction + prompt; assistant = answer; all three language fields must be `en` |
| SlimOrca | `oo-labeled_correct.gpt4.sharegpt.jsonl` | ShareGPT system/human/gpt → system/user/assistant |
| MetaMathQA | `MetaMathQA-395K.json` | query → user; response → assistant |
| UltraChat | `data/train_sft-*.parquet` only | Retain multi-turn messages |
| Glaive | `glaive_code_assistant_v2.json` | question → user; answer → assistant |

The script normalizes Unicode and line endings, checks role alternation, rejects empty messages and embedded ChatML delimiters, and preserves code indentation. Long conversations are rejected and counted; there is no silent truncation or example packing.

Every legal-MMLU dev/validation/test question is indexed from the three legal subjects. Examples containing at least 80% of a question's unique five-word shingles are removed, including matching text in assistant responses. Shorter questions use shorter shingles. This catches lexical copies/near copies; it is not a semantic paraphrase detector. Source subsets explicitly identifying MMLU are also excluded.

SQLite on shared storage holds candidates and tokenized records, so all of LawInstruct need not fit in RAM. Exact normalized prompt/response pairs are deduplicated globally. If one normalized prompt has different response targets, all surviving members of that conflict group are quarantined from selection. Source precedence for identical pairs follows the order in the table above.

Validation is assigned by deterministic group hash before sampling: LawInstruct raw prompt, MetaMath original_question when present, otherwise conversation user text. Exact prompt overlap propagates validation membership across sources and groups. This does not guarantee grouping every paraphrase or every task derived from the same underlying document.

Legal quotas are proportional to eligible source/subset counts; general quotas are proportional to eligible source counts. Largest-remainder rounding gives the exact requested example totals, without replacement. Insufficient candidates cause an explicit failure. Data order is deterministically shuffled by hash priority.

**Implemented limits relative to the broader planning note:** full-pool MinHash near deduplication, semantic decontamination, and a full LegalBench example-overlap audit are not implemented here. The manifest states the actual policy. Do not describe these artifacts as completely contamination-free. LegalBench-related LawInstruct sources are retained; report the existing dataset-held-out and task-held-out LegalBench views after SFT. Existing source license terms still apply.

## Formatting and loss

The saved tokenizer uses an explicit Qwen ChatML template:

```text
<|im_start|>user
Question text<|im_end|>
<|im_start|>assistant
Answer text<|im_end|>
```

Only assistant content and its `<|im_end|>` token contribute to loss. System text, user text, role headers and padding receive label `-100`. Multi-turn conversations supervise each assistant turn. The code requires the ChatML tokens to already exist in the Qwen tokenizer and does not expand the vocabulary.

This is plain answer SFT without a thinking prefix or synthetic reasoning tags. The exact template is saved with the final model. For subsequent SFT evaluation/inference, apply this saved template with `add_generation_prompt=True`; the current raw-completion Base/CPT evaluation path does not automatically become chat evaluation. Keep benchmark content unchanged and explicitly document prompt differences when comparing pre-SFT and post-SFT checkpoints.

## Install additional dependencies

In the existing training Conda environment:

```bash
conda activate lex_legal_rl_post_training_env
python -m pip install 'datasets>=4.4' 'pyarrow>=16' 'ijson>=3.2' 'jinja2>=3.1'
```

PyTorch, Transformers >=4.51, Accelerate, DeepSpeed, TensorBoard and FlashAttention are already needed by the CPT environment. `requirements-sft.txt` lists the full requirements through `requirements-training.txt`; avoid unnecessarily reinstalling the working CUDA/PyTorch stack.

## Submit preparation and dependent training

Use the project path on your server (the `dips_project` checkout if that is where the code is synced). Update that checkout with these files before submitting.

```bash
cd "${HOME}/scratch/dips_project/lex-legal-post-training"
mkdir -p logs
PROJECT_ROOT="$(pwd)"

PREP_JOB_ID="$(
    sbatch --parsable --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
        scripts/training/submit_prepare_sft.sh
)"
PREP_JOB_ID="${PREP_JOB_ID%%;*}"

SFT_JOB_ID="$(
    sbatch --parsable \
        --dependency="afterok:${PREP_JOB_ID}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
        scripts/training/submit_train_sft.sh \
        --attn-implementation flash_attention_2
)"
echo "Preparation: ${PREP_JOB_ID}; SFT: ${SFT_JOB_ID}"
```

Preparation uses one process on `amd`; training requests four GPUs on `i7_h200`. Partition/resource defaults may be overridden through sbatch before the script path. Do not run full preparation on a login node.

Prepared data defaults to:

```text
~/scratch/legal_data_process/datasets/SFT-processed/LawInstruct-General-1.2M-Qwen3/
  manifest.json             # Written last: completed preparation
  tokenizer/                # Fixed tokenizer/template for all comparisons
  train.jsonl               # Selected canonical conversations and provenance
  validation.jsonl
  dataset/                  # Hugging Face train/validation Arrow data
  rejections.jsonl
  conflicting_prompts.jsonl
  candidates.sqlite         # Large preparation intermediate, retained for audit
  arrow_cache/              # Preparation intermediate
```

The manifest reports source hashes, selected quotas, rejection counts, tokenizer signature, total and supervised tokens, and train/validation composition. The mixture is 50:50 by examples, not necessarily by tokens. Intermediates can be large; put preparation on shared scratch with sufficient quota. No automatic deletion occurs. An interrupted preparation is not resumable; use a new preparation directory after diagnosing the failure.

When preparation already completed, submit only `submit_train_sft.sh`. Every Base/CPT comparison should use the same `--prepared-dir` and hyperparameters. The tokenizer signature is checked against each input model.

## Defaults and overrides

| Argument | Default | Purpose |
|---|---|---|
| `--model-path` | Legislation-3B CPT | Initialization checkpoint |
| `--prepared-dir` | Path above | Shared reusable SFT data |
| `--output-dir` | `sft-model/<input model name>-SFT` | Final model |
| `--checkpoint-dir` | `checkpoints/<output folder name>` | Resume checkpoints |
| `--legal-examples` | 600000 | Legal training quota, preparation only |
| `--general-examples` | 600000 | General training quota, preparation only |
| `--validation-fraction` | 0.01 | Group holdout probability |
| `--validation-examples-per-domain` | 2000 | Maximum held-out sample per pool |
| `--max-length` | 4096 | Reject longer conversations; must match prepared manifest |
| `--num-train-epochs` | 1 | Starting experimental SFT budget |
| `--max-steps` | -1 | Positive value overrides epochs |
| `--learning-rate` | 2e-5 | Experimental initial learning rate |
| `--warmup-ratio` | 0.03 | Converted to integer warmup steps |
| `--weight-decay` | 0 | AdamW decay |
| `--per-device-train-batch-size` | 1 | Conversations per GPU microbatch |
| `--gradient-accumulation-steps` | 16 | Global batch = 64 on four GPUs |
| `--eval-steps`, `--save-steps` | 250 | Validation/checkpoint frequency |
| `--save-total-limit` | 2 | Retained checkpoint count; a new save temporarily needs extra disk |
| `--attn-implementation` | flash_attention_2 | Also supports sdpa/eager |
| `--resume-from-checkpoint` | unset | Explicit checkpoint or auto |
| `--seed` | 42 | Preparation, shuffle and training seed |

Run `python scripts/training/train_sft.py --help` for the full CLI. Full-parameter BF16 training uses ZeRO-3 and gradient checkpointing. No GPU training begins during `--prepare-only`.

For another input checkpoint, export both paths/names for the training wrapper:

```bash
MODEL_PATH="${HOME}/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base" \
MODEL_RUN_NAME=Qwen3-8B-Base-SFT \
sbatch scripts/training/submit_train_sft.sh --attn-implementation flash_attention_2
```

## Smoke test and resumption

For a CPU preparation smoke test, use a distinct directory and small quotas:

```bash
sbatch scripts/training/submit_prepare_sft.sh \
    --prepared-dir "${HOME}/scratch/legal_data_process/datasets/SFT-processed/smoke-Qwen3" \
    --max-records-per-source 10000 \
    --legal-examples 100 --general-examples 100 \
    --validation-fraction 0.1 --validation-examples-per-domain 20
```

If too few English legal candidates fit within this prefix, increase `--max-records-per-source`. Smoke artifacts are explicitly marked and training requires `--allow-smoke-data` for them. Use a separate output/checkpoint name and `--max-steps 2` when GPU smoke testing.

After a training time limit, resume with unchanged model, prepared data and training schedule:

```bash
sbatch scripts/training/submit_train_sft.sh \
    --attn-implementation flash_attention_2 \
    --resume-from-checkpoint auto
```

Auto selects the highest numbered checkpoint containing `trainer_state.json` and checks for DeepSpeed optimizer state. It does not repair corrupted checkpoint files. Resume checks the manifest hash, initial model, world size, global batch and optimization settings. An unrelated failure such as disk exhaustion must be fixed before resuming.

The final folder contains the HF model, tokenizer/chat template, `sft_run_configuration.json`, and `sft_training_summary.json`. The summary records training steps, losses and data statistics. It is separate from task-benchmark evaluation.

## Verification

`tests/test_sft.py` exercises raw schema adapters, MMLU lexical filtering, exact quota selection, group-separated validation, multi-turn assistant masks, EOS/padding behavior, tokenizer round trips, checkpoint selection, and tiny Qwen CPU training/save/reload. Full 8B CUDA/DeepSpeed training must be verified on the HPC allocation.
