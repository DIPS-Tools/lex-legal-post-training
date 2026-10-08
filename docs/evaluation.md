# Base-versus-CPT legal evaluation

## 1. Purpose

This evaluation measures what legal continued pretraining (CPT) changes before supervised fine-tuning. It compares:

- `Qwen3-8B-Base`; and
- `Qwen3-8B-CPT-Proportional-5B`, the checkpoint produced from the 5B-token CPT mixture.

The two entry points are intentionally thin wrappers around one implementation:

```text
scripts/evaluation/evaluation_base_model.py
scripts/evaluation/evaluation_cpted_model.py
                  │
                  └── src/evaluation/
                      ├── common.py       data, prompts, metrics, manifests
                      ├── modeling.py     model loading, likelihoods, generation
                      └── runner.py       orchestration and result writing
```

Using one implementation is essential: a comparison is meaningful only when prompt construction, candidate scoring, decoding, and metrics are identical for both checkpoints.

## 2. Benchmarks and local data

The default evaluation root is:

```text
/home/mh1f25/scratch/legal_data_process/datasets/evaluation
```

The evaluator uses:

1. **Legal MMLU**: only `international_law`, `professional_law`, and `jurisprudence`. Test rows are scored; each subject's dev rows are used only as in-context demonstrations.
2. **LegalBench-Instruct**: the already-curated `inputs` prompt is passed to the model as stored. The code does not add a second instruction template.

The original LegalBench checkout under `datasets/evaluation/legalbench` is not read by these scripts. The scripts evaluate the converted LegalBench-Instruct table under `legalbench_instruct/data`. Perplexity evaluation is separate and remains available through `scripts/evaluation/evaluate_cpt_perplexity.py`.

Input filenames may be JSONL or Parquet. JSONL is preferred when both exist. Every consumed input file is SHA256-hashed in the result metadata.

## 3. Prompt and scoring protocol

### 3.1 Why no chat template is used

Both compared checkpoints are base causal language models, not instruction-tuned chat models. The evaluator therefore sends plain benchmark text and does not call `apply_chat_template`. This isolates the effect of CPT and avoids introducing an instruction format that neither checkpoint was trained to follow.

After SFT, create a separate evaluation setting that applies that model's exact chat template. Do not compare a chat-formatted SFT run directly with the plain-text base/CPT numbers and call the difference an SFT gain; report it as a separate protocol.

### 3.2 MMLU

The default is 5-shot evaluation. A prompt has this form:

```text
The following are multiple-choice questions about professional law.

Question: <dev question>
A. <choice>
B. <choice>
C. <choice>
D. <choice>
Answer: <gold dev letter>

... four more demonstrations ...

Question: <test question>
A. <choice>
B. <choice>
C. <choice>
D. <choice>
Answer:
```

For the test item the evaluator calculates the conditional log-likelihood of ` A`, ` B`, ` C`, and ` D`, then selects the highest score. Generation and regex parsing are deliberately avoided for multiple choice because candidate likelihood is deterministic and does not confound knowledge with output-format compliance.

Reported metrics are accuracy and balanced accuracy per subject, plus macro averages across the three subjects. The primary MMLU comparison should use macro accuracy.

### 3.3 LegalBench-Instruct

Each row's `inputs` value is already the complete prompt. The evaluator groups rows by `task_name` and resolves one task-level scoring configuration from the complete task:

- If the task has 2 through 10 distinct gold labels, it is treated as closed-label classification. Every permitted label is scored by conditional likelihood. Balanced accuracy is the task's primary score; ordinary accuracy is also retained.
- Otherwise it is treated as open generation. Decoding is greedy and the primary score is conservative normalized exact match. Normalization applies Unicode NFKC, removes a leading `Answer:`-style prefix, keeps the first line, collapses whitespace, and case-folds. It does not search an explanation for a convenient answer.

The resolved mode and label set for every task are written to `legalbench/resolved_task_configuration.json`. Deriving candidate labels from the task schema is convenient for the current converted dataset, but it is not a substitute for a fixed official task configuration. Before publishing final results, inspect this file once, freeze it with the experiment artifacts, and use the same configuration for both models.

The LegalBench headline is the unweighted macro average of task primary scores. This prevents large tasks from dominating the benchmark. The evaluator also reports macro scores by `task_type`.

## 4. LegalBench overlap views

Because LawInstruct may later be used for SFT, one undifferentiated LegalBench score is insufficient. Each run reports three views:

| View | Meaning |
|---|---|
| `full` | All evaluated LegalBench-Instruct tasks; useful but not leakage-free after LawInstruct SFT. |
| `dataset_held_out` | Removes task families whose source dataset occurs in LawInstruct, following Appendix C.2.1/Table 3. |
| `task_held_out` | Removes semantically overlapping task families, following Appendix C.2.2/Table 4. |

The dataset-held-out selectors are `contract_nli_*`, `cuad_*`, `international_citizenship_questions`, `maud_*`, `opp115_*`, `overruling`, `privacy_policy_qa`, `sara_*`, and `unfair_tos`.

The task-held-out selectors are `function_of_decision_section`, `oral_argument_question_purpose`, `diversity_*`, `personal_jurisdiction`, `contract_nli_*`, `cuad_*`, `jcrew_blocker`, `unfair_tos`, and `contract_qa`.

With the current local 162-task snapshot, the expected retained counts are 61 for dataset-held-out and 98 for task-held-out. These differ slightly from historical paper counts because the benchmark snapshots differ. The evaluator writes exact retained and excluded manifests for the actual run; those files and the input hash are authoritative.

These views mainly matter once LawInstruct SFT has occurred. For the pre-SFT base-versus-CPT experiment, they are still useful diagnostics and ensure continuity with the later study.

## 5. Model locations and output locations

Defaults are:

| Entry point | Model | Output |
|---|---|---|
| `evaluation_base_model.py` | `/home/mh1f25/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base` | `results/evaluation/Qwen3-8B-Base` |
| `evaluation_cpted_model.py` | `/home/mh1f25/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Proportional-5B` | `results/evaluation/Qwen3-8B-CPT-Proportional-5B` |

Override either location with `--model-path` or `--output-dir`. A model directory must contain `config.json`.

Each result directory contains:

```text
evaluation_summary.json                 final completion marker and environment
mmlu/
├── predictions/<subject>.jsonl
├── metrics/<subject>.json
└── summary.json
legalbench/
├── predictions/<task>.jsonl
├── task_metrics/<task>.json
├── manifests/<view>_{retained,excluded}.jsonl
├── resolved_task_configuration.json
└── summary.json
```

Predictions include candidate scores, gold answers, selected predictions, and correctness. Keep them for error analysis and significance testing. `evaluation_summary.json` is written last and is the marker of a completed run. A completed directory is not overwritten unless `--overwrite` is explicitly supplied.

## 6. Installation and preflight checks

Activate the same environment used by the project, then verify imports and paths:

```bash
cd /home/mh1f25/scratch/lex-legal-post-training
conda activate legal-data-process

python -c "import torch, transformers, pyarrow; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
test -f /home/mh1f25/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base/config.json
test -f /home/mh1f25/scratch/legal_data_process/datasets/evaluation/legalbench_instruct/data/train-00000-of-00001.jsonl
```

The evaluator requires CUDA and loads weights as BF16 with `device_map="auto"`. `flash_attention_2` is the default attention backend. If FlashAttention is unavailable, pass `--attn-implementation sdpa`.

## 7. Recommended execution order

### 7.1 Run a small smoke test

Use a disposable output directory. This confirms model loading, prompt construction, scoring, and output writing without running the complete benchmark:

```bash
python scripts/evaluation/evaluation_base_model.py \
  --benchmarks mmlu legalbench \
  --mmlu-subject jurisprudence \
  --legalbench-task abercrombie \
  --example-limit 2 \
  --attn-implementation sdpa \
  --output-dir results/evaluation/smoke-base
```

`--example-limit` limits scored rows per subject/task. LegalBench task mode and candidate labels are still inferred from the complete task, so the smoke limit cannot accidentally change the scoring protocol.

### 7.2 Evaluate the base model

```bash
python scripts/evaluation/evaluation_base_model.py
```

### 7.3 Evaluate the CPT checkpoint

Run this only after the CPT checkpoint is complete and loadable:

```bash
python scripts/evaluation/evaluation_cpted_model.py
```

If the CPT checkpoint is at a different path:

```bash
python scripts/evaluation/evaluation_cpted_model.py \
  --model-path /absolute/path/to/final-cpt-checkpoint
```

The two full commands must use identical values for all settings other than model label, model path, and output directory. In particular, keep `--num-fewshot`, `--candidate-score`, attention implementation, context limit, and benchmark selection fixed.

## 8. Useful options

```text
--benchmarks mmlu legalbench   run both or select one
--num-fewshot 5               MMLU dev demonstrations per subject
--batch-size 8                likelihood/generation batch size
--candidate-score sum         sum or mean continuation log-likelihood
--max-input-tokens 32768      fail instead of silently truncating
--open-max-new-tokens 64      maximum LegalBench open-answer output
--attn-implementation ...     flash_attention_2, sdpa, or eager
--legalbench-task NAME        repeat to select tasks
--mmlu-subject NAME           repeat to select legal MMLU subjects
--example-limit N             per-task development/smoke limit
--allow-download              permit missing tokenizer/model files to download
--overwrite                   replace results in a completed output directory
```

The default candidate score is summed continuation log-likelihood, which is standard for choosing among MMLU letters. LegalBench labels can have different token lengths; therefore retain the default for the main experiment but run `--candidate-score mean` as a sensitivity check if label length is materially unbalanced. Do not choose whichever version produces the preferred result after seeing both.

## 9. Cluster guidance

One evaluation process can use one GPU. Do not launch it with multiple Slurm tasks: that would duplicate the complete evaluation. Request one task, suitable CPU/memory, and a GPU with enough memory for an 8B BF16 model plus activations. If memory is tight, reduce `--batch-size`; this should not change deterministic predictions.

For unattended runs, direct Slurm's standard output to a persistent log and keep Hugging Face in offline mode when all model files are local:

```bash
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
srun python scripts/evaluation/evaluation_base_model.py
```

The scripts are not distributed-evaluation launchers. `device_map="auto"` may place one model across multiple visible GPUs, but data-parallel multi-process evaluation and cross-process result merging are intentionally not implemented.

## 10. Reading and reporting results

For the base-versus-CPT table, report at minimum:

- MMLU accuracy for each of the three legal subjects and macro accuracy;
- LegalBench full, dataset-held-out, and task-held-out macro task scores;
- the number of tasks/examples in every view;
- checkpoint identifiers, input SHA256 values, Git commit, package versions, few-shot count, and candidate-scoring rule.

Calculate change as `CPT score - base score` using the two `evaluation_summary.json` files. Do not treat small point differences as conclusive without uncertainty analysis. Since both models answer exactly the same items, paired bootstrap confidence intervals over MMLU examples and over LegalBench tasks are appropriate follow-up analyses.

Inspect these failure modes before interpreting gains:

- a high classification score but low `format_success_rate` on generated tasks;
- prompts rejected for exceeding the configured context rather than silently truncated;
- different resolved LegalBench task configurations between runs;
- different evaluation file hashes;
- accidental use of a chat template for only one checkpoint;
- evaluation examples included in CPT or later SFT data.

## 11. Reproducibility policy

Keep the base run, CPT run, resolved LegalBench task configuration, manifests, and exact dataset hashes together with the experiment. Never modify an evaluation set after one model has been evaluated without rerunning every comparison model. The CPT corpus must be audited for benchmark contamination independently; held-out LegalBench views address later LawInstruct overlap, not memorization introduced during CPT.

For final paper numbers, run each command from a clean Git commit, preserve the Slurm job log, and archive the checkpoint's training manifest alongside these evaluation outputs.
