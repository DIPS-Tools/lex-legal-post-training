# Base-versus-CPT legal evaluation

## 1. Purpose

This evaluation measures what legal continued pretraining (CPT) changes before supervised fine-tuning. It compares:

- `Qwen3-8B-Base`; and
- `Qwen3-8B-CPT-Legislation-3B`, the checkpoint produced from the legislation-focused 3B-token CPT mixture.

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

The evaluator uses three separately reported benchmarks:

1. **Original LegalBench**: each local `test.tsv` row is rendered with the task's pinned official `base_prompt.txt`; these prompts already contain the task demonstrations. Generations are scored by the pinned official `evaluation.py`. `rule_qa` remains manual-only, as specified upstream.
2. **LegalBench-Instruct**: the already-curated `inputs` prompt is passed to the model as stored. The code does not add a second instruction template.
3. **Legal MMLU**: only `international_law`, `professional_law`, and `jurisprudence`. Test rows are scored; each subject's dev rows are used only as in-context demonstrations.

The Hugging Face LegalBench snapshot under `datasets/evaluation/legalbench` supplies test data and `task_metadata.json`. A separate pinned HazyResearch checkout under `datasets/evaluation/legalbench_official` supplies official base prompts and scoring code. LegalBench-Instruct comes from `legalbench_instruct/data`. Perplexity evaluation is separate and remains available through `scripts/evaluation/evaluate_cpt_perplexity.py`.

An `evaluation_ready` directory is intentionally not created. Original LegalBench templates are rendered dynamically, LegalBench-Instruct already stores complete prompts in `inputs`, and MMLU prompts are assembled from canonical `dev` and `test` records. Prompt construction and tokenization happen at runtime so the same immutable evaluation data can be used for base, CPT, and later SFT protocols without duplicating or silently drifting datasets.

The pinned official LegalBench checkout used when this note was updated is commit:

```text
b46bf4ffae90524b2b72aaa30e7745fe9db64481
```

If it is absent, create it once and do not pull a different revision between model runs:

```bash
git clone https://github.com/HazyResearch/legalbench.git \
  "${HOME}/scratch/legal_data_process/datasets/evaluation/legalbench_official"
git -C "${HOME}/scratch/legal_data_process/datasets/evaluation/legalbench_official" \
  checkout b46bf4ffae90524b2b72aaa30e7745fe9db64481
```

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

#### 3.2.1 Exact MMLU example from the local data

The first test record in
`datasets/evaluation/mmlu/jurisprudence/test-00000-of-00001.jsonl` is:

```json
{
  "question": "Which statement best explains the purpose of Hart's distinction between 'being obliged' and 'having an obligation'?",
  "choices": [
    "It demonstrates the difference between the internal and the external aspect of a rule.",
    "It refutes the natural lawyer' view of the role of morality in law.",
    "It explains the nature of power-conferring rules.",
    "It illuminates the concept of a rule."
  ],
  "answer": "D",
  "answer_index": 3,
  "subject": "jurisprudence"
}
```

With the default `--num-fewshot 5`, the exact plain-text prompt constructed for
this record is:

```text
The following are multiple-choice questions about jurisprudence.

Question: Which position does Rawls claim is the least likely to be adopted by the POP (people in the original position)?
A. The POP would choose equality above liberty.
B. The POP would opt for the 'maximin' strategy.
C. The POP would opt for the 'difference principle'.
D. The POP would reject the 'system of natural liberty.'
Answer: A

Question: Functions of the law include all but which of the following?
A. maximizing individual freedom
B. providing a basis for compromise
C. keeping the peace
D. promoting the principles of the free enterprise system
Answer: D

Question: Which word best summarizes Weber's explanation of the development of formally rational law?
A. Authority.
B. Charisma.
C. Co-operation.
D. Capitalism.
Answer: D

Question: The ________ School of jurisprudence postulates that the law is based on what is "correct."
A. Natural Law
B. Analytical
C. Historical
D. Sociological
Answer: A

Question: Iverson Jewelers wrote a letter to Miller, 'We have received an exceptionally fine self winding Rolox watch which we will sell to you at a very favorable price.'
A. The letter is an offer to sell
B. A valid offer cannot be made by letter.
C. The letter contains a valid offer which will terminate within a reasonable time.
D. The letter lacks one of the essential elements of an offer.
Answer: D

Question: Which statement best explains the purpose of Hart's distinction between 'being obliged' and 'having an obligation'?
A. It demonstrates the difference between the internal and the external aspect of a rule.
B. It refutes the natural lawyer' view of the role of morality in law.
C. It explains the nature of power-conferring rules.
D. It illuminates the concept of a rule.
Answer:
```

No chat or dialogue wrapper is added. The tokenizer receives this prompt with
`add_special_tokens=False`. The evaluator then performs four conditional-
likelihood calculations by appending one of these continuations (each includes
the leading space):

```text
 A
 B
 C
 D
```

Thus, the four scored sequences end in `Answer: A`, `Answer: B`, `Answer: C`,
and `Answer: D`. The model's MMLU prediction is the letter whose continuation
has the highest selected log-probability; it does not generate a sentence. For
this record, the expected output is `D`, because `answer` is `D` and the
zero-based `answer_index` is `3`. A correct saved prediction has the essential
fields:

```json
{
  "benchmark": "mmlu",
  "subject": "jurisprudence",
  "gold": "D",
  "prediction": "D",
  "correct": true
}
```

The actual prediction record additionally contains all four candidate scores,
their summed and mean log-probabilities, and token counts. The Parquet MMLU
files store the zero-based integer directly in `answer` (for example,
`"answer": 3`), while the converted JSONL files store the letter in `answer`
and the integer in `answer_index`. The loader normalizes both representations
to the same A-D label.

To reproduce and inspect this exact prompt without loading a model, run from
the project root:

```bash
python - <<'PY'
from pathlib import Path
from src.evaluation.common import build_mmlu_prompt, find_split_file, load_table

root = Path.home() / "scratch/legal_data_process/datasets/evaluation/mmlu"
subject = "jurisprudence"
dev = load_table(find_split_file(root / subject, "dev", prefer_jsonl=False))
test = load_table(find_split_file(root / subject, "test", prefer_jsonl=True))
print(build_mmlu_prompt(subject, test[0], dev[:5]))
print("Expected output:", test[0]["answer"])
PY
```

### 3.3 LegalBench-Instruct

Each row's `inputs` value is already the complete prompt. The evaluator groups rows by `task_name` and resolves one task-level scoring configuration from the complete task:

- If the task has 2 through 10 distinct gold labels, it is treated as closed-label classification. Every permitted label is scored by conditional likelihood. Balanced accuracy is the task's primary score; ordinary accuracy is also retained.
- Otherwise it is treated as open generation. Decoding is greedy and the primary score is conservative normalized exact match. Normalization applies Unicode NFKC, removes a leading `Answer:`-style prefix, keeps the first line, collapses whitespace, and case-folds. It does not search an explanation for a convenient answer.

The resolved mode and label set for every task are written to `legalbench_instruct/resolved_task_configuration.json`. Deriving candidate labels from the task schema is convenient for the current converted dataset, but it is not a substitute for a fixed official task configuration. Before publishing final results, inspect this file once, freeze it with the experiment artifacts, and use the same configuration for both models.

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
| `evaluation_cpted_model.py` | `/home/mh1f25/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Legislation-3B` | `results/evaluation/Qwen3-8B-CPT-Legislation-3B` |

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
└── summary.json
legalbench_instruct/
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
conda activate lex_legal_rl_post_training_env

python -m pip install nltk
python -c "import torch, transformers, pyarrow, sklearn, nltk; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
test -f /home/mh1f25/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base/config.json
test -f /home/mh1f25/scratch/legal_data_process/datasets/evaluation/legalbench_instruct/data/train-00000-of-00001.jsonl
```

The evaluator requires CUDA and loads weights as BF16 with `device_map="auto"`. `flash_attention_2` is the default attention backend. If FlashAttention is unavailable, pass `--attn-implementation sdpa`.

## 7. Recommended execution order

All commands below use these exact checkpoints:

```text
Base: /home/mh1f25/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base
CPT:  /home/mh1f25/scratch/llms_model/lexlegal_llms/cpt-model/Qwen-Lex-Legal-CPT-using-Legislation-3B
```

### 7.1 Run a small smoke test

Use a disposable output directory. This confirms model loading, prompt construction, scoring, and output writing without running the complete benchmark:

```bash
python scripts/evaluation/evaluation_base_model.py \
  --benchmarks mmlu legalbench legalbench_instruct \
  --mmlu-subject jurisprudence \
  --legalbench-task abercrombie \
  --example-limit 2 \
  --attn-implementation sdpa \
  --output-dir results/evaluation/smoke-base
```

`--example-limit` limits scored rows per subject/task. LegalBench task mode and candidate labels are still inferred from the complete task, so the smoke limit cannot accidentally change the scoring protocol.

### 7.2 Evaluate the base model through Slurm

```bash
mkdir -p logs
sbatch scripts/evaluation/submit_evaluation.sh base
```

### 7.3 Evaluate the CPT checkpoint through Slurm

Run this only after the CPT checkpoint is complete and loadable:

```bash
mkdir -p logs
sbatch scripts/evaluation/submit_evaluation.sh cpt
```

If the CPT checkpoint is at a different path:

```bash
sbatch scripts/evaluation/submit_evaluation.sh cpt \
  --model-path /absolute/path/to/final-cpt-checkpoint \
  --output-dir results/evaluation/custom-cpt
```

Both commands run all three benchmarks by default. They must use identical values for all settings other than model label, model path, and output directory. In particular, keep `--num-fewshot`, `--candidate-score`, attention implementation, context limit, generation budget, and benchmark selection fixed.

The explicit equivalent commands are:

```bash
sbatch scripts/evaluation/submit_evaluation.sh base \
  --benchmarks legalbench legalbench_instruct mmlu \
  --attn-implementation flash_attention_2

sbatch scripts/evaluation/submit_evaluation.sh cpt \
  --benchmarks legalbench legalbench_instruct mmlu \
  --attn-implementation flash_attention_2
```

```bash
#running: qwen3-legal-eval-1801846.out
BASE_JOB_ID="$(
  sbatch --parsable \
      scripts/evaluation/submit_evaluation.sh base \
      --benchmarks legalbench legalbench_instruct mmlu \
      --attn-implementation flash_attention_2
)"
echo "Base job: ${BASE_JOB_ID}"

#qwen3-legal-eval-1801847.out
CPT_JOB_ID="$(
  sbatch --parsable \
      scripts/evaluation/submit_evaluation.sh cpt \
      --benchmarks legalbench legalbench_instruct mmlu \
      --attn-implementation flash_attention_2
)"
echo "CPT job: ${CPT_JOB_ID}"

```


### 7.4 Safer benchmark-split submissions

Original LegalBench and LegalBench-Instruct each contain roughly 90,000 test
examples. If one combined job risks reaching the 60-hour Slurm limit, submit
each benchmark separately. Distinct output roots prevent concurrent jobs from
overwriting `evaluation_summary.json`:

```bash
for benchmark in legalbench legalbench_instruct mmlu; do
  sbatch scripts/evaluation/submit_evaluation.sh base \
    --benchmarks "${benchmark}" \
    --output-dir "results/evaluation/Qwen3-8B-Base-${benchmark}"

  sbatch scripts/evaluation/submit_evaluation.sh cpt \
    --benchmarks "${benchmark}" \
    --output-dir "results/evaluation/Qwen3-8B-CPT-Legislation-3B-${benchmark}"
done
```



Use the same split strategy later for SFT. Do not combine LegalBench and
LegalBench-Instruct scores into one average because they are related benchmark
variants rather than independent datasets.

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
