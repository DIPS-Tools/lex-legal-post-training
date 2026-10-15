# Evaluation comparison: Qwen3-8B Base and CPT models

## 1. Purpose and models

This note compares three checkpoints before supervised fine-tuning (SFT):

1. **Qwen3-8B-Base** — unchanged base model.
2. **Qwen3-8B-CPT-Legislation-3B** — continued pretraining (CPT) on approximately 3B legislation tokens.
3. **Qwen3-8B-CPT-Proportional-5B** — CPT on approximately 5B tokens from the proportional legal/general replay mixture.

The result files are:

- `results/evaluation/Qwen3-8B-Base/evaluation_summary.json`
- `results/evaluation/Qwen3-8B-CPT-Legislation-3B/evaluation_summary.json`
- `results/evaluation/Qwen3-8B-CPT-Proportional-5B/evaluation_summary.json`

## 2. How the comparison was performed

### 2.1 Comparability checks

Before comparing scores, verify that all runs use the same:

- evaluation code commit;
- benchmark files and dataset hashes;
- benchmark/task selection;
- prompt construction and number of few-shot examples;
- candidate-scoring method;
- random seed;
- generation limits and answer parsing;
- model precision and attention implementation, where possible.

For these runs, the dataset hashes, code commit, five-shot setting, `candidate_score=sum`, and seed 42 match. Base and Legislation-3B used `flash_attention_2`; Proportional-5B used `sdpa`. That backend difference should be removed for the final paper comparison, especially when differences are small.

### 2.2 Aggregate comparison

For each model, read the aggregate fields from `evaluation_summary.json`:

- LegalBench-Instruct: full, dataset-held-out, and task-held-out macro scores;
- MMLU: macro accuracy and each selected subject's accuracy;
- original LegalBench: full and held-out macro official scores, subject to the scoring warning in Section 6.

Convert a score in `[0, 1]` to a percentage by multiplying it by 100. Compute the percentage-point change from Base as:

```text
delta_pp = (cpt_score - base_score) * 100
```

Example:

```text
Legislation-3B LegalBench-Instruct full delta
= (0.5020513 - 0.5824927) * 100
= -8.044 percentage points
```

Use **percentage points**, not percent, when subtracting two accuracy values.

### 2.3 Subject- and task-level comparison

Aggregate means can hide where changes occurred. The analysis therefore also compares:

- MMLU correct answers and accuracy for each subject;
- the score of every LegalBench-Instruct task;
- counts of tasks that improved, worsened, or tied relative to Base;
- the scoring mode used by each task (`candidate_logprob` or generation).

For each task:

```text
task_delta = cpt_task_score - base_task_score
```

- `task_delta > 0`: improved;
- `task_delta < 0`: worsened;
- `task_delta == 0`: tied.

Small floating-point differences should normally use a tolerance, for example `abs(task_delta) < 1e-12` as a tie.

### 2.4 Interpretation rules

- Prefer held-out LegalBench variants when training/evaluation overlap is possible.
- Treat tiny differences cautiously. On finite test sets, one or two changed answers can move a subject score.
- Do not claim that 5B is better or worse than 3B purely because of token count: these two CPT runs also use different corpus mixtures.
- CPT is unsupervised language-model training, not instruction tuning. A CPT checkpoint may improve legal representations or perplexity without immediately improving instruction-following benchmarks.
- A complete CPT assessment should include held-out legal loss/perplexity, general-domain loss/perplexity, downstream evaluation before SFT, and the same downstream evaluation after identical SFT.

## 3. Main results

All values are accuracies or macro task scores. Parentheses show percentage-point changes relative to Base.

| Benchmark | Base | Legislation-3B | Proportional-5B |
|---|---:|---:|---:|
| LegalBench-Instruct, full | **58.249%** | 50.205% (-8.044) | 50.418% (-7.831) |
| LegalBench-Instruct, dataset-held-out | **55.742%** | 51.054% (-4.689) | 50.063% (-5.679) |
| LegalBench-Instruct, task-held-out | **50.109%** | 45.570% (-4.540) | 44.469% (-5.640) |
| MMLU, three-subject macro accuracy | **73.132%** | 73.049% (-0.084) | 73.022% (-0.110) |
| MMLU international law | 80.992% (98/121) | **82.645% (100/121)** | **82.645% (100/121)** |
| MMLU jurisprudence | **82.407% (89/108)** | 81.481% (88/108) | 80.556% (87/108) |
| MMLU professional law | **55.997% (859/1534)** | 55.020% (844/1534) | 55.867% (857/1534) |

### Main interpretation

- Base is clearly strongest on all three LegalBench-Instruct aggregates.
- Both CPT models answer two more international-law questions correctly than Base.
- The three-subject MMLU macro result is effectively flat: both CPT changes are close to one tenth of a percentage point.
- Proportional-5B preserves professional-law performance better than Legislation-3B, while Legislation-3B preserves jurisprudence better.
- These results do not show a monotonic downstream improvement from the 3B run to the 5B run.

## 4. LegalBench-Instruct task-level results

| CPT model versus Base | Improved | Worse | Tied | Mean task-score change |
|---|---:|---:|---:|---:|
| Legislation-3B | 23 | 118 | 21 | -8.044 pp |
| Proportional-5B | 34 | 109 | 19 | -7.831 pp |

Of 162 LegalBench-Instruct tasks, 155 use candidate log-probability scoring and 7 use generation. The decline is therefore present mainly in constrained candidate scoring and cannot be explained only by generated-answer formatting.

Examples of Proportional-5B improvements include `diversity_2`, `citation_prediction_classification`, `textualism_tool_dictionaries`, and `diversity_1`. Large regressions include `contract_nli_notice`, `maud_includes_past_practice`, `canada_tax_court_outcomes`, and `legal_reasoning_causality`.

## 5. Meaning of the important JSON fields

Exact nesting can vary slightly by benchmark, but the following meanings apply to the current summaries.

### 5.1 Run and model metadata

| Field | Meaning | How to use it |
|---|---|---|
| `model_name_or_path` | Filesystem path or Hugging Face identifier loaded for evaluation. | Confirms which checkpoint was tested. |
| `model_label` | Human-readable name written into the results. | Use in tables; do not rely on it to identify checkpoint contents. |
| `model_type` or evaluation target | Base/CPT/SFT classification used by the runner. | Descriptive metadata only. |
| `git_commit` | Source-code revision used for the run. | Must match across a controlled comparison. |
| `timestamp` | Evaluation time. | Useful for provenance, not a quality metric. |
| `environment` | Python, PyTorch, Transformers, CUDA, GPU, and related versions. | Check reproducibility and protocol differences. |
| `attention_implementation` | Attention backend, such as `flash_attention_2` or `sdpa`. | Keep identical for final controlled runs. |
| `seed` | Random seed controlling sampling/order where applicable. | Keep identical across models. |

### 5.2 Evaluation configuration

| Field | Meaning | How to use it |
|---|---|---|
| `benchmarks` | Benchmarks selected for the run. | Verify no benchmark is missing. |
| `num_fewshot` | Number of demonstrations inserted before each test question. | Five-shot and zero-shot results are not directly comparable. |
| `batch_size` | Number of examples evaluated together. | Primarily performance-related; it should not normally change deterministic scores. |
| `max_new_tokens` | Maximum generated continuation length. | Too large a value can cause extra text and exact-match failures. |
| `candidate_score` | How candidate-token log probabilities are aggregated. `sum` adds token log probabilities. | Must match across models. `sum` can favor shorter answers relative to length-normalized methods. |
| `example_limit` | Optional cap used for smoke tests. | A limited run must not be compared with a full run. |
| `dataset_hashes` | Hashes of the actual evaluation inputs. | Matching hashes prove the same data version was used. |

### 5.3 Common score and count fields

| Field | Meaning | Interpretation |
|---|---|---|
| `score` | Generic task score, whose definition depends on the task/evaluator. | Read together with `metric` or task metadata. |
| `accuracy` | Correct predictions divided by examples. | Micro result within that task or subject. |
| `balanced_accuracy` | Mean recall across classes. | Useful for imbalanced labels; each class contributes equally. |
| `correct` | Number of correctly answered examples. | Use with `total` to audit accuracy. |
| `total`, `examples`, or `num_examples` | Number of evaluated examples. | Larger counts generally yield more stable estimates. |
| `macro_accuracy` | Unweighted mean of subject/task accuracies. | Every subject/task has equal weight regardless of size. |
| `macro_balanced_accuracy` | Unweighted mean of subject/task balanced accuracies. | Combines equal task weighting with class-balance correction. |
| `weighted` or micro aggregate, if present | Aggregate weighted by example count. | Large tasks contribute more heavily. |

For example, MMLU professional law Base has `correct=859`, `total=1534`, and:

```text
accuracy = 859 / 1534 = 0.5599739 = 55.997%
```

### 5.4 MMLU fields

| Field | Meaning |
|---|---|
| `subject` | MMLU subset, here `international_law`, `jurisprudence`, or `professional_law`. |
| `answer` | Gold answer, normally a letter A–D after normalization. |
| `prediction` | Model-selected answer. |
| `answer_index` | Zero-based index of the gold choice, when stored. |
| `choice_scores` | Model scores for A, B, C, and D. The highest score is selected. |
| `macro_accuracy` | Arithmetic mean of the three subject accuracies, not total correct divided by total questions. |

Because the subjects have very different sizes, macro accuracy gives international law, jurisprudence, and professional law equal weight. A micro accuracy would be dominated by professional law.

### 5.5 LegalBench and LegalBench-Instruct fields

| Field | Meaning |
|---|---|
| `task_name` | Unique LegalBench task identifier. |
| `task_type` | Broad form such as rule, classification, or generation. |
| `evaluation_mode` | Usually `candidate_logprob` for constrained labels or generation for free-form answers. |
| `official_score` | Score returned by the official LegalBench task evaluator. Its metric depends on the task. |
| `macro_official_score` | Unweighted mean of official task scores. Each task contributes equally. |
| `automated_tasks` | Tasks for which automatic official scoring was available. |
| `full` | All selected evaluation tasks. |
| `dataset_held_out` | Tasks removed when their source dataset overlaps with LawInstruct training data. |
| `task_held_out` | Tasks removed when the legal capability/instruction itself overlaps with LawInstruct. |
| `prediction_raw` | Unmodified generated model text. | Useful for diagnosing extra explanations or prompt continuation. |
| `prediction_normalized` | Cleaned answer used for comparison, if the runner records it. | This should follow the official evaluator's rules. |

The held-out variants answer different leakage questions:

- **dataset-held-out:** removes evaluation tasks whose source dataset occurs in training;
- **task-held-out:** more conservatively removes tasks teaching a similar capability, even if examples are not identical.

The full score is useful for comparability with broad LegalBench reporting, but held-out scores are more defensible when LawInstruct is used for SFT.

### 5.6 Timing fields

| Field | Meaning |
|---|---|
| `runtime` | Total evaluation time for that unit. |
| `samples_per_second` | Evaluation throughput. |
| `steps_per_second` | Batch-step throughput. |

Timing fields describe efficiency, not model quality. Compare them only when hardware, batch size, precision, and software configuration match.

## 6. Original LegalBench scoring warning

The raw original LegalBench aggregates are:

| Split | Base | Legislation-3B | Proportional-5B |
|---|---:|---:|---:|
| Full | 5.949% | 4.681% | 1.411% |
| Dataset-held-out | 11.419% | 9.077% | 3.714% |
| Task-held-out | 9.763% | 7.254% | 2.339% |

These values should **not** be used as final paper results. The current evaluator can generate up to 64 tokens and sends the complete output to the official exact-match scorer. Many outputs begin with the correct label but continue into an explanation or another few-shot `Q:` block, causing exact match to fail.

As a diagnostic only, normalizing and scoring the first generated line on exact-match tasks produces:

| Diagnostic first-line score | Base | Legislation-3B | Proportional-5B |
|---|---:|---:|---:|
| Full | **68.726%** | 59.862% | 65.881% |
| Dataset-held-out | 61.878% | **65.486%** | 64.786% |
| Task-held-out | 57.276% | 57.147% | **57.633%** |

This shows that the extremely low raw scores, especially Proportional-5B's 1.411%, are heavily contaminated by stopping and parsing behavior. The diagnostic values are not final official results. Fix answer stopping/extraction in the evaluator and rerun all three models under one protocol.

## 7. Conclusions and next steps

1. **Current pre-SFT downstream winner:** Base, mainly because of its stronger LegalBench-Instruct results.
2. **General legal knowledge retention:** both CPT models preserve the MMLU macro result and improve international law by two questions.
3. **CPT success is not fully measured by instruction evaluation:** also compare held-out legal and replay/general validation loss or perplexity.
4. **Repair original LegalBench evaluation:** stop after the answer or extract only the expected label before official scoring.
5. **Standardize the protocol:** rerun Proportional-5B and the other models with the same attention backend and all other settings fixed.
6. **Perform the decisive post-SFT test:** apply the identical SFT dataset, sampling mixture, hyperparameters, and training budget to Base, Legislation-3B, and Proportional-5B, then reevaluate.
7. **Report uncertainty:** for the paper, add paired bootstrap confidence intervals or an appropriate paired significance test using per-example predictions.

