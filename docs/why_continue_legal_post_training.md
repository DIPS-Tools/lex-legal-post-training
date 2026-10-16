# Why Continue the Legal Post-Training Project?

## 1. Short answer

This project is worth continuing, but its objective should not be limited to building a model that beats every state-of-the-art commercial model.

The stronger and more defensible objective is:

> Determine whether controlled legal continued pretraining (CPT), general-domain replay, and legal supervised fine-tuning (SFT) can produce a smaller, open, reproducible, and deployable legal model with measurable benefits over its base model.

The next decisive experiment is not simply a larger CPT run. It is a controlled comparison of Base + SFT against CPT + the same SFT.

## 2. Why continue when state-of-the-art models are already strong?

State-of-the-art proprietary models can perform many legal tasks, but strong aggregate performance does not solve every research or deployment requirement.

### 2.1 Scientific understanding

The project can answer questions that using an existing API cannot answer:

- Does legislation-only CPT behave differently from proportionally sampled legal CPT?
- How many legal training tokens are useful before returns diminish?
- Does general-domain replay reduce catastrophic forgetting?
- Does CPT improve downstream legal performance directly, or only after SFT?
- Which legal data types and jurisdictions produce useful transfer?
- Can an open 8B model become competitive without requiring frontier-model scale?

The value lies in controlled evidence about legal domain adaptation, not only in the final leaderboard position.

### 2.2 Open and reproducible research

An open model and pipeline allow researchers to inspect:

- training datasets and their proportions;
- preprocessing and deduplication;
- replay strategy;
- CPT and SFT hyperparameters;
- checkpoint behavior;
- benchmark prompts and answer extraction;
- overlap between training and evaluation data.

This makes results easier to verify, reproduce, and extend than results from a closed model or external API.

### 2.3 Privacy and deployment control

Legal documents may contain confidential, privileged, commercially sensitive, or personally identifiable information. A locally deployable model can offer:

- on-premises inference;
- control over data retention;
- no dependency on an external API;
- predictable availability and versioning;
- jurisdiction- or organization-specific adaptation;
- lower inference cost at sufficient usage volume.

These properties may justify a specialized model even when a larger proprietary model has higher average benchmark performance.

### 2.4 Efficiency rather than absolute dominance

A useful result does not require an 8B model to defeat a much larger frontier model. The project may demonstrate that a smaller model achieves:

- competitive legal performance per parameter;
- competitive performance per training token;
- lower deployment cost;
- less general-domain forgetting;
- stronger performance in a selected jurisdiction or document type;
- better improvement after a limited legal SFT budget.

## 3. Where the strongest contribution lies

### 3.1 Data-mixture research

The current datasets support a meaningful comparison among:

- legislation-focused CPT;
- proportionally sampled legal CPT;
- legal data mixed with SlimPajama replay;
- different CPT token budgets, such as 3B, 5B, and 10B.

This can show whether data composition matters more than simply increasing token count.

### 3.2 Catastrophic-forgetting analysis

The replay component provides a basis for studying whether legal specialization damages general capability. This requires both:

- held-out legal loss or perplexity; and
- held-out replay/general loss or broader general benchmarks.

A model that gains legal ability while retaining general capability may be more valuable than one with a slightly higher legal score but severe forgetting.

### 3.3 CPT-to-SFT transfer

CPT optimizes next-token prediction, not instruction following. Therefore, pre-SFT benchmark results do not fully determine whether CPT was useful.

The central experiment should compare:

```text
Qwen3-8B-Base + identical SFT
Legislation-3B CPT + identical SFT
Proportional-5B CPT + identical SFT
Proportional-10B CPT + identical SFT
```

The SFT dataset, sampling mixture, number of examples, epochs, sequence length, learning rate, and evaluation procedure must remain identical. This isolates the effect of the CPT initialization.

### 3.4 Evaluation integrity

The project already identified an important original LegalBench scoring problem: correct labels followed by prompt continuation were treated as incorrect by whole-string exact matching. Correcting this while preserving raw generations improves evaluation validity.

The project can also make a useful methodological contribution by reporting:

- full LegalBench results;
- dataset-held-out results;
- task-held-out results;
- LegalBench-Instruct results;
- legal MMLU subject results;
- training/evaluation overlap controls;
- per-task rather than aggregate-only behavior.

## 4. What the current evidence says

The current pre-SFT evaluations suggest:

- Qwen3-8B-Base is stronger on the aggregate LegalBench-Instruct results.
- The CPT models remain approximately level with Base on the three legal MMLU subjects.
- Both CPT models improve international-law MMLU by two correct answers.
- Corrected original LegalBench results are mixed rather than showing a universal CPT gain.
- Increasing from the legislation-focused 3B run to the proportional 5B run does not establish a pure scaling trend because both the size and corpus composition changed.

These results do not yet prove that CPT is beneficial, but they also do not justify stopping before the controlled SFT comparison. A CPT checkpoint may lower legal language-modeling loss and provide a better SFT initialization without improving instruction-following benchmarks on its own.

## 5. What should be done next?

### Stage 1: Finish the controlled CPT comparison

Complete and evaluate:

- Qwen3-8B-Base;
- Legislation-3B;
- Proportional-5B with the standardized FlashAttention configuration;
- Proportional-10B.

Use the same corrected evaluation code and settings for every checkpoint.

### Stage 2: Measure language-model adaptation

For each checkpoint, measure:

- held-out legal validation loss/perplexity;
- held-out replay/general validation loss/perplexity;
- loss by legal type and jurisdiction where possible.

This determines whether CPT changed the intended modeling behavior even when downstream prompt-based scores are flat.

### Stage 3: Run identical SFT experiments

At minimum, run the same SFT recipe from:

1. Qwen3-8B-Base;
2. the best proportional CPT checkpoint;
3. Legislation-3B as a data-mixture ablation.

Then rerun LegalBench, LegalBench-Instruct, and the selected MMLU subjects.

### Stage 4: Compare with external models

Compare the best final model with:

- open legal models of a similar parameter scale;
- strong general instruction models of a similar scale;
- larger frontier models only as clearly marked reference points.

External comparison establishes practical competitiveness, while the internal Base → CPT → SFT comparison establishes causality.

## 6. Where not to spend resources yet

Do not automatically proceed to 20B, 30B, or the entire legal corpus immediately after completing 10B.

First determine:

- whether 10B improves over 5B;
- whether legal validation loss continues to improve;
- whether replay prevents general-domain degradation;
- whether CPT produces a measurable advantage after identical SFT;
- whether gains justify the additional compute and storage.

If 10B provides no meaningful benefit over 5B, increasing token count without changing data quality, mixture, or training design may only increase cost.

## 7. Go/no-go decision criteria

### Continue scaling CPT when

- legal held-out loss improves consistently;
- general/replay loss remains acceptable;
- CPT + SFT outperforms Base + SFT;
- improvements appear across held-out tasks rather than only overlapping tasks;
- the 10B result improves meaningfully over 5B;
- the expected gain justifies compute and storage costs.

### Pause scaling and improve the pipeline when

- legal loss improves but downstream evaluation does not;
- performance varies strongly by prompt or answer format;
- some data categories cause disproportionate regressions;
- evaluation overlap prevents a trustworthy conclusion;
- SFT data quality appears to be the principal limitation.

In this case, focus on SFT design, data quality, sampling, prompt construction, and task-level error analysis before adding CPT tokens.

### Stop further large CPT runs when

- Base + SFT consistently matches or beats CPT + SFT;
- 10B fails to improve over 5B under a controlled setup;
- general capability degrades substantially;
- improved performance is limited to contaminated or overlapping tasks;
- deployment requirements can be met more reliably and cheaply by an existing model.

Stopping additional scaling would not make the work worthless. A carefully demonstrated negative result can show that indiscriminate legal CPT is not cost-effective at this scale or with this mixture.

## 8. Potential final research claims

The project should avoid an unsupported claim such as:

> We created the strongest legal language model.

More defensible claims include:

> We systematically evaluate how legal-domain composition, general-domain replay, and CPT token budget affect an open 8B language model before and after legal instruction tuning.

> We show whether proportional legal CPT provides a better initialization for legal SFT than legislation-only CPT or direct SFT of the base model.

> We provide an overlap-aware and reproducible evaluation of legal post-training, including corrected LegalBench answer handling and held-out evaluation variants.

Even a negative finding can be important:

> Under the studied compute and data conditions, legal CPT reduces domain validation loss but does not reliably improve downstream legal reasoning after controlled instruction tuning.

## 9. Recommendation

Continue the project through the completed 5B/10B evaluation and one controlled SFT comparison. This is the minimum evidence needed to decide whether legal CPT creates downstream value.

Do not commit to 20B, 30B, or full-corpus CPT until those experiments show a clear benefit. The immediate priority should be:

```text
corrected evaluation
→ held-out legal/general loss analysis
→ identical Base-versus-CPT SFT
→ post-SFT evaluation
→ decision on further scaling
```

The project's strongest value lies in rigorous, reproducible evidence about when legal post-training works, not merely in competing with the largest available model.
