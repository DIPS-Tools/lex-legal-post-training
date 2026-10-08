"""Shared LegalBench-Instruct and legal-MMLU evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.common import (
    LEGAL_MMLU_SUBJECTS,
    EvaluationError,
    accuracy,
    answer_letter,
    balanced_accuracy,
    build_mmlu_prompt,
    find_split_file,
    load_table,
    mean,
    normalize_answer,
    sha256_file,
    summarize_legalbench,
    write_json,
    write_jsonl,
)
from src.evaluation.modeling import ModelEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_ROOT = Path(
    "/home/mh1f25/scratch/legal_data_process/datasets/evaluation"
)


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model: Path,
    default_output: Path,
    default_label: str,
) -> None:
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--model-label", default=default_label)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("mmlu", "legalbench"),
        default=["mmlu", "legalbench"],
    )
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-legalbench-candidates", type=int, default=10)
    parser.add_argument("--open-max-new-tokens", type=int, default=64)
    parser.add_argument("--candidate-score", choices=("sum", "mean"), default="sum")
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--example-limit", type=int, default=None, help="Per-task smoke-test limit")
    parser.add_argument("--legalbench-task", action="append", default=[])
    parser.add_argument("--mmlu-subject", action="append", choices=LEGAL_MMLU_SUBJECTS, default=[])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def score_key(score: Any, mode: str) -> float:
    return score.sum_logprob if mode == "sum" else score.mean_logprob


def ensure_output(output_dir: Path, overwrite: bool) -> None:
    marker = output_dir / "evaluation_summary.json"
    if marker.exists() and not overwrite:
        raise EvaluationError(
            f"Completed evaluation already exists at {output_dir}; use a different directory or --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def evaluate_mmlu(
    evaluator: ModelEvaluator,
    *,
    mmlu_root: Path,
    output_dir: Path,
    model_label: str,
    subjects: Sequence[str],
    num_fewshot: int,
    candidate_mode: str,
    example_limit: int | None,
) -> dict[str, Any]:
    subject_metrics: dict[str, Any] = {}
    input_files: list[dict[str, Any]] = []
    for subject in subjects:
        subject_dir = mmlu_root / subject
        test_path = find_split_file(subject_dir, "test", prefer_jsonl=True)
        dev_path = find_split_file(subject_dir, "dev", prefer_jsonl=False)
        test_records = load_table(test_path)
        dev_records = load_table(dev_path)
        if num_fewshot > len(dev_records):
            raise EvaluationError(
                f"Requested {num_fewshot} demonstrations for {subject}, but dev has {len(dev_records)}"
            )
        demonstrations = dev_records[:num_fewshot]
        if example_limit is not None:
            test_records = test_records[:example_limit]
        prompts = [build_mmlu_prompt(subject, record, demonstrations) for record in test_records]
        pairs = [(prompt, f" {letter}") for prompt in prompts for letter in "ABCD"]
        scores = evaluator.score_pairs(pairs)

        predictions: list[dict[str, Any]] = []
        predicted_letters: list[str] = []
        gold_letters: list[str] = []
        for index, record in enumerate(test_records):
            option_scores = scores[index * 4 : (index + 1) * 4]
            numeric_scores = {
                letter: score_key(value, candidate_mode)
                for letter, value in zip("ABCD", option_scores, strict=True)
            }
            prediction = max(numeric_scores, key=numeric_scores.get)
            gold = answer_letter(record)
            predicted_letters.append(prediction)
            gold_letters.append(gold)
            predictions.append(
                {
                    "model": model_label,
                    "benchmark": "mmlu",
                    "subject": subject,
                    "index": index,
                    "gold": gold,
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "candidate_score_mode": candidate_mode,
                    "candidate_scores": {
                        letter: {
                            "selected": numeric_scores[letter],
                            "sum_logprob": option_scores[position].sum_logprob,
                            "mean_logprob": option_scores[position].mean_logprob,
                            "tokens": option_scores[position].tokens,
                        }
                        for position, letter in enumerate("ABCD")
                    },
                }
            )
        metrics = {
            "subject": subject,
            "examples": len(test_records),
            "num_fewshot": num_fewshot,
            "accuracy": accuracy(gold_letters, predicted_letters),
            "balanced_accuracy": balanced_accuracy(gold_letters, predicted_letters),
            "gold_distribution": dict(sorted(Counter(gold_letters).items())),
            "prediction_distribution": dict(sorted(Counter(predicted_letters).items())),
        }
        write_jsonl(output_dir / "predictions" / f"{subject}.jsonl", predictions)
        write_json(output_dir / "metrics" / f"{subject}.json", metrics)
        subject_metrics[subject] = metrics
        input_files.extend(
            {"path": str(path), "sha256": sha256_file(path)} for path in (test_path, dev_path)
        )

    summary = {
        "subjects": subject_metrics,
        "macro_accuracy": mean([value["accuracy"] for value in subject_metrics.values()]),
        "macro_balanced_accuracy": mean(
            [value["balanced_accuracy"] for value in subject_metrics.values()]
        ),
        "input_files": input_files,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def resolved_legalbench_mode(answers: Sequence[str], max_candidates: int) -> tuple[str, list[str]]:
    candidates = sorted(set(answers), key=lambda value: (value.casefold(), value))
    if 2 <= len(candidates) <= max_candidates:
        return "candidate_logprob", candidates
    return "generation", []


def evaluate_legalbench_task(
    evaluator: ModelEvaluator,
    *,
    records: Sequence[Mapping[str, Any]],
    model_label: str,
    candidate_mode: str,
    max_candidates: int,
    max_new_tokens: int,
    resolved_mode: str | None = None,
    resolved_candidates: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    task_name = str(records[0]["task_name"])
    task_type = str(records[0]["task_type"])
    if any(str(record.get("task_name")) != task_name for record in records):
        raise EvaluationError("Mixed task names passed to evaluate_legalbench_task")
    answers = [str(record["answer"]).strip() for record in records]
    if resolved_mode is None:
        mode, candidates = resolved_legalbench_mode(answers, max_candidates)
    else:
        mode = resolved_mode
        candidates = list(resolved_candidates or [])
        if mode == "candidate_logprob" and len(candidates) < 2:
            raise EvaluationError(f"Classification task {task_name} has fewer than two candidates")
    predictions: list[dict[str, Any]] = []

    if mode == "candidate_logprob":
        pairs = [
            (str(record["inputs"]), " " + candidate)
            for record in records
            for candidate in candidates
        ]
        all_scores = evaluator.score_pairs(pairs)
        predicted: list[str | None] = []
        for row_index, record in enumerate(records):
            row_scores = all_scores[
                row_index * len(candidates) : (row_index + 1) * len(candidates)
            ]
            selected_scores = {
                candidate: score_key(value, candidate_mode)
                for candidate, value in zip(candidates, row_scores, strict=True)
            }
            prediction = max(selected_scores, key=selected_scores.get)
            predicted.append(prediction)
            gold = str(record["answer"]).strip()
            predictions.append(
                {
                    "model": model_label,
                    "benchmark": "legalbench_instruct",
                    "task_name": task_name,
                    "task_type": task_type,
                    "index": str(record.get("index", row_index)),
                    "mode": mode,
                    "gold": gold,
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "candidate_score_mode": candidate_mode,
                    "candidate_scores": {
                        candidate: {
                            "selected": selected_scores[candidate],
                            "sum_logprob": row_scores[position].sum_logprob,
                            "mean_logprob": row_scores[position].mean_logprob,
                            "tokens": row_scores[position].tokens,
                        }
                        for position, candidate in enumerate(candidates)
                    },
                }
            )
        primary_metric = "balanced_accuracy"
        metric_values = {
            "accuracy": accuracy(answers, predicted),
            "balanced_accuracy": balanced_accuracy(answers, predicted),
            "format_success_rate": 1.0,
        }
    else:
        prompts = [str(record["inputs"]) for record in records]
        raw_outputs = evaluator.generate(prompts, max_new_tokens=max_new_tokens)
        normalized_gold = [normalize_answer(answer) for answer in answers]
        normalized_predictions = [normalize_answer(output) or None for output in raw_outputs]
        for row_index, (record, raw_output, prediction, gold) in enumerate(
            zip(records, raw_outputs, normalized_predictions, normalized_gold, strict=True)
        ):
            predictions.append(
                {
                    "model": model_label,
                    "benchmark": "legalbench_instruct",
                    "task_name": task_name,
                    "task_type": task_type,
                    "index": str(record.get("index", row_index)),
                    "mode": mode,
                    "gold": str(record["answer"]).strip(),
                    "normalized_gold": gold,
                    "raw_generation": raw_output,
                    "normalized_prediction": prediction,
                    "correct": prediction == gold,
                }
            )
        primary_metric = "normalized_exact_match"
        metric_values = {
            "normalized_exact_match": accuracy(normalized_gold, normalized_predictions),
            "format_success_rate": sum(value is not None for value in normalized_predictions)
            / len(normalized_predictions),
        }

    metrics = {
        "task_name": task_name,
        "task_type": task_type,
        "mode": mode,
        "examples": len(records),
        "candidates": candidates,
        "primary_metric": primary_metric,
        "primary_score": metric_values[primary_metric],
        **metric_values,
    }
    task_config = {
        "task_name": task_name,
        "task_type": task_type,
        "mode": mode,
        "candidates": candidates,
        "max_new_tokens": max_new_tokens if mode == "generation" else None,
        "primary_metric": primary_metric,
    }
    return predictions, metrics, task_config


def evaluate_legalbench(
    evaluator: ModelEvaluator,
    *,
    data_path: Path,
    output_dir: Path,
    model_label: str,
    candidate_mode: str,
    max_candidates: int,
    max_new_tokens: int,
    requested_tasks: Sequence[str],
    example_limit: int | None,
) -> dict[str, Any]:
    records = load_table(data_path)
    required_fields = {"answer", "index", "task_type", "task_name", "inputs"}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, record in enumerate(records):
        missing = required_fields - set(record)
        if missing:
            raise EvaluationError(f"LegalBench row {row_index} is missing fields: {sorted(missing)}")
        grouped[str(record["task_name"])].append(record)
    if requested_tasks:
        unknown = set(requested_tasks) - set(grouped)
        if unknown:
            raise EvaluationError(f"Unknown LegalBench tasks requested: {sorted(unknown)}")
        grouped = {name: grouped[name] for name in requested_tasks}

    task_metrics: dict[str, Any] = {}
    task_configs: dict[str, Any] = {}
    for task_number, task_name in enumerate(sorted(grouped), start=1):
        all_task_records = grouped[task_name]
        full_answers = [str(record["answer"]).strip() for record in all_task_records]
        task_mode, task_candidates = resolved_legalbench_mode(full_answers, max_candidates)
        task_records = all_task_records
        if example_limit is not None:
            task_records = task_records[:example_limit]
        print(
            f"LegalBench task {task_number}/{len(grouped)}: {task_name} ({len(task_records):,} examples)",
            flush=True,
        )
        predictions, metrics, config = evaluate_legalbench_task(
            evaluator,
            records=task_records,
            model_label=model_label,
            candidate_mode=candidate_mode,
            max_candidates=max_candidates,
            max_new_tokens=max_new_tokens,
            resolved_mode=task_mode,
            resolved_candidates=task_candidates,
        )
        write_jsonl(output_dir / "predictions" / f"{task_name}.jsonl", predictions)
        write_json(output_dir / "task_metrics" / f"{task_name}.json", metrics)
        task_metrics[task_name] = metrics
        task_configs[task_name] = config

    summary = {
        "input_file": {"path": str(data_path), "sha256": sha256_file(data_path)},
        "rows": sum(metric["examples"] for metric in task_metrics.values()),
        "tasks": len(task_metrics),
        "views": summarize_legalbench(task_metrics),
    }
    manifest_dir = output_dir / "manifests"
    for view_name, view in summary["views"].items():
        write_jsonl(
            manifest_dir / f"{view_name}_retained.jsonl",
            ({"task_name": task_name} for task_name in view["retained_tasks"]),
        )
        write_jsonl(
            manifest_dir / f"{view_name}_excluded.jsonl",
            ({"task_name": task_name} for task_name in view["excluded_tasks"]),
        )
    write_json(output_dir / "resolved_task_configuration.json", task_configs)
    write_json(output_dir / "summary.json", summary)
    return summary


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_fewshot < 0:
        raise EvaluationError("--num-fewshot cannot be negative")
    if args.max_legalbench_candidates < 2:
        raise EvaluationError("--max-legalbench-candidates must be at least 2")
    if args.example_limit is not None and args.example_limit <= 0:
        raise EvaluationError("--example-limit must be positive")

    model_path = args.model_path.expanduser().resolve()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise EvaluationError(f"Model config is missing: {model_path / 'config.json'}")
    if not evaluation_root.is_dir():
        raise EvaluationError(f"Evaluation root does not exist: {evaluation_root}")
    ensure_output(output_dir, args.overwrite)
    random.seed(args.seed)

    evaluator = ModelEvaluator(
        model_path,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        attn_implementation=args.attn_implementation,
        local_files_only=not args.allow_download,
    )
    summaries: dict[str, Any] = {}
    if "mmlu" in args.benchmarks:
        subjects = tuple(args.mmlu_subject) or LEGAL_MMLU_SUBJECTS
        summaries["mmlu"] = evaluate_mmlu(
            evaluator,
            mmlu_root=evaluation_root / "mmlu",
            output_dir=output_dir / "mmlu",
            model_label=args.model_label,
            subjects=subjects,
            num_fewshot=args.num_fewshot,
            candidate_mode=args.candidate_score,
            example_limit=args.example_limit,
        )
    if "legalbench" in args.benchmarks:
        legalbench_path = find_split_file(
            evaluation_root / "legalbench_instruct" / "data",
            "train",
            prefer_jsonl=True,
        )
        summaries["legalbench"] = evaluate_legalbench(
            evaluator,
            data_path=legalbench_path,
            output_dir=output_dir / "legalbench",
            model_label=args.model_label,
            candidate_mode=args.candidate_score,
            max_candidates=args.max_legalbench_candidates,
            max_new_tokens=args.open_max_new_tokens,
            requested_tasks=args.legalbench_task,
            example_limit=args.example_limit,
        )

    import torch
    import transformers

    final = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": args.model_label,
        "model_path": str(model_path),
        "evaluation_root": str(evaluation_root),
        "git_commit": git_commit(),
        "arguments": vars(args),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        },
        "summaries": summaries,
    }
    # Convert Paths in parsed arguments before serializing.
    final["arguments"] = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    write_json(output_dir / "evaluation_summary.json", final)
    print(json.dumps(final, indent=2, default=str), flush=True)
    return final


def cli(
    *,
    default_model: Path,
    default_output: Path,
    default_label: str,
    description: str,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(
        parser,
        default_model=default_model,
        default_output=default_output,
        default_label=default_label,
    )
    args = parser.parse_args()
    try:
        run_evaluation(args)
    except (EvaluationError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
