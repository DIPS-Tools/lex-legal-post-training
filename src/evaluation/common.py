"""Data loading, prompt construction, metrics, and result I/O for evaluation."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LEGAL_MMLU_SUBJECTS = ("international_law", "professional_law", "jurisprudence")

DATASET_HELD_OUT_PATTERNS = (
    "contract_nli_*",
    "cuad_*",
    "international_citizenship_questions",
    "maud_*",
    "opp115_*",
    "overruling",
    "privacy_policy_qa",
    "sara_*",
    "unfair_tos",
)

TASK_HELD_OUT_PATTERNS = (
    "function_of_decision_section",
    "oral_argument_question_purpose",
    "diversity_*",
    "personal_jurisdiction",
    "contract_nli_*",
    "cuad_*",
    "jcrew_blocker",
    "unfair_tos",
    "contract_qa",
)


class EvaluationError(ValueError):
    """Raised when evaluation inputs or configuration are invalid."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise EvaluationError(f"Blank line at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise EvaluationError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise EvaluationError(f"No records found in {path}")
    return records


def load_table(path: Path) -> list[dict[str, Any]]:
    """Load JSONL or Parquet records without importing PyArrow unless needed."""
    if not path.is_file():
        raise EvaluationError(f"Evaluation data file does not exist: {path}")
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read Parquet evaluation files") from exc
        return pq.read_table(path).to_pylist()
    raise EvaluationError(f"Unsupported evaluation format: {path}")


def find_split_file(directory: Path, split: str, *, prefer_jsonl: bool = True) -> Path:
    suffixes = ("jsonl", "parquet") if prefer_jsonl else ("parquet", "jsonl")
    for suffix in suffixes:
        matches = sorted(directory.glob(f"{split}-*.{suffix}"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise EvaluationError(f"Expected one {split} {suffix} file in {directory}, found {len(matches)}")
    raise EvaluationError(f"No {split} JSONL or Parquet file found in {directory}")


def answer_letter(record: Mapping[str, Any]) -> str:
    answer = record.get("answer")
    if isinstance(answer, str) and answer.upper() in {"A", "B", "C", "D"}:
        return answer.upper()
    answer_index = record.get("answer_index")
    if isinstance(answer_index, int) and 0 <= answer_index <= 3:
        return "ABCD"[answer_index]
    raise EvaluationError(f"Invalid MMLU answer: {answer!r} / {answer_index!r}")


def format_mmlu_example(record: Mapping[str, Any], *, include_answer: bool) -> str:
    question = record.get("question")
    choices = record.get("choices")
    if not isinstance(question, str) or not question.strip():
        raise EvaluationError("MMLU record has no valid question")
    if not isinstance(choices, list) or len(choices) != 4 or not all(isinstance(item, str) for item in choices):
        raise EvaluationError("MMLU record must contain exactly four string choices")
    lines = [f"Question: {question.strip()}"]
    lines.extend(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices, strict=True))
    lines.append(f"Answer: {answer_letter(record)}" if include_answer else "Answer:")
    return "\n".join(lines)


def build_mmlu_prompt(
    subject: str,
    test_record: Mapping[str, Any],
    demonstrations: Sequence[Mapping[str, Any]],
) -> str:
    title = subject.replace("_", " ")
    header = f"The following are multiple-choice questions about {title}."
    sections = [header]
    sections.extend(format_mmlu_example(record, include_answer=True) for record in demonstrations)
    sections.append(format_mmlu_example(test_record, include_answer=False))
    return "\n\n".join(sections)


_ANSWER_PREFIX = re.compile(r"^(?:answer|label|option|target)\s*:\s*", flags=re.IGNORECASE)


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    value = _ANSWER_PREFIX.sub("", value).strip()
    first_line = value.splitlines()[0].strip() if value else ""
    return " ".join(first_line.split()).casefold()


def accuracy(gold: Sequence[str], predictions: Sequence[str | None]) -> float:
    if len(gold) != len(predictions) or not gold:
        raise EvaluationError("Accuracy requires equally sized, non-empty gold and prediction lists")
    return sum(prediction == target for target, prediction in zip(gold, predictions, strict=True)) / len(gold)


def balanced_accuracy(gold: Sequence[str], predictions: Sequence[str | None]) -> float:
    if len(gold) != len(predictions) or not gold:
        raise EvaluationError("Balanced accuracy requires equally sized, non-empty lists")
    recalls = []
    for label in sorted(set(gold)):
        indices = [index for index, target in enumerate(gold) if target == label]
        recalls.append(sum(predictions[index] == label for index in indices) / len(indices))
    return sum(recalls) / len(recalls)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise EvaluationError("Cannot calculate the mean of an empty sequence")
    return math.fsum(values) / len(values)


def task_matches(task_name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(task_name, pattern) for pattern in patterns)


def legalbench_views(task_names: Iterable[str]) -> dict[str, set[str]]:
    all_tasks = set(task_names)
    return {
        "full": all_tasks,
        "dataset_held_out": {
            task for task in all_tasks if not task_matches(task, DATASET_HELD_OUT_PATTERNS)
        },
        "task_held_out": {
            task for task in all_tasks if not task_matches(task, TASK_HELD_OUT_PATTERNS)
        },
    }


def summarize_legalbench(
    task_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    views = legalbench_views(task_metrics)
    summary: dict[str, Any] = {}
    for view_name, retained_tasks in views.items():
        selected = {name: task_metrics[name] for name in sorted(retained_tasks)}
        if not selected:
            summary[view_name] = {
                "task_count": 0,
                "example_count": 0,
                "macro_task_score": None,
                "macro_by_task_type": {},
                "retained_tasks": [],
                "excluded_tasks": sorted(task_metrics),
            }
            continue
        category_scores: dict[str, list[float]] = defaultdict(list)
        for metric in selected.values():
            category_scores[str(metric["task_type"])].append(float(metric["primary_score"]))
        summary[view_name] = {
            "task_count": len(selected),
            "example_count": sum(int(metric["examples"]) for metric in selected.values()),
            "macro_task_score": mean([float(metric["primary_score"]) for metric in selected.values()]),
            "macro_by_task_type": {
                category: mean(scores) for category, scores in sorted(category_scores.items())
            },
            "retained_tasks": sorted(selected),
            "excluded_tasks": sorted(set(task_metrics) - set(selected)),
        }
    return summary
