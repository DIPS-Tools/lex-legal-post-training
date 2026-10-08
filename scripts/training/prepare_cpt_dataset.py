#!/usr/bin/env python3
"""Validate final CPT JSONL shards and tokenize them into flat uint32 files."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cpt_data import (  # noqa: E402
    FORMAT_VERSION,
    CptDataError,
    discover_jsonl_files,
    sha256_file,
    tokenize_jsonl_dataset,
    validate_jsonl_dataset,
    write_json,
)


DEFAULT_MODEL = Path("/home/mh1f25/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base")
DEFAULT_DATASET = Path("/home/mh1f25/scratch/legal_data_process/datasets/CPT-Proportional-5B")
DEFAULT_OUTPUT = Path(
    "/home/mh1f25/scratch/legal_data_process/datasets/CPT-tokenized/CPT-Proportional-5B-Qwen3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--input-pattern", default="*.jsonl")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-per-mille", type=int, default=1)
    parser.add_argument(
        "--replay-regex",
        default=r"(?:replay|general|fineweb|wikipedia|openwebtext|dolma|slimpajama)",
        help=(
            "Case-insensitive regex matched against corpus/source/type/domain/category/"
            "source_group/dataset_role/mixture. Override it to match the frozen corpus manifest."
        ),
    )
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-jsonl-bytes", type=int)
    parser.add_argument("--expected-estimated-tokens", type=int)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-validation-index",
        action="store_true",
        help="Keep the SQLite ID index after validation (can be large).",
    )
    return parser.parse_args()


def load_source_manifest(path: Path) -> tuple[dict, dict]:
    if not path.is_file():
        raise CptDataError(f"Final dataset manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise CptDataError(f"Expected a JSON object in manifest: {path}")
    return payload, {"path": str(path.resolve()), "sha256": sha256_file(path)}


def find_numeric(payload: dict, aliases: tuple[str, ...]) -> int | None:
    """Find a uniquely-valued numeric manifest field, including nested sections."""
    for alias in aliases:
        found: list[int] = []

        def visit(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == alias and isinstance(child, int) and not isinstance(child, bool):
                        found.append(child)
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
        if found and all(item == found[0] for item in found):
            return found[0]
    return None


def verify_expected_counts(args: argparse.Namespace, source_manifest: dict, summary: dict) -> dict:
    expected = {
        "rows": args.expected_rows,
        "jsonl_bytes": args.expected_jsonl_bytes,
        "estimated_tokens": args.expected_estimated_tokens,
    }
    aliases = {
        "rows": ("total_rows", "rows", "record_count", "records", "total_records"),
        "jsonl_bytes": ("total_jsonl_bytes", "jsonl_bytes"),
        "estimated_tokens": ("total_estimated_tokens", "estimated_tokens"),
    }
    for name, value in list(expected.items()):
        if value is None:
            expected[name] = find_numeric(source_manifest, aliases[name])

    mismatches = {
        name: {"expected": value, "actual": summary[name]}
        for name, value in expected.items()
        if value is not None and value != summary[name]
    }
    if mismatches:
        raise CptDataError(f"Dataset does not match its expected counts: {mismatches}")
    return expected


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise CptDataError(f"Output already exists: {metadata_path}; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_jsonl_files(args.input_dir, args.input_pattern)
    manifest_path = (args.manifest or (args.input_dir / "manifest.json")).expanduser().resolve()
    source_manifest, manifest_identity = load_source_manifest(manifest_path)
    if ".work" in manifest_path.parts:
        raise CptDataError(f"Refusing intermediate manifest under .work: {manifest_path}")

    sqlite_path = output_dir / "validation_ids.sqlite3"
    print(f"Validating {len(paths)} final JSONL shard(s) from {args.input_dir}", flush=True)
    summary_object = validate_jsonl_dataset(
        paths,
        validation_per_mille=args.validation_per_mille,
        replay_regex=args.replay_regex,
        sqlite_path=sqlite_path,
    )
    summary = summary_object.as_dict()
    expected = verify_expected_counts(args, source_manifest, summary)
    write_json(output_dir / "source_statistics.json", summary)
    frozen_manifest_path = output_dir / "source_manifest.json"
    shutil.copy2(manifest_path, frozen_manifest_path)
    manifest_identity["frozen_copy"] = str(frozen_manifest_path)
    source_dataset_statistics = args.input_dir.expanduser().resolve() / "statistics.json"
    statistics_identity = None
    if source_dataset_statistics.is_file():
        frozen_statistics_path = output_dir / "source_dataset_statistics.json"
        shutil.copy2(source_dataset_statistics, frozen_statistics_path)
        statistics_identity = {
            "path": str(source_dataset_statistics),
            "sha256": sha256_file(source_dataset_statistics),
            "frozen_copy": str(frozen_statistics_path),
        }

    try:
        from transformers import AutoTokenizer
        import transformers
    except ImportError as exc:
        raise RuntimeError("transformers is required to prepare CPT data") from exc

    tokenizer_path = args.tokenizer_path.expanduser().resolve()
    if not (tokenizer_path / "config.json").is_file():
        raise CptDataError(f"Local Qwen model config is missing: {tokenizer_path / 'config.json'}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise CptDataError("Qwen tokenizer does not define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing raw documents without special/chat tokens", flush=True)
    split_statistics = tokenize_jsonl_dataset(
        paths,
        tokenizer=tokenizer,
        output_dir=output_dir,
        validation_per_mille=args.validation_per_mille,
        replay_regex=args.replay_regex,
        progress_every=args.progress_every,
    )
    metadata = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "token_dtype": "uint32-little-endian",
        "document_separator": "eos_token_id",
        "chat_template_applied": False,
        "add_special_tokens": False,
        "input_dir": str(args.input_dir.expanduser().resolve()),
        "input_pattern": args.input_pattern,
        "source_manifest": manifest_identity,
        "source_dataset_statistics": statistics_identity,
        "manifest_expected_counts": expected,
        "validation_per_mille": args.validation_per_mille,
        "split_key": "dedup_cluster_id when present, otherwise id",
        "replay_regex": args.replay_regex,
        "tokenizer": {
            "path": str(tokenizer_path),
            "name_or_path": tokenizer.name_or_path,
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "splits": split_statistics,
        "validation_summary": summary,
        "software": {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
        },
    }
    write_json(metadata_path, metadata)
    if not args.keep_validation_index:
        sqlite_path.unlink(missing_ok=True)
    print(f"Prepared tokenized CPT dataset: {output_dir}", flush=True)
    print(json.dumps(split_statistics, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (CptDataError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
