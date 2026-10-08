"""Streaming validation, tokenization, and memory-mapped datasets for CPT."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


FORMAT_VERSION = 1
TOKEN_DTYPE = np.dtype("<u4")


class CptDataError(ValueError):
    """Raised when CPT source data or tokenized metadata is invalid."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def choose_split(document_id: str, validation_per_mille: int = 1) -> str:
    """Choose a stable document-level split from an ID or duplicate-cluster ID."""
    if not 0 <= validation_per_mille <= 1000:
        raise CptDataError("validation_per_mille must be between 0 and 1000")
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return "validation" if value % 1000 < validation_per_mille else "train"


def discover_jsonl_files(input_dir: Path, pattern: str = "*.jsonl") -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    if ".work" in input_dir.parts:
        raise CptDataError(f"Refusing to read an intermediate .work directory: {input_dir}")
    if not input_dir.is_dir():
        raise CptDataError(f"CPT dataset directory does not exist: {input_dir}")

    files = sorted(path.resolve() for path in input_dir.glob(pattern) if path.is_file())
    unsafe = [path for path in files if ".work" in path.parts]
    if unsafe:
        raise CptDataError(f"Input pattern selected files under .work: {unsafe[0]}")
    if not files:
        raise CptDataError(f"No JSONL shards matched {pattern!r} in {input_dir}")
    return files


def iter_jsonl(paths: Sequence[Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise CptDataError(f"Blank row at {path}:{line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CptDataError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise CptDataError(f"Expected a JSON object at {path}:{line_number}")
                yield path, line_number, record


def validate_record(record: Mapping[str, Any], path: Path, line_number: int) -> tuple[str, str]:
    document_id = record.get("id")
    text = record.get("text")
    if not isinstance(document_id, str) or not document_id.strip():
        raise CptDataError(f"Missing or invalid 'id' at {path}:{line_number}")
    if not isinstance(text, str) or not text.strip():
        raise CptDataError(f"Missing or empty 'text' at {path}:{line_number}")
    return document_id, text


def split_key(record: Mapping[str, Any], document_id: str) -> str:
    cluster_id = record.get("dedup_cluster_id")
    return cluster_id if isinstance(cluster_id, str) and cluster_id.strip() else document_id


def is_replay_record(record: Mapping[str, Any], replay_pattern: re.Pattern[str]) -> bool:
    values = (
        record.get("corpus"),
        record.get("source"),
        record.get("type"),
        record.get("domain"),
        record.get("category"),
        record.get("source_group"),
        record.get("dataset_role"),
        record.get("mixture"),
    )
    return any(isinstance(value, str) and replay_pattern.search(value) for value in values)


@dataclass
class ValidationSummary:
    rows: int
    jsonl_bytes: int
    estimated_tokens: int
    files: list[dict[str, Any]]
    corpus_rows: dict[str, int]
    type_rows: dict[str, int]
    jurisdiction_rows: dict[str, int]
    train_documents: int
    validation_documents: int
    validation_legal_documents: int
    validation_replay_documents: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def validate_jsonl_dataset(
    paths: Sequence[Path],
    *,
    validation_per_mille: int,
    replay_regex: str,
    sqlite_path: Path,
) -> ValidationSummary:
    """Validate every row and enforce globally unique IDs with an on-disk index."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()
    connection = sqlite3.connect(sqlite_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
    replay_pattern = re.compile(replay_regex, flags=re.IGNORECASE)

    rows = estimated_tokens = 0
    train_documents = validation_documents = 0
    validation_legal_documents = validation_replay_documents = 0
    corpus_rows: Counter[str] = Counter()
    type_rows: Counter[str] = Counter()
    jurisdiction_rows: Counter[str] = Counter()
    per_file_rows: Counter[Path] = Counter()
    try:
        for path, line_number, record in iter_jsonl(paths):
            document_id, _ = validate_record(record, path, line_number)
            try:
                connection.execute("INSERT INTO ids VALUES (?)", (document_id,))
            except sqlite3.IntegrityError as exc:
                raise CptDataError(f"Duplicate document ID {document_id!r} at {path}:{line_number}") from exc

            rows += 1
            per_file_rows[path] += 1
            estimate = record.get("estimated_tokens")
            if isinstance(estimate, int) and estimate >= 0:
                estimated_tokens += estimate
            corpus_rows[str(record.get("corpus", "UNKNOWN"))] += 1
            type_rows[str(record.get("type", "UNKNOWN"))] += 1
            jurisdiction_rows[str(record.get("jurisdiction", "UNKNOWN"))] += 1

            split = choose_split(split_key(record, document_id), validation_per_mille)
            if split == "train":
                train_documents += 1
            else:
                validation_documents += 1
                if is_replay_record(record, replay_pattern):
                    validation_replay_documents += 1
                else:
                    validation_legal_documents += 1
        connection.commit()
    finally:
        connection.close()

    if rows == 0:
        raise CptDataError("The selected JSONL shards contain no records")
    files = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "rows": per_file_rows[path],
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return ValidationSummary(
        rows=rows,
        jsonl_bytes=sum(item["bytes"] for item in files),
        estimated_tokens=estimated_tokens,
        files=files,
        corpus_rows=dict(corpus_rows.most_common()),
        type_rows=dict(type_rows.most_common()),
        jurisdiction_rows=dict(jurisdiction_rows.most_common()),
        train_documents=train_documents,
        validation_documents=validation_documents,
        validation_legal_documents=validation_legal_documents,
        validation_replay_documents=validation_replay_documents,
    )


class TokenWriter:
    def __init__(self, path: Path) -> None:
        self.final_path = path
        self.partial_path = path.with_suffix(path.suffix + ".partial")
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.partial_path.open("wb")
        self.tokens = 0

    def write(self, token_ids: Sequence[int]) -> None:
        if not token_ids:
            return
        values = np.asarray(token_ids, dtype=TOKEN_DTYPE)
        values.tofile(self.handle)
        self.tokens += len(values)

    def finish(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.partial_path.replace(self.final_path)

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def tokenize_jsonl_dataset(
    paths: Sequence[Path],
    *,
    tokenizer: Any,
    output_dir: Path,
    validation_per_mille: int,
    replay_regex: str,
    progress_every: int = 10_000,
) -> dict[str, dict[str, int]]:
    """Stream raw documents to EOS-delimited uint32 token files."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise CptDataError("Tokenizer has no EOS token ID")
    replay_pattern = re.compile(replay_regex, flags=re.IGNORECASE)
    writers = {
        "train": TokenWriter(output_dir / "train_tokens.bin"),
        "validation": TokenWriter(output_dir / "validation_tokens.bin"),
        "validation_legal": TokenWriter(output_dir / "validation_legal_tokens.bin"),
        "validation_replay": TokenWriter(output_dir / "validation_replay_tokens.bin"),
    }
    documents: Counter[str] = Counter()
    try:
        for row_index, (path, line_number, record) in enumerate(iter_jsonl(paths), start=1):
            document_id, text = validate_record(record, path, line_number)
            split = choose_split(split_key(record, document_id), validation_per_mille)
            token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            if not isinstance(token_ids, list) or any(not isinstance(value, int) for value in token_ids):
                raise CptDataError(f"Tokenizer returned invalid input_ids at {path}:{line_number}")
            token_ids.append(eos_token_id)
            writers[split].write(token_ids)
            documents[split] += 1
            if split == "validation":
                subset = "validation_replay" if is_replay_record(record, replay_pattern) else "validation_legal"
                writers[subset].write(token_ids)
                documents[subset] += 1
            if progress_every > 0 and row_index % progress_every == 0:
                print(f"Tokenized {row_index:,} documents", flush=True)
        for writer in writers.values():
            writer.finish()
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise

    return {
        name: {"documents": documents[name], "tokens": writer.tokens}
        for name, writer in writers.items()
    }


def load_metadata(tokenized_dir: Path) -> dict[str, Any]:
    path = tokenized_dir / "metadata.json"
    if not path.is_file():
        raise CptDataError(f"Tokenized dataset metadata is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("format_version") != FORMAT_VERSION:
        raise CptDataError(
            f"Unsupported tokenized dataset format {metadata.get('format_version')!r}; expected {FORMAT_VERSION}"
        )
    return metadata


class PackedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose consecutive fixed-length blocks from a flat uint32 token stream."""

    def __init__(self, path: Path | str, sequence_length: int) -> None:
        self.path = Path(path)
        if sequence_length < 2:
            raise CptDataError("sequence_length must be at least 2")
        if not self.path.is_file():
            raise CptDataError(f"Token file is missing: {self.path}")
        if self.path.stat().st_size % TOKEN_DTYPE.itemsize:
            raise CptDataError(f"Token file size is not aligned to uint32: {self.path}")
        self.tokens = np.memmap(self.path, mode="r", dtype=TOKEN_DTYPE)
        self.sequence_length = sequence_length
        self.num_blocks = len(self.tokens) // sequence_length
        if self.num_blocks == 0:
            raise CptDataError(
                f"Token file {self.path} has {len(self.tokens)} tokens, fewer than sequence length {sequence_length}"
            )

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.num_blocks
        if index < 0 or index >= self.num_blocks:
            raise IndexError(index)
        start = index * self.sequence_length
        end = start + self.sequence_length
        # Copy makes the array writable and avoids undefined behaviour in Torch.
        input_ids = torch.from_numpy(np.array(self.tokens[start:end], dtype=np.int64, copy=True))
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)
