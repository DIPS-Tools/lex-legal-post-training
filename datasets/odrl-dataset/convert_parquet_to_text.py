#!/usr/bin/env python3
"""Convert Hugging Face Parquet splits into plain UTF-8 text files.

By default, this script reads Parquet files from
``text-to-odrl-dataset-v1/data`` beside this script and writes the ``input``
column to ``text-to-odrl-dataset-v1/text/{split}.txt``. Each dataset record is
normalised to one line so record boundaries remain unambiguous.

Install the required Parquet reader:

  python -m pip install pyarrow

  Then convert the policy input descriptions:

  python open_data/convert_parquet_to_text.py

  It reads:

  open_data/text-to-odrl-dataset-v1/data/*.parquet

  And generates:

  open_data/text-to-odrl-dataset-v1/text/train.txt
  open_data/text-to-odrl-dataset-v1/text/validation.txt
  open_data/text-to-odrl-dataset-v1/text/test.txt

  Each policy description occupies one line. Other columns can also be exported:

  python open_data/convert_parquet_to_text.py --column output
  python open_data/convert_parquet_to_text.py --column instruction

  The script passed syntax and CLI validation. pyarrow is not currently installed in this virtual
  environment, so install it before conversion.



"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "text-to-odrl-dataset-v1" / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "text-to-odrl-dataset-v1" / "text"
SUPPORTED_COLUMNS = ("instruction", "input", "output")


def _load_pyarrow():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "This converter requires pyarrow. Install it in the active "
            "environment with: python -m pip install pyarrow"
        ) from exc
    return parquet


def _split_name(path: Path) -> str:
    """Return 'train' from names such as train-00000-of-00001.parquet."""
    split = path.name.split("-", 1)[0].strip()
    if not split:
        raise ValueError(f"Cannot infer dataset split from filename: {path.name}")
    return split


def _normalise_record(value: object) -> str:
    """Put one textual dataset record on exactly one output line."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def discover_splits(input_dir: Path) -> Dict[str, List[Path]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Parquet input directory not found: {input_dir}")

    grouped: DefaultDict[str, List[Path]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.parquet")):
        grouped[_split_name(path)].append(path)
    if not grouped:
        raise FileNotFoundError(f"No .parquet files found in: {input_dir}")
    return dict(grouped)


def convert_split(
    parquet,
    split: str,
    shards: Iterable[Path],
    output_dir: Path,
    column: str,
) -> tuple[Path, int]:
    output_path = output_dir / f"{split}.txt"
    temporary_path = output_dir / f".{split}.txt.tmp"
    row_count = 0

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for shard in shards:
                available_columns = set(parquet.ParquetFile(shard).schema.names)
                if column not in available_columns:
                    raise ValueError(
                        f"Column {column!r} is absent from {shard}. "
                        f"Available columns: {sorted(available_columns)}"
                    )
                table = parquet.read_table(shard, columns=[column])
                values = table.column(column).to_pylist()
                for value in values:
                    handle.write(_normalise_record(value) + "\n")
                    row_count += 1
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return output_path, row_count


def convert_dataset(input_dir: Path, output_dir: Path, column: str) -> int:
    parquet = _load_pyarrow()
    split_files = discover_splits(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in sorted(split_files):
        output_path, rows = convert_split(
            parquet=parquet,
            split=split,
            shards=split_files[split],
            output_dir=output_dir,
            column=column,
        )
        total += rows
        print(f"{split}: {rows} records -> {output_path}")
    print(f"Completed: {total} records from column {column!r}")
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing Parquet shards (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for split .txt files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--column",
        choices=SUPPORTED_COLUMNS,
        default="input",
        help="Text column to export; 'input' contains policy descriptions.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    convert_dataset(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        column=args.column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
