from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training.cpt_data import (
    CptDataError,
    PackedTokenDataset,
    choose_split,
    discover_jsonl_files,
    tokenize_jsonl_dataset,
    validate_jsonl_dataset,
)


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool, truncation: bool) -> dict:
        assert add_special_tokens is False
        assert truncation is False
        return {"input_ids": [ord(character) for character in text]}


class CptDataTests(unittest.TestCase):
    def test_choose_split_is_stable(self) -> None:
        self.assertEqual(choose_split("same-id"), choose_split("same-id"))
        self.assertEqual(choose_split("anything", validation_per_mille=0), "train")
        self.assertEqual(choose_split("anything", validation_per_mille=1000), "validation")

    def test_work_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / ".work" / "preprocessed"
            work.mkdir(parents=True)
            (work / "accepted-0.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(CptDataError):
                discover_jsonl_files(work)

    def test_validation_detects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "final.jsonl"
            rows = [{"id": "x", "text": "one"}, {"id": "x", "text": "two"}]
            shard.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(CptDataError, "Duplicate document ID"):
                validate_jsonl_dataset(
                    [shard],
                    validation_per_mille=1,
                    replay_regex="replay",
                    sqlite_path=root / "ids.sqlite3",
                )

    def test_tokenization_adds_eos_and_packed_dataset_returns_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "final.jsonl"
            records = [
                {"id": "a", "text": "ab", "corpus": "legal"},
                {"id": "b", "text": "cd", "corpus": "general_replay"},
            ]
            shard.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            stats = tokenize_jsonl_dataset(
                [shard],
                tokenizer=FakeTokenizer(),
                output_dir=root / "tokens",
                validation_per_mille=0,
                replay_regex="replay",
                progress_every=0,
            )
            self.assertEqual(stats["train"], {"documents": 2, "tokens": 6})
            values = np.fromfile(root / "tokens/train_tokens.bin", dtype="<u4").tolist()
            self.assertEqual(values, [ord("a"), ord("b"), 99, ord("c"), ord("d"), 99])

            dataset = PackedTokenDataset(root / "tokens/train_tokens.bin", sequence_length=3)
            self.assertEqual(len(dataset), 2)
            example = dataset[0]
            self.assertEqual(example["input_ids"].tolist(), [ord("a"), ord("b"), 99])
            self.assertEqual(example["labels"].tolist(), example["input_ids"].tolist())
            self.assertEqual(example["attention_mask"].tolist(), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
