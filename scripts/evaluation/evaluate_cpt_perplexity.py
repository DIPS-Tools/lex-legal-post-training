#!/usr/bin/env python3
"""Evaluate legal, replay, and combined held-out CPT perplexity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cpt_data import CptDataError, PackedTokenDataset, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096, choices=(4096, 8192))
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2")
    parser.add_argument("--deepspeed", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Evaluation requires a CUDA GPU with BF16 support")
    model_path = args.model_path.expanduser().resolve()
    tokenized_dir = args.tokenized_dir.expanduser().resolve()
    datasets = {}
    for name, filename in (
        ("all", "validation_tokens.bin"),
        ("legal", "validation_legal_tokens.bin"),
        ("replay", "validation_replay_tokens.bin"),
    ):
        path = tokenized_dir / filename
        if path.is_file() and path.stat().st_size >= args.sequence_length * 4:
            datasets[name] = PackedTokenDataset(path, args.sequence_length)
    if not datasets:
        raise CptDataError(f"No complete validation blocks found in {tokenized_dir}")

    training_args = TrainingArguments(
        output_dir=str(args.output.expanduser().resolve().parent / ".eval-tmp"),
        do_train=False,
        do_eval=True,
        bf16=True,
        tf32=True,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        deepspeed=str(args.deepspeed.expanduser().resolve()) if args.deepspeed else None,
        report_to=[],
        remove_unused_columns=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    trainer = Trainer(model=model, args=training_args, data_collator=default_data_collator)
    metrics = trainer.evaluate(eval_dataset=datasets)
    for key, value in list(metrics.items()):
        if key.endswith("_loss") and isinstance(value, (float, int)):
            metrics[key.removesuffix("_loss") + "_perplexity"] = (
                math.exp(float(value)) if value < math.log(sys.float_info.max) else math.inf
            )
    if trainer.is_world_process_zero():
        write_json(args.output.expanduser().resolve(), metrics)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (CptDataError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
