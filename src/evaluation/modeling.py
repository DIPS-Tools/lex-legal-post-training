"""Qwen model loading, conditional likelihood scoring, and deterministic generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.evaluation.common import EvaluationError


@dataclass(frozen=True)
class ContinuationScore:
    sum_logprob: float
    mean_logprob: float
    tokens: int


def batched(values: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


class ModelEvaluator:
    def __init__(
        self,
        model_path: Path,
        *,
        batch_size: int,
        max_input_tokens: int,
        attn_implementation: str,
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for benchmark evaluation") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Benchmark evaluation requires a CUDA GPU")
        if batch_size <= 0:
            raise EvaluationError("batch_size must be positive")
        if max_input_tokens < 2:
            raise EvaluationError("max_input_tokens must be at least 2")

        self.torch = torch
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map="auto",
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device

    def _encode_pair(self, prompt: str, continuation: str) -> tuple[list[int], list[int]]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        continuation_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
        if not prompt_ids:
            raise EvaluationError("Cannot score an empty prompt")
        if not continuation_ids:
            raise EvaluationError(f"Continuation tokenized to no tokens: {continuation!r}")
        if len(prompt_ids) + len(continuation_ids) > self.max_input_tokens:
            raise EvaluationError(
                "Evaluation prompt exceeds --max-input-tokens; refusing silent truncation: "
                f"{len(prompt_ids)} prompt + {len(continuation_ids)} continuation > {self.max_input_tokens}"
            )
        return prompt_ids, continuation_ids

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[ContinuationScore]:
        torch = self.torch
        results: list[ContinuationScore] = []
        for pair_batch in batched(pairs, self.batch_size):
            encoded = [self._encode_pair(prompt, continuation) for prompt, continuation in pair_batch]
            lengths = [len(prompt_ids) + len(continuation_ids) for prompt_ids, continuation_ids in encoded]
            max_length = max(lengths)
            input_ids = torch.full(
                (len(encoded), max_length),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.input_device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, (prompt_ids, continuation_ids) in enumerate(encoded):
                sequence = prompt_ids + continuation_ids
                input_ids[row, : len(sequence)] = torch.tensor(sequence, device=self.input_device)
                attention_mask[row, : len(sequence)] = 1

            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)

            for row, (prompt_ids, continuation_ids) in enumerate(encoded):
                start = len(prompt_ids) - 1
                positions = torch.arange(start, start + len(continuation_ids), device=log_probs.device)
                targets = torch.tensor(continuation_ids, device=log_probs.device)
                selected = log_probs[row, positions, targets]
                results.append(
                    ContinuationScore(
                        sum_logprob=float(selected.sum().item()),
                        mean_logprob=float(selected.mean().item()),
                        tokens=len(continuation_ids),
                    )
                )
        return results

    def generate(self, prompts: Sequence[str], *, max_new_tokens: int) -> list[str]:
        if max_new_tokens <= 0:
            raise EvaluationError("max_new_tokens must be positive")
        outputs: list[str] = []
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for prompt_batch in batched(prompts, self.batch_size):
                token_lengths = [len(self.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompt_batch]
                if max(token_lengths) + max_new_tokens > self.max_input_tokens:
                    raise EvaluationError(
                        "Generation prompt plus requested output exceeds --max-input-tokens; "
                        "refusing silent truncation"
                    )
                encoded = self.tokenizer(
                    list(prompt_batch),
                    add_special_tokens=False,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                ).to(self.input_device)
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                prefix_length = encoded["input_ids"].shape[1]
                for row in range(generated.shape[0]):
                    outputs.append(
                        self.tokenizer.decode(generated[row, prefix_length:], skip_special_tokens=True)
                    )
        finally:
            self.tokenizer.padding_side = original_padding_side
        return outputs
