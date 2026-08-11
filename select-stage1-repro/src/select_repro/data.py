from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


UINT32_BYTES = np.dtype(np.uint32).itemsize


class PackedTokenDataset(Dataset[torch.Tensor]):
    """Fixed-length views over a contiguous uint32 token stream."""

    def __init__(
        self,
        token_file: str | Path,
        *,
        sequence_length: int,
        offset_tokens: int = 0,
        num_tokens: int | None = None,
    ) -> None:
        self.token_file = Path(token_file)
        self.sequence_length = int(sequence_length)
        self.offset_tokens = int(offset_tokens)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.offset_tokens < 0:
            raise ValueError("offset_tokens must be non-negative")

        file_tokens, remainder = divmod(self.token_file.stat().st_size, UINT32_BYTES)
        if remainder:
            raise ValueError(f"{self.token_file} is not a valid uint32 token file")
        available = file_tokens - self.offset_tokens
        if available < 0:
            raise ValueError("offset_tokens exceeds token file length")
        self.num_tokens = available if num_tokens is None else int(num_tokens)
        if self.num_tokens < 0 or self.num_tokens > available:
            raise ValueError("num_tokens is outside the available token range")
        if self.num_tokens % self.sequence_length:
            raise ValueError("num_tokens must be a multiple of sequence_length")

        self._tokens = np.memmap(self.token_file, mode="r", dtype=np.uint32)

    def __len__(self) -> int:
        return self.num_tokens // self.sequence_length

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = self.offset_tokens + index * self.sequence_length
        end = start + self.sequence_length
        # Copy because tensors backed by a read-only memmap are unsafe to mutate.
        return torch.from_numpy(np.array(self._tokens[start:end], dtype=np.int64))

    def close(self) -> None:
        mmap = getattr(getattr(self, "_tokens", None), "_mmap", None)
        if mmap is not None and not mmap.closed:
            mmap.close()

    def __del__(self) -> None:
        self.close()


class MLMCollator:
    """BERT-style dynamic 15% masking with the standard 80/10/10 policy."""

    def __init__(
        self,
        *,
        mask_token_id: int,
        vocab_size: int,
        special_token_ids: Iterable[int],
        mlm_probability: float = 0.15,
        seed: int = 42,
    ) -> None:
        if not 0.0 < mlm_probability < 1.0:
            raise ValueError("mlm_probability must be between zero and one")
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError("mask_token_id must be inside the vocabulary")
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)
        self.special_token_ids = tuple(sorted(set(map(int, special_token_ids))))
        self.mlm_probability = float(mlm_probability)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def state_dict(self) -> dict[str, Any]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.generator.set_state(state["generator_state"])

    def __call__(self, examples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("MLMCollator received an empty batch")
        inputs = torch.stack(examples).to(dtype=torch.long)
        labels = inputs.clone()
        probability = torch.full(inputs.shape, self.mlm_probability)

        special = torch.zeros_like(inputs, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special |= inputs.eq(token_id)
        probability.masked_fill_(special, 0.0)
        masked = torch.bernoulli(probability, generator=self.generator).bool()

        # Tiny unit-test sequences can sample no targets. Keep every row trainable.
        for row in range(masked.shape[0]):
            if not masked[row].any():
                eligible = (~special[row]).nonzero(as_tuple=False).flatten()
                if eligible.numel() == 0:
                    raise ValueError("a sequence contains no mask-eligible tokens")
                choice = torch.randint(
                    eligible.numel(), (1,), generator=self.generator
                ).item()
                masked[row, eligible[choice]] = True

        labels[~masked] = -100

        replace_with_mask = (
            torch.bernoulli(torch.full(inputs.shape, 0.8), generator=self.generator).bool()
            & masked
        )
        inputs[replace_with_mask] = self.mask_token_id

        # Half of the remaining 20% become random tokens; the other half stay unchanged.
        replace_with_random = (
            torch.bernoulli(torch.full(inputs.shape, 0.5), generator=self.generator).bool()
            & masked
            & ~replace_with_mask
        )
        random_tokens = torch.randint(
            self.vocab_size, inputs.shape, dtype=torch.long, generator=self.generator
        )
        inputs[replace_with_random] = random_tokens[replace_with_random]

        return {"input_ids": inputs, "labels": labels}


def load_dataset_metadata(data_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(data_dir) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)
