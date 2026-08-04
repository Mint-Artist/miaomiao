#!/usr/bin/env python3
"""Run ProX stage-one document filtering over JSONL or JSONL.GZ data."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO


DEFAULT_MODEL_ID = "gair-prox/web-doc-refining-lm"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."


@contextmanager
def open_text_reader(path: Path) -> Iterator[TextIO]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield handle


@contextmanager
def open_text_writer(path: Path, gzip_enabled: bool = False) -> Iterator[TextIO]:
    if gzip_enabled:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open("w", encoding="utf-8") as handle:
            yield handle


def get_by_path(record: dict, dotted_path: str):
    value = record
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_llama2_prompt(
    text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    return (
        "<s>[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        "<</SYS>>\n\n"
        f"{text} [/INST]"
    )


def parse_doc_program(program: str, mode: str = "strict") -> str:
    normalized = program.strip().lower()

    if mode == "repo-compatible":
        return "drop" if "drop" in normalized else "keep"

    normalized = re.sub(r"[.!。！]+$", "", normalized).strip()
    if normalized == "keep":
        return "keep"
    if normalized == "drop":
        return "drop"
    return "unknown"


@dataclass
class InputRecord:
    line_number: int
    record: dict
    normalized_text: str


@dataclass
class GenerationResult:
    program: str
    truncated: bool
    error: str | None = None


class TransformersDocModel:
    def __init__(
        self,
        model_path: str,
        device: str,
        dtype_name: str,
        max_new_tokens: int,
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "缺少推理依赖，请先安装兼容 CUDA 的 PyTorch，再执行 "
                "pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("指定了 CUDA，但当前 PyTorch 无法使用 CUDA。")
        self.device = torch.device(device)

        if dtype_name == "auto":
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif self.device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[dtype_name]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

        context_length = int(getattr(self.model.config, "max_position_embeddings", 2048))
        self.max_prompt_tokens = context_length - max_new_tokens
        if self.max_prompt_tokens <= 0:
            raise ValueError("max_new_tokens 必须小于模型上下文长度。")

    def generate(self, texts: Sequence[str]) -> list[GenerationResult]:
        prompts = [build_llama2_prompt(text) for text in texts]
        prompt_lengths = [
            len(self.tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in prompts
        ]
        truncated = [length > self.max_prompt_tokens for length in prompt_lengths]

        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens,
            return_tensors="pt",
        ).to(self.device)
        input_width = encoded["input_ids"].shape[1]

        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for index, sequence in enumerate(generated):
            new_tokens = sequence[input_width:]
            program = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()
            results.append(
                GenerationResult(program=program, truncated=truncated[index])
            )
        return results


def batched(items: Iterable[InputRecord], size: int) -> Iterator[list[InputRecord]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def make_output_record(
    item: InputRecord,
    result: GenerationResult,
    parse_mode: str,
    output_prefix: str,
) -> dict:
    record = dict(item.record)
    decision = parse_doc_program(result.program, mode=parse_mode)
    output_text = "" if decision == "drop" else item.normalized_text

    record[f"{output_prefix}program"] = result.program
    record[f"{output_prefix}decision"] = decision
    record[f"{output_prefix}text"] = output_text
    record[f"{output_prefix}truncated"] = result.truncated
    record[f"{output_prefix}error"] = result.error
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 ProX 文档级模型给 JSONL 文档标记 keep/drop/unknown。"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--text-key",
        default="content",
        help="文本字段；嵌套字段可写成 data.cleaned_text。",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face 模型 ID 或离线模型目录。",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="每处理多少条打印一次进度；0 表示关闭。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--parse-mode",
        choices=["strict", "repo-compatible"],
        default="strict",
        help="strict 将非标准输出标为 unknown；repo-compatible 复现原仓库逻辑。",
    )
    parser.add_argument("--output-prefix", default="prox_doc_")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="禁止访问网络，只从本地目录或 Hugging Face 缓存加载。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的输出文件。",
    )
    return parser.parse_args()


def iter_records(path: Path, text_key: str) -> Iterator[InputRecord]:
    with open_text_reader(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON：{exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"第 {line_number} 行 JSON 顶层必须是对象。")
            try:
                text = get_by_path(record, text_key)
            except KeyError as exc:
                raise ValueError(
                    f"第 {line_number} 行缺少文本字段 {text_key!r}。"
                ) from exc
            if not isinstance(text, str):
                raise ValueError(f"第 {line_number} 行文本字段不是字符串。")
            yield InputRecord(
                line_number=line_number,
                record=record,
                normalized_text=normalize_text(text),
            )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch-size 必须大于 0。")
    if args.log_every < 0:
        raise SystemExit("log-every 不能小于 0。")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"输出文件已存在：{args.output}；如需覆盖请添加 --overwrite。")

    records = iter_records(args.input, args.text_key)
    model = TransformersDocModel(
        model_path=args.model,
        device=args.device,
        dtype_name=args.dtype,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    stats: Counter[str] = Counter()

    try:
        with open_text_writer(
            temporary_output,
            gzip_enabled=args.output.name.endswith(".gz"),
        ) as writer:
            for batch in batched(records, args.batch_size):
                stats["total"] += len(batch)
                nonempty = [item for item in batch if item.normalized_text]
                generated_by_line: dict[int, GenerationResult] = {}

                if nonempty:
                    try:
                        generated = model.generate(
                            [item.normalized_text for item in nonempty]
                        )
                        generated_by_line.update(
                            (item.line_number, result)
                            for item, result in zip(nonempty, generated)
                        )
                    except RuntimeError as exc:
                        # 批次失败时逐条重试，单条仍失败则保留原文并记录错误。
                        print(f"批次推理失败，改为逐条重试：{exc}", file=sys.stderr)
                        if model.device.type == "cuda":
                            model.torch.cuda.empty_cache()
                        for item in nonempty:
                            try:
                                generated_by_line[item.line_number] = model.generate(
                                    [item.normalized_text]
                                )[0]
                            except Exception as item_exc:  # noqa: BLE001
                                generated_by_line[item.line_number] = GenerationResult(
                                    program="",
                                    truncated=False,
                                    error=f"{type(item_exc).__name__}: {item_exc}",
                                )
                                if model.device.type == "cuda":
                                    model.torch.cuda.empty_cache()

                for item in batch:
                    result = generated_by_line.get(
                        item.line_number,
                        GenerationResult(
                            program="",
                            truncated=False,
                            error="empty_input" if not item.normalized_text else "missing_result",
                        ),
                    )
                    output_record = make_output_record(
                        item=item,
                        result=result,
                        parse_mode=args.parse_mode,
                        output_prefix=args.output_prefix,
                    )
                    decision = output_record[f"{args.output_prefix}decision"]
                    stats[decision] += 1
                    stats["truncated"] += int(result.truncated)
                    stats["error"] += int(result.error is not None)
                    writer.write(json.dumps(output_record, ensure_ascii=False) + "\n")

                if args.log_every and stats["total"] % args.log_every < len(batch):
                    print(
                        "已处理 "
                        f"{stats['total']} 条：keep={stats['keep']}，"
                        f"drop={stats['drop']}，unknown={stats['unknown']}，"
                        f"error={stats['error']}",
                        file=sys.stderr,
                    )

        os.replace(temporary_output, args.output)
    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    print(f"输出完成：{args.output}")


if __name__ == "__main__":
    main()
