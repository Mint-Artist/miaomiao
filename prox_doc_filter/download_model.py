#!/usr/bin/env python3
"""Download the ProX web document refining model for offline use."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL_ID = "gair-prox/web-doc-refining-lm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载 ProX 文档级精炼模型，供离线环境使用。"
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/web-doc-refining-lm"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "缺少 huggingface-hub，请先执行：pip install -r requirements.txt"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(args.output_dir),
    )
    print(f"模型下载完成：{downloaded_path}")


if __name__ == "__main__":
    main()
