#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the V100 multi-node SELECT stage-1 trainer via torchrun."
    )
    parser.add_argument("--config", required=True, help="Path to the V100 JSON config")
    parser.add_argument(
        "--platform",
        default="cuda",
        choices=("cuda",),
        help="Backend platform exposed to the shared distributed trainer.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_rank is not None:
        existing_local_rank = os.environ.get("LOCAL_RANK")
        if existing_local_rank is not None and int(existing_local_rank) != args.local_rank:
            raise ValueError("--local-rank disagrees with the LOCAL_RANK environment")
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    os.environ.setdefault("SELECT_DEPLOY_TARGET", "v100")
    os.environ.setdefault(
        "SELECT_MODEL_DIR", str(ROOT / "artifacts" / "models" / "Qwen3-0.6B-Base")
    )
    os.environ.setdefault(
        "SELECT_DATA_DIR", str(ROOT / "artifacts" / "data" / "fineweb_select_10b_8k")
    )
    os.environ.setdefault(
        "SELECT_OUTPUT_DIR", str(ROOT / "outputs" / "v100-stage1-8k-10b")
    )
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    try:
        module = importlib.import_module("select_repro.distributed")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "select_repro.distributed is not available yet. "
            "This deploy wrapper expects the shared distributed trainer interface "
            "run_training(platform='cuda', config_path=...) to exist."
        ) from exc

    run_training = getattr(module, "run_training", None)
    if run_training is None:
        raise RuntimeError(
            "select_repro.distributed.run_training was not found. "
            "Expected interface: run_training(platform='cuda', config_path='...')."
        )

    run_training(platform=args.platform, config_path=str(config_path))


if __name__ == "__main__":
    main()
