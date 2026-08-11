#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ascend 910C multi-node launcher wrapper for SELECT stage-1 training."
    )
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument(
        "--platform",
        default="npu",
        choices=("npu",),
        help="Execution backend. Ascend deployment only supports npu here.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def resolve_config(path: str) -> str:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    return str(config_path)


def main() -> None:
    args = parse_args()
    if args.local_rank is not None:
        existing_local_rank = os.environ.get("LOCAL_RANK")
        if existing_local_rank is not None and int(existing_local_rank) != args.local_rank:
            raise ValueError("--local-rank disagrees with the LOCAL_RANK environment")
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank))

    os.environ.setdefault("SELECT_DISTRIBUTED_PLATFORM", args.platform)
    os.environ.setdefault("SELECT_DEVICE_TYPE", "npu")
    os.environ.setdefault(
        "SELECT_MODEL_DIR",
        str(REPO_ROOT / "artifacts" / "models" / "Qwen3-0.6B-Base"),
    )
    os.environ.setdefault(
        "SELECT_DATA_DIR",
        str(REPO_ROOT / "artifacts" / "data" / "fineweb_select_10b_8k"),
    )
    os.environ.setdefault(
        "SELECT_OUTPUT_DIR",
        str(REPO_ROOT / "outputs" / "ascend910c-stage1-8k-10b"),
    )

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torch-npu is not installed in the current environment. "
            "Use deploy/ascend910c/requirements-ascend.txt or the documented container image."
        ) from exc

    try:
        from select_repro.distributed import run_training
    except ImportError as exc:
        raise RuntimeError(
            "The shared distributed entrypoint `select_repro.distributed.run_training` "
            "is not present in this checkout. This deployment folder assumes the main "
            "codebase exposes that interface. Sync the branch that contains the shared "
            "distributed trainer before launching on Ascend."
        ) from exc

    run_training(platform=args.platform, config_path=resolve_config(args.config))


if __name__ == "__main__":
    main()
