#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
ENV_PATTERN = re.compile(r"\$(?:([A-Za-z_][A-Za-z0-9_]*)|\{([^}]+)\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for multi-node V100 training.")
    parser.add_argument("--config", required=True, help="Path to the V100 JSON config")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings that normally pass")
    return parser.parse_args()


def expand_environment(value):
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value
    missing = [
        first or braced
        for first, braced in ENV_PATTERN.findall(value)
        if not os.environ.get(first or braced)
    ]
    if missing:
        raise ValueError(f"未设置配置所需环境变量: {', '.join(sorted(set(missing)))}")
    return os.path.expanduser(os.path.expandvars(value))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return expand_environment(json.load(stream))


def main() -> None:
    args = parse_args()
    os.environ.setdefault(
        "SELECT_MODEL_DIR", str(ROOT / "artifacts" / "models" / "Qwen3-0.6B-Base")
    )
    os.environ.setdefault(
        "SELECT_DATA_DIR", str(ROOT / "artifacts" / "data" / "fineweb_select_10b_8k")
    )
    os.environ.setdefault(
        "SELECT_OUTPUT_DIR", str(ROOT / "outputs" / "v100-stage1-8k-10b")
    )
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    config = load_config(config_path)

    failures: list[str] = []
    warnings: list[str] = []

    if not torch.cuda.is_available():
        failures.append("CUDA 不可用。V100 部署必须运行在启用 NVIDIA 驱动的机器上。")

    visible_devices = torch.cuda.device_count()
    if visible_devices == 0:
        failures.append("当前节点没有可见 GPU。请检查 CUDA_VISIBLE_DEVICES。")

    device_names: list[str] = []
    compute_caps: list[str] = []
    for index in range(visible_devices):
        props = torch.cuda.get_device_properties(index)
        device_names.append(props.name)
        compute_caps.append(f"{props.major}.{props.minor}")
        if props.major < 7:
            failures.append(
                f"GPU {index} ({props.name}) 计算能力 "
                f"{props.major}.{props.minor} 低于 Volta。"
            )
        elif props.major == 7 and props.minor == 0 and "V100" not in props.name.upper():
            warnings.append(f"GPU {index} 是 SM70 设备但名称不是 V100：{props.name}")

    if config.get("precision") != "fp16":
        failures.append("V100 配置必须使用 precision=fp16 并启用 GradScaler。")
    if config.get("parameter_dtype") not in (None, "float32"):
        warnings.append("建议 V100 保持 parameter_dtype=float32，以避免 FP16 master weight 精度损失。")
    if config.get("distributed_backend", "nccl") != "nccl":
        failures.append("V100 多机多卡应使用 distributed_backend=nccl。")
    if not torch.distributed.is_nccl_available():
        failures.append("当前 PyTorch 构建不包含 NCCL backend。")
    if config.get("distributed_strategy", "ddp") != "ddp":
        warnings.append("0.6B 模型在 32GB V100 上优先使用 DDP；无需先切 FSDP。")
    if not config.get("use_grad_scaler", True):
        failures.append("V100 FP16 训练必须启用 GradScaler。")

    base_model = Path(config["base_model"])
    data_dir = Path(config["data_dir"])
    if not (base_model / "config.json").is_file():
        failures.append(f"模型目录缺少 config.json: {base_model}")
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        failures.append(f"数据目录缺少 metadata.json: {data_dir}")
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("sequence_length", -1)) != int(config["sequence_length"]):
            failures.append("数据 metadata 的 sequence_length 与训练配置不一致。")
        if int(metadata.get("train_tokens", -1)) < int(config["train_tokens"]):
            failures.append("数据集不足 10B 训练 token，请先生成完整 packed 数据。")

    required_env = ("MASTER_ADDR", "MASTER_PORT", "NNODES", "NODE_RANK", "GPUS_PER_NODE")
    missing_env = [name for name in required_env if not os.environ.get(name)]
    if missing_env:
        warnings.append(f"缺少 torchrun 环境变量：{', '.join(missing_env)}")
    nnodes = int(os.environ.get("NNODES", "1"))
    if nnodes > 1:
        for variable in ("MASTER_ADDR", "MASTER_PORT", "NODE_RANK"):
            if not os.environ.get(variable):
                failures.append(f"NNODES > 1 时必须设置 {variable}。")
        if os.environ.get("MASTER_ADDR") in {"127.0.0.1", "localhost"}:
            failures.append("多机训练的 MASTER_ADDR 不能是 loopback 地址。")

    if os.environ.get("NCCL_SOCKET_IFNAME") is None:
        warnings.append("未设置 NCCL_SOCKET_IFNAME。多机训练建议显式绑定 RoCE/IB 网卡。")
    if os.environ.get("NCCL_IB_DISABLE") == "1":
        warnings.append("NCCL_IB_DISABLE=1 会禁用 InfiniBand/RDMA，只适合纯以太网环境。")
    if os.environ.get("OMP_NUM_THREADS") is None:
        warnings.append("未设置 OMP_NUM_THREADS。建议先设为每卡 4~8。")
    if os.environ.get("GPUS_PER_NODE"):
        requested_devices = int(os.environ["GPUS_PER_NODE"])
        if requested_devices > visible_devices:
            failures.append(
                f"GPUS_PER_NODE={requested_devices} 超过可见 GPU 数 {visible_devices}。"
            )

    hostname = socket.gethostname()
    summary = {
        "hostname": hostname,
        "config": str(config_path),
        "cuda_available": torch.cuda.is_available(),
        "visible_devices": visible_devices,
        "device_names": device_names,
        "compute_capabilities": compute_caps,
        "nccl_available": torch.distributed.is_nccl_available(),
        "env": {
            name: os.environ.get(name)
            for name in required_env
            + ("NCCL_SOCKET_IFNAME", "CUDA_VISIBLE_DEVICES")
        },
        "failures": failures,
        "warnings": warnings,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if failures:
        raise SystemExit(2)
    if args.strict and warnings:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
