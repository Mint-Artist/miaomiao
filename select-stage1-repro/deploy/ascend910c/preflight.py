#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


EXPECTED_COMPATIBILITY = {
    "torch": os.environ.get("SELECT_EXPECTED_TORCH_VERSION"),
    "torch_npu": os.environ.get("SELECT_EXPECTED_TORCH_NPU_VERSION"),
    "cann": os.environ.get("SELECT_EXPECTED_CANN_VERSION"),
}
ENV_PATTERN = re.compile(r"\$(?:([A-Za-z_][A-Za-z0-9_]*)|\{([^}]+)\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Ascend 910C environment before distributed training."
    )
    parser.add_argument("--config", help="Optional JSON training config to validate.")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    missing = [
        first or braced
        for first, braced in ENV_PATTERN.findall(value)
        if not os.environ.get(first or braced)
    ]
    if missing:
        raise ValueError(
            "unset config environment variables: " + ", ".join(sorted(set(missing)))
        )
    path = Path(os.path.expanduser(os.path.expandvars(value)))
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def find_cann_root() -> Path | None:
    candidates = [
        os.environ.get("ASCEND_HOME_PATH"),
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend-toolkit",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def detect_cann_version(cann_root: Path | None) -> str | None:
    if cann_root is None:
        return None
    for relative in ("version.info", "latest/version.info"):
        candidate = cann_root / relative
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip().lower() in {"version", "innerversion"}:
                        return value.strip()
                if "version" in line.lower():
                    return line.split(":", 1)[-1].strip()
    return None


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_keys = [
        "base_model",
        "data_dir",
        "output_dir",
        "sequence_length",
        "train_tokens",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "precision",
    ]
    for key in required_keys:
        if key not in config:
            errors.append(f"missing config key: {key}")
    if "base_model" in config and not resolve_path(config["base_model"]).exists():
        errors.append(f"base_model not found: {resolve_path(config['base_model'])}")
    if "data_dir" in config and not resolve_path(config["data_dir"]).exists():
        errors.append(f"data_dir not found: {resolve_path(config['data_dir'])}")
    if config.get("sequence_length") and config.get("train_tokens"):
        sequence_length = int(config["sequence_length"])
        train_tokens = int(config["train_tokens"])
        if train_tokens % sequence_length:
            errors.append("train_tokens must be a multiple of sequence_length")
    if config.get("precision") != "bf16":
        errors.append("Ascend 910C recipe requires precision=bf16")
    if config.get("distributed_backend") != "hccl":
        errors.append("Ascend distributed training requires distributed_backend=hccl")
    if config.get("use_grad_scaler") is not False:
        errors.append("BF16 recipe must set use_grad_scaler=false")
    if "data_dir" in config and resolve_path(config["data_dir"]).exists():
        metadata_path = resolve_path(config["data_dir"]) / "metadata.json"
        if not metadata_path.is_file():
            errors.append(f"data metadata not found: {metadata_path}")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata.get("sequence_length", -1)) != int(config["sequence_length"]):
                errors.append("dataset and config sequence_length do not match")
            if int(metadata.get("train_tokens", -1)) < int(config["train_tokens"]):
                errors.append("dataset contains fewer than the requested 10B train tokens")
    return errors


def main() -> None:
    args = parse_args()
    os.environ.setdefault(
        "SELECT_MODEL_DIR", str(REPO_ROOT / "artifacts" / "models" / "Qwen3-0.6B-Base")
    )
    os.environ.setdefault(
        "SELECT_DATA_DIR",
        str(REPO_ROOT / "artifacts" / "data" / "fineweb_select_10b_8k"),
    )
    os.environ.setdefault(
        "SELECT_OUTPUT_DIR", str(REPO_ROOT / "outputs" / "ascend910c-stage1-8k-10b")
    )

    report: dict[str, Any] = {
        "expected_stack_from_environment": EXPECTED_COMPATIBILITY,
        "compatibility_note": (
            "Use the official Ascend image/version matrix for the installed driver "
            "and CANN release. Set SELECT_EXPECTED_*_VERSION to enforce local pins."
        ),
        "warnings": [],
        "errors": [],
    }

    cann_root = find_cann_root()
    report["cann_root"] = str(cann_root) if cann_root else None
    report["cann_version"] = detect_cann_version(cann_root)

    report["npu_smi"] = run_command(["npu-smi", "info"])

    try:
        import torch
    except ImportError as exc:
        report["errors"].append(f"PyTorch import failed: {exc}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(1) from exc

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        report["errors"].append(f"torch-npu import failed: {exc}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(1) from exc

    report["python_version"] = sys.version.split()[0]
    report["torch_version"] = torch.__version__
    report["torch_npu_version"] = getattr(sys.modules.get("torch_npu"), "__version__", "unknown")
    report["device_count"] = int(torch.npu.device_count())
    report["hccl_available"] = bool(torch.distributed.is_backend_available("hccl"))
    report["distributed_env"] = {
        "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
        "MASTER_PORT": os.environ.get("MASTER_PORT"),
        "NNODES": os.environ.get("NNODES"),
        "NODE_RANK": os.environ.get("NODE_RANK"),
        "NPROC_PER_NODE": os.environ.get("NPROC_PER_NODE"),
        "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "ASCEND_VISIBLE_DEVICES": os.environ.get("ASCEND_VISIBLE_DEVICES"),
        "PYTORCH_NPU_ALLOC_CONF": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
        "TASK_QUEUE_ENABLE": os.environ.get("TASK_QUEUE_ENABLE"),
        "ASCEND_LAUNCH_BLOCKING": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
    }

    if report["device_count"] <= 0:
        report["errors"].append("torch.npu.device_count() returned 0")
    if not report["hccl_available"]:
        report["errors"].append("the current TorchNPU build has no HCCL backend")
    requested_processes = int(os.environ.get("NPROC_PER_NODE", "1"))
    if requested_processes > report["device_count"]:
        report["errors"].append(
            f"NPROC_PER_NODE={requested_processes} exceeds visible NPU count "
            f"{report['device_count']}"
        )
    if EXPECTED_COMPATIBILITY["torch"] and not str(report["torch_version"]).startswith(
        EXPECTED_COMPATIBILITY["torch"]
    ):
        report["warnings"].append(
            f"expected torch version is {EXPECTED_COMPATIBILITY['torch']}, "
            f"current is {report['torch_version']}"
        )
    if EXPECTED_COMPATIBILITY["torch_npu"] and not str(
        report["torch_npu_version"]
    ).startswith(EXPECTED_COMPATIBILITY["torch_npu"]):
        report["warnings"].append(
            f"expected torch-npu version is {EXPECTED_COMPATIBILITY['torch_npu']}, "
            f"current is {report['torch_npu_version']}"
        )
    if (
        EXPECTED_COMPATIBILITY["cann"]
        and report["cann_version"]
        and EXPECTED_COMPATIBILITY["cann"] not in report["cann_version"]
    ):
        report["warnings"].append(
            f"expected CANN version is {EXPECTED_COMPATIBILITY['cann']}, "
            f"current is {report['cann_version']}"
        )
    if cann_root is None:
        report["errors"].append(
            "could not locate CANN toolkit; source "
            "/usr/local/Ascend/ascend-toolkit/set_env.sh first"
        )
    nnodes = int(os.environ.get("NNODES", "1"))
    if nnodes > 1:
        for variable in ("MASTER_ADDR", "MASTER_PORT", "NODE_RANK"):
            if os.environ.get(variable) is None:
                report["errors"].append(
                    f"{variable} is required when NNODES is greater than one"
                )
        if os.environ.get("MASTER_ADDR") in {"127.0.0.1", "localhost"}:
            report["errors"].append(
                "MASTER_ADDR cannot be loopback for multi-node training"
            )
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is None:
        report["warnings"].append(
            "ASCEND_RT_VISIBLE_DEVICES is unset; all visible NPUs may be exposed to each process"
        )

    if args.config:
        config = load_config(args.config)
        report["config_path"] = str(resolve_path(args.config))
        report["config_validation_errors"] = validate_config(config)
        report["errors"].extend(report["config_validation_errors"])

    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
