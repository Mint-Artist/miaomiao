import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from select_repro.distributed import (
    build_rank_data_plan,
    compare_resume_invariants,
    load_distributed_config,
    make_rank_sequence_order,
)


def valid_config(**overrides):
    config = {
        "platform": "cuda",
        "distributed_backend": "nccl",
        "distributed_strategy": "ddp",
        "precision": "fp16",
        "parameter_dtype": "float32",
        "use_grad_scaler": True,
        "base_model": "${TEST_MODEL_DIR}",
        "data_dir": "${TEST_DATA_DIR}",
        "output_dir": "${TEST_OUTPUT_DIR}",
        "sequence_length": 8,
        "train_tokens": 80,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "mlm_probability": 0.15,
        "mask_token": "<|mask|>",
        "learning_rate": 5e-5,
        "min_learning_rate": 1e-6,
        "warmup_steps": 1,
        "weight_decay": 0.1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "seed": 42,
        "shuffle": True,
        "gradient_checkpointing": True,
        "fused_optimizer": False,
        "log_every_steps": 1,
        "save_every_steps": 0,
        "max_optimizer_steps": None,
        "resume_from": None,
    }
    config.update(overrides)
    return config


class DistributedDataPlanTests(unittest.TestCase):
    def test_rank_shards_are_equal_disjoint_and_deterministic(self) -> None:
        shards = [
            make_rank_sequence_order(
                total_sequences=10,
                usable_sequences=9,
                rank=rank,
                world_size=3,
                seed=17,
                shuffle=True,
            )
            for rank in range(3)
        ]
        self.assertEqual([len(shard) for shard in shards], [3, 3, 3])
        flattened = torch.cat(shards).tolist()
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(len(flattened), 9)
        repeated = make_rank_sequence_order(
            total_sequences=10,
            usable_sequences=9,
            rank=1,
            world_size=3,
            seed=17,
            shuffle=True,
        )
        self.assertTrue(torch.equal(shards[1], repeated))

    def test_plan_drops_only_distributed_microbatch_tail(self) -> None:
        plan = build_rank_data_plan(
            train_tokens=80,
            sequence_length=8,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            world_size=3,
        )
        self.assertEqual(plan.total_sequences, 10)
        self.assertEqual(plan.usable_sequences, 9)
        self.assertEqual(plan.dropped_sequences, 1)
        self.assertEqual(plan.micro_batches_per_rank, 3)
        self.assertEqual(plan.optimizer_steps, 2)
        self.assertEqual(plan.effective_train_tokens, 72)
        self.assertEqual(plan.effective_executed_tokens, 72)

    def test_max_steps_caps_global_token_count(self) -> None:
        plan = build_rank_data_plan(
            train_tokens=80,
            sequence_length=8,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            world_size=3,
            max_optimizer_steps=1,
        )
        self.assertEqual(plan.optimizer_steps, 1)
        self.assertEqual(plan.effective_executed_tokens, 48)


class DistributedConfigTests(unittest.TestCase):
    def test_shipped_deployment_configs_validate(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        configs = [
            (repository / "deploy" / "v100" / "config-10b.json", "cuda"),
            (repository / "deploy" / "v100" / "config-smoke.json", "cuda"),
            (
                repository
                / "deploy"
                / "ascend910c"
                / "configs"
                / "stage1_ascend910c_8k_10b.json",
                "npu",
            ),
            (
                repository
                / "deploy"
                / "ascend910c"
                / "configs"
                / "stage1_ascend910c_8k_smoke.json",
                "npu",
            ),
        ]
        with TemporaryDirectory() as directory:
            environment = {
                "SELECT_MODEL_DIR": directory,
                "SELECT_DATA_DIR": directory,
                "SELECT_OUTPUT_DIR": str(Path(directory) / "output"),
            }
            with patch.dict(os.environ, environment, clear=False):
                for config_path, platform in configs:
                    with self.subTest(config=config_path.name):
                        config = load_distributed_config(
                            config_path, platform=platform
                        )
                        self.assertEqual(config["platform"], platform)

    def test_environment_paths_are_expanded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(valid_config()), encoding="utf-8")
            environment = {
                "TEST_MODEL_DIR": str(root / "model"),
                "TEST_DATA_DIR": str(root / "data"),
                "TEST_OUTPUT_DIR": str(root / "output"),
            }
            with patch.dict(os.environ, environment, clear=False):
                config = load_distributed_config(config_path, platform="cuda")
            self.assertEqual(config["base_model"], str((root / "model").resolve()))
            self.assertEqual(config["data_dir"], str((root / "data").resolve()))
            self.assertEqual(config["output_dir"], str((root / "output").resolve()))

    def test_unset_environment_path_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(valid_config()), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "TEST_MODEL_DIR": "",
                    "TEST_DATA_DIR": "",
                    "TEST_OUTPUT_DIR": "",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "unset environment"):
                    load_distributed_config(config_path, platform="cuda")

    def test_platform_precision_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(valid_config(precision="bf16")), encoding="utf-8"
            )
            environment = {
                "TEST_MODEL_DIR": directory,
                "TEST_DATA_DIR": directory,
                "TEST_OUTPUT_DIR": directory,
            }
            with patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(ValueError, "requires precision='fp16'"):
                    load_distributed_config(config_path, platform="cuda")

    def test_resume_invariants_report_world_size_change(self) -> None:
        saved = {"world_size": 8, "precision": "fp16"}
        expected = {"world_size": 16, "precision": "fp16"}
        mismatches = compare_resume_invariants(saved, expected)
        self.assertEqual(
            mismatches["world_size"], {"checkpoint": 8, "current": 16}
        )


if __name__ == "__main__":
    unittest.main()
