import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from select_repro.training import cosine_learning_rate, validate_training_config
from scripts.prepare_fineweb import resolve_tokenizer_reference
from scripts.train_stage1 import resolve_training_dtypes, validate_resume_checkpoint


class TrainingUtilityTests(unittest.TestCase):
    def test_cosine_schedule_hits_peak_and_floor(self) -> None:
        values = [
            cosine_learning_rate(
                step,
                total_steps=6,
                peak_lr=5e-5,
                min_lr=1e-6,
                warmup_steps=2,
            )
            for step in range(6)
        ]
        self.assertAlmostEqual(values[0], 2.5e-5)
        self.assertAlmostEqual(values[1], 5e-5)
        self.assertAlmostEqual(values[-1], 1e-6)
        self.assertEqual(values[2:], sorted(values[2:], reverse=True))

    def test_config_requires_token_alignment(self) -> None:
        config = {
            "sequence_length": 8,
            "train_tokens": 15,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "log_every_steps": 1,
            "mlm_probability": 0.15,
            "learning_rate": 5e-5,
            "min_learning_rate": 1e-6,
            "warmup_steps": 0,
        }
        with self.assertRaisesRegex(ValueError, "multiple"):
            validate_training_config(config)

    def test_fp32_parameters_are_independent_from_bf16_compute(self) -> None:
        parameter_dtype, compute_dtype = resolve_training_dtypes(
            {"bf16": True, "parameter_dtype": "float32"}
        )
        self.assertIs(parameter_dtype, torch.float32)
        self.assertIs(compute_dtype, torch.bfloat16)

    def test_resume_invariant_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            invariants = {
                "dataset_token_file_sha256": "abc123",
                "mlm_probability": 0.15,
            }
            (checkpoint / "stage1_manifest.json").write_text(
                json.dumps({"resume_invariants": invariants}), encoding="utf-8"
            )
            validate_resume_checkpoint(checkpoint, invariants)
            changed = dict(invariants, mlm_probability=0.30)
            with self.assertRaisesRegex(ValueError, "mlm_probability"):
                validate_resume_checkpoint(checkpoint, changed)

    def test_local_tokenizer_path_is_canonicalized(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            local = Path(directory)
            relative = local.relative_to(Path.cwd())
            for candidate in (relative, local.resolve()):
                reference = resolve_tokenizer_reference(str(candidate))
                self.assertIsInstance(reference, Path)
                self.assertEqual(reference, local.resolve())
        self.assertEqual(
            resolve_tokenizer_reference("Qwen/Qwen3-0.6B-Base"),
            "Qwen/Qwen3-0.6B-Base",
        )


if __name__ == "__main__":
    unittest.main()
