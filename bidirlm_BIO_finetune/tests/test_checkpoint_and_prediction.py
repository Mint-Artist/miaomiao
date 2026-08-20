import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from bidirlm_BIO_finetune.predict import (
    PredictionJsonlDataset,
    build_prediction_output,
    labels_to_token_spans,
)
from bidirlm_BIO_finetune.train import (
    next_training_position,
    prune_step_checkpoints,
    save_training_checkpoint,
)


class FakeTokenizer:
    def decode(self, input_ids, skip_special_tokens=True):
        return "".join(chr(96 + int(item)) for item in input_ids)


class FakeModel:
    def save_artifacts(self, output_dir, tokenizer):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "model.marker").write_text("saved\n", encoding="utf-8")


class CheckpointTests(unittest.TestCase):
    def test_next_training_position(self):
        self.assertEqual(next_training_position(2, 3, 10), (2, 4))
        self.assertEqual(next_training_position(2, 9, 10), (3, 0))

    def test_step_checkpoint_contains_mid_epoch_resume_position(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = AdamW([parameter], lr=1e-3)
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-step-00000020"
            save_training_checkpoint(
                checkpoint,
                model=FakeModel(),
                tokenizer=FakeTokenizer(),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=1,
                global_step=20,
                best_validation_loss=0.8,
                patience=1,
                resume_epoch=1,
                resume_batch_index=40,
                running_loss=12.0,
                running_classification_loss=5.0,
                running_transition_loss=7.0,
                running_batches=40,
                world_size=1,
            )
            state = torch.load(
                checkpoint / "trainer_state.pt",
                map_location="cpu",
                weights_only=False,
            )
        self.assertEqual(state["global_step"], 20)
        self.assertEqual(state["resume_epoch"], 1)
        self.assertEqual(state["resume_batch_index"], 40)
        self.assertEqual(state["running_batches"], 40)
        self.assertEqual(state["world_size"], 1)

    def test_prune_step_checkpoints_does_not_touch_best_or_last(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for step in (10, 20, 30):
                (output / f"checkpoint-step-{step:08d}").mkdir()
            (output / "best").mkdir()
            (output / "last").mkdir()
            prune_step_checkpoints(output, 2)
            remaining = sorted(path.name for path in output.iterdir())
        self.assertEqual(
            remaining,
            ["best", "checkpoint-step-00000020", "checkpoint-step-00000030", "last"],
        )


class PredictionOutputTests(unittest.TestCase):
    def test_audit_jsonl_is_flattened_for_prediction(self):
        audit = {
            "id": "doc-1",
            "source_text": "甲乙丙丁戊己",
            "teacher_refined_text": "乙丙戊己",
            "adjusted_refined_text": "乙丙戊己",
            "tokenization": {
                "input_ids": [1, 2, 3, 4, 5, 6],
                "attention_mask": [1, 1, 1, 1, 1, 1],
                "offset_mapping": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
            },
            "supervision": {"labels": [0, 1, 2, 0, 1, 2]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            path.write_text(
                json.dumps(audit, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            dataset = PredictionJsonlDataset(path, input_format="auto", max_length=8)
            record = dataset[0]
        self.assertEqual(record["input_ids"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(record["labels"], [0, 1, 2, 0, 1, 2])
        self.assertEqual(record["prediction_metadata"]["source_text"], "甲乙丙丁戊己")

    def test_prediction_output_contains_exact_source_segments(self):
        predicted_labels = [0, 1, 2, 0, 2, 2]
        output = build_prediction_output(
            sample_id="doc-1",
            input_ids=[1, 2, 3, 4, 5, 6],
            predicted_labels=predicted_labels,
            gold_labels=[0, 1, 2, 0, 1, 2],
            metadata={
                "input_format": "audit",
                "source_text": "甲乙丙丁戊己",
                "offset_mapping": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
                "teacher_refined_text": "乙丙戊己",
                "adjusted_refined_text": "乙丙戊己",
            },
            tokenizer=FakeTokenizer(),
        )
        self.assertEqual(labels_to_token_spans(predicted_labels), [(1, 3), (4, 6)])
        self.assertEqual(output["source_text"], "甲乙丙丁戊己")
        self.assertEqual(output["predicted_refined_text"], "乙丙戊己")
        self.assertEqual(
            [item["text"] for item in output["predicted_retained_segments"]],
            ["乙丙", "戊己"],
        )
        self.assertEqual(
            output["predicted_retained_segments"][1]["starts_with_tag"], "I"
        )


if __name__ == "__main__":
    unittest.main()
