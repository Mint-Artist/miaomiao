import json
import tempfile
import unittest
from pathlib import Path

import torch

from bidirlm_BIO_finetune.data import BioDataCollator, BioJsonlDataset
from bidirlm_BIO_finetune.decoding import (
    bio_spans,
    compute_bio_metrics,
    pad_and_cat,
    viterbi_decode_batch,
)
from bidirlm_BIO_finetune.modeling import select_loss
from bidirlm_BIO_finetune.train import DistributedEvalSampler, merge_evaluation_parts


class DataAndDecodingTests(unittest.TestCase):
    def test_dataset_and_dynamic_padding(self):
        records = [
            {
                "id": "a",
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "labels": [1, 2],
            },
            {"id": "b", "input_ids": [6], "attention_mask": [1], "labels": [0]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
            )
            dataset = BioJsonlDataset(path, max_length=8)
            batch = BioDataCollator(0, pad_to_multiple_of=None)(
                [dataset[0], dataset[1]]
            )
        self.assertEqual(batch.get("input_ids").tolist(), [[4, 5], [6, 0]])
        self.assertEqual(batch.get("labels").tolist(), [[1, 2], [0, -100]])

    def test_viterbi_uses_transition_scores(self):
        classification = torch.zeros((1, 3, 3))
        classification[0, 0, 1] = 4
        classification[0, 1, 0] = 1
        classification[0, 2, 0] = 1
        transitions = torch.zeros((1, 3, 3, 3))
        transitions[0, 0, 1, 2] = 8
        transitions[0, 1, 2, 2] = 8
        decoded = viterbi_decode_batch(
            classification, transitions, torch.ones((1, 3), dtype=torch.bool)
        )
        self.assertEqual(decoded.tolist(), [[1, 2, 2]])

    def test_select_loss_is_finite_with_ignored_boundaries(self):
        classification = torch.randn(2, 4, 3, requires_grad=True)
        transitions = torch.randn(2, 4, 3, 3, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, -100], [0, 1, 2, 0]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        result = select_loss(classification, transitions, labels, mask)
        self.assertTrue(torch.isfinite(result.get("loss")))
        result.get("loss").backward()

    def test_metrics_and_spans(self):
        labels = torch.tensor([[0, 1, 2, 0, 1, 2]])
        metrics = compute_bio_metrics(labels, labels)
        self.assertEqual(metrics.get("span_f1"), 1.0)
        self.assertEqual(bio_spans(labels[0].tolist()), [(1, 3), (4, 6)])

    def test_pad_and_cat_handles_dynamic_batch_widths(self):
        result = pad_and_cat([torch.tensor([[1, 2]]), torch.tensor([[0]])])
        self.assertEqual(result.tolist(), [[1, 2], [0, -100]])

    def test_distributed_eval_sampler_has_no_duplicates(self):
        dataset = list(range(10))
        shards = [list(DistributedEvalSampler(dataset, rank, 3)) for rank in range(3)]
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(10)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_merge_distributed_evaluation_parts(self):
        parts = [
            {
                "loss_sum": 1.0,
                "classification_loss_sum": 0.4,
                "transition_loss_sum": 0.6,
                "batches": 1,
                "predictions": [[1, 2]],
                "labels": [[1, 2]],
            },
            {
                "loss_sum": 3.0,
                "classification_loss_sum": 1.0,
                "transition_loss_sum": 2.0,
                "batches": 1,
                "predictions": [[0]],
                "labels": [[0]],
            },
        ]
        metrics = merge_evaluation_parts(parts)
        self.assertEqual(metrics.get("loss"), 2.0)
        self.assertEqual(metrics.get("token_accuracy"), 1.0)


if __name__ == "__main__":
    unittest.main()
