import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bidirlm_BIO_finetune import onnx_inference as oi
from bidirlm_BIO_finetune import standalone_inference as si


class FakeRunner:
    """Logits depend only on the token id, so windowing must not change them."""

    def __init__(self):
        self.seen_lengths = []

    def run(self, input_ids, attention_mask):
        ids = np.asarray(input_ids)[0]
        self.seen_lengths.append(len(ids))
        classification = np.zeros((len(ids), 3), dtype=np.float32)
        classification[:, 0] = 1.0
        hit = ids % 7 == 0
        classification[hit, 0] = -1.0
        classification[hit, 2] = 2.0
        transition = np.zeros((len(ids), 3, 3), dtype=np.float32)
        transition[:, 2, 2] = 0.5
        return classification, transition


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(char) for char in text],
            "attention_mask": [1] * len(text),
            "offset_mapping": [[index, index + 1] for index in range(len(text))],
        }


class NumpyParityTests(unittest.TestCase):
    """The numpy reference must decode exactly like the torch implementation."""

    def test_log_softmax_matches_torch(self):
        # Route through lists: some environments have a broken torch<->numpy
        # bridge, and the parity being checked here does not involve it.
        values = np.random.RandomState(0).randn(9, 3).astype(np.float32)
        expected = torch.tensor(values.tolist()).log_softmax(-1).tolist()
        self.assertTrue(np.allclose(oi.log_softmax(values).tolist(), expected, atol=1e-6))

    def test_viterbi_matches_torch_implementation(self):
        generator = np.random.RandomState(7)
        for _ in range(5):
            cls_lp = generator.randn(20, 3).astype(np.float32)
            tr_lp = generator.randn(20, 3, 3).astype(np.float32)
            valid = [True] * 20
            numpy_labels = oi.viterbi(cls_lp, tr_lp, valid)
            torch_labels = si.viterbi(
                torch.tensor(cls_lp.tolist()), torch.tensor(tr_lp.tolist()), valid
            )
            self.assertEqual(numpy_labels, torch_labels)

    def test_viterbi_marks_invalid_positions(self):
        cls_lp = np.zeros((4, 3), dtype=np.float32)
        tr_lp = np.zeros((4, 3, 3), dtype=np.float32)
        labels = oi.viterbi(cls_lp, tr_lp, [False, True, True, False])
        self.assertEqual(labels[0], -100)
        self.assertEqual(labels[3], -100)

    def test_window_plan_matches_torch_implementation(self):
        for length, window, stride in ((5, 8, 6), (20, 8, 6), (16, 8, 4), (21, 8, 3)):
            self.assertEqual(
                oi.window_plan(length, window, stride),
                si.window_plan(length, window, stride),
            )

    def test_span_helpers_match_torch_implementation(self):
        labels = [0, 1, 2, 0, 2, 2, 1]
        self.assertEqual(
            oi.labels_to_token_spans(labels), si.labels_to_token_spans(labels)
        )
        offsets = [(0, 0), (0, 2), (2, 5), (5, 5)]
        self.assertEqual(
            oi.token_spans_to_char_spans(offsets, [(0, 3)]),
            si.token_spans_to_char_spans(offsets, [(0, 3)]),
        )

    def test_refine_text_is_verbatim_and_window_invariant(self):
        text = "".join(chr(code) for code in range(60, 120))
        whole = oi.refine_text(text, FakeTokenizer(), FakeRunner(), window=256, stride=192)
        windowed = oi.refine_text(text, FakeTokenizer(), FakeRunner(), window=16, stride=12)
        self.assertEqual(whole["refined_text"], windowed["refined_text"])
        expected = "".join(char for char in text if ord(char) % 7 == 0)
        self.assertEqual(whole["refined_text"], expected)
        for start, end in whole["char_spans"]:
            self.assertIn(text[start:end], text)


class BucketPaddingTests(unittest.TestCase):
    def test_pad_to_bucket_right_pads_and_masks(self):
        ids = np.arange(1, 6, dtype=np.int64).reshape(1, 5)
        padded, mask, length = oi.pad_to_bucket(ids, [4, 8, 16], pad_token_id=0)
        self.assertEqual(padded.shape, (1, 8))
        self.assertEqual(length, 5)
        self.assertEqual(padded[0, :5].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(padded[0, 5:].tolist(), [0, 0, 0])
        self.assertEqual(mask[0].tolist(), [1, 1, 1, 1, 1, 0, 0, 0])

    def test_no_buckets_passes_through(self):
        ids = np.arange(3, dtype=np.int64).reshape(1, 3)
        padded, mask, length = oi.pad_to_bucket(ids, None)
        self.assertEqual(padded.shape, (1, 3))
        self.assertEqual(length, 3)
        self.assertEqual(mask[0].tolist(), [1, 1, 1])

    def test_sequence_longer_than_largest_bucket_is_rejected(self):
        ids = np.zeros((1, 20), dtype=np.int64)
        with self.assertRaises(ValueError):
            oi.pad_to_bucket(ids, [4, 8, 16])

    def test_bucketed_run_gives_the_same_text_and_uses_bucket_shapes(self):
        text = "".join(chr(code) for code in range(60, 100))
        plain_runner, bucket_runner = FakeRunner(), FakeRunner()
        plain = oi.refine_text(text, FakeTokenizer(), plain_runner, window=16, stride=12)
        bucketed = oi.refine_text(
            text,
            FakeTokenizer(),
            bucket_runner,
            window=16,
            stride=12,
            buckets=[32, 64],
        )
        self.assertEqual(plain["refined_text"], bucketed["refined_text"])
        self.assertEqual(set(bucket_runner.seen_lengths), {32})


class CliTests(unittest.TestCase):
    def test_main_writes_refined_records(self):
        text = "".join(chr(code) for code in range(60, 90))
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.jsonl"
            output_path = Path(directory) / "refined.jsonl"
            input_path.write_text(
                json.dumps({"id": "a", "source_text": text}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            original_runner, original_loader = oi.OnnxRunner, oi.load_tokenizer
            oi.OnnxRunner = lambda *args, **kwargs: FakeRunner()
            oi.load_tokenizer = lambda *args, **kwargs: FakeTokenizer()
            try:
                code = oi.main(
                    [
                        "--model", "unused.onnx",
                        "--tokenizer", "unused",
                        "--input", str(input_path),
                        "--output", str(output_path),
                    ]
                )
            finally:
                oi.OnnxRunner, oi.load_tokenizer = original_runner, original_loader
            written = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(written["id"], "a")
        self.assertEqual(
            written["refined_text"],
            "".join(char for char in text if ord(char) % 7 == 0),
        )


if __name__ == "__main__":
    unittest.main()
