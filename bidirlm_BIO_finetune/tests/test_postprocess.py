import json
import tempfile
import unittest
from pathlib import Path

from bidirlm_BIO_finetune.postprocess import (
    main,
    postprocess_char_spans,
    postprocess_record,
    snap_span,
    trim_span,
)


class PostprocessTests(unittest.TestCase):
    def test_trim_drops_unpaired_trailing_opener(self):
        text = "正文内容。【广告】"
        self.assertEqual(trim_span(text, 0, 6), (0, 5))

    def test_trim_keeps_paired_brackets(self):
        text = "见【说明】。"
        self.assertEqual(trim_span(text, 0, 6), (0, 6))

    def test_trim_drops_unpaired_leading_closer_and_whitespace(self):
        text = "】 正文。"
        self.assertEqual(trim_span(text, 0, 5), (2, 5))

    def test_snap_recovers_lost_prefix(self):
        text = "上一句。这是丢了开头的句子。"
        self.assertEqual(snap_span(text, 6, len(text), window=5), (4, len(text)))

    def test_snap_extends_truncated_end(self):
        text = "第一句。第二句没写完呢。"
        self.assertEqual(snap_span(text, 0, 10, window=5), (0, 12))

    def test_snap_leaves_span_without_nearby_boundary(self):
        text = "abcdefghijklmnopqrst"
        self.assertEqual(snap_span(text, 5, 15, window=3), (5, 15))

    def test_postprocess_snaps_and_merges_touching_spans(self):
        text = "第一句。第二句。"
        self.assertEqual(
            postprocess_char_spans(text, [(0, 5), (3, 8)], snap_window=2),
            [(0, 8)],
        )

    def test_postprocess_min_chars_drops_short_segments(self):
        text = "第一句。第二句。"
        self.assertEqual(
            postprocess_char_spans(text, [(0, 4)], snap_window=0, min_chars=4),
            [],
        )

    def test_postprocess_rejects_out_of_range_span(self):
        with self.assertRaises(ValueError):
            postprocess_char_spans("短文", [(0, 9)])

    def test_postprocess_record_adds_fields(self):
        record = {
            "source_text": "正文内容。【广告】",
            "predicted_retained_segments": [{"char_span": [0, 6]}],
        }
        result = postprocess_record(
            record,
            snap_window=5,
            boundary_chars="。\n",
            min_chars=0,
            separator="",
        )
        self.assertEqual(result["postprocessed_char_spans"], [[0, 5]])
        self.assertEqual(result["postprocessed_refined_text"], "正文内容。")

    def test_postprocess_record_requires_prediction_fields(self):
        with self.assertRaises(ValueError):
            postprocess_record(
                {"source_text": "x"},
                snap_window=5,
                boundary_chars="。",
                min_chars=0,
                separator="",
            )

    def test_main_round_trip(self):
        record = {
            "source_text": "上一句。这是正文。【广告】",
            "predicted_retained_segments": [{"char_span": [4, 10]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "predictions.jsonl"
            output_path = Path(directory) / "predictions.post.jsonl"
            input_path.write_text(
                json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            self.assertEqual(
                main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            written = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(written["postprocessed_refined_text"], "这是正文。")


if __name__ == "__main__":
    unittest.main()
