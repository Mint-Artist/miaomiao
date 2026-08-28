from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sequence_BIO.alignment import (
    align_deletion_only,
    extract_refined_text,
    find_longest_match_segments,
)
from sequence_BIO.cli import main
from sequence_BIO.labeling import derive_transition_supervision, label_aligned_pair


class CharacterTokenizer:
    """Small fast-tokenizer stand-in with one token per character."""

    is_fast = True

    def __call__(self, text, **kwargs):
        max_length = kwargs.get("max_length")
        add_special_tokens = kwargs.get("add_special_tokens", True)
        character_count = len(text)
        if kwargs.get("truncation") and max_length is not None:
            reserved = 2 if add_special_tokens else 0
            character_count = min(character_count, max(0, max_length - reserved))
        input_ids = list(range(10, 10 + character_count))
        offsets = [(index, index + 1) for index in range(character_count)]
        if add_special_tokens:
            input_ids = [1] + input_ids + [2]
            offsets = [(0, 0)] + offsets + [(0, 0)]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "offset_mapping": offsets,
        }


class ListOffsetTokenizer(CharacterTokenizer):
    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        encoded["offset_mapping"] = [
            list(item) for item in encoded.get("offset_mapping", [])
        ]
        return encoded


class AlignmentTest(unittest.TestCase):
    def test_extracts_teacher_payload_without_stripping(self):
        response = (
            "refinement_reason:\n[doc]删除导航[/doc]\n\n"
            "refined_text:\n[doc]正文一\n正文二[/doc]"
        )
        self.assertEqual(extract_refined_text(response), "正文一\n正文二")

    def test_appendix_b_longest_match_segments(self):
        source = "A" * 20 + "NOISE" + "B" * 20
        target = "A" * 20 + "B" * 20
        matches = find_longest_match_segments(source, target)
        self.assertEqual(
            [
                (
                    item.source_start,
                    item.source_end,
                    item.target_start,
                    item.target_end,
                )
                for item in matches
            ],
            [(0, 20, 0, 20), (25, 45, 20, 40)],
        )

    def test_aligned_deletion_has_two_retained_spans(self):
        source = "A" * 20 + "NOISE" + "B" * 20
        target = "A" * 20 + "B" * 20
        result = align_deletion_only(source, target)
        self.assertEqual(result.status, "aligned")
        self.assertTrue(result.is_accepted)
        self.assertEqual(result.character_spans, ((0, 20), (25, 45)))
        self.assertEqual(result.target_coverage, 1.0)

    def test_adjusted_repairs_small_internal_modification(self):
        source = "A" * 20 + "cat" + "B" * 20
        target = "A" * 20 + "dogs" + "B" * 20
        result = align_deletion_only(source, target)
        self.assertEqual(result.status, "adjusted")
        self.assertEqual(result.character_spans, ((0, len(source)),))
        self.assertEqual(result.adjusted_refined_text, source)
        self.assertEqual(len(result.adjustments), 1)
        self.assertEqual(result.adjustments[0].source_text, "cat")
        self.assertEqual(result.adjustments[0].target_text, "dogs")

    def test_large_internal_modification_is_unaligned(self):
        source = "A" * 20 + "cat" + "B" * 20
        target = "A" * 20 + "X" * 10 + "B" * 20
        result = align_deletion_only(source, target)
        self.assertEqual(result.status, "unaligned")
        self.assertFalse(result.is_accepted)

    def test_unmatched_prefix_is_unaligned(self):
        source = "A" * 20 + "B" * 20
        target = "oops" + source
        result = align_deletion_only(source, target)
        self.assertEqual(result.status, "unaligned")

    def test_exact_text_shorter_than_paper_threshold_is_unaligned(self):
        result = align_deletion_only("A" * 19, "A" * 19)
        self.assertEqual(result.status, "unaligned")

    def test_empty_refined_text_is_aligned_and_keeps_nothing(self):
        result = align_deletion_only("导航和广告", "")
        self.assertEqual(result.status, "aligned")
        self.assertEqual(result.character_spans, ())
        self.assertEqual(result.adjusted_refined_text, "")


class LabelingTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_two_retained_spans_get_two_b_labels(self):
        first = "A" * 20
        second = "B" * 20
        source = first + "\nAD\n" + second
        target = first + "\n" + second
        alignment, result = label_aligned_pair(source, target, self.tokenizer)
        self.assertEqual(alignment.status, "aligned")
        assert result is not None
        self.assertEqual(result.bio_tags.count("B"), 2)
        self.assertEqual(result.bio_tags[0], "IGN")
        self.assertEqual(result.bio_tags[-1], "IGN")
        self.assertEqual(result.labels[0], -100)
        self.assertEqual(len(result.input_ids), len(result.attention_mask))
        self.assertEqual(len(result.input_ids), len(result.labels))

    def test_adjusted_gap_becomes_one_continuous_bio_span(self):
        source = "A" * 20 + "cat" + "B" * 20
        target = "A" * 20 + "dogs" + "B" * 20
        alignment, result = label_aligned_pair(source, target, self.tokenizer)
        self.assertEqual(alignment.status, "adjusted")
        assert result is not None
        self.assertEqual(result.bio_tags.count("B"), 1)
        self.assertTrue(all(tag == "I" for tag in result.bio_tags[2:-1]))

    def test_empty_refined_text_labels_real_tokens_o(self):
        alignment, result = label_aligned_pair("全部删除", "", self.tokenizer)
        self.assertEqual(alignment.status, "aligned")
        assert result is not None
        self.assertEqual(result.bio_tags, ("IGN", "O", "O", "O", "O", "IGN"))

    def test_newline_input_is_used_verbatim(self):
        body = "有效正文" * 5
        source = "导航\n" + body + "\n版权"
        alignment, result = label_aligned_pair(source, body, self.tokenizer)
        self.assertEqual(alignment.source_text, source)
        self.assertEqual(alignment.status, "aligned")
        assert result is not None
        self.assertEqual(result.bio_tags.count("B"), 1)

    def test_derives_conditional_transition_targets(self):
        left, right, mask = derive_transition_supervision([-100, 0, 1, 2, -100])
        self.assertEqual(left, [-100, 0, 1, 2])
        self.assertEqual(right, [0, 1, 2, -100])
        self.assertEqual(mask, [False, True, True, False])

    def test_reports_truncation(self):
        text = "a" * 20
        _, result = label_aligned_pair(
            text,
            text,
            self.tokenizer,
            max_length=7,
        )
        assert result is not None
        self.assertTrue(result.truncated)
        self.assertEqual(result.tokenized_char_end, 5)

    def test_accepts_list_based_offset_pairs(self):
        text = "abc" * 7
        alignment, result = label_aligned_pair(text, text, ListOffsetTokenizer())
        self.assertTrue(alignment.is_accepted)
        assert result is not None
        self.assertEqual(result.bio_tags[1], "B")


class CliTest(unittest.TestCase):
    def test_writes_accepted_sft_rejected_and_manifest(self):
        aligned_source = "A" * 20 + "NOISE" + "B" * 20
        aligned_target = "A" * 20 + "B" * 20
        adjusted_source = "C" * 20 + "cat" + "D" * 20
        adjusted_target = "C" * 20 + "dogs" + "D" * 20
        unaligned_source = "E" * 20 + "cat" + "F" * 20
        unaligned_target = "E" * 20 + "X" * 10 + "F" * 20

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "pairs.jsonl"
            accepted_path = root / "accepted.jsonl"
            rows = [
                {
                    "id": "a",
                    "source_text": aligned_source,
                    "refined_text": aligned_target,
                },
                {
                    "id": "b",
                    "source_text": adjusted_source,
                    "refined_text": adjusted_target,
                },
                {
                    "id": "c",
                    "source_text": unaligned_source,
                    "refined_text": unaligned_target,
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            with patch(
                "sequence_BIO.cli._load_tokenizer",
                return_value=CharacterTokenizer(),
            ):
                code = main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(accepted_path),
                        "--tokenizer",
                        "fake-tokenizer",
                    ]
                )
            self.assertEqual(code, 0)

            accepted = _read_jsonl(accepted_path)
            sft = _read_jsonl(root / "accepted.sft.jsonl")
            rejected = _read_jsonl(root / "accepted.rejected.jsonl")
            manifest = json.loads((root / "accepted.meta.json").read_text())

            self.assertEqual(
                [row.get("alignment", {}).get("status") for row in accepted],
                ["aligned", "adjusted"],
            )
            self.assertEqual(len(sft), 2)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0].get("rejection_reason"), "unaligned")
            self.assertEqual(manifest.get("counts", {}).get("accepted"), 2)
            self.assertEqual(manifest.get("counts", {}).get("unaligned"), 1)
            self.assertNotIn("transition_label_ids", json.dumps(accepted))


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
