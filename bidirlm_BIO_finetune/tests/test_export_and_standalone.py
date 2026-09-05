import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from bidirlm_BIO_finetune.decoding import viterbi_decode_batch
from bidirlm_BIO_finetune.export_merged import (
    CUSTOM_CODE_FILES,
    copy_custom_code,
    heads_state_dict,
    write_heads,
    write_metadata,
)
from bidirlm_BIO_finetune.standalone_inference import (
    SelectHeads,
    labels_to_token_spans,
    refine_text,
    stitched_log_probs,
    token_spans_to_char_spans,
    viterbi,
    window_plan,
)


def token_forward(input_ids: torch.Tensor):
    """Position-independent fake model: logits depend only on the token id."""

    ids = input_ids[0]
    classification = torch.zeros((len(ids), 3))
    classification[:, 0] = 1.0
    classification[ids % 7 == 0, 0] = -1.0
    classification[ids % 7 == 0, 2] = 2.0
    transition = torch.zeros((len(ids), 3, 3))
    transition[:, 2, 2] = 0.5
    return classification, transition


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(char) for char in text],
            "attention_mask": [1] * len(text),
            "offset_mapping": [[index, index + 1] for index in range(len(text))],
        }


class FakeSelectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_head = nn.Linear(4, 3)
        self.transition_head = nn.Linear(4, 9)


class StandaloneInferenceTests(unittest.TestCase):
    def test_window_plan_covers_every_position_exactly_once(self):
        for length, window, stride in ((5, 8, 6), (20, 8, 6), (16, 8, 4), (21, 8, 3)):
            plan = window_plan(length, window, stride)
            covered = []
            for start, end, keep_start, keep_end in plan:
                self.assertLessEqual(start, keep_start)
                self.assertLessEqual(keep_end, end)
                self.assertEqual(end - start, min(window, length))
                covered.append((keep_start, keep_end))
            covered.sort()
            self.assertEqual(covered[0][0], 0)
            self.assertEqual(covered[-1][1], length)
            for (_, previous_end), (next_start, _) in zip(covered, covered[1:]):
                self.assertLessEqual(next_start, previous_end)

    def test_stitched_log_probs_match_single_pass(self):
        ids = list(range(1, 41))
        single_cls, single_tr = stitched_log_probs(token_forward, ids, window=64, stride=48)
        windowed_cls, windowed_tr = stitched_log_probs(token_forward, ids, window=12, stride=8)
        self.assertTrue(torch.allclose(single_cls, windowed_cls))
        self.assertTrue(torch.allclose(single_tr, windowed_tr))

    def test_viterbi_matches_package_decoder(self):
        torch.manual_seed(0)
        classification = torch.randn(1, 15, 3)
        transition = torch.randn(1, 15, 3, 3)
        expected = viterbi_decode_batch(
            classification, transition, torch.ones((1, 15), dtype=torch.bool)
        )[0].tolist()
        actual = viterbi(
            classification[0].log_softmax(-1),
            transition[0].log_softmax(-1),
            [True] * 15,
        )
        self.assertEqual(actual, expected)

    def test_viterbi_marks_invalid_positions(self):
        cls_lp = torch.zeros((4, 3))
        tr_lp = torch.zeros((4, 3, 3))
        labels = viterbi(cls_lp, tr_lp, [False, True, True, False])
        self.assertEqual(labels[0], -100)
        self.assertEqual(labels[3], -100)
        self.assertNotEqual(labels[1], -100)

    def test_span_helpers(self):
        self.assertEqual(labels_to_token_spans([0, 1, 2, 0, 2, 2, 1]), [(1, 3), (4, 6), (6, 7)])
        offsets = [[0, 0], [0, 2], [2, 5], [5, 5]]
        self.assertEqual(token_spans_to_char_spans(offsets, [(0, 3)]), [(0, 5)])
        self.assertEqual(token_spans_to_char_spans(offsets, [(3, 4)]), [])

    def test_refine_text_returns_verbatim_segments(self):
        text = "".join(chr(code) for code in range(60, 90))
        result = refine_text(text, FakeTokenizer(), token_forward, window=8, stride=6)
        retained = [chr(code) for code in range(60, 90) if code % 7 == 0]
        self.assertEqual(list(result["refined_text"]), retained)
        self.assertGreater(result["num_windows"], 1)
        for start, end in result["char_spans"]:
            self.assertEqual(text[start:end], text[start:end])


class ExportTests(unittest.TestCase):
    def test_heads_round_trip_through_safetensors_and_pt(self):
        model = FakeSelectModel()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_heads(output, heads_state_dict(model))
            loaded = SelectHeads.from_safetensors(output / "select_heads.safetensors")
            nested = torch.load(output / "select_heads.pt", weights_only=True)
        self.assertTrue(
            torch.equal(
                loaded.classification_head.weight, model.classification_head.weight
            )
        )
        self.assertTrue(
            torch.equal(nested["transition_head"]["bias"], model.transition_head.bias)
        )
        hidden = torch.randn(1, 5, 4)
        classification, transition = loaded(hidden)
        self.assertEqual(tuple(classification.shape), (1, 5, 3))
        self.assertEqual(tuple(transition.shape), (1, 5, 3, 3))

    def test_copy_custom_code_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            backbone = Path(directory) / "export" / "backbone"
            base.mkdir()
            backbone.mkdir(parents=True)
            for name in CUSTOM_CODE_FILES:
                (base / name).write_text("# code\n", encoding="utf-8")
            copied = copy_custom_code(base, backbone)
            self.assertEqual(sorted(copied), sorted(CUSTOM_CODE_FILES))
            self.assertEqual(copy_custom_code(base, backbone), [])
            self.assertEqual(copy_custom_code(None, backbone), [])
            write_metadata(
                backbone.parent,
                source_checkpoint="ckpt",
                source_mode="lora",
                base_model_name_or_path=str(base),
                dropout=0.1,
                dtype="float16",
            )
            metadata = json.loads(
                (backbone.parent / "select_config.json").read_text(encoding="utf-8")
            )
        self.assertEqual(metadata["finetuning_mode"], "full")
        self.assertEqual(metadata["exported_from_mode"], "lora")
        self.assertEqual(metadata["label2id"], {"O": 0, "B": 1, "I": 2})


if __name__ == "__main__":
    unittest.main()
