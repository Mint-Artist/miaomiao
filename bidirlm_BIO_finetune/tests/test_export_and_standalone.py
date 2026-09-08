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


def pytest_importorskip_onnx():
    try:
        import onnx
    except ImportError:  # pragma: no cover - environment dependent
        raise unittest.SkipTest("onnx not installed")
    return onnx


def _numpy_bridge_works() -> bool:
    try:
        torch.zeros(1).numpy()
    except Exception:
        return False
    return True


class FakeBackbone(nn.Module):
    """Length-agnostic stand-in: hidden state depends only on the token id."""

    def __init__(self, hidden_size=4, vocab_size=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.config = type("cfg", (), {"vocab_size": vocab_size, "hidden_size": hidden_size})()
        self.seen_mask = None

    def forward(self, input_ids=None, attention_mask=None, return_dict=True):
        self.seen_mask = attention_mask
        return type("out", (), {"last_hidden_state": self.embedding(input_ids)})()


class OnnxWrapperTests(unittest.TestCase):
    def _module(self):
        from bidirlm_BIO_finetune.export_onnx import SelectOnnxModule

        torch.manual_seed(0)
        return SelectOnnxModule(FakeBackbone(), nn.Linear(4, 3), nn.Linear(4, 9))

    def test_four_d_mask_is_additive_and_key_only(self):
        module = self._module()
        ids = torch.randint(0, 50, (2, 6))
        mask = torch.ones((2, 6), dtype=torch.long)
        mask[1, 4:] = 0
        module(ids, mask)
        seen = module.backbone.seen_mask
        self.assertEqual(tuple(seen.shape), (2, 1, 1, 6))
        minimum = torch.finfo(torch.float32).min
        self.assertEqual(float(seen[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(seen[1, 0, 0, 3]), 0.0)
        self.assertEqual(float(seen[1, 0, 0, 5]), minimum)

    def test_two_d_mask_mode_passes_mask_through(self):
        from bidirlm_BIO_finetune.export_onnx import SelectOnnxModule

        module = SelectOnnxModule(
            FakeBackbone(), nn.Linear(4, 3), nn.Linear(4, 9), mask_mode="2d"
        )
        ids = torch.randint(0, 50, (1, 5))
        mask = torch.ones((1, 5), dtype=torch.long)
        module(ids, mask)
        self.assertTrue(torch.equal(module.backbone.seen_mask, mask))

    def test_mask_mode_is_validated(self):
        from bidirlm_BIO_finetune.export_onnx import SelectOnnxModule

        with self.assertRaises(ValueError):
            SelectOnnxModule(FakeBackbone(), nn.Linear(4, 3), nn.Linear(4, 9), mask_mode="3d")

    def test_output_shapes_follow_input_length(self):
        module = self._module()
        for length in (1, 7, 64):
            ids = torch.randint(0, 50, (1, length))
            classification, transition = module(ids, torch.ones_like(ids))
            self.assertEqual(tuple(classification.shape), (1, length, 3))
            self.assertEqual(tuple(transition.shape), (1, length, 3, 3))

    def test_transition_reshape_matches_head_layout(self):
        module = self._module()
        ids = torch.randint(0, 50, (2, 5))
        _, transition = module(ids, torch.ones_like(ids))
        hidden = module.backbone(input_ids=ids).last_hidden_state
        expected = module.transition_head(hidden).reshape(2, 5, 3, 3)
        self.assertTrue(torch.equal(transition, expected))

    @unittest.skipUnless(_numpy_bridge_works(), "torch<->numpy bridge unavailable")
    def test_onnx_forward_fn_matches_torch_forward_fn(self):
        module = self._module()

        class FakeSession:
            def run(self, names, feeds):
                ids = torch.from_numpy(feeds["input_ids"])
                mask = torch.from_numpy(feeds["attention_mask"])
                with torch.no_grad():
                    classification, transition = module(ids, mask)
                return [classification.numpy(), transition.numpy()]

        from bidirlm_BIO_finetune.standalone_inference import make_onnx_forward_fn

        forward = make_onnx_forward_fn(FakeSession())
        ids = torch.randint(0, 50, (1, 9))
        classification, transition = forward(ids)
        with torch.no_grad():
            expected_cls, expected_tr = module(ids, torch.ones_like(ids))
        self.assertTrue(torch.allclose(classification, expected_cls[0]))
        self.assertTrue(torch.allclose(transition, expected_tr[0]))

    def test_exported_files_lists_scattered_external_data(self):
        from bidirlm_BIO_finetune.export_onnx import exported_files

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            output.write_bytes(b"graph")
            # torch names per-tensor files after the tensors, not the model.
            (Path(directory) / "onnx__MatMul_8481").write_bytes(b"w" * 8)
            (Path(directory) / "backbone.embed_tokens.weight").write_bytes(b"w" * 4)
            listed = {item["file"] for item in exported_files(output)}
        self.assertEqual(
            listed,
            {"model.onnx", "onnx__MatMul_8481", "backbone.embed_tokens.weight"},
        )

    def test_consolidate_merges_scattered_external_data(self):
        onnx = pytest_importorskip_onnx()
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            weights = {
                # Above save_model's 1024-byte threshold, so they land in the
                # .data file rather than being inlined into the model.
                "backbone.embed_tokens.weight": np.arange(2048, dtype=np.float32).reshape(64, 32),
                "onnx__MatMul_8481": np.full((32, 32), 7.0, dtype=np.float32),
            }
            initializers = []
            for name, array in weights.items():
                # Mimic torch: one file per tensor, named after the tensor.
                (Path(directory) / name).write_bytes(array.tobytes())
                tensor = numpy_helper.from_array(array, name)
                tensor.ClearField("raw_data")
                tensor.data_location = TensorProto.EXTERNAL
                entry = tensor.external_data.add()
                entry.key, entry.value = "location", name
                initializers.append(tensor)
            graph = helper.make_graph(
                [helper.make_node("Identity", ["x"], ["y"])],
                "g",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
                initializer=initializers,
            )
            onnx.save(helper.make_model(graph), str(output))

            from bidirlm_BIO_finetune.export_onnx import consolidate_external_data

            result = consolidate_external_data(output)
            remaining = sorted(path.name for path in Path(directory).iterdir())
            reloaded = onnx.load(str(output))
            restored = {
                item.name: numpy_helper.to_array(item)
                for item in reloaded.graph.initializer
            }

        self.assertTrue(result["consolidated"])
        self.assertEqual(result["inlined_tensors"], 2)
        self.assertEqual(remaining, ["model.onnx", "model.onnx.data"])
        for name, array in weights.items():
            self.assertTrue(np.array_equal(restored[name], array))

    def test_consolidate_is_a_noop_without_external_data(self):
        from bidirlm_BIO_finetune.export_onnx import consolidate_external_data

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            output.write_bytes(b"graph")
            result = consolidate_external_data(output)
        self.assertFalse(result["consolidated"])


if __name__ == "__main__":
    unittest.main()
