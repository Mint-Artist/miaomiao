from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from select_repro.modeling_bidirectional_qwen3 import Qwen3ForBidirectionalMaskedLM


def make_config(attn_implementation: str = "eager") -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
        use_cache=False,
        use_sliding_window=False,
        sliding_window=None,
    )
    config._attn_implementation = attn_implementation
    return config


def make_model(attn_implementation: str = "eager") -> Qwen3ForBidirectionalMaskedLM:
    torch.manual_seed(0)
    model = Qwen3ForBidirectionalMaskedLM(make_config(attn_implementation))
    model.eval()
    return model


class BidirectionalQwen3Tests(unittest.TestCase):
    def test_right_context_changes_left_hidden_states_and_logits(self):
        model = make_model("eager")
        input_ids = torch.tensor([[1, 5, 6, 7], [1, 5, 6, 8]])
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_delta = (
            outputs.hidden_states[-1][0, 0] - outputs.hidden_states[-1][1, 0]
        ).abs().max()
        logits_delta = (outputs.logits[0, 0] - outputs.logits[1, 0]).abs().max()
        self.assertGreater(hidden_delta, 1e-6)
        self.assertGreater(logits_delta, 1e-6)
        self.assertTrue(outputs.logits_are_full_sequence)

    def test_sdpa_uses_explicit_non_causal_attention_and_masks_padding(self):
        model = make_model("sdpa")
        input_ids_a = torch.tensor([[1, 2, 0, 0]])
        input_ids_b = torch.tensor([[1, 2, 9, 10]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])
        calls = []
        original_sdpa = F.scaled_dot_product_attention

        def recording_sdpa(*args, **kwargs):
            calls.append(kwargs["is_causal"])
            return original_sdpa(*args, **kwargs)

        with patch.object(F, "scaled_dot_product_attention", side_effect=recording_sdpa):
            outputs_a = model(
                input_ids=input_ids_a,
                attention_mask=attention_mask,
                return_full_logits=True,
            )
            outputs_b = model(
                input_ids=input_ids_b,
                attention_mask=attention_mask,
                return_full_logits=True,
            )

        self.assertTrue(calls)
        self.assertTrue(all(call is False for call in calls))
        torch.testing.assert_close(
            outputs_a.logits[:, :2], outputs_b.logits[:, :2], atol=1e-6, rtol=0.0
        )

    def test_masked_only_logits_and_loss_match_full_cross_entropy(self):
        model = make_model("eager")
        input_ids = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]])
        labels = torch.tensor([[-100, 3, -100, 4], [5, -100, 6, -100]])
        masked_output = model(input_ids=input_ids, labels=labels)
        full_output = model(input_ids=input_ids, labels=labels, return_full_logits=True)
        selected_logits = full_output.logits[labels != -100]
        selected_labels = labels[labels != -100]
        manual_loss = F.cross_entropy(
            selected_logits.float(), selected_labels, reduction="mean"
        )

        self.assertFalse(masked_output.logits_are_full_sequence)
        self.assertEqual(
            masked_output.masked_token_count, int((labels != -100).sum().item())
        )
        self.assertEqual(
            masked_output.logits.shape,
            (masked_output.masked_token_count, model.config.vocab_size),
        )
        torch.testing.assert_close(masked_output.logits, selected_logits)
        torch.testing.assert_close(masked_output.loss, full_output.loss)
        torch.testing.assert_close(masked_output.loss, manual_loss)

    def test_state_dict_keys_round_trip_and_resize_embeddings(self):
        config = make_config("eager")
        reference_model = Qwen3ForCausalLM(config)
        model = make_model("eager")
        self.assertEqual(
            list(model.state_dict().keys()), list(reference_model.state_dict().keys())
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        original_outputs = model(input_ids=input_ids)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded = Qwen3ForBidirectionalMaskedLM.from_pretrained(tmp_dir)
            reloaded.eval()
            reloaded_outputs = reloaded(input_ids=input_ids)
            torch.testing.assert_close(reloaded_outputs.logits, original_outputs.logits)

        model.resize_token_embeddings(model.config.vocab_size + 3)
        self.assertEqual(
            model.get_input_embeddings().weight.shape[0], config.vocab_size + 3
        )
        self.assertEqual(
            model.get_output_embeddings().weight.shape[0], config.vocab_size + 3
        )


if __name__ == "__main__":
    unittest.main()
