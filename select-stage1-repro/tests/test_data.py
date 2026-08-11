from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from select_repro.data import MLMCollator, PackedTokenDataset


class PackedTokenDatasetTests(unittest.TestCase):
    def test_reads_fixed_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(24, dtype=np.uint32).tofile(path)
            dataset = PackedTokenDataset(
                path, sequence_length=4, offset_tokens=4, num_tokens=16
            )
            self.assertEqual(len(dataset), 4)
            self.assertTrue(torch.equal(dataset[0], torch.tensor([4, 5, 6, 7])))
            self.assertTrue(
                torch.equal(dataset[-1], torch.tensor([16, 17, 18, 19]))
            )
            dataset.close()

    def test_rejects_partial_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(9, dtype=np.uint32).tofile(path)
            with self.assertRaisesRegex(ValueError, "multiple"):
                PackedTokenDataset(path, sequence_length=4)


class MLMCollatorTests(unittest.TestCase):
    def test_masks_only_non_special_tokens(self) -> None:
        collator = MLMCollator(
            mask_token_id=99,
            vocab_size=100,
            special_token_ids=[0, 2, 99],
            mlm_probability=0.9,
            seed=7,
        )
        original = torch.tensor([0, 10, 11, 2, 12, 99, 13, 14])
        batch = collator([original])
        self.assertEqual(batch["labels"][0, 0].item(), -100)
        self.assertEqual(batch["labels"][0, 3].item(), -100)
        self.assertEqual(batch["labels"][0, 5].item(), -100)
        target_positions = batch["labels"].ne(-100)
        self.assertTrue(target_positions.any())
        self.assertTrue(
            torch.equal(batch["labels"][target_positions], original[target_positions[0]])
        )

    def test_rng_round_trip(self) -> None:
        kwargs = dict(
            mask_token_id=99,
            vocab_size=100,
            special_token_ids=[0, 2, 99],
            mlm_probability=0.4,
            seed=11,
        )
        first = MLMCollator(**kwargs)
        example = torch.arange(3, 35)
        state = first.state_dict()
        expected = first([example])
        restored = MLMCollator(**kwargs)
        restored.load_state_dict(state)
        actual = restored([example])
        self.assertTrue(torch.equal(actual["input_ids"], expected["input_ids"]))
        self.assertTrue(torch.equal(actual["labels"], expected["labels"]))


if __name__ == "__main__":
    unittest.main()
