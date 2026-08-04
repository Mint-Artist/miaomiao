import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "run_doc_filter.py"
SPEC = importlib.util.spec_from_file_location("run_doc_filter", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CoreTests(unittest.TestCase):
    def test_parse_strict_program(self):
        self.assertEqual(MODULE.parse_doc_program("keep"), "keep")
        self.assertEqual(MODULE.parse_doc_program("DROP!"), "drop")
        self.assertEqual(MODULE.parse_doc_program("do not drop"), "unknown")
        self.assertEqual(MODULE.parse_doc_program(""), "unknown")

    def test_parse_repo_compatible_program(self):
        self.assertEqual(
            MODULE.parse_doc_program("please drop", mode="repo-compatible"),
            "drop",
        )
        self.assertEqual(
            MODULE.parse_doc_program("unexpected", mode="repo-compatible"),
            "keep",
        )

    def test_nested_text_key(self):
        record = {"data": {"cleaned_text": "hello"}}
        self.assertEqual(
            MODULE.get_by_path(record, "data.cleaned_text"),
            "hello",
        )

    def test_normalize_text(self):
        self.assertEqual(
            MODULE.normalize_text("a\r\n\r\n\r\nb\x00"),
            "a\n\nb",
        )

    def test_prompt_has_expected_llama2_markers(self):
        prompt = MODULE.build_llama2_prompt("hello")
        self.assertTrue(prompt.startswith("<s>[INST] <<SYS>>"))
        self.assertTrue(prompt.endswith("hello [/INST]"))

    def test_unknown_program_preserves_text(self):
        item = MODULE.InputRecord(
            line_number=1,
            record={"id": "one", "content": "hello"},
            normalized_text="hello",
        )
        result = MODULE.GenerationResult(program="unexpected", truncated=False)
        output = MODULE.make_output_record(
            item=item,
            result=result,
            parse_mode="strict",
            output_prefix="prox_doc_",
        )
        self.assertEqual(output["prox_doc_decision"], "unknown")
        self.assertEqual(output["prox_doc_text"], "hello")


if __name__ == "__main__":
    unittest.main()
