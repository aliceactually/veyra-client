import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_dialogue_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_dialogue_corpus", MODULE_PATH)
corpus = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = corpus
SPEC.loader.exec_module(corpus)


class DialogueCorpusTests(unittest.TestCase):
    def test_extracts_exact_dialogue_without_reasoning_or_environment(self):
        events = [
            self.message("user", "<environment_context>secret path</environment_context>"),
            self.message("user", "somei mes I type quickly"),
            {"type": "response_item", "payload": {"type": "reasoning"}},
            self.message("assistant", "brief update", phase="commentary"),
            self.message("assistant", "I understood the typo.", phase="final_answer"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rollout.jsonl"
            source.write_text("\n".join(json.dumps(event) for event in events) + "\n")
            records = corpus.extract_records([source])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["alice"], "somei mes I type quickly")
        self.assertEqual(records[0]["veyra"]["commentary"], ["brief update"])
        self.assertEqual(records[0]["veyra"]["final"], ["I understood the typo."])
        self.assertEqual(records[0]["privacy"], corpus.PRIVATE_ONLY)
        self.assertEqual(records[0]["quality"], "candidate")
        self.assertGreater(records[0]["source"]["rollout_prefix_bytes"], 0)

    @staticmethod
    def message(role, text, phase=None):
        content_type = "input_text" if role == "user" else "output_text"
        return {
            "timestamp": "2026-09-04T00:00:00Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": role,
                "phase": phase,
                "content": [{"type": content_type, "text": text}],
            },
        }


if __name__ == "__main__":
    unittest.main()
