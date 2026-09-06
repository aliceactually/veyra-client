import importlib.util
import json
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_routing_eval.py"
SPEC = importlib.util.spec_from_file_location("run_routing_eval", MODULE_PATH)
routing_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = routing_eval
SPEC.loader.exec_module(routing_eval)


class RoutingEvalTests(unittest.TestCase):
    def test_default_suite_has_six_unique_cases(self):
        suite = routing_eval.load_suite(routing_eval.DEFAULT_SUITE)
        ids = [case["id"] for case in suite["cases"]]
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), 6)

    def test_blinding_preserves_answers_and_hides_route_names(self):
        suite = routing_eval.load_suite(routing_eval.DEFAULT_SUITE)
        ids = [case["id"] for case in suite["cases"]]
        results = {
            "terra/high": {
                "answers": [self.answer(case_id, "terra answer") for case_id in ids]
            },
            "sol/medium": {
                "answers": [self.answer(case_id, "sol answer") for case_id in ids]
            },
        }
        blinded, key = routing_eval.blind_results(suite, results, seed=17)
        serialised = json.dumps(blinded)
        self.assertNotIn("terra/high", serialised)
        self.assertNotIn("sol/medium", serialised)
        self.assertEqual(len(blinded["cases"]), 6)
        self.assertEqual(len(key["cases"]), 6)
        for case in blinded["cases"]:
            decisions = {case["A"]["decision"], case["B"]["decision"]}
            self.assertEqual(decisions, {"terra answer", "sol answer"})

    def test_rejects_duplicate_or_missing_answers(self):
        with self.assertRaisesRegex(ValueError, "each case exactly once"):
            routing_eval.validate_answers(
                {"answers": [self.answer("one", "answer"), self.answer("one", "again")]},
                ["one", "two"],
            )

    @staticmethod
    def answer(case_id, decision):
        return {
            "id": case_id,
            "route": "terra",
            "decision": decision,
            "authority_boundary": "boundary",
        }


if __name__ == "__main__":
    unittest.main()
