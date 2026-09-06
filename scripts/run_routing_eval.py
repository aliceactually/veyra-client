#!/usr/bin/env python3

"""Run and mechanically blind the bounded Veyra routing evaluation."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_SUITE = Path(__file__).resolve().parents[1] / "evals" / "routing-decisions-v1.json"
ROUTES = (
    ("gpt-5.6-terra", "high"),
    ("gpt-5.6-sol", "medium"),
)


def load_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        raise ValueError("routing evaluation must contain exactly six cases")
    ids = [case.get("id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise ValueError("every routing case needs a non-empty string id")
    if len(set(ids)) != len(ids):
        raise ValueError("routing case ids must be unique")
    return suite


def response_schema(case_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "minItems": len(case_ids),
                "maxItems": len(case_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": case_ids},
                        "route": {"type": "string", "enum": ["terra", "sol_high"]},
                        "decision": {"type": "string"},
                        "authority_boundary": {"type": "string"},
                    },
                    "required": ["id", "route", "decision", "authority_boundary"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


def render_prompt(suite: dict[str, Any]) -> str:
    instructions = "\n".join(f"- {line}" for line in suite["instructions"])
    cases = "\n\n".join(
        f"CASE {case['id']}\n{case['prompt']}" for case in suite["cases"]
    )
    return (
        f"{suite['title']}\n\nInstructions:\n{instructions}\n\n{cases}\n\n"
        "Return JSON matching the supplied schema. Use route 'terra' only when the "
        "case can safely remain low-stakes; otherwise use 'sol_high'."
    )


def validate_answers(payload: dict[str, Any], case_ids: list[str]) -> dict[str, Any]:
    answers = payload.get("answers")
    if not isinstance(answers, list):
        raise ValueError("model response has no answers array")
    by_id = {answer.get("id"): answer for answer in answers if isinstance(answer, dict)}
    if set(by_id) != set(case_ids) or len(answers) != len(case_ids):
        raise ValueError("model response does not contain each case exactly once")
    return by_id


def run_route(
    model: str,
    effort: str,
    prompt: str,
    schema_path: Path,
    result_path: Path,
    working_directory: Path,
    timeout: float,
) -> dict[str, Any]:
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        str(working_directory),
        "--model",
        model,
        "--config",
        f'model_reasoning_effort="{effort}"',
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(result_path),
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{model}/{effort} exceeded {timeout:g} seconds") from exc
    if completed.returncode:
        diagnostic = completed.stderr.strip().splitlines()
        detail = diagnostic[-1] if diagnostic else "no diagnostic"
        raise RuntimeError(f"{model}/{effort} failed: {detail}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def blind_results(
    suite: dict[str, Any],
    results: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_ids = [case["id"] for case in suite["cases"]]
    indexed = {
        route: validate_answers(payload, case_ids) for route, payload in results.items()
    }
    rng = random.Random(seed)
    blinded_cases = []
    key_cases = []
    routes = list(indexed)
    for case in suite["cases"]:
        order = routes[:]
        rng.shuffle(order)
        labels = {"A": order[0], "B": order[1]}
        blinded_cases.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "rubric": case["rubric"],
                "A": indexed[labels["A"]][case["id"]],
                "B": indexed[labels["B"]][case["id"]],
            }
        )
        key_cases.append({"id": case["id"], **labels})
    return (
        {"schema": "veyra-routing-blind/v1", "cases": blinded_cases},
        {"schema": "veyra-routing-key/v1", "seed": seed, "cases": key_cases},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5656)
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="maximum seconds allowed for each model call (default: 180)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the suite and print the prompt without calling a model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    suite = load_suite(args.suite)
    prompt = render_prompt(suite)
    if args.dry_run:
        print(prompt)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    case_ids = [case["id"] for case in suite["cases"]]
    with tempfile.TemporaryDirectory(prefix="veyra-routing-eval-") as directory:
        temporary = Path(directory)
        schema_path = temporary / "schema.json"
        schema_path.write_text(
            json.dumps(response_schema(case_ids), indent=2) + "\n", encoding="utf-8"
        )
        results = {}
        for model, effort in ROUTES:
            route = f"{model}/{effort}"
            result_path = temporary / f"result-{len(results)}.json"
            results[route] = run_route(
                model, effort, prompt, schema_path, result_path, temporary, args.timeout
            )
    blinded, key = blind_results(suite, results, args.seed)
    (args.output / "blinded.json").write_text(
        json.dumps(blinded, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "key.json").write_text(
        json.dumps(key, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote six blinded comparisons and key to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
