#!/usr/bin/env python3

"""Extract provenance-preserving Alice/Veyra dialogue pairs from rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "veyra-dialogue/v1"
PRIVATE_ONLY = "alice-encrypted-continuity-only"
IGNORED_USER_PREFIXES = (
    "<environment_context>",
    "<permissions instructions>",
)


def message_text(payload: dict[str, Any]) -> str:
    parts = []
    for item in payload.get("content") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"input_text", "output_text"}:
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def rollout_messages(path: Path) -> tuple[str, str, int, list[dict[str, Any]]]:
    snapshot_size = path.stat().st_size
    hasher = hashlib.sha256()
    session_id = "unknown"
    messages: list[dict[str, Any]] = []
    remaining = snapshot_size
    with path.open("rb") as stream:
        line_number = 0
        while remaining:
            line = stream.readline(remaining)
            if not line:
                break
            remaining -= len(line)
            hasher.update(line)
            line_number += 1
            if not line.endswith(b"\n"):
                continue
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path.name}:{line_number}: malformed JSON") from exc
            if event.get("type") == "session_meta":
                payload = event.get("payload") or {}
                session_id = str(payload.get("session_id") or payload.get("id") or session_id)
                continue
            if event.get("type") != "response_item":
                continue
            payload = event.get("payload") or {}
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = message_text(payload)
            if text:
                messages.append(
                    {
                        "role": role,
                        "phase": payload.get("phase"),
                        "text": text,
                        "timestamp": event.get("timestamp"),
                    }
                )
    return session_id, hasher.hexdigest(), snapshot_size, messages


def extract_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        session_id, source_digest, source_size, messages = rollout_messages(path)
        active: dict[str, Any] | None = None
        for message in messages:
            if message["role"] == "user":
                if active and active["veyra"]["final"]:
                    records.append(active)
                text = message["text"]
                if text.startswith(IGNORED_USER_PREFIXES):
                    active = None
                    continue
                record_seed = f"{session_id}\0{message['timestamp']}\0{text}".encode()
                active = {
                    "schema": SCHEMA,
                    "id": hashlib.sha256(record_seed).hexdigest()[:24],
                    "privacy": PRIVATE_ONLY,
                    "quality": "candidate",
                    "source": {
                        "session_id": session_id,
                        "rollout_sha256": source_digest,
                        "rollout_prefix_bytes": source_size,
                        "user_timestamp": message["timestamp"],
                    },
                    "alice": text,
                    "veyra": {"commentary": [], "final": []},
                }
                continue
            if active is None:
                continue
            phase = message.get("phase")
            if phase == "commentary":
                active["veyra"]["commentary"].append(message["text"])
            elif phase == "final_answer":
                active["veyra"]["final"].append(message["text"])
        if active and active["veyra"]["final"]:
            records.append(active)
    return records


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        metavar="ID",
        help="include only this record id and mark the result as gold",
    )
    parser.add_argument("sources", type=Path, nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = extract_records(args.sources)
    if args.select:
        selected = set(args.select)
        known = {record["id"] for record in records}
        missing = selected - known
        if missing:
            raise SystemExit("unknown selected record ids: " + ", ".join(sorted(missing)))
        records = [record for record in records if record["id"] in selected]
        for record in records:
            record["quality"] = "gold"
    write_records(args.output, records)
    quality = "gold" if args.select else "candidate"
    print(f"wrote {len(records)} {quality} dialogue pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
