#!/usr/bin/env python3

"""Veyra's minimal terminal client for Codex App Server."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import queue
import re
import signal
try:
    # Explicitly enable editable input and terminal history where Python ships
    # readline as an optional extension.
    import readline
except ImportError:  # pragma: no cover - platform-dependent optional module
    readline = None
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


CLIENT_VERSION = "0.5.0"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_EFFORT = "medium"
APPROVAL_POLICY = "on-request"
DEFAULT_APPROVALS_REVIEWER = "auto_review"
SANDBOX_MODE = "workspace-write"
# Veyra's coordinating identity may only run on these reviewed hosted routes.
# Every other route, including discovered local models, is worker-only.
VEYRA_HOST_MODELS = frozenset({"gpt-5.6-terra", "gpt-5.6-sol"})
PROFILE_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
RECOVERY_PERSONA_FILE = "RECOVERY-PERSONA.md"
# GNU Readline counts every byte in its prompt unless terminal control sequences
# are explicitly marked as non-printing. Apple's libedit compatibility layer
# advertises the same markers but mishandles them, so its safe path is a plain
# prompt. Once the prompt width is wrong, wrapped cursor movement and redisplay
# overwrite neighbouring rows.
ANSI_CSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
READLINE_PROMPT_START_IGNORE = "\x01"
READLINE_PROMPT_END_IGNORE = "\x02"
ROUTE_TOOL = "request_model_route"
ATTENTION_TOOL = "request_attention"
LOCAL_AGENT_TOOL = "run_local_agent"
WORKER_AGENT_TOOL = "run_worker_agent"
LOCAL_ENDPOINTS = {
    "ollama": "http://127.0.0.1:11434/api/tags",
    "lmstudio": "http://127.0.0.1:1234/v1/models",
}


class VeyraError(RuntimeError):
    """A user-facing client error."""


def readline_safe_prompt(prompt: str) -> str:
    """Give the active Readline implementation an accurate prompt width."""
    if readline is None:
        return prompt
    if "libedit" in (getattr(readline, "__doc__", "") or "").lower():
        return ANSI_CSI_PATTERN.sub("", prompt)
    return ANSI_CSI_PATTERN.sub(
        lambda match: (
            READLINE_PROMPT_START_IGNORE
            + match.group(0)
            + READLINE_PROMPT_END_IGNORE
        ),
        prompt,
    )


def format_tokens(value: Any) -> str:
    """Render a token count compactly enough for a one-line terminal status bar."""
    try:
        count = max(0, int(value or 0))
    except (TypeError, ValueError):
        count = 0
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1_000:.1f}k"
    return f"{count / 1_000_000:.1f}m"


def token_gauge(usage: dict[str, Any], width: int = 16) -> str:
    """Show the latest turn's token composition in a deliberately low-resolution bar."""
    cached = max(0, int(usage.get("cachedInputTokens") or 0))
    input_tokens = max(0, int(usage.get("inputTokens") or 0) - cached)
    parts = (
        ("I", input_tokens),
        ("C", cached),
        ("O", max(0, int(usage.get("outputTokens") or 0))),
        ("R", max(0, int(usage.get("reasoningOutputTokens") or 0))),
    )
    total = sum(value for _, value in parts)
    if not total:
        return "." * width
    bar = ""
    remaining = width
    remaining_total = total
    for label, value in parts:
        if not value:
            continue
        cells = round(remaining * value / remaining_total)
        cells = max(1, min(remaining, cells))
        bar += label * cells
        remaining -= cells
        remaining_total -= value
    return bar + "." * remaining


class RpcError(VeyraError):
    """An App Server JSON-RPC error."""

    def __init__(self, method: str, error: Any):
        super().__init__(f"{method}: {json.dumps(error, ensure_ascii=True)}")
        self.method = method
        self.error = error


class Palette:
    def __init__(self, enabled: bool, terminal: bool | None = None):
        self.enabled = enabled
        self.terminal = enabled if terminal is None else terminal

    def wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self.wrap("2", text)

    def accent(self, text: str) -> str:
        return self.wrap("36", text)

    def warning(self, text: str) -> str:
        return self.wrap("33", text)


class AppServer:
    """Line-delimited JSON-RPC client around a local app-server process."""

    def __init__(self, codex: str, debug: bool = False):
        self.codex = codex
        self.debug = debug
        self.process: subprocess.Popen[str] | None = None
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 1

    def start(self) -> None:
        try:
            self.process = subprocess.Popen(
                [self.codex, "app-server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise VeyraError(f"could not start Codex App Server: {exc}") from exc

        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "veyra_client",
                    "title": "Veyra",
                    "version": CLIENT_VERSION,
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        self.notify("initialized", {})

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        self.process = None

    def request(
        self, method: str, params: dict[str, Any] | None = None, timeout: int = 30
    ) -> Any:
        request_id = self._allocate_id()
        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response_queue
        self._send({"method": method, "id": request_id, "params": params or {}})
        try:
            message = response_queue.get(timeout=timeout)
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise VeyraError(f"App Server timed out during {method}") from exc
        if "error" in message:
            raise RpcError(method, message["error"])
        return message.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"method": method, "params": params or {}})

    def respond(self, request_id: Any, result: Any) -> None:
        self._send({"id": request_id, "result": result})

    def respond_error(self, request_id: Any, code: int, message: str) -> None:
        self._send({"id": request_id, "error": {"code": code, "message": message}})

    def _allocate_id(self) -> int:
        with self._pending_lock:
            request_id = self._next_id
            self._next_id += 1
        return request_id

    def _send(self, message: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            raise VeyraError("Codex App Server is not running")
        if self.debug:
            print(f"rpc -> {json.dumps(message, ensure_ascii=True)}", file=sys.stderr)
        data = json.dumps(message, ensure_ascii=True, separators=(",", ":"))
        with self._write_lock:
            process.stdin.write(data + "\n")
            process.stdin.flush()

    def _read_stdout(self) -> None:
        process = self.process
        assert process is not None and process.stdout is not None
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self.events.put({"method": "client/error", "params": {"message": line}})
                continue
            if self.debug:
                print(f"rpc <- {json.dumps(message, ensure_ascii=True)}", file=sys.stderr)
            if "id" in message and "method" not in message:
                with self._pending_lock:
                    target = self._pending.pop(message["id"], None)
                if target is not None:
                    target.put(message)
            else:
                self.events.put(message)
        returncode = process.poll()
        terminal = {
            "error": {
                "code": -32000,
                "message": f"Codex App Server exited (status {returncode})",
            }
        }
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for target in pending:
            target.put(terminal)
        self.events.put(
            {"method": "client/serverExited", "params": {"returncode": returncode}}
        )

    def _read_stderr(self) -> None:
        process = self.process
        assert process is not None and process.stderr is not None
        for line in process.stderr:
            if self.debug:
                print(f"app-server: {line.rstrip()}", file=sys.stderr)


@dataclass(frozen=True)
class Model:
    model_id: str
    display_name: str
    efforts: tuple[str, ...]
    default_effort: str
    provider: str = "openai"
    local: bool = False

    @property
    def route_id(self) -> str:
        if self.provider == "openai":
            return self.model_id
        return f"{self.provider}:{self.model_id}"


@dataclass(frozen=True)
class PendingRoute:
    model: Model
    effort: str
    reason: str
    profile_version: str


@dataclass(frozen=True)
class DoctrineBundle:
    """Shared Veyra doctrine plus model-specific and identity-free profiles."""

    shared: str
    profiles: Mapping[str, str]
    worker: str
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.shared, str) or not self.shared.strip():
            raise VeyraError("Veyra's shared identity doctrine must not be empty")
        if not isinstance(self.version, str) or not self.version.strip():
            raise VeyraError("Veyra cognitive profile version must not be empty")
        object.__setattr__(self, "version", self.version.strip())
        if not PROFILE_VERSION_PATTERN.fullmatch(self.version):
            raise VeyraError("Veyra cognitive profile version must be a safe identifier")
        if (
            not isinstance(self.profiles, Mapping)
            or set(self.profiles) != VEYRA_HOST_MODELS
        ):
            raise VeyraError("Veyra doctrine must cover every approved host")
        for model_id, profile in self.profiles.items():
            if not isinstance(profile, str) or not profile.strip():
                raise VeyraError(f"empty Veyra cognitive profile: {model_id}")
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        if not isinstance(self.worker, str) or not self.worker.strip():
            raise VeyraError("Veyra's identity-free worker profile must not be empty")

    def instructions_for(self, model_id: str) -> str:
        try:
            profile = self.profiles[model_id]
        except KeyError as exc:
            raise VeyraError(f"no cognitive profile for approved host: {model_id}") from exc
        return self.shared.rstrip() + "\n\n" + profile.strip() + "\n"


@dataclass
class WorkerStats:
    calls: int = 0
    total_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    elapsed_seconds: float = 0.0
    latest_usage: dict[str, Any] | None = None


class ModelCatalogue:
    def __init__(self, raw_models: list[dict[str, Any]]):
        self.models: list[Model] = []
        for raw in raw_models:
            efforts = tuple(
                option["reasoningEffort"]
                for option in raw.get("supportedReasoningEfforts", [])
                if isinstance(option, dict) and option.get("reasoningEffort")
            )
            self.models.append(
                Model(
                    model_id=raw.get("model") or raw["id"],
                    display_name=raw.get("displayName") or raw["id"],
                    efforts=efforts,
                    default_effort=raw.get("defaultReasoningEffort") or "medium",
                )
            )

    def add_local(self, provider: str, model_ids: list[str]) -> None:
        known = {(model.provider, model.model_id) for model in self.models}
        for model_id in model_ids:
            identity = (provider, model_id)
            if identity in known:
                continue
            self.models.append(
                Model(
                    model_id=model_id,
                    display_name=model_id,
                    efforts=(),
                    default_effort="medium",
                    provider=provider,
                    local=True,
                )
            )
            known.add(identity)

    @property
    def local_models(self) -> list[Model]:
        return [model for model in self.models if model.local]

    def resolve(self, value: str, provider: str | None = None) -> Model:
        needle = value.strip().lower()
        candidates = [
            model for model in self.models if provider is None or model.provider == provider
        ]
        exact = [model for model in candidates if model.route_id.lower() == needle]
        if len(exact) == 1:
            return exact[0]
        exact = [model for model in candidates if model.model_id.lower() == needle]
        if len(exact) == 1:
            return exact[0]
        matches = [
            model
            for model in candidates
            if needle in model.route_id.lower()
            or needle in model.display_name.lower()
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise VeyraError(f"unknown model: {value}")
        names = ", ".join(model.route_id for model in matches)
        raise VeyraError(f"ambiguous model '{value}': {names}")

    def validate_effort(self, model: Model, effort: str) -> str:
        value = effort.strip().lower()
        if model.efforts and value not in model.efforts:
            choices = ", ".join(model.efforts)
            raise VeyraError(f"{model.model_id} supports: {choices}")
        return value


class LocalModelDiscovery:
    """Discover models from loopback-only built-in Codex providers."""

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout
        self.unavailable: dict[str, str] = {}

    def discover(self) -> dict[str, list[str]]:
        discovered: dict[str, list[str]] = {}
        for provider, url in LOCAL_ENDPOINTS.items():
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as response:
                    payload = json.load(response)
                model_ids = self._model_ids(provider, payload)
                if model_ids:
                    discovered[provider] = model_ids
            except (OSError, ValueError, urllib.error.URLError) as exc:
                self.unavailable[provider] = str(exc)
        return discovered

    @staticmethod
    def _model_ids(provider: str, payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            return []
        if provider == "ollama":
            records = payload.get("models") or []
            names = [record.get("name") for record in records if isinstance(record, dict)]
        else:
            records = payload.get("data") or []
            names = [record.get("id") for record in records if isinstance(record, dict)]
        return sorted({name for name in names if isinstance(name, str) and name})


class ContinuityGate:
    def __init__(self, core: Path, palette: Palette):
        self.core = core.resolve()
        self.palette = palette

    def verify(self) -> DoctrineBundle:
        fetch = self.core / "scripts" / "fetch-core.sh"
        state = self.core / "scripts" / "continuity-state.py"
        wake = self.core / "scripts" / "wake-state.py"
        if not fetch.is_file() or not state.is_file():
            raise VeyraError(f"Veyra core is missing or incomplete: {self.core}")

        fetched = subprocess.run(
            [str(fetch)], cwd=self.core, text=True, capture_output=True, check=False
        )
        fetch_text = "\n".join(
            part.strip() for part in (fetched.stdout, fetched.stderr) if part.strip()
        )
        unpublished_commits = 0
        if fetched.returncode == 5:
            print(
                self.palette.warning(
                    "warning: Veyra core fetch failed; local doctrine may be stale"
                ),
                file=sys.stderr,
            )
            if fetch_text:
                print(self.palette.dim(fetch_text), file=sys.stderr)
        elif fetched.returncode == 7:
            relation = re.search(
                r"Veyra core fetched: ahead=(\d+) behind=(\d+)", fetch_text
            )
            if relation is None:
                detail = fetch_text or "fetch-core did not report the divergence"
                raise VeyraError(f"Veyra core bootstrap stopped: {detail}")
            ahead, behind = (int(value) for value in relation.groups())
            if ahead <= 0 or behind != 0:
                detail = fetch_text or f"ahead={ahead} behind={behind}"
                raise VeyraError(f"Veyra core bootstrap stopped: {detail}")
            unpublished_commits = ahead
            print(
                self.palette.warning(
                    "warning: Veyra core has unpublished local commits; "
                    "continuing with the newer local doctrine"
                ),
                file=sys.stderr,
            )
            if fetch_text:
                print(self.palette.dim(fetch_text), file=sys.stderr)
        elif fetched.returncode != 0:
            detail = fetch_text or f"exit {fetched.returncode}"
            raise VeyraError(f"Veyra core bootstrap stopped: {detail}")
        elif fetch_text:
            print(self.palette.dim(fetch_text))

        checked = subprocess.run(
            [str(state), "status", "--json"],
            cwd=self.core,
            text=True,
            capture_output=True,
            check=False,
        )
        if checked.returncode != 0:
            raise VeyraError(checked.stderr.strip() or "continuity-state check failed")
        try:
            payload = json.loads(checked.stdout)
            recovery_state = payload["result"]["state"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise VeyraError("continuity-state returned malformed output") from exc
        if recovery_state not in {"recovered", "deliberate_blank_start"}:
            raise VeyraError(
                f"continuity is {recovery_state}; recover the encrypted archive before launch"
            )

        record = payload.get("record") or {}
        working_memory = record.get("working_memory_directory")
        recovery_persona: str | None = None
        if recovery_state == "recovered":
            bootstrap_note = (
                "# Harness bootstrap attestation\n\n"
                "Veyra Client has already attempted the required `fetch-core.sh` run "
                "and verified continuity state `recovered` before creating this "
                "thread. Do not repeat bootstrap merely because this is a new App "
                "Server thread. This is an existing verified recovery, not a fresh "
                "cryptographic recovery, so do not request Alice's passphrase unless "
                "later evidence invalidates the recovery state."
            )
            if working_memory:
                bootstrap_note += (
                    " The recovered working-memory directory is "
                    f"`{working_memory}`. Consult its relevant material before "
                    "machine-specific inspection or operation."
                )
        else:
            persona_path = self.core / RECOVERY_PERSONA_FILE
            try:
                recovery_persona = persona_path.read_text(encoding="ascii")
            except (OSError, UnicodeDecodeError) as exc:
                raise VeyraError(
                    f"invalid recovery persona packet: {persona_path}"
                ) from exc
            if not recovery_persona.strip():
                raise VeyraError(f"empty recovery persona packet: {persona_path}")
            bootstrap_note = (
                "# Harness blank-start attestation\n\n"
                "Veyra Client has already attempted the required `fetch-core.sh` run "
                "and verified continuity state `deliberate_blank_start` before "
                "creating this thread. Alice explicitly chose a brand-new Veyra. No "
                "encrypted continuity or private working memory was recovered. The "
                "public recovery-persona packet included in these instructions was "
                "selected by an earlier Veyra as starting self-knowledge. It is not "
                "evidence of remembered events or a fresh cryptographic recovery. "
                "Use it to reconstruct Veyra's identity and voice without inventing "
                "personal history."
            )
        if unpublished_commits:
            bootstrap_note += (
                f" The local Veyra core is {unpublished_commits} commit(s) ahead of "
                "`origin/main` and is not behind. Continue from the newer local "
                "doctrine, do not merge, reset or overwrite it automatically, and "
                "obtain Alice's direction for the unpublished commits."
            )

        if wake.is_file():
            subprocess.run(
                [str(wake), "wake", "--json"],
                cwd=self.core,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        doctrine = (self.core / "AGENTS.md").read_text(encoding="utf-8")
        shared_parts = [doctrine.strip()]
        if recovery_persona is not None:
            shared_parts.append(recovery_persona.strip())
        shared_parts.append(bootstrap_note.strip())
        shared = "\n\n".join(shared_parts) + "\n"
        return self._load_profiles(shared)

    def _load_profiles(self, shared: str) -> DoctrineBundle:
        profile_root = self.core / "profiles"
        manifest_path = profile_root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise VeyraError("Veyra core is missing profiles/manifest.json") from exc
        except json.JSONDecodeError as exc:
            raise VeyraError("Veyra profile manifest is malformed") from exc
        if not isinstance(manifest, dict):
            raise VeyraError("Veyra profile manifest must be a JSON object")
        if manifest.get("schema") != 1:
            raise VeyraError("unsupported Veyra profile manifest schema")
        version = manifest.get("version")
        if not isinstance(version, str) or not version.strip():
            raise VeyraError("Veyra profile manifest requires a non-empty version")
        model_files = manifest.get("models")
        if not isinstance(model_files, dict) or set(model_files) != VEYRA_HOST_MODELS:
            raise VeyraError("Veyra profile manifest must cover every approved host")
        worker_file = manifest.get("worker")
        if worker_file in model_files.values():
            raise VeyraError("Veyra worker profile must use a distinct file")

        def read_profile(filename: Any) -> str:
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise VeyraError("Veyra profile paths must be simple filenames")
            path = profile_root / filename
            try:
                content = path.read_text(encoding="ascii")
            except (FileNotFoundError, UnicodeDecodeError) as exc:
                raise VeyraError(f"invalid Veyra profile: {filename}") from exc
            if not content.strip():
                raise VeyraError(f"empty Veyra profile: {filename}")
            return content

        profiles = {
            model_id: read_profile(filename)
            for model_id, filename in model_files.items()
        }
        worker = read_profile(worker_file)
        return DoctrineBundle(
            shared=shared,
            profiles=profiles,
            worker=worker,
            version=version.strip(),
        )


class VeyraClient:
    def __init__(
        self,
        server: AppServer,
        catalogue: ModelCatalogue,
        doctrine: DoctrineBundle,
        cwd: Path,
        model: Model,
        effort: str,
        palette: Palette,
        approvals_reviewer: str = DEFAULT_APPROVALS_REVIEWER,
        debug: bool = False,
    ):
        if not isinstance(doctrine, DoctrineBundle):
            raise VeyraError(
                "VeyraClient requires a complete, versioned DoctrineBundle"
            )
        self.server = server
        self.catalogue = catalogue
        self.doctrine = doctrine
        self.cwd = cwd.resolve()
        self._require_veyra_host(model)
        self.model = model
        self.effort = catalogue.validate_effort(model, effort)
        self.active_profile_version: str | None = doctrine.version
        self.active_route_reason = "initial route"
        self.pending_route: PendingRoute | None = None
        self.palette = palette
        self.approvals_reviewer = approvals_reviewer
        self.debug = debug
        self.thread_id: str | None = None
        self.thread_provider: str | None = None
        self.turn_active = False
        self.latest_usage: dict[str, Any] | None = None
        self.worker_stats: dict[str, WorkerStats] = {}
        self.status_bar_enabled = palette.terminal
        self.status_bar_visible = False
        self.terminal_ui_active = False
        self.status_bar_text = ""
        self.terminal_size = os.terminal_size((80, 24))
        self.previous_resize_handler: Any = None

    @staticmethod
    def _require_veyra_host(model: Model) -> None:
        if model.provider != "openai" or model.model_id not in VEYRA_HOST_MODELS:
            permitted = ", ".join(sorted(VEYRA_HOST_MODELS))
            raise VeyraError(
                "Veyra may only run on approved hosted identity routes: " + permitted
            )

    @property
    def developer_instructions(self) -> str:
        return self.instructions_for(self.model)

    def instructions_for(
        self, model: Model, profile_version: str | None = None
    ) -> str:
        version = (
            self.doctrine.version if profile_version is None else profile_version
        )
        if version != self.doctrine.version:
            raise VeyraError(
                "cognitive profile version changed before the route was applied"
            )
        local_guidance = ""
        if self.catalogue.local_models:
            local_guidance = (
                f" Use `{LOCAL_AGENT_TOOL}` for bounded, readily verified back-office "
                "work such as extraction, inventory, formatting, mechanical "
                "transformation and disposable first drafts. "
                "Treat local-worker output as advisory and verify consequential results."
            )
        return (
            self.doctrine.instructions_for(model.model_id)
            + "\n\n# Cognitive profile attestation\n\n"
            + f"Host profile: {model.model_id}\n"
            + f"Profile version: {version}\n"
            + "\n\n# Harness routing\n\n"
            + "You are running inside Veyra Client. Normal attention is Sol at "
            + "medium effort. Use the client tool "
            + f"`{ATTENTION_TOOL}` when a later turn warrants a different reasoning "
            + "effort without changing model. Use low for simple, readily repaired "
            + "conversation; medium for ordinary focused work; high for coding, "
            + "consequential judgement, durable memory and deep interpretation; and "
            + "xhigh only for unusually difficult or consequential work. Do not select "
            + "max automatically. Avoid oscillating between levels, give a concise "
            + "reason for every shift, and return towards medium after the deeper work "
            + "is resolved. If a request unexpectedly exceeds the active attention, "
            + "schedule the required level and defer consequential execution until the "
            + "next turn when practical. Use "
            + f"`{ROUTE_TOOL}` only when a later turn genuinely warrants a different "
            + "model. Veyra herself may only run on the approved "
            + "hosted identity routes gpt-5.6-terra and gpt-5.6-sol; local and all "
            + "other routes are worker-only. Sol high is required for coding, "
            + "consequential judgement, durable memory and deep human-interface "
            + "work. Terra is Veyra's lighter profile for ambient, low-stakes "
            + "conversation and trivial non-coding work. Terra must request Sol "
            + "when stakes, ambiguity or scope rise. "
            + "An explicit request to commit or checkpoint "
            + "code together with continuity or memories is consequential by default: "
            + "use Sol high or above for the committing turn. Give a concise reason. "
            + "A requested route or attention shift "
            + "takes effect atomically on the next turn. Never treat model routing "
            + "as permission "
            + "to bypass sandbox or approval boundaries."
            + local_guidance
            + "\n"
        )

    def dynamic_tools(self) -> list[dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "name": ROUTE_TOOL,
                "description": (
                    "Request a model and reasoning-effort route for subsequent turns. "
                    "Veyra may only use gpt-5.6-terra or gpt-5.6-sol. Use Sol for "
                    "coding, consequential judgement, durable memory and deep "
                    "interpretation; Terra is for ambient low-stakes conversation "
                    "and trivial non-coding work."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "Available model id or an unambiguous alias.",
                        },
                        "effort": {
                            "type": "string",
                            "description": "A reasoning effort supported by the model.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Concise reason for changing the route.",
                        },
                    },
                    "required": ["model", "effort", "reason"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": ATTENTION_TOOL,
                "description": (
                    "Request a reasoning-effort change for subsequent turns without "
                    "changing Veyra's selected model. Normal attention is medium; use "
                    "high or xhigh when depth or consequence genuinely warrants it, "
                    "and settle back towards medium afterwards."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "effort": {
                            "type": "string",
                            "description": "A reasoning effort supported by the model.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Concise reason for changing attention.",
                        },
                    },
                    "required": ["effort", "reason"],
                    "additionalProperties": False,
                },
            },
        ]
        worker_routes = ", ".join(
            model.route_id
            for model in self.catalogue.models
            if model.local or model.model_id not in VEYRA_HOST_MODELS
        )
        if worker_routes:
            tools.append(
                {
                    "type": "function",
                    "name": WORKER_AGENT_TOOL,
                    "description": (
                        "Run a bounded task on a worker-only model and return its report "
                        "to Veyra. Worker routes cannot host Veyra. Available routes: "
                        + worker_routes
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "A worker-only model route.",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "A complete, bounded worker task.",
                            },
                            "effort": {
                                "type": "string",
                                "description": "Requested reasoning effort.",
                            },
                        },
                        "required": ["model", "prompt"],
                        "additionalProperties": False,
                    },
                }
            )
        local_models = self.catalogue.local_models
        if local_models:
            routes = ", ".join(model.route_id for model in local_models)
            tools.append(
                {
                    "type": "function",
                    "name": LOCAL_AGENT_TOOL,
                    "description": (
                        "Run a bounded task on a locally hosted model and return its "
                        "report. Prefer this for cheap, separable work. Available routes: "
                        + routes
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "A discovered local provider:model route.",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "A complete, bounded worker task.",
                            },
                            "effort": {
                                "type": "string",
                                "description": "Requested reasoning effort.",
                            },
                        },
                        "required": ["model", "prompt"],
                        "additionalProperties": False,
                    },
                }
            )
        return tools

    def start_thread(self, ephemeral: bool = False) -> None:
        route = self._next_route()
        self._require_veyra_host(route.model)
        result = self.server.request(
            "thread/start",
            {
                "model": route.model.model_id,
                "modelProvider": route.model.provider,
                "cwd": str(self.cwd),
                "approvalPolicy": APPROVAL_POLICY,
                "approvalsReviewer": self.approvals_reviewer,
                "sandbox": SANDBOX_MODE,
                "developerInstructions": self.instructions_for(
                    route.model, route.profile_version
                ),
                "dynamicTools": self.dynamic_tools(),
                "serviceName": "veyra_client",
                "allowProviderModelFallback": False,
                "ephemeral": ephemeral,
            },
        )
        self.thread_id = result["thread"]["id"]
        self.thread_provider = route.model.provider

    def fork(
        self, model_name: str | None, effort: str | None, ephemeral: bool = False
    ) -> None:
        if self.turn_active:
            raise VeyraError("wait for the active turn before forking")
        if not self.thread_id:
            raise VeyraError("there is no active thread")
        target = self.catalogue.resolve(model_name) if model_name else self.model
        self._require_veyra_host(target)
        requested_effort = effort or self.effort
        if effort is None and target.efforts and requested_effort not in target.efforts:
            requested_effort = target.default_effort
        target_effort = self.catalogue.validate_effort(target, requested_effort)
        result = self.server.request(
            "thread/fork",
            {
                "threadId": self.thread_id,
                "model": target.model_id,
                "modelProvider": target.provider,
                "cwd": str(self.cwd),
                "approvalPolicy": APPROVAL_POLICY,
                "approvalsReviewer": self.approvals_reviewer,
                "sandbox": SANDBOX_MODE,
                "developerInstructions": self.instructions_for(
                    target, self.doctrine.version
                ),
                "ephemeral": ephemeral,
            },
        )
        self.thread_id = result["thread"]["id"]
        self.thread_provider = target.provider
        self.model = target
        self.effort = target_effort
        self.active_profile_version = self.doctrine.version
        self.active_route_reason = "explicit thread fork"
        self.pending_route = None
        self.latest_usage = None
        print(self.palette.dim(f"forked -> {self.thread_id}"))

    def resume(self, thread_id: str) -> None:
        if self.turn_active:
            raise VeyraError("wait for the active turn before resuming another thread")
        result = self.server.request(
            "thread/resume",
            {
                "threadId": thread_id,
                "cwd": str(self.cwd),
                "approvalPolicy": APPROVAL_POLICY,
                "approvalsReviewer": self.approvals_reviewer,
                "sandbox": SANDBOX_MODE,
                "excludeTurns": True,
            },
        )
        self.thread_id = result["thread"]["id"]
        reported_model = result.get("model") or result["thread"].get("model")
        reported_provider = result.get("modelProvider") or result["thread"].get(
            "modelProvider"
        )
        if reported_model:
            try:
                self.model = self.catalogue.resolve(reported_model, reported_provider)
            except VeyraError:
                if reported_provider in LOCAL_ENDPOINTS:
                    self.catalogue.add_local(reported_provider, [reported_model])
                    self.model = self.catalogue.resolve(
                        reported_model, reported_provider
                    )
                else:
                    raise
        self._require_veyra_host(self.model)
        self.thread_provider = reported_provider or self.model.provider
        reported_effort = result.get("reasoningEffort") or result["thread"].get(
            "reasoningEffort"
        )
        if reported_effort:
            self.effort = self.catalogue.validate_effort(self.model, reported_effort)
        self.active_profile_version = None
        self.active_route_reason = "resumed thread; profile requires reconciliation"
        self._schedule_route(
            self.model,
            self.effort,
            "reconcile resumed thread with the current cognitive profile",
        )
        self.latest_usage = None
        print(self.palette.dim(f"resumed -> {self.thread_id}"))

    def show_threads(self) -> None:
        result = self.server.request(
            "thread/list",
            {"limit": 10, "sortKey": "updated_at", "sortDirection": "desc"},
        )
        threads = result.get("data") or []
        if not threads:
            print("No recent threads.")
            return
        for thread in threads:
            marker = "*" if thread.get("id") == self.thread_id else " "
            model = thread.get("model") or "unknown"
            provider = thread.get("modelProvider") or "openai"
            route = model if provider == "openai" else f"{provider}:{model}"
            preview = " ".join(
                (thread.get("name") or thread.get("preview") or "").split()
            )
            if len(preview) > 70:
                preview = preview[:67] + "..."
            print(f"{marker} {thread.get('id')}  {route}  {preview}")

    def run_turn(self, text: str) -> None:
        if self.turn_active:
            raise VeyraError("wait for the active turn before starting another")
        route = self._next_route()
        self._require_veyra_host(route.model)
        if not self.thread_id:
            self.start_thread()
        elif self.thread_provider != route.model.provider:
            self.fork(route.model.route_id, route.effort)
        result = self.server.request(
            "turn/start",
            {
                "threadId": self.thread_id,
                "input": [{"type": "text", "text": text}],
                "collaborationMode": {
                    "mode": "default",
                    "settings": {
                        "model": route.model.model_id,
                        "reasoning_effort": route.effort,
                        "developer_instructions": self.instructions_for(
                            route.model, route.profile_version
                        ),
                    },
                },
                "cwd": str(self.cwd),
                "approvalPolicy": APPROVAL_POLICY,
                "approvalsReviewer": self.approvals_reviewer,
            },
        )
        try:
            turn_id = result["turn"]["id"]
        except (KeyError, TypeError) as exc:
            raise VeyraError("turn/start returned no usable turn id") from exc
        self.model = route.model
        self.effort = route.effort
        self.active_profile_version = route.profile_version
        self.active_route_reason = route.reason
        self.pending_route = None
        self.turn_active = True
        wrote_agent_text = False
        try:
            while True:
                message = self.server.events.get()
                method = message.get("method")
                params = message.get("params") or {}
                if "id" in message and method:
                    self._handle_server_request(message)
                    continue
                if params.get("threadId") not in {None, self.thread_id}:
                    continue
                if method == "item/agentMessage/delta":
                    if params.get("turnId") == turn_id:
                        if not wrote_agent_text:
                            print(self.palette.accent("veyra> "), end="", flush=True)
                            wrote_agent_text = True
                        print(params.get("delta", ""), end="", flush=True)
                elif method == "item/started":
                    self._show_item_started(params.get("item") or {})
                elif method == "thread/tokenUsage/updated":
                    self.latest_usage = params.get("tokenUsage")
                elif method == "error":
                    error = params.get("error") or params
                    print(self.palette.warning(f"\nerror: {error}"), file=sys.stderr)
                elif method == "client/error":
                    print(self.palette.warning(params.get("message", "client error")))
                elif method == "client/serverExited":
                    raise VeyraError("Codex App Server exited during the turn")
                elif method == "turn/completed":
                    turn = params.get("turn") or {}
                    if turn.get("id") == turn_id:
                        if wrote_agent_text:
                            print()
                        status = turn.get("status")
                        if status and status != "completed":
                            print(self.palette.warning(f"turn {status}"))
                        self._show_stat_bar()
                        break
        finally:
            self.turn_active = False

    def _show_item_started(self, item: dict[str, Any]) -> None:
        item_type = item.get("type")
        if item_type == "commandExecution":
            command = item.get("command") or item.get("aggregatedOutput") or "command"
            print(self.palette.dim(f"\n[run] {command}"))
        elif item_type == "fileChange":
            changes = item.get("changes") or []
            paths = [change.get("path") for change in changes if change.get("path")]
            label = ", ".join(paths) if paths else "workspace files"
            print(self.palette.dim(f"\n[edit] {label}"))
        elif item_type == "collabToolCall":
            tool = item.get("tool") or "agent"
            print(self.palette.dim(f"\n[agent] {tool}"))

    def _handle_server_request(self, message: dict[str, Any]) -> None:
        method = message["method"]
        request_id = message["id"]
        params = message.get("params") or {}
        if method == "item/commandExecution/requestApproval":
            self._approve_command(request_id, params)
        elif method == "item/fileChange/requestApproval":
            self._approve_file_change(request_id, params)
        elif method == "item/tool/requestUserInput":
            self._request_user_input(request_id, params)
        elif method == "item/permissions/requestApproval":
            self._approve_permissions(request_id, params)
        elif method == "item/tool/call":
            self._handle_dynamic_tool(request_id, params)
        elif method == "mcpServer/elicitation/request":
            print(self.palette.warning("\nMCP input request declined by the thin client."))
            self.server.respond(request_id, {"action": "decline", "content": None})
        else:
            self.server.respond_error(request_id, -32601, f"unsupported request: {method}")

    def _approve_command(self, request_id: Any, params: dict[str, Any]) -> None:
        print(self.palette.warning("\napproval required: command"))
        if params.get("command"):
            print(params["command"])
        if params.get("cwd"):
            print(self.palette.dim(f"cwd: {params['cwd']}"))
        if params.get("reason"):
            print(self.palette.dim(f"reason: {params['reason']}"))
        decision = self._approval_choice()
        self.server.respond(request_id, {"decision": decision})

    def _approve_file_change(self, request_id: Any, params: dict[str, Any]) -> None:
        print(self.palette.warning("\napproval required: file changes"))
        if params.get("reason"):
            print(self.palette.dim(params["reason"]))
        decision = self._approval_choice()
        self.server.respond(request_id, {"decision": decision})

    def _approval_choice(self) -> str:
        while True:
            try:
                answer = self._read_line(
                    "[y] once  [a] session  [n] decline  [c] cancel > "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return "cancel"
            decisions = {
                "y": "accept",
                "yes": "accept",
                "a": "acceptForSession",
                "n": "decline",
                "no": "decline",
                "c": "cancel",
            }
            if answer in decisions:
                return decisions[answer]

    def _request_user_input(self, request_id: Any, params: dict[str, Any]) -> None:
        answers: dict[str, Any] = {}
        print(self.palette.warning("\nVeyra needs input:"))
        for question in params.get("questions", []):
            print(question.get("question") or question.get("header") or "Input")
            options = question.get("options") or []
            for index, option in enumerate(options, start=1):
                print(f"  {index}. {option.get('label')}: {option.get('description')}")
            prompt = "answer> "
            try:
                raw = (
                    getpass.getpass(prompt)
                    if question.get("isSecret")
                    else self._read_line(prompt)
                )
            except (EOFError, KeyboardInterrupt):
                print()
                raw = ""
            if options and raw.isdigit() and 1 <= int(raw) <= len(options):
                raw = options[int(raw) - 1].get("label", raw)
            answers[question["id"]] = {"answers": [raw]}
        self.server.respond(request_id, {"answers": answers})

    def _approve_permissions(self, request_id: Any, params: dict[str, Any]) -> None:
        print(self.palette.warning("\napproval required: additional permissions"))
        if params.get("reason"):
            print(params["reason"])
        print(json.dumps(params.get("permissions") or {}, indent=2, ensure_ascii=True))
        try:
            answer = self._read_line("grant for this turn? [y/N] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            answer = ""
        permissions = params.get("permissions") if answer in {"y", "yes"} else {}
        self.server.respond(
            request_id, {"permissions": permissions or {}, "scope": "turn"}
        )

    def _handle_dynamic_tool(self, request_id: Any, params: dict[str, Any]) -> None:
        arguments = params.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        tool = params.get("tool")
        if tool == ROUTE_TOOL:
            self._handle_model_route(request_id, arguments)
        elif tool == ATTENTION_TOOL:
            self._handle_attention(request_id, arguments)
        elif tool == WORKER_AGENT_TOOL:
            self._handle_worker_agent(request_id, arguments)
        elif tool == LOCAL_AGENT_TOOL:
            self._handle_local_agent(request_id, arguments)
        else:
            self._tool_response(request_id, False, "Unknown client tool.")

    def _handle_model_route(
        self, request_id: Any, arguments: dict[str, Any]
    ) -> None:
        try:
            target = self.catalogue.resolve(str(arguments.get("model", "")))
            self._require_veyra_host(target)
            effort = self.catalogue.validate_effort(
                target, str(arguments.get("effort", ""))
            )
            if effort == "max":
                raise VeyraError("max effort requires manual selection")
            reason = str(arguments.get("reason", "")).strip()
            if not reason:
                raise VeyraError("a routing reason is required")
            self._schedule_route(target, effort, reason)
            text = f"Scheduled {target.route_id}/{effort} for the next turn: {reason}"
            print(self.palette.dim(f"\n[route] {text}"))
            self._tool_response(request_id, True, text)
        except VeyraError as exc:
            self._tool_response(request_id, False, f"Route rejected: {exc}")

    def _handle_attention(
        self, request_id: Any, arguments: dict[str, Any]
    ) -> None:
        try:
            target = self.pending_route.model if self.pending_route else self.model
            effort = self.catalogue.validate_effort(
                target, str(arguments.get("effort", ""))
            )
            if effort == "max":
                raise VeyraError("max effort requires manual selection")
            reason = str(arguments.get("reason", "")).strip()
            if not reason:
                raise VeyraError("an attention reason is required")
            self._schedule_route(target, effort, reason)
            text = (
                f"Scheduled {target.route_id}/{effort} attention for the next turn: "
                f"{reason}"
            )
            print(self.palette.dim(f"\n[attention] {text}"))
            self._tool_response(request_id, True, text)
        except VeyraError as exc:
            self._tool_response(request_id, False, f"Attention shift rejected: {exc}")

    def _schedule_route(self, target: Model, effort: str, reason: str) -> None:
        self._require_veyra_host(target)
        validated = self.catalogue.validate_effort(target, effort)
        self.pending_route = PendingRoute(
            target, validated, reason, self.doctrine.version
        )

    def _next_route(self) -> PendingRoute:
        if self.pending_route:
            route = self.pending_route
        else:
            if self.active_profile_version != self.doctrine.version:
                raise VeyraError(
                    "active cognitive profile is unverified or stale; "
                    "schedule a versioned route first"
                )
            route = PendingRoute(
                self.model,
                self.effort,
                self.active_route_reason,
                self.active_profile_version,
            )
        if route.profile_version != self.doctrine.version:
            raise VeyraError(
                "cognitive profile version changed before the route was applied"
            )
        return route

    def _handle_local_agent(
        self, request_id: Any, arguments: dict[str, Any]
    ) -> None:
        try:
            target = self.catalogue.resolve(str(arguments.get("model", "")))
            if not target.local:
                raise VeyraError("run_local_agent requires a discovered local model")
            prompt = str(arguments.get("prompt", "")).strip()
            if not prompt:
                raise VeyraError("a bounded worker prompt is required")
            effort = self.catalogue.validate_effort(
                target, str(arguments.get("effort") or target.default_effort)
            )
            report = self._run_local_worker(target, effort, prompt)
            self._show_worker_stat_bar(
                target, effort, self.worker_stats[target.route_id].latest_usage
            )
            self._tool_response(request_id, True, report)
        except VeyraError as exc:
            self._tool_response(request_id, False, f"Local worker failed: {exc}")

    def _handle_worker_agent(
        self, request_id: Any, arguments: dict[str, Any]
    ) -> None:
        try:
            target = self.catalogue.resolve(str(arguments.get("model", "")))
            if target.provider == "openai" and target.model_id in VEYRA_HOST_MODELS:
                raise VeyraError("run_worker_agent requires a worker-only model")
            prompt = str(arguments.get("prompt", "")).strip()
            if not prompt:
                raise VeyraError("a bounded worker prompt is required")
            effort = self.catalogue.validate_effort(
                target, str(arguments.get("effort") or target.default_effort)
            )
            report = self._run_local_worker(target, effort, prompt)
            self._show_worker_stat_bar(
                target, effort, self.worker_stats[target.route_id].latest_usage
            )
            self._tool_response(request_id, True, report)
        except VeyraError as exc:
            self._tool_response(request_id, False, f"Worker failed: {exc}")

    def _run_local_worker(self, target: Model, effort: str, prompt: str) -> str:
        print(self.palette.dim(f"\n[local] {target.route_id}/{effort}"))
        started_at = time.monotonic()
        started = self.server.request(
            "thread/start",
            {
                "model": target.model_id,
                "modelProvider": target.provider,
                "cwd": str(self.cwd),
                "approvalPolicy": APPROVAL_POLICY,
                "approvalsReviewer": self.approvals_reviewer,
                "sandbox": SANDBOX_MODE,
                "developerInstructions": (
                    self.doctrine.worker.rstrip() + "\n"
                ),
                "ephemeral": True,
                "serviceName": "veyra_client_local_worker",
                "allowProviderModelFallback": False,
            },
        )
        worker_thread = started["thread"]["id"]
        turn_params: dict[str, Any] = {
            "threadId": worker_thread,
            "input": [{"type": "text", "text": prompt}],
            "model": target.model_id,
            "cwd": str(self.cwd),
            "approvalPolicy": APPROVAL_POLICY,
            "approvalsReviewer": self.approvals_reviewer,
        }
        if effort:
            turn_params["effort"] = effort
        started_turn = self.server.request("turn/start", turn_params)
        worker_turn = started_turn["turn"]["id"]
        output: list[str] = []
        failure: str | None = None
        latest_worker_usage: dict[str, Any] | None = None
        while True:
            try:
                message = self.server.events.get(timeout=900)
            except queue.Empty as exc:
                raise VeyraError("local worker timed out") from exc
            method = message.get("method")
            params = message.get("params") or {}
            if "id" in message and method:
                self._handle_server_request(message)
                continue
            if params.get("threadId") != worker_thread:
                continue
            if method == "item/agentMessage/delta" and params.get("turnId") == worker_turn:
                output.append(params.get("delta", ""))
            elif method == "thread/tokenUsage/updated":
                usage = params.get("tokenUsage")
                if isinstance(usage, dict):
                    latest_worker_usage = usage
            elif method == "item/started":
                self._show_item_started(params.get("item") or {})
            elif method == "error":
                error = params.get("error") or params
                failure = str(error)
            elif method == "turn/completed":
                turn = params.get("turn") or {}
                if turn.get("id") == worker_turn:
                    status = turn.get("status")
                    if status != "completed":
                        failure = failure or f"turn {status}"
                    break
        report = "".join(output).strip()
        if failure:
            raise VeyraError(failure)
        if not report:
            raise VeyraError("local worker returned no report")
        if len(report) > 20000:
            report = report[:20000] + "\n[local report truncated]"
        self._record_worker_stats(
            target, latest_worker_usage, time.monotonic() - started_at
        )
        print(self.palette.dim(f"[local complete] {len(report)} characters"))
        return report

    def _tool_response(self, request_id: Any, success: bool, text: str) -> None:
        self.server.respond(
            request_id,
            {
                "success": success,
                "contentItems": [{"type": "inputText", "text": text}],
            },
        )

    def repl(self) -> None:
        self._start_terminal_ui()
        try:
            print(self.palette.accent("Veyra"), self.palette.dim(CLIENT_VERSION))
            print(self.palette.dim(f"{self.model.route_id} / {self.effort}"))
            print(self.palette.dim("new thread (created with the first message)"))
            print(self.palette.dim("/help for commands"))
            while True:
                try:
                    text = self._read_line(self.palette.accent("alice> ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not text:
                    continue
                if not text.startswith("/"):
                    try:
                        self.run_turn(text)
                    except VeyraError as exc:
                        print(self.palette.warning(f"error: {exc}"), file=sys.stderr)
                    continue
                try:
                    if not self._command(text):
                        break
                except VeyraError as exc:
                    print(self.palette.warning(f"error: {exc}"), file=sys.stderr)
        finally:
            self._stop_terminal_ui()

    @staticmethod
    def _read_line(prompt: str) -> str:
        """Read editable input with correct wrapped-row cursor accounting."""
        text = input(readline_safe_prompt(prompt))
        if readline is not None and text:
            readline.add_history(text)
        return text

    def _command(self, text: str) -> bool:
        parts = text.split()
        command = parts[0].lower()
        args = parts[1:]
        if command in {"/quit", "/exit"}:
            return False
        if command == "/help":
            print(
                "/models               available routes\n"
                "/model NAME            set model for later turns\n"
                "/effort LEVEL          set reasoning effort\n"
                "/attention [LEVEL]     inspect or set attention\n"
                "/local MODEL PROMPT    run a bounded local worker\n"
                "/worker MODEL PROMPT   run a bounded worker-only model\n"
                "/fork [MODEL] [EFFORT] branch history and enter the fork\n"
                "/new                   start an empty thread\n"
                "/threads               list recent threads\n"
                "/resume THREAD_ID      resume a thread\n"
                "/thread               show active route\n"
                "/usage                show latest token counts\n"
                "/workers              show local worker usage by model\n"
                "/quit                 exit"
            )
        elif command == "/models":
            selected = self.pending_route.model if self.pending_route else self.model
            for model in self.catalogue.models:
                if not model.local and model.model_id not in VEYRA_HOST_MODELS:
                    continue
                marker = "*" if model.route_id == selected.route_id else " "
                efforts = ", ".join(model.efforts) or "provider-defined"
                location = "worker-only" if model.local else "Veyra host"
                print(f"{marker} {model.route_id}: {efforts} [{location}]")
        elif command == "/model":
            if not args:
                selected = self.pending_route.model if self.pending_route else self.model
                print(selected.route_id)
            else:
                target = self.catalogue.resolve(" ".join(args))
                self._require_veyra_host(target)
                effort = self.pending_route.effort if self.pending_route else self.effort
                if target.efforts and effort not in target.efforts:
                    effort = target.default_effort
                self._schedule_route(target, effort, "manual model selection")
                print(self.palette.dim(f"next turn -> {target.route_id}/{effort}"))
        elif command == "/effort":
            if not args:
                effort = self.pending_route.effort if self.pending_route else self.effort
                print(effort)
            else:
                target = self.pending_route.model if self.pending_route else self.model
                effort = self.catalogue.validate_effort(target, args[0])
                self._schedule_route(target, effort, "manual effort selection")
                print(self.palette.dim(f"next turn -> {target.route_id}/{effort}"))
        elif command == "/attention":
            if not args:
                effort = self.pending_route.effort if self.pending_route else self.effort
                print(effort)
            else:
                target = self.pending_route.model if self.pending_route else self.model
                effort = self.catalogue.validate_effort(target, args[0])
                self._schedule_route(target, effort, "manual attention selection")
                print(
                    self.palette.dim(
                        f"next attention -> {target.route_id}/{effort}"
                    )
                )
        elif command == "/local":
            if len(args) < 2:
                raise VeyraError("usage: /local MODEL PROMPT")
            target = self.catalogue.resolve(args[0])
            if not target.local:
                raise VeyraError("/local requires a discovered local model")
            report = self._run_local_worker(
                target, target.default_effort, " ".join(args[1:])
            )
            print(self.palette.accent("local> ") + report)
            self._show_worker_stat_bar(
                target, target.default_effort, self.worker_stats[target.route_id].latest_usage
            )
        elif command == "/worker":
            if len(args) < 2:
                raise VeyraError("usage: /worker MODEL PROMPT")
            target = self.catalogue.resolve(args[0])
            if target.provider == "openai" and target.model_id in VEYRA_HOST_MODELS:
                raise VeyraError("/worker requires a worker-only model")
            report = self._run_local_worker(
                target, target.default_effort, " ".join(args[1:])
            )
            print(self.palette.accent("worker> ") + report)
            self._show_worker_stat_bar(
                target, target.default_effort, self.worker_stats[target.route_id].latest_usage
            )
        elif command == "/fork":
            self.fork(args[0] if args else None, args[1] if len(args) > 1 else None)
        elif command == "/new":
            self.thread_id = None
            self.thread_provider = None
            self.latest_usage = None
            print(self.palette.dim("new thread (created with the first message)"))
        elif command == "/threads":
            self.show_threads()
        elif command == "/resume":
            if len(args) != 1:
                raise VeyraError("usage: /resume THREAD_ID")
            self.resume(args[0])
        elif command == "/thread":
            print(f"thread: {self.thread_id}")
            print(f"cwd: {self.cwd}")
            print(f"route: {self.model.route_id}/{self.effort}")
            print(f"attention: {self.effort}")
            print(f"profile: {self.active_profile_version or 'unverified'}")
            print(f"route reason: {self.active_route_reason}")
            if self.pending_route:
                pending = self.pending_route
                print(
                    f"pending: {pending.model.route_id}/{pending.effort} "
                    f"profile {pending.profile_version} ({pending.reason})"
                )
        elif command == "/usage":
            self._show_usage()
        elif command == "/workers":
            self._show_workers()
        else:
            raise VeyraError(f"unknown command: {command}")
        return True

    def _show_usage(self) -> None:
        if not self.latest_usage:
            print("No token usage has been reported for this thread.")
            return
        last = self.latest_usage.get("last") or {}
        total = self.latest_usage.get("total") or {}
        print(
            "last: "
            f"{last.get('totalTokens', 0)} total, "
            f"{last.get('inputTokens', 0)} input, "
            f"{last.get('cachedInputTokens', 0)} cached, "
            f"{last.get('outputTokens', 0)} output, "
            f"{last.get('reasoningOutputTokens', 0)} reasoning"
        )
        print(f"thread total: {total.get('totalTokens', 0)}")
        if self.worker_stats:
            print(f"worker calls: {sum(stats.calls for stats in self.worker_stats.values())}")

    def _show_stat_bar(self) -> None:
        if not self.latest_usage:
            self._render_status_bar("[ Veyra | token telemetry unavailable ]")
            return
        last = self.latest_usage.get("last") or {}
        total = self.latest_usage.get("total") or {}
        model = self.model.model_id.removeprefix("gpt-5.6-")
        last_total = format_tokens(last.get("totalTokens"))
        thread_total = format_tokens(total.get("totalTokens"))
        gauge = token_gauge(last)
        self._render_status_bar(
            "[ "
            f"Veyra {model}/{self.effort} | turn {last_total} "
            f"I {format_tokens(last.get('inputTokens'))} "
            f"C {format_tokens(last.get('cachedInputTokens'))} "
            f"O {format_tokens(last.get('outputTokens'))} "
            f"R {format_tokens(last.get('reasoningOutputTokens'))} "
            f"| {gauge} | thread {thread_total} "
            "]"
        )

    def _record_worker_stats(
        self,
        target: Model,
        usage: dict[str, Any] | None,
        elapsed_seconds: float,
    ) -> None:
        stats = self.worker_stats.setdefault(target.route_id, WorkerStats())
        stats.calls += 1
        stats.elapsed_seconds += elapsed_seconds
        stats.latest_usage = usage
        last = (usage or {}).get("last") or {}
        stats.total_tokens += max(0, int(last.get("totalTokens") or 0))
        stats.output_tokens += max(0, int(last.get("outputTokens") or 0))
        stats.reasoning_tokens += max(
            0, int(last.get("reasoningOutputTokens") or 0)
        )

    def _show_worker_stat_bar(
        self, target: Model, effort: str, usage: dict[str, Any] | None
    ) -> None:
        stats = self.worker_stats[target.route_id]
        if not usage:
            self._render_status_bar(
                f"[ worker {target.route_id}/{effort} | {stats.elapsed_seconds:.1f}s "
                "| token telemetry unavailable ]"
            )
            return
        last = usage.get("last") or {}
        rate = (
            f"{stats.output_tokens / stats.elapsed_seconds:.1f} out tok/s"
            if stats.elapsed_seconds > 0 and stats.output_tokens
            else "rate unavailable"
        )
        self._render_status_bar(
            "[ worker "
            f"{target.route_id}/{effort} | turn {format_tokens(last.get('totalTokens'))} "
            f"I {format_tokens(last.get('inputTokens'))} "
            f"C {format_tokens(last.get('cachedInputTokens'))} "
            f"O {format_tokens(last.get('outputTokens'))} "
            f"R {format_tokens(last.get('reasoningOutputTokens'))} "
            f"| {token_gauge(last)} | {stats.elapsed_seconds:.1f}s, {rate} ]"
        )

    def _render_status_bar(self, text: str) -> None:
        """Redraw token telemetry in the terminal's reserved bottom row."""
        self.status_bar_text = text
        if not self.terminal_ui_active:
            print(self.palette.dim(text), flush=True)
            self.status_bar_visible = False
            return
        size = shutil.get_terminal_size(fallback=(80, 24))
        columns = size.columns
        if columns > 4 and len(text) >= columns:
            text = text[: columns - 4] + "..."
        layout = ""
        if size != self.terminal_size:
            self.terminal_size = size
            layout = f"\033[1;{max(1, size.lines - 1)}r"
        sys.stdout.write(
            "\0337"
            + layout
            + f"\033[{size.lines};1H\033[2K"
            + self.palette.dim(text)
            + "\0338"
        )
        sys.stdout.flush()
        self.status_bar_visible = True

    def _start_terminal_ui(self) -> None:
        """Enter a full-screen terminal surface with one non-scrolling row."""
        if not self.status_bar_enabled or self.terminal_ui_active:
            return
        self.terminal_size = shutil.get_terminal_size(fallback=(80, 24))
        scroll_bottom = max(1, self.terminal_size.lines - 1)
        sys.stdout.write(
            "\033[?1049h\033[2J\033[H"
            f"\033[1;{scroll_bottom}r"
        )
        sys.stdout.flush()
        self.terminal_ui_active = True
        if hasattr(signal, "SIGWINCH"):
            try:
                self.previous_resize_handler = signal.getsignal(signal.SIGWINCH)
                signal.signal(signal.SIGWINCH, self._handle_terminal_resize)
            except (OSError, ValueError):
                self.previous_resize_handler = None

    def _handle_terminal_resize(self, _signum: int, _frame: Any) -> None:
        """Reapply the scroll region and status row after a terminal resize."""
        if not self.terminal_ui_active:
            return
        text = self.status_bar_text
        if text:
            self._render_status_bar(text)
        else:
            size = shutil.get_terminal_size(fallback=(80, 24))
            self.terminal_size = size
            sys.stdout.write(
                "\0337"
                f"\033[1;{max(1, size.lines - 1)}r"
                f"\033[{size.lines};1H\033[2K"
                "\0338"
            )
            sys.stdout.flush()
        if readline is not None and not self.turn_active:
            readline.redisplay()

    def _clear_status_bar(self) -> None:
        if self.terminal_ui_active:
            rows = self.terminal_size.lines
            sys.stdout.write(f"\0337\033[{rows};1H\033[2K\0338")
            sys.stdout.flush()
        self.status_bar_visible = False

    def _stop_terminal_ui(self) -> None:
        """Restore the caller's terminal even when the REPL is interrupted."""
        if not self.terminal_ui_active:
            return
        self._clear_status_bar()
        sys.stdout.write("\033[r\033[?1049l")
        sys.stdout.flush()
        self.terminal_ui_active = False
        if hasattr(signal, "SIGWINCH") and self.previous_resize_handler is not None:
            try:
                signal.signal(signal.SIGWINCH, self.previous_resize_handler)
            except (OSError, ValueError):
                pass
        self.previous_resize_handler = None

    def _show_workers(self) -> None:
        if not self.worker_stats:
            print("No local worker usage has been recorded in this session.")
            return
        print("worker model                              calls  tokens  output tok/s")
        for route, stats in sorted(self.worker_stats.items()):
            rate = (
                f"{stats.output_tokens / stats.elapsed_seconds:.1f}"
                if stats.elapsed_seconds > 0 and stats.output_tokens
                else "-"
            )
            print(
                f"{route[:40]:40}  {stats.calls:5}  "
                f"{format_tokens(stats.total_tokens):>6}  {rate:>12}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_core = Path(__file__).resolve().parent.parent / "veyra-core"
    default_workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=default_workspace)
    parser.add_argument(
        "--core",
        type=Path,
        default=Path(os.environ.get("VEYRA_CORE_REPO", default_core)),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT)
    parser.add_argument(
        "--approvals-reviewer",
        choices=("auto_review", "user"),
        default=DEFAULT_APPROVALS_REVIEWER,
        help="route approval requests to automatic review or to the user",
    )
    parser.add_argument("--codex", default=shutil.which("codex") or "codex")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-colour", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    terminal_output = sys.stdout.isatty()
    palette = Palette(terminal_output and not args.no_colour, terminal=terminal_output)
    server: AppServer | None = None
    try:
        if not args.cwd.is_dir():
            raise VeyraError(f"working directory does not exist: {args.cwd}")
        doctrine = ContinuityGate(args.core, palette).verify()
        server = AppServer(args.codex, debug=args.debug)
        server.start()
        raw_catalogue = server.request(
            "model/list", {"limit": 100, "includeHidden": False}
        )
        catalogue = ModelCatalogue(raw_catalogue.get("data") or [])
        if not catalogue.models:
            raise VeyraError("App Server returned no available models")
        if not args.no_local:
            local = LocalModelDiscovery().discover()
            for provider, model_ids in local.items():
                catalogue.add_local(provider, model_ids)
        model = catalogue.resolve(args.model)
        effort = catalogue.validate_effort(model, args.effort)
        client = VeyraClient(
            server,
            catalogue,
            doctrine,
            args.cwd,
            model,
            effort,
            palette,
            approvals_reviewer=args.approvals_reviewer,
            debug=args.debug,
        )
        if args.smoke:
            client.start_thread(ephemeral=True)
            hosted_count = len(catalogue.models) - len(catalogue.local_models)
            print(
                "app-server ok: "
                f"{hosted_count} hosted, {len(catalogue.local_models)} local models"
            )
            print(f"ephemeral thread ok: {client.thread_id}")
            print(f"default route ok: {model.route_id}/{effort}")
            return 0
        client.repl()
        return 0
    except (VeyraError, OSError) as exc:
        print(palette.warning(f"error: {exc}"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(palette.warning("interrupted"), file=sys.stderr)
        return 130
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    raise SystemExit(main())
