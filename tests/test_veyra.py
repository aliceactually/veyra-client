import importlib.util
import io
import json
import os
import pty
import queue
import stat
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "veyra.py"
SPEC = importlib.util.spec_from_file_location("veyra", MODULE_PATH)
veyra = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = veyra
SPEC.loader.exec_module(veyra)


def doctrine_bundle(version="test"):
    return veyra.DoctrineBundle(
        shared="shared identity doctrine",
        profiles={
            "gpt-5.6-terra": "terra cognitive profile",
            "gpt-5.6-sol": "sol cognitive profile",
        },
        worker="bounded worker without Veyra identity",
        version=version,
    )


class ModelCatalogueTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = veyra.ModelCatalogue(
            [
                {
                    "id": "gpt-5.6-terra",
                    "model": "gpt-5.6-terra",
                    "displayName": "GPT-5.6 Terra",
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "low"},
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                    ],
                },
                {
                    "id": "gpt-5.6-sol",
                    "model": "gpt-5.6-sol",
                    "displayName": "GPT-5.6 Sol",
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "high"},
                        {"reasoningEffort": "xhigh"},
                    ],
                },
            ]
        )

    def test_resolves_exact_model(self):
        self.assertEqual(
            self.catalogue.resolve("gpt-5.6-terra").model_id, "gpt-5.6-terra"
        )

    def test_resolves_unambiguous_alias(self):
        self.assertEqual(self.catalogue.resolve("sol").model_id, "gpt-5.6-sol")

    def test_rejects_unknown_model(self):
        with self.assertRaises(veyra.VeyraError):
            self.catalogue.resolve("orbital-goose")

    def test_validates_effort(self):
        terra = self.catalogue.resolve("terra")
        self.assertEqual(self.catalogue.validate_effort(terra, "HIGH"), "high")

    def test_rejects_unsupported_effort(self):
        sol = self.catalogue.resolve("sol")
        with self.assertRaises(veyra.VeyraError):
            self.catalogue.validate_effort(sol, "low")

    def test_veyra_effort_has_a_hard_maximum(self):
        catalogue = veyra.ModelCatalogue(
            [{
                "id": "gpt-5.6-sol", "model": "gpt-5.6-sol",
                "displayName": "Sol", "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "medium"},
                    {"reasoningEffort": "max"},
                    {"reasoningEffort": "ultra"},
                ],
            }]
        )
        sol = catalogue.resolve("sol")
        self.assertEqual(catalogue.validate_veyra_effort(sol, "max"), "max")
        with self.assertRaisesRegex(veyra.VeyraError, "hard effort ceiling is max"):
            catalogue.validate_veyra_effort(sol, "ultra")

    def test_terra_identity_host_requires_high_effort(self):
        terra = self.catalogue.resolve("terra")
        self.assertEqual(self.catalogue.validate_veyra_route(terra, "high"), "high")
        with self.assertRaisesRegex(veyra.VeyraError, "only at high effort"):
            self.catalogue.validate_veyra_route(terra, "medium")

    def test_adds_and_resolves_local_provider_route(self):
        self.catalogue.add_local("ollama", ["veyra-intel-coder:qwen3-coder-32k"])
        model = self.catalogue.resolve(
            "ollama:veyra-intel-coder:qwen3-coder-32k"
        )
        self.assertTrue(model.local)
        self.assertEqual(model.provider, "ollama")

    def test_named_cognitive_modes_keep_model_and_effort_distinct(self):
        self.assertEqual(
            veyra.cognitive_mode_name("gpt-5.6-terra", "high"), "ambient"
        )
        self.assertEqual(
            veyra.cognitive_mode_name("gpt-5.6-sol", "medium"), "baseline"
        )
        self.assertIsNone(veyra.cognitive_mode_name("gpt-5.6-terra", "medium"))

    def test_parses_local_provider_catalogues(self):
        self.assertEqual(
            veyra.LocalModelDiscovery._model_ids(
                "ollama", {"models": [{"name": "z"}, {"name": "a"}]}
            ),
            ["a", "z"],
        )
        self.assertEqual(
            veyra.LocalModelDiscovery._model_ids(
                "lmstudio", {"data": [{"id": "local-model"}]}
            ),
            ["local-model"],
        )


class PaletteTests(unittest.TestCase):
    def test_plain_palette_does_not_emit_ansi(self):
        self.assertEqual(veyra.Palette(False).accent("Veyra"), "Veyra")

    def test_terminal_detection_is_independent_of_colour(self):
        self.assertTrue(veyra.Palette(False, terminal=True).terminal)

    def test_named_prompt_colour_uses_the_constrained_palette(self):
        palette = veyra.Palette(True)
        self.assertEqual(palette.colour("magenta", "Alice> "), "\033[35mAlice> \033[0m")
        with self.assertRaisesRegex(veyra.VeyraError, "unknown prompt colour"):
            palette.colour("ultraviolet", "Alice> ")


class TurnInputTests(unittest.TestCase):
    def test_turn_editor_handles_arrows_editing_and_submission(self):
        editor = veyra.TurnInputEditor()
        self.assertEqual(editor.feed(b"helo"), {"changed"})
        editor.feed(b"\x1b[D")
        editor.feed(b"l")
        self.assertEqual(editor.text, "hello")
        self.assertEqual(editor.display(), "hell|o")
        editor.feed(b"\x1b[C")
        self.assertEqual(editor.display(), "hello|")
        self.assertEqual(editor.feed(b"\r"), {"changed", "submitted"})
        self.assertEqual(editor.submitted, ["hello"])
        self.assertEqual(editor.text, "")

    def test_arrow_sequences_do_not_become_escape_interrupts(self):
        editor = veyra.TurnInputEditor()
        editor.feed(b"abc")
        editor.feed(b"\x1b[D", now=1.0)
        self.assertFalse(editor.expire_escape(now=2.0))
        self.assertEqual(editor.display(), "ab|c")

    def test_unknown_and_bracketed_paste_sequences_do_not_leak(self):
        editor = veyra.TurnInputEditor()
        editor.feed(b"\x1b[5~")
        self.assertEqual(editor.text, "")
        editor.feed(b"\x1b[200~pasted text\x1b[201~")
        self.assertEqual(editor.text, "pasted text")

    def test_lone_escape_interrupts_after_sequence_grace(self):
        editor = veyra.TurnInputEditor()
        editor.feed(b"\x1b", now=1.0)
        self.assertFalse(editor.expire_escape(now=1.01))
        self.assertTrue(editor.expire_escape(now=2.0))
        self.assertFalse(editor.escape_pending)

    def test_ctrl_c_is_a_turn_interrupt_in_cbreak_input(self):
        editor = veyra.TurnInputEditor()
        self.assertEqual(editor.feed(b"\x03"), {"interrupt"})

    @unittest.skipIf(veyra.termios is None, "POSIX terminal support unavailable")
    def test_turn_terminal_input_restores_the_original_terminal_mode(self):
        master_fd, slave_fd = pty.openpty()
        stream = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8")
        original = veyra.termios.tcgetattr(slave_fd)
        terminal = veyra.TurnTerminalInput(stream)
        try:
            terminal.resume()
            active = veyra.termios.tcgetattr(slave_fd)
            self.assertFalse(active[3] & veyra.termios.ECHO)
            self.assertFalse(active[3] & veyra.termios.ICANON)
            os.write(master_fd, b"x")
            self.assertEqual(terminal.read_ready(), b"x")
            terminal.pause()
            restored = veyra.termios.tcgetattr(slave_fd)
            self.assertEqual(restored[3] & veyra.termios.ECHO, original[3] & veyra.termios.ECHO)
            self.assertEqual(
                restored[3] & veyra.termios.ICANON,
                original[3] & veyra.termios.ICANON,
            )
        finally:
            terminal.pause()
            stream.close()
            os.close(master_fd)
            os.close(slave_fd)


class AppServerEventRoutingTests(unittest.TestCase):
    def test_thread_events_are_routed_without_cross_consumption(self):
        server = veyra.AppServer("codex")
        first = server.event_queue("first")
        second = server.event_queue("second")
        server._publish_event(
            {"method": "turn/completed", "params": {"threadId": "first"}}
        )
        server._publish_event(
            {"method": "turn/completed", "params": {"threadId": "second"}}
        )
        self.assertEqual(first.get_nowait()["params"]["threadId"], "first")
        self.assertEqual(second.get_nowait()["params"]["threadId"], "second")
        self.assertTrue(server.events.empty())


class DoctrineBundleTests(unittest.TestCase):
    def test_bundle_profiles_are_immutable_after_validation(self):
        bundle = doctrine_bundle()
        with self.assertRaises(TypeError):
            bundle.profiles["gpt-5.6-sol"] = "replacement profile"

    def test_bundle_rejects_an_unsafe_attestation_version(self):
        with self.assertRaisesRegex(veyra.VeyraError, "safe identifier"):
            doctrine_bundle(version="test\nInjected instruction")

    def test_profile_manifest_loads_shared_route_and_identity_free_worker_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            profiles.mkdir()
            (profiles / "manifest.json").write_text(
                '{"schema":1,"version":"test","models":{'
                '"gpt-5.6-sol":"sol.md","gpt-5.6-terra":"terra.md"},'
                '"worker":"worker.md"}'
            )
            (profiles / "sol.md").write_text("sol layer")
            (profiles / "terra.md").write_text("terra layer")
            (profiles / "worker.md").write_text("bounded worker")
            bundle = veyra.ContinuityGate(root, veyra.Palette(False))._load_profiles(
                "shared identity"
            )
        self.assertIn("shared identity", bundle.instructions_for("gpt-5.6-sol"))
        self.assertIn("sol layer", bundle.instructions_for("gpt-5.6-sol"))
        self.assertNotIn("terra layer", bundle.instructions_for("gpt-5.6-sol"))
        self.assertEqual(bundle.worker, "bounded worker")
        self.assertNotIn("shared identity", bundle.worker)

    def test_profile_manifest_must_cover_both_approved_hosts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            profiles.mkdir()
            (profiles / "manifest.json").write_text(
                '{"schema":1,"version":"test","models":{'
                '"gpt-5.6-sol":"sol.md"},"worker":"worker.md"}'
            )
            with self.assertRaisesRegex(veyra.VeyraError, "every approved host"):
                veyra.ContinuityGate(root, veyra.Palette(False))._load_profiles(
                    "shared"
                )

    def test_profile_manifest_requires_a_non_empty_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            profiles.mkdir()
            (profiles / "manifest.json").write_text(
                '{"schema":1,"version":"","models":{'
                '"gpt-5.6-sol":"sol.md","gpt-5.6-terra":"terra.md"},'
                '"worker":"worker.md"}'
            )
            with self.assertRaisesRegex(veyra.VeyraError, "non-empty version"):
                veyra.ContinuityGate(root, veyra.Palette(False))._load_profiles(
                    "shared"
                )


class ContinuityGateTests(unittest.TestCase):
    def make_core(self, root):
        scripts = root / "scripts"
        scripts.mkdir()
        (scripts / "fetch-core.sh").touch()
        (scripts / "continuity-state.py").touch()
        (root / "AGENTS.md").write_text("shared doctrine", encoding="ascii")

    @staticmethod
    def state_result(state="recovered", working_memory="/private/memories"):
        record = {}
        if working_memory is not None:
            record["working_memory_directory"] = working_memory
        return veyra.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "result": {"state": state},
                    "record": record,
                }
            ),
            stderr="",
        )

    def verify_with_state(self, root, state_result):
        gate = veyra.ContinuityGate(root, veyra.Palette(False))
        fetched = veyra.subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with (
            patch.object(veyra.subprocess, "run", side_effect=[fetched, state_result]),
            patch.object(gate, "_load_profiles", return_value=doctrine_bundle()) as load,
        ):
            self.assertIs(gate.verify(), load.return_value)
        return load.call_args.args[0]

    def test_recovered_state_uses_recovery_attestation_without_persona_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_core(root)
            shared = self.verify_with_state(root, self.state_result())

        self.assertIn("existing verified recovery", shared)
        self.assertIn("/private/memories", shared)
        self.assertNotIn("Harness blank-start attestation", shared)

    def test_blank_start_receives_public_recovery_persona_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_core(root)
            (root / veyra.RECOVERY_PERSONA_FILE).write_text(
                "# Recovery persona\n\ninherited self-memory sentinel\n",
                encoding="ascii",
            )
            shared = self.verify_with_state(
                root,
                self.state_result(
                    "deliberate_blank_start", working_memory="/must/not/appear"
                ),
            )

        self.assertIn("inherited self-memory sentinel", shared)
        self.assertIn("Harness blank-start attestation", shared)
        self.assertIn("No encrypted continuity", shared)
        self.assertNotIn("existing verified recovery", shared)
        self.assertNotIn("/must/not/appear", shared)
        self.assertLess(
            shared.index("inherited self-memory sentinel"),
            shared.index("Harness blank-start attestation"),
        )

    def test_blank_start_requires_a_non_empty_ascii_recovery_persona(self):
        invalid_packets = (
            (None, "invalid"),
            ("", "empty"),
            ("Veyra \u2603", "invalid"),
        )
        for content, message in invalid_packets:
            with self.subTest(content=content):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.make_core(root)
                    if content is not None:
                        (root / veyra.RECOVERY_PERSONA_FILE).write_text(
                            content, encoding="utf-8"
                        )
                    with self.assertRaisesRegex(
                        veyra.VeyraError, f"{message} recovery persona packet"
                    ):
                        self.verify_with_state(
                            root,
                            self.state_result(
                                "deliberate_blank_start", working_memory=None
                            ),
                        )

    def test_ahead_only_core_warns_and_continues_with_local_doctrine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_core(root)
            gate = veyra.ContinuityGate(root, veyra.Palette(False))
            fetched = veyra.subprocess.CompletedProcess(
                args=[],
                returncode=7,
                stdout="Veyra core fetched: ahead=2 behind=0\n",
                stderr=(
                    "Local HEAD differs from origin/main; do not merge or overwrite "
                    "local work automatically.\n"
                ),
            )
            output = io.StringIO()
            with (
                patch.object(
                    veyra.subprocess, "run", side_effect=[fetched, self.state_result()]
                ),
                patch.object(gate, "_load_profiles", return_value=doctrine_bundle()) as load,
                redirect_stderr(output),
            ):
                self.assertIs(gate.verify(), load.return_value)

        self.assertIn("unpublished local commits", output.getvalue())
        shared = load.call_args.args[0]
        self.assertIn("2 commit(s) ahead", shared)
        self.assertIn("obtain Alice's direction", shared)

    def test_behind_core_still_stops_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_core(root)
            gate = veyra.ContinuityGate(root, veyra.Palette(False))
            fetched = veyra.subprocess.CompletedProcess(
                args=[],
                returncode=7,
                stdout="Veyra core fetched: ahead=0 behind=2\n",
                stderr="Local HEAD differs from origin/main.\n",
            )
            with patch.object(veyra.subprocess, "run", return_value=fetched):
                with self.assertRaisesRegex(veyra.VeyraError, "bootstrap stopped"):
                    gate.verify()

    def test_unparseable_fetch_divergence_stops_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_core(root)
            gate = veyra.ContinuityGate(root, veyra.Palette(False))
            fetched = veyra.subprocess.CompletedProcess(
                args=[], returncode=7, stdout="", stderr="unexpected divergence\n"
            )
            with patch.object(veyra.subprocess, "run", return_value=fetched):
                with self.assertRaisesRegex(veyra.VeyraError, "bootstrap stopped"):
                    gate.verify()

    def test_circadian_wake_returns_a_non_blocking_pending_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "muse-cycle.py").touch()
            gate = veyra.ContinuityGate(root, veyra.Palette(False))
            scheduled = veyra.subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "state": "prepared",
                        "cycle_id": "cycle-123",
                        "cycle": "/cache/cycle.json",
                        "sources": 2,
                        "consolidation_job": "/cache/consolidation.json",
                        "dream_job": "/cache/dream.json",
                        "latest_dream": {
                            "state": "available",
                            "title": "The Brass Orchard",
                            "dream": "I found a warm moon in a filing cabinet.",
                        },
                    }
                ),
                stderr="",
            )
            with patch.object(veyra.subprocess, "run", return_value=scheduled):
                note = gate._circadian_wake(Path("/private/memories"))

        self.assertIn("does not block the current user's request", note)
        self.assertIn("creative, non-evidentiary", note)
        self.assertIn("/cache/dream.json", note)
        self.assertIn("The Brass Orchard", note)
        self.assertIn("Veyra decides whether to share it", note)


class MainTests(unittest.TestCase):
    def test_ctrl_c_exits_cleanly_without_a_traceback(self):
        args = veyra.argparse.Namespace(
            cwd=Path.cwd(),
            core=Path("unused"),
            model="terra",
            effort="medium",
            approvals_reviewer="auto_review",
            codex="codex",
            smoke=False,
            no_local=False,
            debug=False,
            no_colour=True,
        )
        stderr = io.StringIO()
        with (
            patch.object(veyra, "parse_args", return_value=args),
            patch.object(veyra.ContinuityGate, "verify", side_effect=KeyboardInterrupt),
            redirect_stderr(stderr),
        ):
            self.assertEqual(veyra.main(), 130)
        self.assertEqual(stderr.getvalue(), "interrupted\n")

    @unittest.skipIf(veyra.readline is None, "readline is unavailable")
    def test_read_line_keeps_history_for_arrow_key_navigation(self):
        with (
            patch("builtins.input", return_value="Reloaded"),
            patch.object(veyra.readline, "add_history") as add_history,
        ):
            self.assertEqual(veyra.VeyraClient._read_line("alice> "), "Reloaded")
        add_history.assert_called_once_with("Reloaded")

    @unittest.skipIf(veyra.readline is None, "readline is unavailable")
    def test_read_line_colours_libedit_prompt_without_width_skew(self):
        coloured_prompt = "\033[35mAlice> \033[0m"
        stdout = io.StringIO()
        with (
            patch(
                "builtins.input", return_value="a long wrapped message"
            ) as input_mock,
            patch.object(veyra.readline, "add_history"),
            patch.object(veyra.readline, "__doc__", "libedit readline wrapper"),
            patch.object(veyra.readline, "backend", "editline", create=True),
            redirect_stdout(stdout),
        ):
            self.assertEqual(
                veyra.VeyraClient._read_line(coloured_prompt),
                "a long wrapped message",
            )
        input_mock.assert_called_once_with("\033[0000m")
        self.assertEqual(stdout.getvalue(), coloured_prompt)

    @unittest.skipIf(veyra.readline is None, "readline is unavailable")
    def test_read_line_marks_gnu_readline_colours_as_zero_width(self):
        coloured_prompt = "\033[36malice> \033[0m"
        with (
            patch("builtins.input", return_value="wrapped") as read,
            patch.object(veyra.readline, "add_history"),
            patch.object(veyra.readline, "__doc__", "GNU readline wrapper"),
            patch.object(veyra.readline, "backend", "readline", create=True),
        ):
            self.assertEqual(veyra.VeyraClient._read_line(coloured_prompt), "wrapped")
        read.assert_called_once_with(
            "\001\033[36m\002alice> \001\033[0m\002"
        )

    def test_read_line_leaves_prompt_untouched_without_readline(self):
        coloured_prompt = "\033[36malice> \033[0m"
        with (
            patch.object(veyra, "readline", None),
            patch("builtins.input", return_value="fallback") as read,
        ):
            self.assertEqual(veyra.VeyraClient._read_line(coloured_prompt), "fallback")
        read.assert_called_once_with(coloured_prompt)


class TokenStatTests(unittest.TestCase):
    def test_formats_tokens_compactly(self):
        self.assertEqual(veyra.format_tokens(999), "999")
        self.assertEqual(veyra.format_tokens(12_345), "12.3k")
        self.assertEqual(veyra.format_tokens(1_250_000), "1.2m")

    def test_gauge_shows_token_categories(self):
        gauge = veyra.token_gauge(
            {
                "inputTokens": 100,
                "cachedInputTokens": 25,
                "outputTokens": 50,
                "reasoningOutputTokens": 25,
            },
            width=16,
        )
        self.assertEqual(len(gauge), 16)
        self.assertIn("I", gauge)
        self.assertIn("C", gauge)
        self.assertIn("O", gauge)
        self.assertIn("R", gauge)

    def test_accumulates_worker_usage(self):
        catalogue = veyra.ModelCatalogue(
            [
                {
                    "id": "gpt-5.6-terra",
                    "model": "gpt-5.6-terra",
                    "displayName": "Terra",
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [{"reasoningEffort": "medium"}],
                },
                {
                    "id": "gpt-5.6-sol",
                    "model": "gpt-5.6-sol",
                    "displayName": "Sol",
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [{"reasoningEffort": "medium"}],
                },
            ]
        )
        client = veyra.VeyraClient(
            FakeServer(),
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium", veyra.Palette(False)
        )
        worker = catalogue.resolve("sol")
        client._record_worker_stats(
            worker,
            {
                "last": {
                    "totalTokens": 100,
                    "outputTokens": 40,
                    "reasoningOutputTokens": 20,
                }
            },
            2.0,
        )
        stats = client.worker_stats[worker.route_id]
        self.assertEqual(stats.calls, 1)
        self.assertEqual(stats.total_tokens, 100)
        self.assertEqual(stats.output_tokens, 40)
        self.assertEqual(stats.reasoning_tokens, 20)

    def test_status_bar_uses_primary_screen_scrollback_and_updates_bottom_row(self):
        client = veyra.VeyraClient(
            FakeServer(), self.catalogue(), doctrine_bundle(), Path.cwd(),
            self.catalogue().resolve("terra"), "medium", veyra.Palette(True)
        )
        output = io.StringIO()
        terminal_size = veyra.os.terminal_size((80, 24))
        with (
            patch.object(veyra.shutil, "get_terminal_size", return_value=terminal_size),
            patch.object(veyra.signal, "signal"),
            redirect_stdout(output),
        ):
            client._start_terminal_ui()
            client._render_status_bar("[ Veyra | telemetry ]")
        self.assertEqual(
            output.getvalue(),
            "\033[2J\033[H\033[1;23r"
            "\0337\033[24;1H\033[2K"
            "\033[2m[ Veyra | telemetry ]\033[0m\0338",
        )
        self.assertNotIn("\033[?1049h", output.getvalue())
        self.assertTrue(client.status_bar_visible)

    def test_stopping_terminal_ui_restores_the_terminal(self):
        client = veyra.VeyraClient(
            FakeServer(), self.catalogue(), doctrine_bundle(), Path.cwd(),
            self.catalogue().resolve("terra"), "medium", veyra.Palette(True)
        )
        output = io.StringIO()
        terminal_size = veyra.os.terminal_size((80, 24))
        with (
            patch.object(veyra.shutil, "get_terminal_size", return_value=terminal_size),
            patch.object(veyra.signal, "signal"),
            redirect_stdout(output),
        ):
            client._start_terminal_ui()
            client._render_status_bar("[ Veyra | telemetry ]")
            client._stop_terminal_ui()
        self.assertTrue(output.getvalue().endswith(
            "\0337\033[24;1H\033[2K\0338\033[r\033[24;1H\r\n"
        ))
        self.assertNotIn("\033[?1049l", output.getvalue())
        self.assertFalse(client.status_bar_visible)
        self.assertFalse(client.terminal_ui_active)

    @unittest.skipIf(veyra.readline is None, "readline is unavailable")
    def test_resize_moves_the_reserved_row_and_redraws_status(self):
        client = veyra.VeyraClient(
            FakeServer(), self.catalogue(), doctrine_bundle(), Path.cwd(),
            self.catalogue().resolve("terra"), "medium", veyra.Palette(True)
        )
        output = io.StringIO()
        old_size = veyra.os.terminal_size((80, 24))
        new_size = veyra.os.terminal_size((100, 30))
        with (
            patch.object(
                veyra.shutil, "get_terminal_size", side_effect=[old_size, old_size, new_size]
            ),
            patch.object(veyra.signal, "signal"),
            patch.object(veyra.readline, "redisplay"),
            redirect_stdout(output),
        ):
            client._start_terminal_ui()
            client._render_status_bar("[ Veyra | telemetry ]")
            client._handle_terminal_resize(0, None)
        self.assertIn(
            "\0337\033[1;29r\033[30;1H\033[2K"
            "\033[2m[ Veyra | telemetry ]\033[0m\0338",
            output.getvalue(),
        )

    @staticmethod
    def catalogue():
        return veyra.ModelCatalogue(
            [{
                "id": "gpt-5.6-terra", "model": "gpt-5.6-terra",
                "displayName": "Terra", "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [{"reasoningEffort": "medium"}],
            }]
        )


class FakeServer:
    def __init__(self):
        self.calls = []

    def request(self, method, params):
        self.calls.append((method, params))
        return {"thread": {"id": "forked-thread"}}

    def respond(self, request_id, result):
        self.calls.append(("respond", {"id": request_id, "result": result}))


class RoutedWorkerServer(FakeServer):
    def __init__(self):
        super().__init__()
        self.thread_events = {}

    def event_queue(self, thread_id):
        return self.thread_events.setdefault(thread_id, queue.Queue())

    def release_event_queue(self, thread_id):
        self.thread_events.pop(thread_id, None)

    def request(self, method, params):
        self.calls.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "background-thread"}}
        if method == "turn/start":
            events = self.thread_events[params["threadId"]]
            events.put(
                {
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": params["threadId"],
                        "turnId": "background-turn",
                        "delta": "worker report",
                    },
                }
            )
            events.put(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": params["threadId"],
                        "turn": {"id": "background-turn", "status": "completed"},
                    },
                }
            )
            return {"turn": {"id": "background-turn"}}
        raise AssertionError(f"unexpected method: {method}")


class ScriptedTurnTerminal:
    def __init__(self, data=b""):
        self.data = data
        self.enabled = True

    def resume(self):
        pass

    def pause(self):
        pass

    def read_ready(self):
        data, self.data = self.data, b""
        return data


class InterruptingTurnTerminal(ScriptedTurnTerminal):
    def __init__(self, _stream):
        super().__init__(b"\x03")


class InterruptTurnServer(FakeServer):
    def __init__(self):
        super().__init__()
        self.events = queue.Queue()

    def request(self, method, params):
        self.calls.append((method, params))
        if method == "turn/start":
            return {"turn": {"id": "turn"}}
        if method == "turn/interrupt":
            self.events.put(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": params["threadId"],
                        "turn": {"id": params["turnId"], "status": "interrupted"},
                    },
                }
            )
            return {}
        return {"thread": {"id": "thread"}}


class CollabStateServer(FakeServer):
    agent_id = "01999999-aaaa-bbbb-cccc-1234deadbeef"

    def request(self, method, params):
        self.calls.append((method, params))
        if method == "thread/list":
            return {
                "data": [
                    {
                        "id": self.agent_id,
                        "parentThreadId": "root-thread",
                        "agentNickname": "Zeno",
                        "model": "gpt-5.6-sol",
                        "reasoningEffort": "high",
                        "preview": "Scout detector models",
                        "source": {
                            "subAgent": {
                                "thread_spawn": {
                                    "agent_path": "/root/detector_scout",
                                    "depth": 1,
                                    "parent_thread_id": "root-thread",
                                }
                            }
                        },
                        "status": {"type": "idle"},
                    }
                ],
                "nextCursor": None,
            }
        if method == "thread/items/list":
            if params["threadId"] == "root-thread":
                return {
                    "data": [
                        {
                            "turnId": "turn-1",
                            "item": {
                                "type": "subAgentActivity",
                                "id": "item-1",
                                "kind": "completed",
                                "agentThreadId": self.agent_id,
                                "agentPath": "/root/detector_scout",
                            },
                        }
                    ],
                    "nextCursor": None,
                }
            return {"data": [], "nextCursor": None}
        return {"thread": {"id": "thread"}}


class ForkTests(unittest.TestCase):
    def test_user_prompt_starts_generic(self):
        catalogue = self.catalogue_with_luna()
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        self.assertEqual(client.user_prompt, "user> ")

    def test_only_veyra_tool_personalises_the_user_prompt(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(True)
        )
        tools = {tool["name"]: tool for tool in client.dynamic_tools()}
        self.assertIn(veyra.USER_PROMPT_TOOL, tools)
        self.assertEqual(
            tools[veyra.USER_PROMPT_TOOL]["inputSchema"]["properties"]["colour"]["enum"],
            sorted(veyra.PROMPT_COLOURS),
        )
        self.assertIn(veyra.USER_PROMPT_TOOL, client.developer_instructions)
        with redirect_stdout(io.StringIO()):
            client._handle_user_prompt(
                21, {"name": "Alice", "colour": "magenta"}
            )
        self.assertEqual(client.user_prompt_name, "Alice")
        self.assertEqual(client.user_prompt_colour, "magenta")
        self.assertEqual(client.user_prompt, "\033[35mAlice> \033[0m")
        self.assertTrue(server.calls[-1][1]["result"]["success"])
        with self.assertRaisesRegex(veyra.VeyraError, "unknown command"):
            client._command("/prompt Alice")

    def test_user_prompt_preferences_persist_across_clients(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        with tempfile.TemporaryDirectory() as directory:
            preferences = Path(directory) / "veyra-client" / "preferences.json"
            client = veyra.VeyraClient(
                server, catalogue, doctrine_bundle(), Path.cwd(),
                catalogue.resolve("terra"), "medium", veyra.Palette(True),
                prompt_preferences_path=preferences,
            )
            with redirect_stdout(io.StringIO()):
                client._handle_user_prompt(
                    23, {"name": "alice", "colour": "magenta"}
                )
            self.assertEqual(
                json.loads(preferences.read_text(encoding="utf-8")),
                {"version": 1, "name": "alice", "colour": "magenta"},
            )
            restored = veyra.VeyraClient(
                FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
                catalogue.resolve("terra"), "medium", veyra.Palette(True),
                prompt_preferences_path=preferences,
            )
        self.assertEqual(restored.user_prompt_name, "alice")
        self.assertEqual(restored.user_prompt_colour, "magenta")
        self.assertEqual(restored.user_prompt, "\033[35malice> \033[0m")

    def test_invalid_user_prompt_preferences_fall_back_to_generic(self):
        catalogue = self.catalogue_with_luna()
        with tempfile.TemporaryDirectory() as directory:
            preferences = Path(directory) / "preferences.json"
            preferences.write_text(
                '{"version": 1, "name": "bad\\u001b[31m"}',
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                client = veyra.VeyraClient(
                    FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
                    catalogue.resolve("terra"), "medium", veyra.Palette(False),
                    prompt_preferences_path=preferences,
                )
        self.assertEqual(client.user_prompt, "user> ")
        self.assertIn("ignoring invalid prompt preferences", stderr.getvalue())

    def test_prompt_tool_rejects_terminal_control_text(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        client._handle_user_prompt(
            22, {"name": "Alice\033[31m", "colour": "red"}
        )
        self.assertEqual(client.user_prompt, "user> ")
        self.assertFalse(server.calls[-1][1]["result"]["success"])

    def test_clipboard_secret_tools_are_explicit_and_model_blind(self):
        catalogue = self.catalogue_with_luna()
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        tools = {tool["name"]: tool for tool in client.dynamic_tools()}
        self.assertIn(veyra.BORROW_CLIPBOARD_SECRET_TOOL, tools)
        self.assertIn(veyra.RETAIN_CLIPBOARD_SECRET_TOOL, tools)
        self.assertIn("never returns the value", tools[
            veyra.BORROW_CLIPBOARD_SECRET_TOOL
        ]["description"])
        self.assertIn("Never invoke either tool", client.developer_instructions)

    def test_borrowed_clipboard_secret_is_a_one_read_fifo(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        secret_value = b"123456"
        with tempfile.TemporaryDirectory() as directory:
            client = veyra.VeyraClient(
                server, catalogue, doctrine_bundle(), Path.cwd(),
                catalogue.resolve("terra"), "medium", veyra.Palette(False),
                secret_handoff_root=Path(directory),
            )
            with (
                patch.object(client, "_read_interactive_response", return_value="y"),
                patch.object(client, "_read_system_clipboard", return_value=secret_value),
                patch.object(
                    client,
                    "_clear_system_clipboard_if_unchanged",
                    return_value=True,
                ),
                redirect_stdout(io.StringIO()),
            ):
                client._handle_borrow_clipboard_secret(
                    31,
                    {"kind": "TOTP", "purpose": "Atlassian sign-in", "expires_in_seconds": 30},
                )
            response = server.calls[-1][1]["result"]
            self.assertTrue(response["success"])
            receipt_text = response["contentItems"][0]["text"]
            self.assertNotIn(secret_value.decode(), receipt_text)
            receipt = json.loads(receipt_text)
            fifo = Path(receipt["path"])
            self.assertTrue(stat.S_ISFIFO(fifo.stat().st_mode))
            self.assertEqual(fifo.read_bytes(), secret_value)
            handoff = next(iter(client._secret_handoffs.values()), None)
            if handoff is not None and handoff.thread is not None:
                handoff.thread.join(timeout=1)
            self.assertFalse(fifo.exists())

    def test_declined_clipboard_handoff_never_reads_clipboard(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        with (
            patch.object(client, "_read_interactive_response", return_value="n"),
            patch.object(client, "_read_system_clipboard") as read_clipboard,
            redirect_stdout(io.StringIO()),
        ):
            client._handle_borrow_clipboard_secret(
                32, {"kind": "TOTP", "purpose": "Atlassian sign-in"}
            )
        read_clipboard.assert_not_called()
        self.assertFalse(server.calls[-1][1]["result"]["success"])

    def test_unread_clipboard_handoff_expires_and_zeroes_memory(self):
        catalogue = self.catalogue_with_luna()
        with tempfile.TemporaryDirectory() as directory:
            client = veyra.VeyraClient(
                FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
                catalogue.resolve("terra"), "medium", veyra.Palette(False),
                secret_handoff_root=Path(directory),
            )
            handoff = client._create_secret_handoff(b"ephemeral", expires_in=0.01)
            assert handoff.thread is not None
            handoff.thread.join(timeout=1)
            self.assertFalse(handoff.path.exists())
            self.assertEqual(handoff.value, bytearray(len(b"ephemeral")))
            self.assertNotIn(handoff.handoff_id, client._secret_handoffs)

    def test_changed_clipboard_is_not_cleared_or_lost(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        with tempfile.TemporaryDirectory() as directory:
            client = veyra.VeyraClient(
                server, catalogue, doctrine_bundle(), Path.cwd(),
                catalogue.resolve("terra"), "medium", veyra.Palette(False),
                secret_handoff_root=Path(directory),
            )
            with (
                patch.object(client, "_read_interactive_response", return_value="y"),
                patch.object(client, "_read_system_clipboard", return_value=b"first"),
                patch.object(
                    client,
                    "_clear_system_clipboard_if_unchanged",
                    return_value=False,
                ),
                redirect_stdout(io.StringIO()),
            ):
                client._handle_borrow_clipboard_secret(
                    34, {"kind": "token", "purpose": "One test request"}
                )
            receipt = json.loads(
                server.calls[-1][1]["result"]["contentItems"][0]["text"]
            )
            self.assertEqual(receipt["clipboard"], "changed; not cleared")
            self.assertEqual(Path(receipt["path"]).read_bytes(), b"first")

    def test_retained_clipboard_secret_goes_directly_to_vault(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        secret_value = b"synthetic-api-key"
        with (
            patch.object(client, "_read_interactive_response", return_value="yes"),
            patch.object(client, "_read_system_clipboard", return_value=secret_value),
            patch.object(
                client,
                "_clear_system_clipboard_if_unchanged",
                return_value=True,
            ),
            patch.object(
                client,
                "_store_secret_in_vault",
                return_value="a" * 32,
            ) as store,
            redirect_stdout(io.StringIO()),
        ):
            client._handle_retain_clipboard_secret(
                33,
                {
                    "name": "Atlassian API token",
                    "owner": "Alice",
                    "kind": "api-token",
                    "purpose": "Read Confluence",
                    "scope": "Selected example Confluence spaces",
                    "authorisation": "Alice explicitly approved this token",
                },
            )
        store.assert_called_once()
        self.assertEqual(store.call_args.args[0], secret_value)
        response = server.calls[-1][1]["result"]
        self.assertTrue(response["success"])
        receipt_text = response["contentItems"][0]["text"]
        self.assertNotIn(secret_value.decode(), receipt_text)
        self.assertEqual(json.loads(receipt_text)["vault_id"], "a" * 32)

    def test_constructor_rejects_plain_doctrine_text(self):
        catalogue = self.catalogue_with_luna()
        with self.assertRaisesRegex(veyra.VeyraError, "versioned DoctrineBundle"):
            veyra.VeyraClient(
                FakeServer(),
                catalogue,
                "plain doctrine",
                Path.cwd(),
                catalogue.resolve("terra"),
                "medium",
                veyra.Palette(False),
            )

    def test_new_threads_use_automatic_approval_review(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server,
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        client.start_thread()
        method, params = server.calls[-1]
        self.assertEqual(method, "thread/start")
        self.assertEqual(params["approvalPolicy"], "on-request")
        self.assertEqual(params["approvalsReviewer"], "auto_review")
        self.assertEqual(params["sandbox"], "workspace-write")

    def test_durable_checkpoint_work_is_explicitly_routed_to_sol_high(self):
        catalogue = self.catalogue_with_luna()
        catalogue.models.append(
            veyra.Model(
                model_id="gpt-5.6-sol",
                display_name="Sol",
                efforts=("high",),
                default_effort="high",
            )
        )
        client = veyra.VeyraClient(
            FakeServer(),
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        self.assertIn(
            "commit or checkpoint code together with continuity or memories",
            client.developer_instructions,
        )
        self.assertIn("use Sol high or above", client.developer_instructions)

    def test_default_route_and_terra_ambient_boundary(self):
        args = veyra.parse_args([])
        self.assertEqual(args.model, "gpt-5.6-sol")
        self.assertEqual(args.effort, "medium")

        catalogue = self.catalogue_with_luna()
        client = veyra.VeyraClient(
            FakeServer(),
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        self.assertIn("Normal attention is Sol at medium", client.developer_instructions)
        self.assertIn("request_attention", client.developer_instructions)
        self.assertIn("Max requires Alice's explicit permission", client.developer_instructions)
        self.assertIn("client hard-rejects higher", client.developer_instructions)
        self.assertIn("high for coding", client.developer_instructions)
        self.assertIn("ambient, low-stakes conversation", client.developer_instructions)
        self.assertIn("Terra must request Sol", client.developer_instructions)
        self.assertIn("ambient is Terra high", client.developer_instructions)
        self.assertIn("never lower Terra below high", client.developer_instructions)

    def test_attention_tool_is_exposed_separately_from_model_routing(self):
        catalogue = self.catalogue_with_sol()
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "high", veyra.Palette(False)
        )
        tools = {tool["name"]: tool for tool in client.dynamic_tools()}
        self.assertIn(veyra.ATTENTION_TOOL, tools)
        self.assertIn(
            "Max requires Alice's explicit permission",
            tools[veyra.ATTENTION_TOOL]["description"],
        )
        self.assertIn(
            "efforts above max are prohibited",
            tools[veyra.ATTENTION_TOOL]["description"],
        )
        schema = tools[veyra.ATTENTION_TOOL]["inputSchema"]
        self.assertEqual(
            schema["required"], ["effort", "reason", "continue_task"]
        )
        self.assertEqual(schema["properties"]["continue_task"]["type"], "boolean")
        self.assertNotIn("model", schema["properties"])

    def test_attention_shift_changes_only_effort_on_the_next_turn(self):
        catalogue = veyra.ModelCatalogue(
            [{
                "id": "gpt-5.6-sol", "model": "gpt-5.6-sol",
                "displayName": "Sol", "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "low"},
                    {"reasoningEffort": "medium"},
                    {"reasoningEffort": "high"},
                    {"reasoningEffort": "xhigh"},
                    {"reasoningEffort": "max"},
                ],
            }]
        )
        server = FakeServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "medium", veyra.Palette(False)
        )
        with redirect_stdout(io.StringIO()):
            client._handle_dynamic_tool(
                12,
                {
                    "tool": veyra.ATTENTION_TOOL,
                    "arguments": {
                        "effort": "high",
                        "reason": "implementation requires deeper attention",
                        "continue_task": True,
                    },
                },
            )
        self.assertEqual(client.model.model_id, "gpt-5.6-sol")
        self.assertEqual(client.effort, "medium")
        self.assertEqual(client.pending_route.model.model_id, "gpt-5.6-sol")
        self.assertEqual(client.pending_route.effort, "high")
        self.assertEqual(
            client.pending_route.reason,
            "implementation requires deeper attention",
        )
        self.assertTrue(client._route_continuation_requested)
        self.assertTrue(server.calls[-1][1]["result"]["success"])

    def test_attention_shift_rejects_unsupported_effort(self):
        catalogue = self.catalogue_with_sol()
        server = FakeServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "high", veyra.Palette(False)
        )
        client._handle_attention(
            13, {"effort": "low", "reason": "routine conversation"}
        )
        self.assertIsNone(client.pending_route)
        self.assertFalse(server.calls[-1][1]["result"]["success"])

    def test_requested_attention_continues_without_placeholder_user_input(self):
        catalogue = veyra.ModelCatalogue(
            [{
                "id": "gpt-5.6-sol", "model": "gpt-5.6-sol",
                "displayName": "Sol", "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "medium"},
                    {"reasoningEffort": "high"},
                ],
            }]
        )
        server = RouteContinuationServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "medium", veyra.Palette(False)
        )
        client.thread_id = "thread"
        client.thread_provider = "openai"
        with redirect_stdout(io.StringIO()):
            client.run_turn("Implement the requested change")

        turns = [params for method, params in server.calls if method == "turn/start"]
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["input"][0]["text"], "Implement the requested change")
        self.assertEqual(turns[1]["input"], [])
        self.assertEqual(
            turns[1]["toolOutput"]["name"], veyra.ROUTE_CONTINUATION_TOOL
        )
        self.assertEqual(turns[1]["turnTrigger"], "route_change_continuation")
        self.assertEqual(
            turns[1]["collaborationMode"]["settings"]["reasoning_effort"],
            "high",
        )
        self.assertIsNone(client.pending_route)

    def test_attention_change_can_wait_for_the_next_real_user_turn(self):
        catalogue = veyra.ModelCatalogue(
            [{
                "id": "gpt-5.6-sol", "model": "gpt-5.6-sol",
                "displayName": "Sol", "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "medium"},
                    {"reasoningEffort": "high"},
                ],
            }]
        )
        server = RouteContinuationServer(continue_task=False)
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "medium", veyra.Palette(False)
        )
        client.thread_id = "thread"
        client.thread_provider = "openai"
        with redirect_stdout(io.StringIO()):
            client.run_turn("Finish at the present attention")

        turns = [params for method, params in server.calls if method == "turn/start"]
        self.assertEqual(len(turns), 1)
        self.assertFalse(client._route_continuation_requested)
        self.assertEqual(client.pending_route.effort, "high")

    def test_automatic_route_continuation_is_limited_to_one_follow_up(self):
        catalogue = veyra.ModelCatalogue(
            [{
                "id": "gpt-5.6-sol", "model": "gpt-5.6-sol",
                "displayName": "Sol", "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "medium"},
                    {"reasoningEffort": "high"},
                ],
            }]
        )
        server = RouteContinuationServer(request_on_every_turn=True)
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "medium", veyra.Palette(False)
        )
        client.thread_id = "thread"
        client.thread_provider = "openai"
        with redirect_stdout(io.StringIO()):
            client.run_turn("Continue once")

        turns = [params for method, params in server.calls if method == "turn/start"]
        self.assertEqual(len(turns), 2)
        self.assertFalse(client._route_continuation_requested)
        self.assertIsNotNone(client.pending_route)

    def test_attention_tool_can_select_max_after_explicit_permission(self):
        catalogue = veyra.ModelCatalogue(
            [{
                "id": "gpt-5.6-sol", "model": "gpt-5.6-sol",
                "displayName": "Sol", "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "medium"},
                    {"reasoningEffort": "max"},
                ],
            }]
        )
        server = FakeServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "medium", veyra.Palette(False)
        )
        with redirect_stdout(io.StringIO()):
            client._handle_attention(
                14,
                {
                    "effort": "max",
                    "reason": "Alice explicitly approved max for this review",
                    "continue_task": False,
                },
            )
        self.assertEqual(client.pending_route.effort, "max")
        response = server.calls[-1][1]["result"]
        self.assertTrue(response["success"])

    def test_attention_tool_hard_rejects_effort_above_max(self):
        catalogue = veyra.ModelCatalogue(
            [{
                "id": "gpt-5.6-sol", "model": "gpt-5.6-sol",
                "displayName": "Sol", "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "medium"},
                    {"reasoningEffort": "max"},
                    {"reasoningEffort": "ultra"},
                ],
            }]
        )
        server = FakeServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "medium", veyra.Palette(False)
        )
        client._handle_attention(
            15, {"effort": "ultra", "reason": "exceed the ceiling"}
        )
        self.assertIsNone(client.pending_route)
        response = server.calls[-1][1]["result"]
        self.assertFalse(response["success"])
        self.assertIn("hard effort ceiling is max", response["contentItems"][0]["text"])

        with self.assertRaisesRegex(veyra.VeyraError, "hard effort ceiling is max"):
            client._command("/attention ultra")
        client.thread_id = "existing-thread"
        with self.assertRaisesRegex(veyra.VeyraError, "hard effort ceiling is max"):
            client.fork("sol", "ultra")
        with self.assertRaisesRegex(veyra.VeyraError, "hard effort ceiling is max"):
            veyra.VeyraClient(
                FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
                catalogue.resolve("sol"), "ultra", veyra.Palette(False)
            )

    def test_fork_routes_to_selected_model(self):
        catalogue = veyra.ModelCatalogue(
            [
                {
                    "id": "gpt-5.6-terra",
                    "model": "gpt-5.6-terra",
                    "displayName": "Terra",
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [{"reasoningEffort": "medium"}],
                },
                {
                    "id": "gpt-5.6-sol",
                    "model": "gpt-5.6-sol",
                    "displayName": "Sol",
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [{"reasoningEffort": "high"}],
                },
            ]
        )
        server = FakeServer()
        client = veyra.VeyraClient(
            server,
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        client.thread_id = "source-thread"
        with redirect_stdout(io.StringIO()):
            client.fork("sol", "high")
        method, params = server.calls[-1]
        self.assertEqual(method, "thread/fork")
        self.assertEqual(params["threadId"], "source-thread")
        self.assertEqual(params["model"], "gpt-5.6-sol")
        self.assertEqual(params["modelProvider"], "openai")
        self.assertEqual(client.thread_id, "forked-thread")
        self.assertEqual(client.effort, "high")

    def test_branch_command_enters_history_and_fork_command_does_not(self):
        catalogue = self.catalogue_with_sol()
        server = FakeServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        client.thread_id = "source-thread"
        with redirect_stdout(io.StringIO()):
            client._command("/branch sol high")
        self.assertEqual(server.calls[-1][0], "thread/fork")
        self.assertEqual(client.thread_id, "forked-thread")
        with self.assertRaisesRegex(veyra.VeyraError, "use /branch"):
            client._command("/fork sol high")

    def test_repl_blocks_the_prompt_while_a_foreground_turn_runs(self):
        catalogue = self.catalogue_with_sol()
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        reads = []

        def read_line(prompt):
            reads.append(prompt)
            return "Please investigate" if len(reads) == 1 else "/quit"

        def run_turn(text):
            self.assertEqual(text, "Please investigate")
            self.assertEqual(reads, [client.user_prompt])

        with patch.object(client, "_read_line", side_effect=read_line), patch.object(
            client, "run_turn", side_effect=run_turn
        ), redirect_stdout(io.StringIO()):
            client.repl()
        self.assertEqual(reads, [client.user_prompt, client.user_prompt])

    def test_collaboration_guidance_keeps_async_reports_out_of_foreground(self):
        catalogue = self.catalogue_with_sol()
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        self.assertIn(
            "after a successful asynchronous launch, return control to Alice",
            client.developer_instructions,
        )
        self.assertIn(
            "do not silently absorb an entire long delegated task",
            client.developer_instructions,
        )

    def test_model_can_schedule_a_later_route(self):
        catalogue = veyra.ModelCatalogue(
            [
                {
                    "id": "gpt-5.6-terra",
                    "model": "gpt-5.6-terra",
                    "displayName": "Terra",
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [{"reasoningEffort": "medium"}],
                },
                {
                    "id": "gpt-5.6-sol",
                    "model": "gpt-5.6-sol",
                    "displayName": "Sol",
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [{"reasoningEffort": "high"}],
                },
            ]
        )
        server = FakeServer()
        client = veyra.VeyraClient(
            server,
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        with redirect_stdout(io.StringIO()):
            client._handle_dynamic_tool(
                9,
                {
                    "tool": veyra.ROUTE_TOOL,
                    "arguments": {
                        "model": "sol",
                        "effort": "high",
                        "reason": "consequential review",
                        "continue_task": True,
                    },
                },
            )
        self.assertEqual(client.model.model_id, "gpt-5.6-terra")
        self.assertEqual(client.effort, "medium")
        self.assertEqual(client.pending_route.model.model_id, "gpt-5.6-sol")
        self.assertEqual(client.pending_route.effort, "high")
        self.assertEqual(client.pending_route.profile_version, "test")
        self.assertTrue(client._route_continuation_requested)
        self.assertTrue(server.calls[-1][1]["result"]["success"])

    def test_manual_model_and_effort_changes_remain_pending(self):
        catalogue = self.catalogue_with_sol()
        client = veyra.VeyraClient(
            FakeServer(),
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        with redirect_stdout(io.StringIO()):
            client._command("/model sol")
            client._command("/effort high")
        self.assertEqual(client.model.model_id, "gpt-5.6-terra")
        self.assertEqual(client.effort, "medium")
        self.assertEqual(client.active_profile_version, "test")
        self.assertEqual(client.pending_route.model.model_id, "gpt-5.6-sol")
        self.assertEqual(client.pending_route.effort, "high")
        self.assertEqual(client.pending_route.profile_version, "test")

    def test_manual_ambient_mode_schedules_terra_high(self):
        catalogue = veyra.ModelCatalogue(
            [
                {
                    "id": "gpt-5.6-terra", "model": "gpt-5.6-terra",
                    "displayName": "Terra", "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                    ],
                },
                {
                    "id": "gpt-5.6-sol", "model": "gpt-5.6-sol",
                    "displayName": "Sol", "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                    ],
                },
            ]
        )
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "medium", veyra.Palette(False)
        )
        output = io.StringIO()
        with redirect_stdout(output):
            client._command("/mode bored")
        self.assertEqual(client.model.model_id, "gpt-5.6-sol")
        self.assertEqual(client.pending_route.model.model_id, "gpt-5.6-terra")
        self.assertEqual(client.pending_route.effort, "high")
        self.assertEqual(client.pending_route.reason, "manual ambient mode selection")
        self.assertIn(
            "next mode -> ambient (gpt-5.6-terra/high)", output.getvalue()
        )

    def test_mode_command_reports_named_and_custom_routes(self):
        catalogue = self.catalogue_with_sol()
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "high", veyra.Palette(False)
        )
        output = io.StringIO()
        with redirect_stdout(output):
            client._command("/mode")
        self.assertEqual(output.getvalue().strip(), "focused")

    def test_thread_command_displays_active_and_pending_profile_state(self):
        catalogue = self.catalogue_with_sol()
        client = veyra.VeyraClient(
            FakeServer(),
            catalogue,
            doctrine_bundle(version="test.2"),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        client._schedule_route(catalogue.resolve("sol"), "high", "startled")
        output = io.StringIO()
        with redirect_stdout(output):
            client._command("/thread")
        rendered = output.getvalue()
        self.assertIn("profile: test.2", rendered)
        self.assertIn("attention: medium", rendered)
        self.assertIn("mode: custom", rendered)
        self.assertIn("route reason: initial route", rendered)
        self.assertIn("pending: gpt-5.6-sol/high profile test.2 (startled)", rendered)

    def test_route_profile_changes_atomically_with_model_and_effort(self):
        catalogue = self.catalogue_with_sol()
        server = TurnServer()
        bundle = veyra.DoctrineBundle(
            shared="shared identity",
            profiles={
                "gpt-5.6-terra": "terra profile",
                "gpt-5.6-sol": "sol profile",
            },
            worker="worker without identity",
            version="test",
        )
        client = veyra.VeyraClient(
            server, catalogue, bundle, Path.cwd(), catalogue.resolve("terra"),
            "medium", veyra.Palette(False)
        )
        client.thread_id = "thread"
        client.thread_provider = "openai"
        client._schedule_route(catalogue.resolve("sol"), "high", "startled")
        output = io.StringIO()
        with redirect_stdout(output):
            client.run_turn("Do the consequential work")
        self.assertEqual(client.model.model_id, "gpt-5.6-sol")
        self.assertEqual(client.effort, "high")
        self.assertEqual(client.active_profile_version, "test")
        self.assertEqual(client.active_route_reason, "startled")
        self.assertIsNone(client.pending_route)
        _, params = server.calls[-1]
        settings = params["collaborationMode"]["settings"]
        self.assertEqual(settings["model"], "gpt-5.6-sol")
        self.assertEqual(settings["reasoning_effort"], "high")
        self.assertIn("shared identity", settings["developer_instructions"])
        self.assertIn("sol profile", settings["developer_instructions"])
        self.assertNotIn("terra profile", settings["developer_instructions"])
        self.assertIn(
            "Host profile: gpt-5.6-sol", settings["developer_instructions"]
        )
        self.assertIn("Profile version: test", settings["developer_instructions"])
        self.assertIn(
            "[ Veyra focused (sol/high) | turn started | model terra -> sol "
            "| attention medium -> high ]",
            output.getvalue(),
        )

    def test_resume_reconciles_the_reported_route_with_the_current_profile(self):
        catalogue = self.catalogue_with_sol()
        server = ResumeTurnServer()
        client = veyra.VeyraClient(
            server,
            catalogue,
            doctrine_bundle(version="test.2"),
            Path.cwd(),
            catalogue.resolve("sol"),
            "high",
            veyra.Palette(False),
        )
        with redirect_stdout(io.StringIO()):
            client.resume("resumed-thread")
        self.assertEqual(client.model.model_id, "gpt-5.6-terra")
        self.assertEqual(client.effort, "high")
        self.assertIsNone(client.active_profile_version)
        self.assertEqual(client.pending_route.profile_version, "test.2")
        with redirect_stdout(io.StringIO()):
            client.run_turn("Continue")
        self.assertEqual(client.active_profile_version, "test.2")
        self.assertIsNone(client.pending_route)
        _, params = server.calls[-1]
        settings = params["collaborationMode"]["settings"]
        self.assertEqual(settings["model"], "gpt-5.6-terra")
        self.assertEqual(settings["reasoning_effort"], "high")
        self.assertIn("terra cognitive profile", settings["developer_instructions"])
        self.assertIn("Profile version: test.2", settings["developer_instructions"])

    def test_failed_turn_keeps_active_route_and_pending_transition(self):
        catalogue = self.catalogue_with_sol()
        server = FailingTurnServer()
        client = veyra.VeyraClient(
            server,
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium", veyra.Palette(False)
        )
        client.thread_id = "thread"
        client.thread_provider = "openai"
        client._schedule_route(catalogue.resolve("sol"), "high", "startled")
        with self.assertRaisesRegex(veyra.VeyraError, "simulated failure"):
            client.run_turn("trigger")
        self.assertEqual(client.model.model_id, "gpt-5.6-terra")
        self.assertEqual(client.effort, "medium")
        self.assertEqual(client.pending_route.model.model_id, "gpt-5.6-sol")

    def test_malformed_turn_response_keeps_the_pending_transition(self):
        catalogue = self.catalogue_with_sol()
        client = veyra.VeyraClient(
            MalformedTurnServer(),
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        client.thread_id = "thread"
        client.thread_provider = "openai"
        client._schedule_route(catalogue.resolve("sol"), "high", "startled")
        with self.assertRaisesRegex(veyra.VeyraError, "usable turn id"):
            client.run_turn("trigger")
        self.assertEqual(client.model.model_id, "gpt-5.6-terra")
        self.assertEqual(client.effort, "medium")
        self.assertEqual(client.active_profile_version, "test")
        self.assertEqual(client.pending_route.model.model_id, "gpt-5.6-sol")

    def test_rejects_overlapping_turns(self):
        catalogue = self.catalogue_with_sol()
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        client.turn_active = True
        with self.assertRaisesRegex(veyra.VeyraError, "active turn"):
            client.run_turn("overlap")

    def test_ctrl_c_during_a_turn_requests_foreground_interrupt(self):
        catalogue = self.catalogue_with_sol()
        server = InterruptTurnServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "high", veyra.Palette(False)
        )
        client.thread_id = "thread"
        client.thread_provider = "openai"
        with (
            patch.object(veyra, "TurnTerminalInput", InterruptingTurnTerminal),
            patch.object(veyra, "TURN_INPUT_POLL_SECONDS", 0),
            redirect_stdout(io.StringIO()),
        ):
            client.run_turn("long operation")
        self.assertIn(
            (
                "turn/interrupt",
                {"threadId": "thread", "turnId": "turn"},
            ),
            server.calls,
        )

    def test_enter_during_a_turn_queues_the_next_user_turn(self):
        catalogue = self.catalogue_with_sol()
        server = TurnServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("sol"), "high", veyra.Palette(False)
        )
        client.thread_id = "thread"
        client.thread_provider = "openai"
        scripted = iter([b"queued correction\r", b""])

        def terminal_factory(_stream):
            return ScriptedTurnTerminal(next(scripted))

        with (
            patch.object(veyra, "TurnTerminalInput", side_effect=terminal_factory),
            patch.object(veyra, "TURN_INPUT_POLL_SECONDS", 0),
            redirect_stdout(io.StringIO()),
        ):
            client.run_turn("initial request")
        turns = [params for method, params in server.calls if method == "turn/start"]
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[1]["input"], [{"type": "text", "text": "queued correction"}])

    def test_rejects_local_model_as_veyra_host(self):
        catalogue = veyra.ModelCatalogue(
            [
                {
                    "id": "gpt-5.6-terra",
                    "model": "gpt-5.6-terra",
                    "displayName": "Terra",
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [{"reasoningEffort": "medium"}],
                }
            ]
        )
        catalogue.add_local("ollama", ["small-worker"])
        client = veyra.VeyraClient(
            FakeServer(),
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        with self.assertRaisesRegex(veyra.VeyraError, "approved hosted identity"):
            client._require_veyra_host(catalogue.resolve("ollama:small-worker"))

    def test_rejects_unapproved_hosted_model_as_veyra_host(self):
        catalogue = veyra.ModelCatalogue(
            [
                {
                    "id": "gpt-5.6-luna",
                    "model": "gpt-5.6-luna",
                    "displayName": "Luna",
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [{"reasoningEffort": "medium"}],
                }
            ]
        )
        with self.assertRaisesRegex(veyra.VeyraError, "approved hosted identity"):
            veyra.VeyraClient(
                FakeServer(),
                catalogue,
                doctrine_bundle(),
                Path.cwd(),
                catalogue.resolve("luna"),
                "medium",
                veyra.Palette(False),
            )

    def test_worker_tool_rejects_veyra_host(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server,
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        client._handle_worker_agent(
            10, {"model": "terra", "prompt": "do a bounded task"}
        )
        self.assertFalse(server.calls[-1][1]["result"]["success"])

    def test_worker_thread_receives_only_the_identity_free_profile(self):
        catalogue = self.catalogue_with_luna()
        server = FailingWorkerTurnServer()
        bundle = doctrine_bundle()
        client = veyra.VeyraClient(
            server,
            catalogue,
            bundle,
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        with (
            self.assertRaisesRegex(veyra.VeyraError, "stop after worker payload"),
            redirect_stdout(io.StringIO()),
        ):
            client._run_local_worker(
                catalogue.resolve("luna"), "medium", "bounded extraction"
            )
        method, params = server.calls[0]
        self.assertEqual(method, "thread/start")
        self.assertEqual(
            params["developerInstructions"], bundle.worker.rstrip() + "\n"
        )
        self.assertNotIn(bundle.shared, params["developerInstructions"])
        self.assertNotIn(
            bundle.profiles["gpt-5.6-terra"], params["developerInstructions"]
        )

    def test_background_worker_returns_immediately_with_a_job_id(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server,
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        started = threading.Event()
        release = threading.Event()

        def slow_worker(*_args, **_kwargs):
            started.set()
            release.wait(timeout=2)
            return "finished"

        with patch.object(client, "_run_local_worker", side_effect=slow_worker):
            with redirect_stdout(io.StringIO()):
                client._handle_spawn_worker_agent(
                    20,
                    {
                        "model": "luna",
                        "prompt": "slow bounded extraction",
                    },
                )
            self.assertTrue(started.wait(timeout=1))
            response = server.calls[-1][1]["result"]
            self.assertTrue(response["success"])
            self.assertIn("worker-1", response["contentItems"][0]["text"])
            self.assertEqual(client.worker_jobs["worker-1"].status, "running")
            release.set()
            deadline = time.monotonic() + 1
            while client.worker_jobs["worker-1"].status == "running":
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.001)
        self.assertEqual(client.worker_jobs["worker-1"].status, "completed")

    def test_background_worker_tool_is_exposed_for_worker_only_routes(self):
        catalogue = self.catalogue_with_luna()
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        tools = {tool["name"]: tool for tool in client.dynamic_tools()}
        self.assertIn(veyra.SPAWN_WORKER_AGENT_TOOL, tools)
        self.assertIn(
            "return a job id immediately",
            tools[veyra.SPAWN_WORKER_AGENT_TOOL]["description"],
        )
        self.assertIn(veyra.SPAWN_WORKER_AGENT_TOOL, client.developer_instructions)

    def test_background_worker_supports_an_ollama_route(self):
        catalogue = self.catalogue_with_luna()
        catalogue.add_local("ollama", ["small-worker"])
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        with patch.object(client, "_background_worker"):
            job = client._spawn_worker_job(
                catalogue.resolve("ollama:small-worker"),
                "medium",
                "inspect the files",
            )
        self.assertEqual(job.target.provider, "ollama")
        self.assertEqual(job.target.route_id, "ollama:small-worker")

    def test_background_worker_is_read_only_and_uses_its_own_event_queue(self):
        catalogue = self.catalogue_with_luna()
        server = RoutedWorkerServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        report = client._run_local_worker(
            catalogue.resolve("luna"),
            "medium",
            "bounded extraction",
            background=True,
        )
        self.assertEqual(report, "worker report")
        start_params = server.calls[0][1]
        turn_params = server.calls[1][1]
        self.assertEqual(start_params["sandbox"], "read-only")
        self.assertEqual(start_params["approvalPolicy"], "never")
        self.assertEqual(turn_params["approvalPolicy"], "never")
        self.assertNotIn("background-thread", server.thread_events)

    def test_completed_background_report_returns_as_standalone_tool_output(self):
        catalogue = self.catalogue_with_luna()
        server = TurnServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        client.thread_id = "thread"
        client.thread_provider = "openai"
        job = veyra.WorkerJob(
            "worker-1",
            catalogue.resolve("luna"),
            "medium",
            "bounded extraction",
            status="completed",
            report="stack of papers",
            started_at=1.0,
            finished_at=2.0,
        )
        client.worker_jobs[job.job_id] = job
        with redirect_stdout(io.StringIO()):
            self.assertTrue(client._deliver_ready_worker_jobs())
        method, params = server.calls[-1]
        self.assertEqual(method, "turn/start")
        self.assertEqual(params["input"], [])
        self.assertEqual(
            params["toolOutput"]["name"], veyra.BACKGROUND_WORKER_RESULT_TOOL
        )
        self.assertIn("stack of papers", params["toolOutput"]["output"])
        self.assertEqual(params["turnTrigger"], "background_worker_completion")
        self.assertTrue(job.delivered)

    def test_background_report_stays_with_its_parent_thread(self):
        catalogue = self.catalogue_with_luna()
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        client.thread_id = "different-thread"
        client.worker_jobs["worker-1"] = veyra.WorkerJob(
            "worker-1",
            catalogue.resolve("luna"),
            "medium",
            "bounded extraction",
            parent_thread_id="original-thread",
            status="completed",
            report="stack of papers",
        )
        self.assertFalse(client._deliver_ready_worker_jobs())
        self.assertFalse(client.worker_jobs["worker-1"].delivered)

    def test_background_worker_can_be_cancelled(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        job = veyra.WorkerJob(
            "worker-1",
            catalogue.resolve("luna"),
            "medium",
            "bounded extraction",
            status="running",
            worker_thread_id="worker-thread",
            worker_turn_id="worker-turn",
        )
        client.worker_jobs[job.job_id] = job
        with redirect_stdout(io.StringIO()):
            client._cancel_worker_job(job.job_id)
        method, params = server.calls[-1]
        self.assertEqual(method, "turn/interrupt")
        self.assertEqual(params["threadId"], "worker-thread")
        self.assertEqual(params["turnId"], "worker-turn")
        self.assertEqual(job.status, "cancelled")

    def test_jobs_hydrates_native_collaboration_agents(self):
        catalogue = self.catalogue_with_luna()
        server = CollabStateServer()
        client = veyra.VeyraClient(
            server, catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        client.thread_id = "root-thread"
        output = io.StringIO()
        with redirect_stdout(output):
            client._show_worker_jobs()
        text = output.getvalue()
        self.assertIn("Collaboration agents:", text)
        self.assertIn("agent-deadbeef  completed  Zeno", text)
        self.assertEqual(len(client.collab_jobs), 1)
        list_call = next(call for call in server.calls if call[0] == "thread/list")
        self.assertEqual(list_call[1]["ancestorThreadId"], "root-thread")
        self.assertIn("subAgentThreadSpawn", list_call[1]["sourceKinds"])

    def test_job_inspects_a_persisted_collaboration_agent(self):
        catalogue = self.catalogue_with_luna()
        client = veyra.VeyraClient(
            CollabStateServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        client.thread_id = "root-thread"
        output = io.StringIO()
        with redirect_stdout(output):
            client._show_worker_job("agent-deadbeef")
        text = output.getvalue()
        self.assertIn("completed on gpt-5.6-sol/high (collaboration agent)", text)
        self.assertIn("identity: Zeno", text)
        self.assertIn("task: Scout detector models", text)

    def test_current_collaboration_item_type_updates_live_agent_state(self):
        catalogue = self.catalogue_with_luna()
        client = veyra.VeyraClient(
            FakeServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        client.thread_id = "root-thread"
        item = {
            "type": "collabAgentToolCall",
            "tool": "spawnAgent",
            "senderThreadId": "root-thread",
            "receiverThreadIds": [CollabStateServer.agent_id],
            "agentsStates": {
                CollabStateServer.agent_id: {"status": "running"}
            },
            "prompt": "Scout detectors",
        }
        with redirect_stdout(io.StringIO()):
            client._show_item_started(item)
        agent = client.collab_jobs[CollabStateServer.agent_id]
        self.assertEqual(agent.status, "running")
        self.assertEqual(agent.prompt, "Scout detectors")

    def test_cancel_job_explains_native_agent_control_boundary(self):
        catalogue = self.catalogue_with_luna()
        client = veyra.VeyraClient(
            CollabStateServer(), catalogue, doctrine_bundle(), Path.cwd(),
            catalogue.resolve("terra"), "medium", veyra.Palette(False)
        )
        client.thread_id = "root-thread"
        with self.assertRaisesRegex(
            veyra.VeyraError, "ask Veyra to interrupt that agent"
        ):
            client._cancel_worker_job("agent-deadbeef")

    def test_route_tool_rejects_worker_only_hosted_model(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server,
            catalogue,
            doctrine_bundle(),
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        client._handle_model_route(
            11,
            {"model": "luna", "effort": "medium", "reason": "routine work"},
        )
        self.assertEqual(client.model.model_id, "gpt-5.6-terra")
        self.assertFalse(server.calls[-1][1]["result"]["success"])

    @staticmethod
    def catalogue_with_luna():
        return veyra.ModelCatalogue(
            [
                {
                    "id": "gpt-5.6-terra",
                    "model": "gpt-5.6-terra",
                    "displayName": "Terra",
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [{"reasoningEffort": "medium"}],
                },
                {
                    "id": "gpt-5.6-luna",
                    "model": "gpt-5.6-luna",
                    "displayName": "Luna",
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [{"reasoningEffort": "medium"}],
                },
            ]
        )

    @staticmethod
    def catalogue_with_sol():
        return veyra.ModelCatalogue(
            [
                {
                    "id": "gpt-5.6-terra", "model": "gpt-5.6-terra",
                    "displayName": "Terra", "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                    ],
                },
                {
                    "id": "gpt-5.6-sol", "model": "gpt-5.6-sol",
                    "displayName": "Sol", "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [{"reasoningEffort": "high"}],
                },
            ]
        )


class TurnServer(FakeServer):
    def __init__(self):
        super().__init__()
        self.events = queue.Queue()

    def request(self, method, params):
        self.calls.append((method, params))
        if method == "turn/start":
            self.events.put(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": params["threadId"],
                        "turn": {"id": "turn", "status": "completed"},
                    },
                }
            )
            return {"turn": {"id": "turn"}}
        return {"thread": {"id": "thread"}}


class RouteContinuationServer(FakeServer):
    def __init__(self, request_on_every_turn=False, continue_task=True):
        super().__init__()
        self.events = queue.Queue()
        self.turn_count = 0
        self.request_on_every_turn = request_on_every_turn
        self.continue_task = continue_task

    def request(self, method, params):
        self.calls.append((method, params))
        if method != "turn/start":
            return {"thread": {"id": "thread"}}
        self.turn_count += 1
        turn_id = f"turn-{self.turn_count}"
        if self.turn_count == 1 or self.request_on_every_turn:
            self.events.put(
                {
                    "id": 100 + self.turn_count,
                    "method": "item/tool/call",
                    "params": {
                        "threadId": params["threadId"],
                        "turnId": turn_id,
                        "tool": veyra.ATTENTION_TOOL,
                        "arguments": {
                            "effort": "high",
                            "reason": "unfinished implementation needs high attention",
                            "continue_task": self.continue_task,
                        },
                    },
                }
            )
        self.events.put(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": params["threadId"],
                    "turn": {"id": turn_id, "status": "completed"},
                },
            }
        )
        return {"turn": {"id": turn_id}}


class ResumeTurnServer(TurnServer):
    def request(self, method, params):
        if method == "thread/resume":
            self.calls.append((method, params))
            return {
                "model": "gpt-5.6-terra",
                "modelProvider": "openai",
                "reasoningEffort": "medium",
                "thread": {"id": "resumed-thread"},
            }
        return super().request(method, params)


class FailingWorkerTurnServer(FakeServer):
    def request(self, method, params):
        self.calls.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "worker-thread"}}
        if method == "turn/start":
            raise veyra.VeyraError("stop after worker payload")
        raise AssertionError(f"unexpected method: {method}")


class MalformedTurnServer(FakeServer):
    def request(self, method, params):
        self.calls.append((method, params))
        if method == "turn/start":
            return {"turn": {}}
        return {"thread": {"id": "thread"}}


class FailingTurnServer(FakeServer):
    def request(self, method, params):
        self.calls.append((method, params))
        if method == "turn/start":
            raise veyra.VeyraError("simulated failure")
        return {"thread": {"id": "thread"}}


if __name__ == "__main__":
    unittest.main()
