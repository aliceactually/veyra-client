import importlib.util
import io
import json
import queue
import sys
import tempfile
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

    def test_adds_and_resolves_local_provider_route(self):
        self.catalogue.add_local("ollama", ["veyra-intel-coder:qwen3-coder-32k"])
        model = self.catalogue.resolve(
            "ollama:veyra-intel-coder:qwen3-coder-32k"
        )
        self.assertTrue(model.local)
        self.assertEqual(model.provider, "ollama")

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
    def state_result():
        return veyra.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "result": {"state": "recovered"},
                    "record": {"working_memory_directory": "/private/memories"},
                }
            ),
            stderr="",
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

    def test_status_bar_reserves_and_updates_the_bottom_row(self):
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
            "\033[?1049h\033[2J\033[H\033[1;23r"
            "\0337\033[24;1H\033[2K"
            "\033[2m[ Veyra | telemetry ]\033[0m\0338",
        )
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
            "\0337\033[24;1H\033[2K\0338\033[r\033[?1049l"
        ))
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


class ForkTests(unittest.TestCase):
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
        self.assertEqual(args.effort, "high")

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
        self.assertIn("Sol high is required", client.developer_instructions)
        self.assertIn("ambient, low-stakes conversation", client.developer_instructions)
        self.assertIn("Terra must request Sol", client.developer_instructions)

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
                    },
                },
            )
        self.assertEqual(client.model.model_id, "gpt-5.6-terra")
        self.assertEqual(client.effort, "medium")
        self.assertEqual(client.pending_route.model.model_id, "gpt-5.6-sol")
        self.assertEqual(client.pending_route.effort, "high")
        self.assertEqual(client.pending_route.profile_version, "test")
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
        with redirect_stdout(io.StringIO()):
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
        self.assertEqual(client.effort, "medium")
        self.assertIsNone(client.active_profile_version)
        self.assertEqual(client.pending_route.profile_version, "test.2")
        with redirect_stdout(io.StringIO()):
            client.run_turn("Continue")
        self.assertEqual(client.active_profile_version, "test.2")
        self.assertIsNone(client.pending_route)
        _, params = server.calls[-1]
        settings = params["collaborationMode"]["settings"]
        self.assertEqual(settings["model"], "gpt-5.6-terra")
        self.assertEqual(settings["reasoning_effort"], "medium")
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
                    "supportedReasoningEfforts": [{"reasoningEffort": "medium"}],
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
