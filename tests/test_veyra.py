import importlib.util
import io
import sys
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
            FakeServer(), catalogue, "doctrine", Path.cwd(), catalogue.resolve("terra"),
            "medium", veyra.Palette(False)
        )
        worker = catalogue.resolve("sol")
        client._record_worker_stats(
            worker,
            {"last": {"totalTokens": 100, "outputTokens": 40, "reasoningOutputTokens": 20}},
            2.0,
        )
        stats = client.worker_stats[worker.route_id]
        self.assertEqual(stats.calls, 1)
        self.assertEqual(stats.total_tokens, 100)
        self.assertEqual(stats.output_tokens, 40)
        self.assertEqual(stats.reasoning_tokens, 20)

    def test_status_bar_reserves_and_updates_the_bottom_row(self):
        client = veyra.VeyraClient(
            FakeServer(), self.catalogue(), "doctrine", Path.cwd(),
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
            FakeServer(), self.catalogue(), "doctrine", Path.cwd(),
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
            FakeServer(), self.catalogue(), "doctrine", Path.cwd(),
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
    def test_new_threads_use_automatic_approval_review(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server,
            catalogue,
            "doctrine",
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
            "doctrine",
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
            "doctrine",
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
            "doctrine",
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
        self.assertEqual(client.model.model_id, "gpt-5.6-sol")
        self.assertEqual(client.effort, "high")
        self.assertTrue(server.calls[-1][1]["result"]["success"])

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
            "doctrine",
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
                "doctrine",
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
            "doctrine",
            Path.cwd(),
            catalogue.resolve("terra"),
            "medium",
            veyra.Palette(False),
        )
        client._handle_worker_agent(
            10, {"model": "terra", "prompt": "do a bounded task"}
        )
        self.assertFalse(server.calls[-1][1]["result"]["success"])

    def test_route_tool_rejects_worker_only_hosted_model(self):
        catalogue = self.catalogue_with_luna()
        server = FakeServer()
        client = veyra.VeyraClient(
            server,
            catalogue,
            "doctrine",
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


if __name__ == "__main__":
    unittest.main()
