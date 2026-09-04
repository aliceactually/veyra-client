# Veyra Client

A small, dependency-free terminal client for Codex App Server. It keeps the
surface deliberately narrow: conversation, model and effort selection, thread
forking, local workers, token usage, and approval prompts.

It uses the existing Codex CLI authentication and sandbox. It does not read or
store API keys.

## Requirements

- Python 3.11 or newer
- A current `codex` CLI, already signed in
- A local checkout of `aliceactually/veyra-core`
- Optional: Ollama on `127.0.0.1:11434` or LM Studio on `127.0.0.1:1234`

## Run

```sh
cd <workspace>/veyra-client
./veyra.py --cwd <workspace>
```

By default, Veyra starts on `gpt-5.6-terra` with medium reasoning. Veyra's
coordinating identity is hard-limited to the reviewed hosted routes
`gpt-5.6-terra` and `gpt-5.6-sol`; every other route, including all local
models, is worker-only. Before App Server starts, the client always runs the
`veyra-core` fetch and continuity checks. It also discovers models from the
built-in Ollama and LM Studio providers. Override the core location with `--core PATH` or
`VEYRA_CORE_REPO`; use `--no-local` to disable local discovery.

Run a non-generating connection check with:

```sh
./veyra.py --smoke
```

## Commands

- `/help` - show commands
- `/models` - list models and supported efforts
- `/model NAME` - route later turns to an approved Veyra host; aliases such as `terra` work
- `/effort LEVEL` - set reasoning effort for later turns
- `/local MODEL PROMPT` - run a bounded local worker directly
- `/worker MODEL PROMPT` - run a bounded task on any worker-only route
- `/fork [MODEL] [EFFORT]` - branch the current history and enter the fork
- `/new` - start an empty thread
- `/threads` - list recent threads
- `/resume THREAD_ID` - resume an existing thread
- `/thread` - show the active thread and routing state
- `/usage` - show the latest token counts
- `/quit` - exit

Veyra can also call the client-provided `request_model_route` tool. A route
requested during a turn takes effect on the next turn because an in-flight
model invocation cannot be replaced. Cross-provider routes automatically fork
the current history. The `run_local_agent` tool lets the coordinating model run
a bounded task on a discovered local model and receive its report. The
`run_worker_agent` tool does the same on any worker-only route, including a
lower-capability hosted route, while keeping Veyra on an approved identity host.

In an interactive terminal, Veyra reserves the bottom row for a compact token
status bar. The conversation scrolls above it, and the terminal is restored when
the client exits. `I` is input,
`C` is cached input, `O` is output, and `R` is reasoning output. The 16-character
gauge shows their relative share of the latest turn; the final figure is the
thread's cumulative token count.

Worker threads receive their own stat bar, including elapsed time and output
tokens per second. Use `/workers` to compare worker-only models for the current
client session. These figures are performance indicators, not a cross-session
benchmark database.

Local routes are written as `provider:model`, for example:

```text
ollama:veyra-intel-coder:qwen3-coder-32k
```

The protocol implementation follows the official [Codex App Server
documentation](https://developers.openai.com/codex/app-server).
