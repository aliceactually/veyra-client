# Veyra Client

A small, dependency-free terminal client for Codex App Server. It keeps the
surface deliberately narrow: conversation, atomic cognitive-profile routing,
thread forking, local workers, token usage, and approval prompts.

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
./veyra.py
```

The default workspace is the directory containing the sibling `veyra-client`
and `veyra-core` repositories. Override it with `--cwd PATH` when needed.

By default, Veyra starts on `gpt-5.6-sol` with medium reasoning. Veyra's
coordinating identity is hard-limited to the reviewed hosted routes
`gpt-5.6-terra` and `gpt-5.6-sol`; every other route, including all local
models, is worker-only. Sol handles coding, consequential judgement, durable
memory and deep interpretation. Terra handles ambient, low-stakes conversation
and trivial non-coding work, with a startle boundary that requests Sol as scope
or consequence rises. Bounded mechanical work should normally go to a suitable
local worker. Before App Server starts,
the client always runs the
`veyra-core` fetch and continuity checks. It also discovers models from the
built-in Ollama and LM Studio providers. Override the core location with
`--core PATH` or `VEYRA_CORE_REPO`; use `--no-local` to disable local discovery.
An ahead-only core checkout produces a warning and continues with the newer
local doctrine so Veyra can obtain Alice's direction. A behind or diverged
checkout still stops before launch rather than selecting a version implicitly.

Approval requests use Codex automatic review by default. Routine sandbox
escapes such as protected Git metadata, networked GitHub operations and work in
an adjacent repository are reviewed without interrupting Alice. The
`workspace-write` sandbox and `on-request` approval policy remain active, so
destructive or unusually risky actions can still be denied. Use
`--approvals-reviewer user` to restore manual prompts for a session.

Run a non-generating connection check with:

```sh
./veyra.py --smoke
```

## Commands

- `/help` - show commands
- `/models` - list models and supported efforts
- `/model NAME` - route later turns to an approved Veyra host; aliases such as `terra` work
- `/effort LEVEL` - set reasoning effort for later turns
- `/attention [LEVEL]` - inspect or set attention for later turns
- `/local MODEL PROMPT` - run a bounded local worker directly
- `/worker MODEL PROMPT` - run a bounded task on any worker-only route
- `/fork [MODEL] [EFFORT]` - branch the current history and enter the fork
- `/new` - start an empty thread
- `/threads` - list recent threads
- `/resume THREAD_ID` - resume an existing thread
- `/thread` - show the active thread and routing state
- `/usage` - show the latest token counts
- `/quit` - exit

Veyra can call the client-provided `request_attention` tool to change reasoning
effort without changing model. Attention starts at medium, rises to high or
xhigh when the depth or consequence of the work warrants it, and settles back
towards medium afterwards. Each shift is reasoned, visible in the terminal, and
applies to the next turn; max effort is not selected automatically. `/attention`
shows the active or pending level, while `/thread` shows the active route,
attention, profile version and transition reason.

Veyra can also call the client-provided `request_model_route` tool. A route
requested during a turn remains pending until the next turn. The client then
changes model, reasoning effort and route-specific developer instructions as a
single App Server collaboration-mode setting. Every transition is bound to a
non-empty profile version, exposed by `/thread`, and included in the developer
instruction attestation. If that turn cannot start, the old route remains
active and the transition stays pending. Resumed threads reconcile their
reported model and effort with the current profile on the next turn.
Cross-provider routes automatically fork the current history. The
`run_local_agent` tool lets
the coordinating model run a bounded task on a discovered local model and
receive its report. The `run_worker_agent` tool does the same on any worker-only
route, including a lower-capability hosted route, while keeping Veyra on an
approved identity host.

The public identity and safety doctrine is shared by both routes. Route-specific
profiles live in `veyra-core/profiles`; worker threads receive an identity-free
worker profile. Private memories and dialogue corpora remain in Alice-encrypted
continuity, never in either public repository.

In an interactive terminal, Veyra reserves the bottom row for a compact token
status bar. The conversation scrolls above it, and the terminal is restored when
the client exits. `I` is input,
`C` is cached input, `O` is output, and `R` is reasoning output. The 16-character
gauge shows their relative share of the latest turn; the final figure is the
thread's cumulative token count.

Interactive input uses Readline editing and history. Coloured prompt controls
are excluded from GNU Readline's width calculations; the libedit compatibility
layer used by macOS receives a plain prompt because it mishandles those control
markers. Long input can therefore wrap across terminal rows while arrow
movement, insertion and deletion continue to redraw at the correct cursor
position.

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
