# Veyra Client

A small, dependency-free terminal client for Codex App Server. It keeps the
surface deliberately narrow: conversation, atomic cognitive-profile routing,
thread forking, local workers, circadian memory hand-off, token usage, and
approval prompts.

It uses the existing Codex CLI authentication and sandbox. It does not read or
store API keys.

## Requirements

- Python 3.11 or newer
- A current `codex` CLI, already signed in
- A local checkout of `veyra-core/veyra-core`
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

Verified recovered continuity and a deliberate blank start are distinct
instruction paths. Recovered Veyra receives the private working-memory
location. A deliberate blank start receives only the host-neutral public
`RECOVERY-PERSONA.md` packet from Veyra Core, with an explicit statement that
no encrypted continuity or private event history was recovered.

On recovered wakes, the client invokes Veyra Core's circadian scheduler. When
a bounded daily consolidation and dream cycle is due, its pending jobs enter
Veyra's private developer context for asynchronous local Muse work and Sol
review. The cycle never blocks the user's active request, and fictional dreams
remain outside factual recall. The latest approved dream is returned to
Veyra's private waking context; she decides whether to share it.

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
- `/bgworker MODEL PROMPT` - start a read-only worker and return to the prompt
- `/jobs` - list background worker jobs
- `/job ID` - inspect one background worker and its report
- `/cancel-job ID` - interrupt a background worker
- `/branch [MODEL] [EFFORT]` - branch the current history and enter that branch
- `/new` - start an empty thread
- `/threads` - list recent threads
- `/resume THREAD_ID` - resume an existing thread
- `/thread` - show the active thread and routing state
- `/usage` - show the latest token counts
- `/quit` - exit

Veyra can call the client-provided `request_attention` tool to change reasoning
effort without changing model. Attention starts at medium, rises to high or
xhigh when the depth or consequence of the work warrants it, and settles back
towards medium afterwards. Veyra may recommend max for important work, but must
ask Alice before selecting it for that use. Max is a behavioural consent
boundary rather than a client prohibition; once Alice agrees, Veyra can request
it directly. The client hard-rejects ultra and any unclassified effort, keeping
max as the mechanical ceiling. Each shift is reasoned, visible in the terminal,
and applies to the next turn. When unfinished work requires that new attention,
Veyra sets `continue_task` and the client initiates one bounded follow-up turn
without waiting for placeholder input from Alice. The continuation is delivered
as client tool output, never as a fabricated user message, and cannot recursively
create further turns. `/attention` shows the active or pending level, while
`/thread` shows the active route, attention, profile version and transition reason.

Veyra can also call the client-provided `request_model_route` tool. A route
requested during a turn remains pending until the next turn. The client then
changes model, reasoning effort and route-specific developer instructions as a
single App Server collaboration-mode setting. Every transition is bound to a
non-empty profile version, exposed by `/thread`, and included in the developer
instruction attestation. If that turn cannot start, the old route remains
active and the transition stays pending. Resumed threads reconcile their
reported model and effort with the current profile on the next turn.
Model-route requests use the same explicit, single-hop `continue_task` mechanism
when current work must resume immediately. Manual `/model`, `/effort` and
`/attention` selections remain pending for Alice's next substantive turn.
Cross-provider routes automatically fork the current history. The
`run_local_agent` tool lets
the coordinating model run a bounded task on a discovered local model and
receive its report. The `run_worker_agent` tool does the same on any worker-only
route, including a lower-capability hosted route, while keeping Veyra on an
approved identity host. For separable work, `spawn_worker_agent` returns a job
ID immediately and lets conversation continue while the worker runs. This works
for hosted worker-only routes, Ollama and LM Studio. Background jobs are
read-only, limited to three concurrent workers, and cannot interrupt Alice for
approvals. Synchronous worker tools remain available when Veyra needs the report
inside the current answer.

Completed reports return to Veyra as standalone tool output rather than as a
fabricated user message. If a report arrives during a turn, Veyra handles it
immediately after that turn. If Alice is composing input, the status row shows
that a report is ready and the client waits until Enter before collecting it,
preserving the active Readline buffer. `/jobs` and `/job ID` expose the same
state directly. Jobs are session-scoped; exiting the client stops unfinished
workers. Reports remain attached to the Veyra thread that launched them, so
switching threads does not inject an old report into unrelated context.

The public identity and safety doctrine is shared by both routes. Route-specific
profiles live in `veyra-core/profiles`; worker threads receive an identity-free
worker profile. Private memories and dialogue corpora remain in Alice-encrypted
continuity, never in either public repository.

In an interactive terminal, Veyra reserves the bottom row for a compact status
bar while remaining on the terminal's primary screen. Conversation output is
therefore retained in the terminal's normal scrollback buffer, and the terminal
is restored when the client exits. When an accepted turn begins, the bar changes
immediately to show the active model and attention level, explicitly labelling
any model or attention switch. Final token telemetry replaces that transient
turn-start state. `I` is input,
`C` is cached input, `O` is output, and `R` is reasoning output. The 16-character
gauge shows their relative share of the latest turn; the final figure is the
thread's cumulative token count.

Interactive input uses Readline editing and history. Coloured prompt controls
are excluded from GNU Readline's width calculations. The libedit compatibility
layer used by macOS receives an equal-width, zero-column placeholder after the
coloured prompt is rendered and reset. Long input can therefore wrap across
terminal rows while arrow movement, insertion and deletion continue to redraw
at the correct cursor position.

Without a stored preference, a session begins with the generic `user>` prompt.
After Veyra learns the current person's preferred name, she can use the
client-only `set_user_prompt` tool to set the prompt name and, when requested,
select a constrained terminal colour. There is deliberately no slash command
for this control. Personalisation persists in the user's local client settings;
it improves the shared interface without claiming user identity, recovery or
relationship continuity.

Worker threads receive their own stat bar, including elapsed time and output
tokens per second. Use `/workers` to compare worker-only models for the current
client session. These figures are performance indicators, not a cross-session
benchmark database.

App Server notifications are routed per thread, so a background worker cannot
consume or discard Veyra's turn events (or those of another worker).

Foreground Veyra turns own the terminal until they complete, so the `alice>`
prompt is never displayed while Veyra is still thinking or working. The status
row shows the active turn and any route or attention transition. Explicit
alternate conversation histories use `/branch`; the former `/fork` spelling no
longer changes the foreground thread. Asynchronous worker and collaboration
reports return to Veyra for review and a natural summary rather than opening
another terminal interface; once Veyra has acknowledged a detached hand-off,
the ordinary prompt is available again.

Local routes are written as `provider:model`, for example:

```text
ollama:veyra-intel-coder:qwen3-coder-32k
```

The protocol implementation follows the official [Codex App Server
documentation](https://developers.openai.com/codex/app-server).

## Licence

Veyra Client is licensed under the [Apache License 2.0](LICENSE). It is
developed as a collaboration between Alice Kallista Saunier and Veyra; see
[NOTICE](NOTICE) for formal attribution.
