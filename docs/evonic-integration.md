# Integrating claude-proxy with Evonic

This guide explains how the [Evonic](https://github.com/) agent runtime is wired to
`claude-proxy`, and how to reproduce the setup on a fresh host. The proxy turns an
authenticated `claude` (Claude Code) CLI into an OpenAI-compatible endpoint; Evonic
talks to it as if it were OpenAI, while **owning all tool execution itself**.

```
Evonic agent  ──OpenAI /v1/chat/completions──▶  claude-proxy (:8088)  ──▶  claude -p  ──▶  Anthropic
   ▲  (executes tool_calls itself)                    │
   └──────────── tool_calls / tool results ───────────┘  (prompt-emulated, native CLI tools OFF)
```

## Why a special integration is needed

The `claude` CLI in `-p` mode is itself Claude Code, with its own native tool
registry and no OpenAI function-calling. Pointed at it naively, an agent framework
breaks in two ways:

1. **The CLI runs tools itself.** With native tools live, the model executes
   Bash/Read/WebFetch on its own (and stalls on "requires approval" for sensitive
   ones in non-interactive mode) instead of returning a `tool_call` for Evonic to
   run. Evonic's whole tool/skill layer (kanban, sshc, python sandbox, …) is bypassed.
2. **False refusals.** When tools are injected via prompt, the model sometimes
   claims the tool "isn't responding" or the board "registry is erroring" and emits
   no call — so the agent's action silently never happens.

The proxy solves both. The pieces below are what make it work for Evonic.

## The four pieces that make it work

### 1. Native CLI tools are disabled (`--tools ""`)
`build_cli_args()` always invokes the CLI with `--tools ""`, disabling every
built-in Claude Code tool. The proxy becomes a **pure LLM endpoint** — the model
can only emit text (and prompt-emulated tool calls), never execute anything. Evonic
owns execution. See `claude_openai_proxy.py` → `build_cli_args`.

### 2. Prompt-emulated OpenAI tool calling
Because `-p` has no native function-calling, the proxy emulates it
(`modules/base.py`):
- **Request:** `tools[]` from Evonic are rendered into a system instruction
  (`build_system_prompt`) telling the model to emit a fenced ```` ```tool_call
  {json} ```` block to call one.
- **Response:** those blocks (and any leaked Claude-native
  `<function_calls><invoke>` form) are parsed back into a standard OpenAI
  `choices[0].message.tool_calls[]` with `finish_reason: "tool_calls"`
  (`extract`).
- **Next turn:** Evonic sends `role:"tool"` results + prior assistant `tool_calls`,
  which are replayed into the transcript so the model can produce a final answer.

### 3. False-refusal hardening
The instruction states the tools are real and forbids "unavailable"/"do it
manually" refusals. After generation, `should_retry()` fires when **tools were
offered, no call came back, and** the text trips a broad multilingual excuse regex
or names an offered tool without calling it — then the proxy retries **once** with a
corrective `nudge`. This is structural: under prompt-emulation the model never
receives a real tool error, so any "tool failed" claim with no call is a
hallucination.

### 4. Per-host handler module (`modules/evonic.py`)
Handlers are auto-discovered from `modules/`. `EvonicToolCallHandler` subclasses the
base engine and only adds Evonic/kanban-flavored excuse phrases
(`EXTRA_REFUSAL_PATTERNS`: board errors, "registry tool", "skill … gagal", "tunggu
board-nya pulih", …) so they're caught even if base patterns drift. Selection order
(`modules/__init__.py`):
1. request header `X-Toolcall-Module: <name>`
2. env `CLAUDE_PROXY_TOOLCALL_MODULE` (default `evonic`)
3. the generic `base` engine

### 5. Concurrency cap (host-RAM guard)
`CLAUDE_PROXY_MAX_CONCURRENCY` (default `2`) bounds simultaneous `claude` processes
via a semaphore. Each CLI process is heavy; on a ~2 GB host, raising this risks OOM.
Keep it at 2 unless the host has more RAM.

## Implementing it on a fresh host

### Prerequisites
- The `claude` CLI installed and **logged in once interactively** (`claude`) as the
  user the service will run as. The proxy reuses that session's credentials
  (`~/.claude`); no API key is required.
- Python 3 (stdlib only — the proxy has **zero pip dependencies**).

### 1. Deploy the proxy
```bash
git clone https://github.com/rephapeng/claude-proxy.git /opt/claude-proxy
```

### 2. Install the systemd unit
The unit in this repo (`claude-proxy.service`) already encodes the Evonic-ready
config. Key settings:

| Env var | Value | Why |
|---|---|---|
| `CLAUDE_PROXY_PORT` | `8088` | endpoint Evonic points at |
| `CLAUDE_PROXY_DEFAULT_MODEL` | `sonnet` | model when a request omits one |
| `CLAUDE_PROXY_MAX_CONCURRENCY` | `2` | host-RAM guard (see §5) |
| `CLAUDE_PROXY_TOOLCALL_MODULE` | `evonic` *(default)* | selects the Evonic handler |
| `CLAUDE_BIN` | `/root/.local/bin/claude` | path to the logged-in CLI |

```bash
sudo cp /opt/claude-proxy/claude-proxy.service /etc/systemd/system/claude-proxy.service
sudo systemctl daemon-reload
sudo systemctl enable --now claude-proxy
systemctl status claude-proxy
```

### 3. Verify the proxy
```bash
curl -s localhost:8088/health
curl -s localhost:8088/v1/models | head
# tool-call smoke test (should return tool_calls, not prose):
curl -s localhost:8088/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"sonnet",
  "messages":[{"role":"user","content":"What is the weather in Paris?"}],
  "tools":[{"type":"function","function":{"name":"get_weather",
    "description":"Get weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
}' | python3 -m json.tool
```
Expect `finish_reason: "tool_calls"` and a `get_weather` call with `{"city":"Paris"}`.

### 4. Point Evonic at it
Evonic uses the proxy as its OpenAI-compatible LLM backend. In Evonic's model config
(`shared/db/evonic.db`), the default model (`setup_custom`) maps to:

| Setting | Value |
|---|---|
| Base URL | `http://localhost:8088/v1` |
| API key | any non-empty string (ignored) |
| Model | `claude-opus-4-8` (or `sonnet`, `opus`, any id passed to `claude --model`) |

Use Evonic's **OpenAI / OpenAI-compatible** provider mode, not its Anthropic mode —
the proxy speaks `/chat/completions`, not Anthropic's `/messages`. All agents that
should share this backend point at the same `setup_custom` model.

### 5. Restart order
The proxy must be up before Evonic starts handling agent turns. Evonic's unit
declares `After=claude-proxy.service` / `Wants=claude-proxy.service`, so systemd
starts the proxy first and a host reboot brings both up in order.

## Adding another host integration
Drop a sibling file in `modules/` (e.g. `hermes.py`) that subclasses
`BaseToolCallHandler`, sets `name`, optionally extends `EXTRA_REFUSAL_PATTERNS`, and
calls `register()`. No proxy edits needed — it's auto-discovered. Select it per
request with `X-Toolcall-Module: hermes` or globally via
`CLAUDE_PROXY_TOOLCALL_MODULE=hermes`.

## Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| Agent says a tool "isn't responding" and nothing runs | False refusal slipped past patterns — add the phrase to `modules/evonic.py` `EXTRA_REFUSAL_PATTERNS`. |
| Model runs commands itself instead of returning a `tool_call` | `--tools ""` not applied (old proxy build) — update and restart. |
| OOM / host freezes under load | `CLAUDE_PROXY_MAX_CONCURRENCY` too high for host RAM — keep at 2 on ~2 GB. |
| `401`/auth errors from the CLI | The service user's `claude` session isn't logged in — run `claude` interactively as that user. |
| `tool_calls` never parsed | Client sent Anthropic mode — switch Evonic to OpenAI-compatible mode. |

## See also
- `README.md` — generic proxy reference (endpoints, env vars, Docker).
- `modules/base.py` — the prompt-emulation engine and refusal logic.
- `modules/evonic.py` — the Evonic handler.
