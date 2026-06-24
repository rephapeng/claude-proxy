# claude-openai-proxy

A ~300-line, **zero-dependency** (stdlib-only) OpenAI-compatible HTTP proxy that
wraps the local `claude` (Claude Code) CLI. It lets any OpenAI-API client (agent
frameworks, IDE plugins, SDKs, your own scripts) use your authenticated Claude
Code session, including a **Claude Max** subscription that has no raw API key.

```
OpenAI-API client ──/v1/chat/completions──▶ claude_openai_proxy.py ──▶ claude CLI ──▶ Anthropic
```

## Run

```bash
python3 claude_openai_proxy.py --port 8088
# -> http://127.0.0.1:8088/v1   (api key is ignored)
```

Requires the `claude` CLI installed and logged in (`claude` once, interactively,
to authenticate). No `pip install` needed.

### Endpoints
| Method | Path                    | Purpose                              |
|--------|-------------------------|--------------------------------------|
| GET    | `/health`               | liveness check                       |
| GET    | `/v1/models`            | list exposed model ids               |
| POST   | `/v1/chat/completions`  | chat completion (stream + non-stream)|

### Env vars
| Var                           | Default  | Meaning                              |
|-------------------------------|----------|--------------------------------------|
| `CLAUDE_PROXY_DEFAULT_MODEL`  | `sonnet` | model when request omits one         |
| `CLAUDE_PROXY_TIMEOUT`        | `600`    | per-call CLI timeout (seconds)       |
| `CLAUDE_PROXY_HOST` / `_PORT` | `127.0.0.1` / `8088` | bind address             |
| `CLAUDE_BIN`                  | `claude` | path to the CLI                      |
| `CLAUDE_PROXY_MAX_CONCURRENCY` | `2` | max concurrent `claude` processes (0=unlimited); cap to host RAM |

## Point a client at it

Configure your client like any OpenAI-compatible endpoint:

| Setting   | Value                                            |
|-----------|--------------------------------------------------|
| Base URL  | `http://localhost:8088/v1`                       |
| API key   | any non-empty string (ignored by the proxy)      |
| Model     | `sonnet` (or `opus`, `haiku`, `claude-sonnet-4-6`, …) |

Common forms:

```bash
# OpenAI Python SDK
export OPENAI_BASE_URL=http://localhost:8088/v1
export OPENAI_API_KEY=ignored
```
```ini
# typical .env-style config
LLM_BASE_URL=http://localhost:8088/v1
LLM_API_KEY=ignored-but-required
LLM_MODEL=sonnet
```

Use the client's **OpenAI / OpenAI-compatible** mode — not its native
"Anthropic" mode. This proxy speaks `/chat/completions`, not Anthropic's
`/messages`. Whatever model id you send is passed straight to `claude --model`,
so aliases (`sonnet`/`opus`/`haiku`) and full ids both work.

## How it maps OpenAI → CLI
- `system` messages → `--append-system-prompt`
- conversation → flattened transcript piped to `claude -p` on stdin
- non-stream → `--output-format json`, parsed into a `chat.completion`
- `stream:true` → `--output-format stream-json --include-partial-messages`,
  text deltas re-emitted as OpenAI SSE `chat.completion.chunk`s + `[DONE]`
- token counts mapped from the CLI's `usage` block

## Function / tool calling
The proxy emulates OpenAI function calling with a prompt protocol (the `claude`
CLI has no native OpenAI tool-calling in `-p` mode):

1. Request `tools` are described in an injected system instruction that tells the
   model to emit a fenced ```` ```tool_call {json} ``` ```` block to call one.
2. Those blocks are parsed back into a standard OpenAI
   `choices[0].message.tool_calls[]` with `finish_reason: "tool_calls"`.
3. On the next turn, incoming `role:"tool"` results and prior assistant
   `tool_calls` are rendered into the transcript so the model can produce a final
   answer.

This is reliable for normal agentic loops but is prompt-based, not a native API
feature — very large/complex tool schemas may need tuning.

**False-refusal hardening.** The `claude` CLI is itself Claude Code, with its own
native tool registry; it occasionally treats a prompt-injected tool as "not
available" and apologizes instead of calling it (so the host app's action never
runs). The proxy guards against this:
1. The injected tool instruction states the tools are real, wired to a live
   backend, and forbids "unavailable"/"do it manually" refusals.
2. The parser accepts both the fenced ```` ```tool_call ````/```` ```json ```` form
   **and** Claude's native `<function_calls><invoke …>` form, so a leaked native
   call still becomes a proper `tool_calls`.
3. If tools were offered but the model refused without calling anything, the proxy
   retries once with a hard corrective nudge — forcing a real tool call, or a
   normal answer when genuinely no tool fits.

## Host integrations
Per-host tool-call handlers live in `modules/` and are auto-discovered (subclass
`BaseToolCallHandler`, call `register()`, select via `X-Toolcall-Module` header or
`CLAUDE_PROXY_TOOLCALL_MODULE`). For a full worked example of wiring a host app to
the proxy — disabling native CLI tools, prompt-emulated tool calls, false-refusal
hardening, and end-to-end setup — see [`docs/evonic-integration.md`](docs/evonic-integration.md).

## Keep it running

### systemd
```bash
sudo cp claude-proxy.service /etc/systemd/system/claude-proxy@.service
sudo systemctl daemon-reload
sudo systemctl enable --now claude-proxy@youruser   # the user that ran `claude` login
systemctl status claude-proxy@youruser
```

### Docker
The `claude` CLI needs your Claude Code credentials (`~/.claude`), so mount them —
never bake them into the image:
```bash
docker compose up -d --build       # uses docker-compose.yml
# or:
docker build -t claude-proxy .
docker run --rm -p 8088:8088 -v "$HOME/.claude:/root/.claude" claude-proxy
```

## Limitations / notes
- Tool calling is **prompt-emulated**, not a native API capability (see above).
- One CLI process per request (stateless); multi-turn context is replayed each call.
- **ToS:** using a Claude Max subscription as a generic API backend for a
  third-party app may violate Anthropic's Terms of Service. Unofficial; no SLA;
  may break on `claude` CLI updates. Use at your own risk. For production, a real
  Anthropic API key (or OpenRouter) is the supported path.
