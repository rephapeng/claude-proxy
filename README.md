# claude-openai-proxy

A ~300-line, **zero-dependency** (stdlib-only) OpenAI-compatible HTTP proxy that
wraps the local `claude` (Claude Code) CLI. It lets any OpenAI-API client — such
as **Evonic** — use your authenticated Claude Code session, including a **Claude
Max** subscription that has no raw API key.

```
Evonic ──OpenAI /v1/chat/completions──▶ claude_openai_proxy.py ──▶ claude CLI ──▶ Anthropic
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

## Point Evonic at it

In `.env` (or the Evonic web UI → Models), use the **OpenAI** provider format
(the default — do **not** pick "anthropic"; this proxy speaks `/chat/completions`,
not `/messages`):

```ini
LLM_BASE_URL=http://localhost:8088/v1
LLM_API_KEY=ignored-but-required
LLM_MODEL=sonnet            # or claude-sonnet-4-6, opus, haiku, etc.
```

Evonic chooses its wire format from the provider's `api_format` field (default
`openai`), not from the model name, so any model id routes correctly here.

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

This is reliable for normal agentic loops (Evonic's included) but is prompt-based,
not a native API feature — very large/complex tool schemas may need tuning.

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
