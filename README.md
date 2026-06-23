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

## Limitations / notes
- **Tool calls / function calling are not translated** — this is a text-in/text-out
  chat proxy. The underlying Claude Code session may still use its own tools.
- One CLI process per request (stateless); multi-turn context is replayed each call.
- **ToS:** using a Claude Max subscription as a generic API backend for a
  third-party app may violate Anthropic's Terms of Service. Unofficial; no SLA;
  may break on `claude` CLI updates. Use at your own risk. For production, a real
  Anthropic API key (or OpenRouter) is the supported path.
# claude-proxy
