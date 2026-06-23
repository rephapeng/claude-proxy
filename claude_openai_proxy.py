#!/usr/bin/env python3
"""
claude_openai_proxy.py
----------------------
A tiny, zero-dependency OpenAI-compatible HTTP proxy that wraps the local
`claude` (Claude Code) CLI. It lets any OpenAI-API client (e.g. Evonic) talk to
your authenticated Claude Code session — including a Claude Max subscription,
which has no raw API key.

Endpoints:
  GET  /v1/models             -> list exposed model ids
  POST /v1/chat/completions   -> OpenAI chat completion (stream + non-stream,
                                 with prompt-based function/tool calling)
  GET  /health                -> {"ok": true}

Usage:
  python3 claude_openai_proxy.py --port 8088
  (then point Evonic at  http://localhost:8088/v1 , any api key)

Stdlib only. No pip install required.

WARNING: Using a Claude Max subscription as a generic API backend for a
third-party app may violate Anthropic's Terms of Service. Use at your own risk.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
DEFAULT_MODEL = os.environ.get("CLAUDE_PROXY_DEFAULT_MODEL", "sonnet")
CALL_TIMEOUT = int(os.environ.get("CLAUDE_PROXY_TIMEOUT", "600"))

EXPOSED_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
    "sonnet",
    "opus",
    "haiku",
]


# ---------------------------------------------------------------------------
# OpenAI message -> text helpers
# ---------------------------------------------------------------------------

def _content_to_text(content):
    """OpenAI message content can be a string or a list of parts. Flatten to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif "text" in p:
                    parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Function/tool calling shim
#
# The claude CLI in -p mode does not expose OpenAI function-calling. We emulate
# it with a prompt protocol: describe the tools, instruct the model to emit a
# fenced ```tool_call <json>``` block when it wants to call one, then parse that
# block back into OpenAI `tool_calls`. Tool *results* coming back from the client
# (role:"tool") and prior assistant tool_calls are rendered into the transcript
# so the model has the full picture.
# ---------------------------------------------------------------------------

TOOL_CALL_RE = re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL)


def build_tools_system_prompt(tools):
    """Render OpenAI tool definitions into a system-prompt instruction block."""
    if not tools:
        return ""
    lines = [
        "You have access to the following tools (functions). When you need to "
        "call one, respond with ONLY a fenced code block of the form:",
        "",
        "```tool_call",
        '{"name": "<function_name>", "arguments": {<json args>}}',
        "```",
        "",
        "Emit one such block per tool call. You may emit multiple blocks to call "
        "several tools. Do not add prose around the block(s) when calling a tool. "
        "If you do not need a tool, just answer normally.",
        "",
        "Available tools:",
    ]
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"- {name}: {desc}")
        try:
            lines.append(f"  parameters schema: {json.dumps(params)}")
        except (TypeError, ValueError):
            pass
    return "\n".join(lines)


def extract_tool_calls(text):
    """
    Find ```tool_call {json}``` blocks in model output.
    Returns (list_of_openai_tool_calls, leftover_text).
    """
    if not text or "tool_call" not in text:
        return [], text
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args)
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    leftover = TOOL_CALL_RE.sub("", text).strip()
    return calls, leftover


def messages_to_prompt(messages, tools=None):
    """
    Split OpenAI messages into (system_prompt, user_prompt).

    System messages are concatenated and handed to the CLI via
    --append-system-prompt, along with the tool-protocol instructions. The
    remaining conversation (including prior tool calls and tool results) is
    flattened into a single prompt with role labels so multi-turn context and
    function-calling history are preserved.
    """
    system_parts = []
    convo_parts = []
    for m in messages or []:
        role = m.get("role", "user")
        text = _content_to_text(m.get("content"))
        if role == "system":
            if text:
                system_parts.append(text)
        elif role == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                rendered = []
                for tc in tcs:
                    fn = tc.get("function", {})
                    rendered.append(
                        f'```tool_call\n{{"name": "{fn.get("name","")}", '
                        f'"arguments": {fn.get("arguments","{}")}}}\n```'
                    )
                joined = "\n".join(rendered)
                convo_parts.append(f"Assistant (tool call):\n{joined}")
            if text:
                convo_parts.append(f"Assistant: {text}")
        elif role == "tool":
            name = m.get("name") or m.get("tool_call_id") or "tool"
            convo_parts.append(f"Tool result ({name}): {text}")
        else:  # user and anything else
            convo_parts.append(f"User: {text}")

    if tools:
        system_parts.append(build_tools_system_prompt(tools))

    system_prompt = "\n\n".join(p for p in system_parts if p).strip()

    non_system = [m for m in (messages or []) if m.get("role") != "system"]
    if len(non_system) == 1 and non_system[0].get("role") == "user" and not tools:
        prompt = _content_to_text(non_system[0].get("content"))
    else:
        prompt = "\n\n".join(convo_parts)
        prompt += "\n\nAssistant:"
    return system_prompt, prompt


# ---------------------------------------------------------------------------
# claude CLI invocation
# ---------------------------------------------------------------------------

def build_cli_args(model, system_prompt, stream):
    args = [CLAUDE_BIN, "-p", "--model", model or DEFAULT_MODEL]
    if stream:
        args += ["--output-format", "stream-json", "--include-partial-messages", "--verbose"]
    else:
        args += ["--output-format", "json"]
    if system_prompt:
        args += ["--append-system-prompt", system_prompt]
    return args


def run_claude_blocking(model, system_prompt, prompt):
    """Run claude in --output-format json mode, return (text, usage_dict, error)."""
    args = build_cli_args(model, system_prompt, stream=False)
    try:
        proc = subprocess.run(
            args, input=prompt, capture_output=True, text=True,
            timeout=CALL_TIMEOUT, cwd=os.environ.get("CLAUDE_PROXY_CWD", "/tmp"),
        )
    except subprocess.TimeoutExpired:
        return None, {}, f"claude CLI timed out after {CALL_TIMEOUT}s"
    except FileNotFoundError:
        return None, {}, f"claude binary not found: {CLAUDE_BIN}"

    if proc.returncode != 0:
        return None, {}, (proc.stderr or proc.stdout or "claude CLI failed").strip()

    raw = proc.stdout.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, {}, None

    if data.get("is_error"):
        return None, {}, data.get("result") or "claude reported an error"

    text = data.get("result", "")
    u = data.get("usage", {}) or {}
    usage = {
        "prompt_tokens": int(u.get("input_tokens", 0) or 0)
        + int(u.get("cache_read_input_tokens", 0) or 0)
        + int(u.get("cache_creation_input_tokens", 0) or 0),
        "completion_tokens": int(u.get("output_tokens", 0) or 0),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return text, usage, None


def run_claude_stream(model, system_prompt, prompt):
    """Generator yielding text chunks as the CLI streams (text answers only)."""
    args = build_cli_args(model, system_prompt, stream=True)
    proc = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=os.environ.get("CLAUDE_PROXY_CWD", "/tmp"),
    )
    proc.stdin.write(prompt)
    proc.stdin.close()

    emitted_any = False
    last_full_text = ""
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = evt.get("type")
            if etype == "stream_event":
                inner = evt.get("event", {})
                if inner.get("type") == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta":
                        chunk = delta.get("text", "")
                        if chunk:
                            emitted_any = True
                            yield chunk
            elif etype == "assistant" and not emitted_any:
                msg = evt.get("message", {})
                for block in msg.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        last_full_text = block.get("text", "")
            elif etype == "result" and not emitted_any:
                final = evt.get("result", "") or last_full_text
                if final:
                    yield final
    finally:
        proc.stdout.close()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        sys.stderr.write("[proxy] " + (fmt % a) + "\n")

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, msg, code=500, etype="server_error"):
        self._send_json({"error": {"message": msg, "type": etype}}, code)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            return self._send_json({"ok": True})
        if self.path.rstrip("/").endswith("/models"):
            now = int(time.time())
            return self._send_json({
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": now, "owned_by": "anthropic"}
                    for m in EXPOSED_MODELS
                ],
            })
        return self._send_error("not found", 404, "invalid_request_error")

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send_error("not found", 404, "invalid_request_error")
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send_error("invalid JSON body", 400, "invalid_request_error")

        model = payload.get("model") or DEFAULT_MODEL
        stream = bool(payload.get("stream"))
        tools = payload.get("tools") or None
        system_prompt, prompt = messages_to_prompt(payload.get("messages", []), tools)
        if not prompt.strip():
            return self._send_error("no prompt content in messages", 400, "invalid_request_error")

        if stream:
            return self._handle_stream(model, system_prompt, prompt, tools)
        return self._handle_blocking(model, system_prompt, prompt, tools)

    def _handle_blocking(self, model, system_prompt, prompt, tools):
        text, usage, err = run_claude_blocking(model, system_prompt, prompt)
        if err:
            return self._send_error(err, 502, "upstream_error")

        message = {"role": "assistant", "content": text}
        finish = "stop"
        if tools:
            calls, leftover = extract_tool_calls(text)
            if calls:
                message = {"role": "assistant", "content": leftover or None,
                           "tool_calls": calls}
                finish = "tool_calls"

        resp = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return self._send_json(resp)

    def _handle_stream(self, model, system_prompt, prompt, tools):
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        def sse(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        def chunk(delta, finish=None):
            return {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        sse(chunk({"role": "assistant"}))

        # When tools are offered, generate fully (blocking) so we can detect and
        # emit tool_calls; streaming token-by-token can't be re-parsed mid-flight.
        if tools:
            text, _usage, err = run_claude_blocking(model, system_prompt, prompt)
            if err:
                sse(chunk({"content": f"[proxy error: {err}]"}))
                sse(chunk({}, finish="stop"))
            else:
                calls, leftover = extract_tool_calls(text or "")
                if calls:
                    tc_delta = [{
                        "index": i, "id": c["id"], "type": "function",
                        "function": c["function"],
                    } for i, c in enumerate(calls)]
                    if leftover:
                        sse(chunk({"content": leftover}))
                    sse(chunk({"tool_calls": tc_delta}))
                    sse(chunk({}, finish="tool_calls"))
                else:
                    if text:
                        sse(chunk({"content": text}))
                    sse(chunk({}, finish="stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        try:
            for piece in run_claude_stream(model, system_prompt, prompt):
                sse(chunk({"content": piece}))
        except Exception as e:  # noqa: BLE001
            sse(chunk({"content": f"\n[proxy error: {e}]"}))
        sse(chunk({}, finish="stop"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser(description="OpenAI-compatible proxy for the claude CLI")
    ap.add_argument("--host", default=os.environ.get("CLAUDE_PROXY_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLAUDE_PROXY_PORT", "8088")))
    args = ap.parse_args()

    if not shutil.which(CLAUDE_BIN):
        sys.exit(f"ERROR: '{CLAUDE_BIN}' not found on PATH. Install/authenticate Claude Code first.")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"claude-openai-proxy listening on http://{args.host}:{args.port}/v1")
    print(f"  default model: {DEFAULT_MODEL}   timeout: {CALL_TIMEOUT}s")
    print("  point your OpenAI client's base_url here; api key is ignored.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
