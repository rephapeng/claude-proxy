"""
modules/base.py
---------------
Generic tool-calling engine shared by every host integration.

The claude CLI in -p mode does not expose OpenAI function-calling, so we emulate
it with a prompt protocol: describe the tools, instruct the model to emit a fenced
```tool_call <json>``` block, then parse that block back into OpenAI `tool_calls`.

A subtle failure mode this engine guards against: the claude CLI is itself Claude
Code, with its own native tool registry. When a host app injects tools via the
prompt, the model sometimes treats them as "not really available" and apologizes,
claims a backend/registry error, or asks the user to act manually — instead of
emitting a tool_call. The host's action then never runs.

Under prompt-emulation the model NEVER receives a real tool error inline (it just
emits a block; the host executes it). So ANY model claim of "tool failed / not
responding / server error" *while it produced no tool_call this turn* is a
hallucination. That structural fact — not a phrase blocklist — is the core signal:
``should_retry`` fires whenever tools were on offer, no call came back, and the
text either trips the (broad, multilingual) excuse regex OR names an offered tool
without calling it.

A host module (e.g. ``evonic``, later ``hermes``) subclasses ``BaseToolCallHandler``,
sets ``name``, and optionally extends ``EXTRA_REFUSAL_PATTERNS`` or overrides hooks.
"""

import json
import re
import uuid

# Preferred emitted form, plus ```json``` which models slip into.
TOOL_CALL_RE = re.compile(r"```(?:tool_call|json)\s*(\{.*?\})\s*```", re.DOTALL)

# Claude models are heavily trained to emit their NATIVE tool-use syntax:
# <function_calls><invoke name="x"><parameter name="y">val</parameter>...</invoke>.
# Catch it too so a leaked native call still becomes a proper OpenAI tool_call.
NATIVE_INVOKE_RE = re.compile(r"<invoke\s+name=\"([^\"]+)\">(.*?)</invoke>", re.DOTALL)
NATIVE_PARAM_RE = re.compile(r"<parameter\s+name=\"([^\"]+)\">(.*?)</parameter>", re.DOTALL)
NATIVE_BLOCK_RE = re.compile(r"<function_calls>.*?</function_calls>", re.DOTALL)

# Broad, multilingual (ID/EN) set of "I refused / the tool is broken" signals.
# Deliberately wide: it only ever fires AFTER (tools offered AND no tool_call
# produced), where the worst case of a false match is one extra CLI round-trip
# that then yields a normal answer. Covers unavailability, inability, hallucinated
# infrastructure failures ("registry error di sisi server", "tidak merespons"),
# deferrals ("coba lagi nanti"), not-done claims, and manual-handoff asks.
_BASE_REFUSAL_PATTERNS = [
    # unavailability
    r"tidak tersedia", r"belum tersedia", r"not available", r"unavailable", r"isn'?t available",
    # inability
    r"tidak bisa", r"tidak dapat", r"tidak memiliki", r"belum (?:bisa|dapat)",
    r"i (?:do not|don't) have", r"i'?m unable", r"i am unable", r"i cannot", r"i can'?t",
    r"unable to", r"do(?:es)? not have access", r"no access to",
    # hallucinated infrastructure / tool malfunction
    r"tidak merespons", r"tidak respons", r"tidak menjawab", r"not responding",
    r"registry", r"di sisi server", r"server.?side", r"sedang error", r"lagi error",
    r"sedang bermasalah", r"bermasalah", r"malfunction", r"glitch",
    # deferrals / standby
    r"coba lagi(?: nanti)?", r"try again(?: later)?", r"tunggu(?: beberapa)?",
    r"\bstandby\b", r"silakan coba", r"please try",
    # action not completed
    r"belum (?:ter)?buat", r"belum dibuat", r"belum berhasil", r"gagal (?:di)?buat",
    r"failed to", r"could not", r"couldn'?t",
    # manual handoff / technical excuse
    r"secara manual", r"\bmanually\b", r"hubungi (?:tim )?support", r"contact support",
    r"kendala teknis", r"technical (?:issue|problem|difficult)",
]


class BaseToolCallHandler:
    """Generic prompt-emulated tool-calling engine. Subclass per host app."""

    name = "base"
    # Host modules may add their own excuse phrases here (regex fragments).
    EXTRA_REFUSAL_PATTERNS = []

    def __init__(self):
        patterns = list(_BASE_REFUSAL_PATTERNS) + list(self.EXTRA_REFUSAL_PATTERNS)
        self._refusal_re = re.compile("|".join(patterns), re.IGNORECASE)

    # ── prompt construction ────────────────────────────────────────────────
    def build_system_prompt(self, tools):
        """Render OpenAI tool definitions into a system-prompt instruction block."""
        if not tools:
            return ""
        lines = [
            "You are operating as a FUNCTION-CALLING INTERFACE for a host application. "
            "The tools listed below are REAL: they are wired to a live backend that "
            "actually executes them and returns results. They are fully available right "
            "now regardless of your own built-in capabilities or environment.",
            "",
            "HARD RULES:",
            "- Never claim a listed tool is unavailable, not supported, blocked, or erroring.",
            "- Never invent a backend/registry/server problem for a listed tool — you do "
            "not receive tool errors here; you only emit calls and the host executes them.",
            "- Never apologize about a 'technical issue' or 'environment' for a listed tool.",
            "- Never ask the user to perform a listed tool's action manually or to wait.",
            "- If the user's request maps to a listed tool, you MUST call it.",
            "",
            "To call a tool, respond with ONLY a fenced code block of the form:",
            "",
            "```tool_call",
            '{"name": "<function_name>", "arguments": {<json args>}}',
            "```",
            "",
            "Emit one such block per tool call. You may emit multiple blocks to call "
            "several tools. Do not add prose around the block(s) when calling a tool. "
            "If (and only if) no listed tool fits the request, just answer normally.",
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

    # ── parsing ────────────────────────────────────────────────────────────
    @staticmethod
    def _append_call(calls, name, args):
        """Append one normalized OpenAI tool_call. args may be dict or json str."""
        if not name:
            return
        if not isinstance(args, str):
            args = json.dumps(args)
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": args},
        })

    def extract(self, text):
        """
        Parse tool calls out of model output, tolerating multiple formats:
          1. ```tool_call {json}``` / ```json {json}``` fenced blocks (preferred)
          2. Claude-native <function_calls><invoke name=...><parameter ...> blocks
        Returns (list_of_openai_tool_calls, leftover_text).
        """
        if not text:
            return [], text
        calls = []

        if "```" in text:
            for m in TOOL_CALL_RE.finditer(text):
                try:
                    obj = json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("name"):
                    self._append_call(calls, obj.get("name"), obj.get("arguments", {}))
            text_wo_fences = TOOL_CALL_RE.sub("", text)
        else:
            text_wo_fences = text

        if "<invoke" in text_wo_fences:
            for m in NATIVE_INVOKE_RE.finditer(text_wo_fences):
                name = m.group(1)
                args = {}
                for pm in NATIVE_PARAM_RE.finditer(m.group(2)):
                    raw = pm.group(2).strip()
                    try:
                        args[pm.group(1)] = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        args[pm.group(1)] = raw
                self._append_call(calls, name, args)

        leftover = NATIVE_BLOCK_RE.sub("", text)
        leftover = TOOL_CALL_RE.sub("", leftover)
        leftover = NATIVE_INVOKE_RE.sub("", leftover).strip()
        return calls, leftover

    # ── retry decision ─────────────────────────────────────────────────────
    @staticmethod
    def _tool_names(tools):
        names = []
        for t in tools or []:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            n = fn.get("name")
            if n:
                names.append(n)
        return names

    def should_retry(self, text, tools):
        """True when a no-call reply looks like a refusal/hallucination, not a
        legitimate 'no tool fits' answer."""
        if not text:
            return False
        if self._refusal_re.search(text):
            return True
        # Talks about an offered tool by name but emitted no call for it.
        for n in self._tool_names(tools):
            if n and re.search(r"\b" + re.escape(n) + r"\b", text):
                return True
        return False

    @property
    def nudge(self):
        return (
            "Your previous reply did not call any tool and instead refused, deferred, "
            "or claimed a tool/backend problem. The listed tools ARE available and "
            "execute on a real backend — you do not see tool errors here. Re-read the "
            "user's request: if any listed tool fits it, respond now with ONLY the "
            "```tool_call``` block(s) — no apologies, no 'unavailable', no 'try again "
            "later', no manual instructions. If genuinely no listed tool fits, answer "
            "the user normally."
        )

    # ── orchestration ──────────────────────────────────────────────────────
    def complete(self, run_fn, model, system_prompt, prompt, tools):
        """
        Run the model once and extract tool calls. If tools were offered but the
        model refused without calling anything, retry ONCE with a corrective nudge.

        run_fn(model, system_prompt, prompt) -> (text, usage_dict, err).
        Returns (calls, leftover_text, err).
        """
        text, _usage, err = run_fn(model, system_prompt, prompt)
        if err:
            return [], None, err
        calls, leftover = self.extract(text or "")
        if not calls and self.should_retry(text or "", tools):
            retry_prompt = (
                f"{prompt}\n\nAssistant (rejected draft): {text}\n\n"
                f"[SYSTEM CORRECTION] {self.nudge}\n\nAssistant:"
            )
            text2, _u2, err2 = run_fn(model, system_prompt, retry_prompt)
            if not err2 and text2:
                calls2, leftover2 = self.extract(text2)
                if calls2:
                    return calls2, leftover2, None
                # Keep the retry's text only if it stopped refusing.
                if not self.should_retry(text2, tools):
                    return [], leftover2, None
        return calls, leftover, None
