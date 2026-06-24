"""
modules/evonic.py
-----------------
Tool-call handler for the Evonic agent runtime.

Evonic sends standard OpenAI `tools` and reads back `tool_calls`; its agents also
load skills (e.g. kanban, pinchtab) whose actions surface as tools. The failure
this targets: an Evonic agent decides to call a tool/skill, but the model (Claude
Code behind the proxy) replies that the tool "isn't responding" / the board
"registry is erroring server-side" and does NOT emit a tool_call — so the action
never runs. That is a hallucination (see base.py), handled generically by
BaseToolCallHandler; this subclass just adds a few Evonic/kanban-flavored excuse
phrases so they are caught even if the base patterns drift.

Add a new host integration by dropping a sibling file (e.g. hermes.py) that
subclasses BaseToolCallHandler and calls register(); no proxy edits needed.
"""

from .base import BaseToolCallHandler
from . import register


class EvonicToolCallHandler(BaseToolCallHandler):
    name = "evonic"

    # Evonic/kanban-specific phrasings of "the tool/board is broken" the agents
    # have been observed to hallucinate.
    EXTRA_REFUSAL_PATTERNS = [
        r"board.{0,20}(?:error|gagal|tidak)",
        r"tool .{0,20}(?:tidak merespons|error|gagal)",
        r"registry tool",
        r"skill .{0,20}(?:tidak|error|gagal)",
        r"pulih",            # "tunggu board-nya pulih"
        r"belum (?:ter)?buat",
    ]


register(EvonicToolCallHandler())
