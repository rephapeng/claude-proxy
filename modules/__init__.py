"""
modules/ — pluggable per-host tool-call handlers.

Each host integration is a file in this package that subclasses
``base.BaseToolCallHandler`` and calls ``register(instance)``. The proxy selects a
handler per request and routes tool-call construction, parsing, and false-refusal
retry through it.

Selection order (first match wins):
  1. per-request header  ``X-Toolcall-Module: <name>``
  2. env  ``CLAUDE_PROXY_TOOLCALL_MODULE``  (default: ``evonic``)
  3. the ``base`` engine, if nothing else resolves.

Adding a host (e.g. ``hermes``) = drop ``hermes.py`` here that registers its
handler. No edits to the proxy or this file are required — submodules are
auto-discovered on first use.
"""

import importlib
import os
import pkgutil

from .base import BaseToolCallHandler

_REGISTRY = {}
_discovered = False


def register(handler):
    """Register a handler instance under its ``name``. Returns it for chaining."""
    _REGISTRY[handler.name] = handler
    return handler


def _discover():
    """Import every sibling module so its register() side effect runs."""
    global _discovered
    if _discovered:
        return
    pkg_dir = os.path.dirname(__file__)
    for mod in pkgutil.iter_modules([pkg_dir]):
        if mod.name in ("base",) or mod.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{mod.name}")
    _discovered = True


def get_handler(name=None):
    """Resolve a handler by explicit name, else env default, else base engine."""
    _discover()
    if name and name in _REGISTRY:
        return _REGISTRY[name]
    default = os.environ.get("CLAUDE_PROXY_TOOLCALL_MODULE", "evonic")
    return _REGISTRY.get(default) or _REGISTRY.get("base") or BaseToolCallHandler()


def available():
    """List registered handler names (for diagnostics)."""
    _discover()
    return sorted(_REGISTRY)


# Always have the generic engine available as a fallback name.
register(BaseToolCallHandler())
