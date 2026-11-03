"""`myelin claude install` — wire Myelin into Claude Code for this vault.

Dry-run by default, like the vault's other installers (`install_obsidian_plugins.py`,
`build_subagents.py`): it prints exactly what it would change and writes only
with `--apply`. Existing hooks and MCP servers are merged, never replaced, and
re-running is a no-op.

Writes:
  .mcp.json                  the `myelin` stdio MCP server
  .claude/settings.json      UserPromptSubmit / PostToolUse(Read) / Stop hooks
"""
from __future__ import annotations

import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import Config
from .store import atomic_write

PKG_PARENT = Path(__file__).resolve().parent.parent   # the directory holding the `myelin` package


def _command(cfg: Config, *args: str) -> list[str]:
    exe = shutil.which("myelin")
    if exe:
        return [exe, "--root", str(cfg.root), *args]
    return [sys.executable, "-m", "myelin", "--root", str(cfg.root), *args]


def _env() -> dict[str, str]:
    return {} if shutil.which("myelin") else {"PYTHONPATH": str(PKG_PARENT)}


def _shell(cfg: Config, *args: str) -> str:
    env = " ".join(f"{k}={shlex.quote(v)}" for k, v in _env().items())
    return (env + " " if env else "") + " ".join(shlex.quote(a) for a in _command(cfg, *args))


def plan(cfg: Config) -> dict[str, Any]:
    mcp_path = cfg.root / ".mcp.json"
    settings_path = cfg.root / ".claude" / "settings.json"
    mcp = _load(mcp_path)
    settings = _load(settings_path)

    server = {"command": _command(cfg, "mcp")[0], "args": _command(cfg, "mcp")[1:]}
    if _env():
        server["env"] = _env()
    new_mcp = json.loads(json.dumps(mcp))
    new_mcp.setdefault("mcpServers", {})["myelin"] = server

    wanted = {
        "UserPromptSubmit": (None, _shell(cfg, "hook", "prompt"), 15),
        "PostToolUse": ("Read", _shell(cfg, "hook", "read"), 10),
        "Stop": (None, _shell(cfg, "hook", "stop"), 60),
    }
    new_settings = json.loads(json.dumps(settings))
    hooks = new_settings.setdefault("hooks", {})
    for event, (matcher, command, timeout) in wanted.items():
        groups = hooks.setdefault(event, [])
        present = any("myelin" in h.get("command", "") and " hook " in h.get("command", "")
                      for g in groups for h in g.get("hooks", []))
        if present:
            continue
        group: dict[str, Any] = {"hooks": [{"type": "command", "command": command, "timeout": timeout}]}
        if matcher:
            group["matcher"] = matcher
        groups.append(group)
    return {"mcp": (mcp_path, mcp, new_mcp), "settings": (settings_path, settings, new_settings)}


def install(cfg: Config, apply: bool = False) -> int:
    if not cfg.db_path.exists():
        print("no brain yet — run `myelin ingest` (and ideally `myelin sleep`) first; hooks stay silent until then")
    changes = 0
    for name, (path, before, after) in plan(cfg).items():
        if before == after:
            print(f"{path.relative_to(cfg.root)}: already configured")
            continue
        changes += 1
        print(f"{path.relative_to(cfg.root)}: {'writing' if apply else 'would write'}")
        print("  " + json.dumps(after, indent=2).replace("\n", "\n  "))
        if apply:
            atomic_write(path, json.dumps(after, indent=2) + "\n")
    if changes and not apply:
        print("\ndry run — re-run with --apply to write these files")
    elif changes:
        print("\ndone. Restart Claude Code in this directory and approve the `myelin` MCP server when asked.")
    return 0


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON; fix it before installing ({exc})") from exc
