"""`myelin` command line. Every subcommand works without a network or an API key
except `ask` and `agent`, which need ANTHROPIC_API_KEY."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import CONFIG_NAME, load_config

TEMPLATE = """# Myelin configuration. Every key is optional; these are the defaults.

[vault]
include = ["**/*.md"]
exclude_dirs = [".git", ".obsidian", ".trash", ".myelin", "node_modules"]
exclude_files = []
# Notes another program regenerates: read, never written, never edit targets.
protected = []
# report_note = "MYELIN_SELF_MODEL.md"   # publish the self-model inside the vault
inbox_dir = "inbox"

[server]
host = "127.0.0.1"
port = 7317

[hooks]
inject_context = true      # UserPromptSubmit hook adds matching notes to Claude's context
inject_min_score = 0.55    # raw cosine; below this nothing is injected
inject_max_notes = 3
"""


def _brain(args):
    from .brain import Brain
    cfg = load_config(Path(args.root).resolve() if args.root else None)
    return Brain(cfg)


def _engine(brain):
    from .consolidate import load_reranker
    from .recall import RecallEngine
    eng = RecallEngine(brain)
    load_reranker(brain, eng)
    return eng


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))
    elif isinstance(obj, str):
        print(obj)
    else:
        print(json.dumps(obj, indent=2, default=str))


def cmd_init(args) -> int:
    root = Path(args.root or ".").resolve()
    path = root / CONFIG_NAME
    if path.exists():
        print(f"{path} already exists; not overwriting")
        return 1
    path.write_text(TEMPLATE, encoding="utf-8")
    print(f"wrote {path}\nnext: myelin ingest && myelin sleep && myelin serve")
    return 0


def cmd_ingest(args) -> int:
    brain = _brain(args)
    report = brain.ingest(refit=args.refit)
    _print(report.__dict__, args.json)
    return 0 if not report.errors else 1


def cmd_recall(args) -> int:
    brain = _brain(args)
    if brain.store.one("SELECT COUNT(*) FROM notes")[0] == 0:
        brain.ingest()
    res = _engine(brain).recall(args.query, k=args.k, source="cli")
    if args.json:
        _print(res, True)
        return 0
    print(f"recall {res['recall_id']} · {res['ms']} ms · best similarity {res['best_similarity']}"
          + ("" if res["confident"] else "  (low confidence — logged as a knowledge gap)"))
    for i, r in enumerate(res["results"], 1):
        via = f"  ← {r['via']['kind']} link from {r['via']['seed_title']}" if r.get("via") else ""
        print(f"{i:>2}. {r['score']:.3f}  {r['title']}  [{r['path']}]{via}")
        if args.snippets and r["snippet"]:
            print(f"      {r['heading']}: {r['snippet'][:220]}")
    return 0


def cmd_sleep(args) -> int:
    from .consolidate import sleep
    brain = _brain(args)
    def show(name, entry):
        if not args.json:
            status = "ok" if entry["ok"] else f"FAILED {entry['error']}"
            print(f"  {name:<16} {entry['seconds']:>6.2f}s  {status}")
    report = sleep(brain, _engine(brain), generations=args.generations, on_step=show)
    if args.json:
        _print(report, True)
    else:
        print(f"sleep finished in {report['seconds']}s · ledger: {report['ledger']['message']}")
    return 0 if all(e["ok"] for e in report["steps"].values()) else 1


def cmd_quiz(args) -> int:
    from .selfeval import QuizRunner, ablations, build_quiz
    brain = _brain(args)
    eng = _engine(brain)
    runner = QuizRunner(eng, build_quiz(eng))
    version, p = brain.champion()
    out = {"config_version": version, "items": len(runner.items),
           "train": runner.evaluate(p, "train"), "holdout": runner.with_ci(p, "holdout"),
           "ablations": ablations(runner, p)}
    _print(out, args.json)
    return 0


def cmd_evolve(args) -> int:
    from .evolve import generation, rollback
    from .selfeval import QuizRunner, build_quiz
    brain = _brain(args)
    if args.rollback:
        _print(rollback(brain), args.json)
        return 0
    eng = _engine(brain)
    runner = QuizRunner(eng, build_quiz(eng))
    for _ in range(args.generations):
        res = generation(brain, runner, lam=brain.cfg.evolution_lambda)
        print(f"gen {res['generation']:>3}: {res['verdict']}" + (f"  {res['diff']}" if res["promoted"] else ""))
    return 0


def cmd_proposals(args) -> int:
    from . import proposals as P
    brain = _brain(args)
    items = P.listing(brain, args.status, args.kind, args.limit)
    if args.json:
        _print(items, True)
        return 0
    for p in items:
        pl = p["payload"]
        if p["kind"] == "link":
            desc = f"{pl['src_title']} → {pl['dst_title']}  ({pl.get('why', '')})"
        elif p["kind"] == "merge":
            desc = f"{pl['a_title']} ≈ {pl['b_title']}  (similarity {pl['similarity']})"
        elif p["kind"] == "gap":
            desc = f"“{pl['example']}” asked {pl['count']}×"
        else:
            desc = pl.get("title", p["key"])
        print(f"#{p['id']:<4} {p['kind']:<6} {p['score']:.3f} {p['method']:<12} {desc}")
    if not items:
        print(f"no {args.status} proposals")
    return 0


def cmd_decide(args) -> int:
    from . import proposals as P
    brain = _brain(args)
    out = P.decide(brain, args.id, accept=args.command == "accept", actor=args.actor, reason=args.reason or "")
    _print(out, args.json)
    return 0


def cmd_revert(args) -> int:
    from . import proposals as P
    _print(P.revert(_brain(args), args.id, args.actor), args.json)
    return 0


def cmd_feedback(args) -> int:
    from .learning import record_feedback
    brain = _brain(args)
    out = record_feedback(brain, args.recall_id, args.note, 1.0 if args.useful else 0.0, "cli")
    _print(out, args.json)
    return 0


def cmd_self(args) -> int:
    brain = _brain(args)
    path = brain.cfg.state / "self_model.json"
    if not path.exists():
        print("no self-model yet — run `myelin sleep` first")
        return 1
    model = json.loads(path.read_text(encoding="utf-8"))
    if args.json:
        _print(model, True)
        return 0
    from .selfmodel import render_markdown
    print(render_markdown(brain, model))
    return 0


def cmd_critique(args) -> int:
    from .quality import critique, rules_table, vault_baseline
    if args.rules:
        for r in rules_table():
            print(f"{r['id']}  [{r['severity']:<6}] thr={r['threshold']}  {r['title']}")
        return 0
    text = sys.stdin.read() if args.path in (None, "-") else Path(args.path).read_text(encoding="utf-8")
    brain = _brain(args)
    report = critique(text, vault_baseline(brain))
    if args.json:
        _print(report, True)
        return 1 if args.strict and report["blocking"] else 0
    print(f"score {report['score']} / 100 — {report['verdict']} ({report['words']} words)")
    for f in report["findings"]:
        lines = ", ".join(str(e["line"]) for e in f["examples"])
        print(f"  {f['rule']} [{f['severity']}] {f['title']} ×{f['count']} (lines {lines})")
        print(f"      e.g. “{f['examples'][0]['text']}” — {f['fix']}")
    v = report["voice"]
    if v.get("available"):
        print(f"  voice distance {v['distance']} from {v['baseline_notes']} of your notes")
        for d in v["deviations"]:
            print(f"      {d['feature']}: this text {d['this_text']}, your notes {d['your_notes']} (z {d['z']})")
    return 1 if args.strict and report["blocking"] else 0


def cmd_verify(args) -> int:
    ok, msg, n = _brain(args).store.verify_ledger()
    _print({"ok": ok, "message": msg, "events": n}, args.json)
    return 0 if ok else 2


def cmd_snapshot(args) -> int:
    from .snapshot import build_snapshot
    brain = _brain(args)
    out = build_snapshot(brain, _engine(brain), Path(args.out).resolve(), args.query or [], args.with_content)
    _print(out, args.json)
    return 0


def cmd_app(args) -> int:
    from .native import readiness, run_app
    if args.check:
        problem = readiness()
        print(problem or "ready: this interpreter can open the Myelin window")
        return 1 if problem else 0
    if args.browser:
        # The same interface in a browser tab, served locally: for machines where
        # no native web view is available.
        from .server import serve
        return serve(_brain(args))
    return run_app(_brain(args), debug=args.debug, tour=args.tour)


def cmd_install_app(args) -> int:
    from .native import plan_launcher, readiness
    cfg = load_config(Path(args.root).resolve() if args.root else None)
    python = args.python or sys.executable
    plan = plan_launcher(cfg, python=python, desktop_shortcut=args.desktop)
    print(plan.describe())
    if not args.apply:
        print("dry run: nothing written. Re-run with --apply to install.")
        return 0
    problem = _readiness_of(python) if args.python else readiness()
    if problem:
        print(f"not installing: {python} cannot open the Myelin window yet.\n{problem}")
        return 1
    plan.apply()
    print("installed. Look for Myelin among your applications.")
    return 0


def _readiness_of(python: str) -> str | None:
    """Ask another interpreter, the one the launcher will run, whether it can
    open the window. A launcher for an interpreter that cannot is a broken icon."""
    import subprocess
    from .native import PACKAGE
    try:
        out = subprocess.run([python, "-m", "myelin", "app", "--check"], cwd=PACKAGE.parent,
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"could not run {python}: {exc}"
    return None if out.returncode == 0 else (out.stdout + out.stderr).strip()


def cmd_serve(args) -> int:
    from .server import serve
    brain = _brain(args)
    return serve(brain, host=args.host, port=args.port, background=not args.no_daemon)


def cmd_daemon(args) -> int:
    from .daemon import run_daemon
    return run_daemon(_brain(args), once=args.once)


def cmd_mcp(args) -> int:
    from .mcp_server import run
    return run(load_config(Path(args.root).resolve() if args.root else None))


def cmd_hook(args) -> int:
    from .hooks import main as hook_main
    return hook_main(args.event, Path(args.root).resolve() if args.root else None)


def cmd_claude(args) -> int:
    from .install import install
    return install(load_config(Path(args.root).resolve() if args.root else None), apply=args.apply)


def cmd_ask(args) -> int:
    from .agent import ask
    brain = _brain(args)
    out = ask(brain, _engine(brain), args.question, model=args.model)
    if args.json:
        _print(out, True)
    else:
        print(out["answer"])
        print(f"\n— {out['turns']} turns · {out['usage']['input_tokens']} in / {out['usage']['output_tokens']} out tokens"
              f" · cited: {', '.join(out['cited']) or 'nothing'}")
    return 0


def cmd_agent(args) -> int:
    from .agent import review_proposals
    brain = _brain(args)
    _print(review_proposals(brain, _engine(brain), limit=args.limit, model=args.model), args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="myelin", description="A self-reinforcing knowledge brain over a markdown vault.")
    ap.add_argument("--root", help="vault root (default: nearest directory with myelin.toml, else cwd)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="write a myelin.toml template").set_defaults(fn=cmd_init)
    p = sub.add_parser("ingest", help="index changed notes")
    p.add_argument("--refit", action="store_true", help="refit the semantic space from scratch")
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("recall", help="ask the brain")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--snippets", action="store_true")
    p.set_defaults(fn=cmd_recall)
    p = sub.add_parser("sleep", help="run one consolidation cycle (train, evolve, publish)")
    p.add_argument("--generations", type=int, default=3)
    p.set_defaults(fn=cmd_sleep)
    sub.add_parser("quiz", help="score the current recall config on the self-quiz").set_defaults(fn=cmd_quiz)
    p = sub.add_parser("evolve", help="run evolution generations, or roll back the last promotion")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--rollback", action="store_true")
    p.set_defaults(fn=cmd_evolve)
    p = sub.add_parser("proposals", help="list staged changes")
    p.add_argument("--status", default="pending")
    p.add_argument("--kind")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(fn=cmd_proposals)
    for name in ("accept", "reject"):
        p = sub.add_parser(name, help=f"{name} a proposal")
        p.add_argument("id", type=int)
        p.add_argument("--reason")
        p.add_argument("--actor", default="cli")
        p.set_defaults(fn=cmd_decide)
    p = sub.add_parser("revert", help="undo an applied link proposal")
    p.add_argument("id", type=int)
    p.add_argument("--actor", default="cli")
    p.set_defaults(fn=cmd_revert)
    p = sub.add_parser("feedback", help="label a recall result")
    p.add_argument("recall_id")
    p.add_argument("note")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--useful", action="store_true")
    g.add_argument("--not-useful", action="store_true")
    p.set_defaults(fn=cmd_feedback)
    sub.add_parser("self", help="print the self-model").set_defaults(fn=cmd_self)
    sub.add_parser("verify", help="verify the hash-chained ledger").set_defaults(fn=cmd_verify)
    p = sub.add_parser("critique", help="authenticity check for prose (file path, or - for stdin)")
    p.add_argument("path", nargs="?")
    p.add_argument("--strict", action="store_true", help="exit 1 on any high-severity finding (CI gate)")
    p.add_argument("--rules", action="store_true", help="list the rules")
    p.set_defaults(fn=cmd_critique)
    p = sub.add_parser("app", help="open Myelin in its own desktop window (also runs the background scheduler)")
    p.add_argument("--check", action="store_true", help="report whether this machine can open the window")
    p.add_argument("--tour", action="store_true", help="play the guided tour on opening")
    p.add_argument("--debug", action="store_true", help="enable the web view's developer tools")
    p.add_argument("--browser", action="store_true", help="no native web view here: serve the same UI locally instead")
    p.set_defaults(fn=cmd_app)
    p = sub.add_parser("install-app", help="add Myelin to the applications menu with its icon (dry run unless --apply)")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--desktop", action="store_true", help="also put a shortcut on the desktop")
    p.add_argument("--python", help="interpreter the launcher should run (default: this one)")
    p.set_defaults(fn=cmd_install_app)
    p = sub.add_parser("serve", help="web UI + API (also runs the background scheduler)")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--no-daemon", action="store_true", help="UI only; no file watching or scheduled sleep")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("snapshot", help="export a read-only, server-free copy of the UI and brain data")
    p.add_argument("out")
    p.add_argument("--query", action="append", help="preset question to include (repeatable)")
    p.add_argument("--with-content", action="store_true", help="include full note text (read it before sharing)")
    p.set_defaults(fn=cmd_snapshot)
    p = sub.add_parser("daemon", help="watch the vault and sleep on schedule, no UI")
    p.add_argument("--once", action="store_true")
    p.set_defaults(fn=cmd_daemon)
    sub.add_parser("mcp", help="MCP server on stdio for Claude Code").set_defaults(fn=cmd_mcp)
    p = sub.add_parser("hook", help="Claude Code hook entry point (reads the payload on stdin)")
    p.add_argument("event", choices=["prompt", "read", "stop"])
    p.set_defaults(fn=cmd_hook)
    p = sub.add_parser("claude", help="wire Myelin into Claude Code (dry-run unless --apply)")
    # `claude install` is the spelling the docs use; bare `claude` came first and stays valid.
    p.add_argument("action", nargs="?", choices=["install"], default="install")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(fn=cmd_claude)
    p = sub.add_parser("ask", help="grounded answer from Claude using the brain as tools")
    p.add_argument("question")
    p.add_argument("--model")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("agent", help="Claude reviews pending link proposals and records its verdicts")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--model")
    p.set_defaults(fn=cmd_agent)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except (KeyError, ValueError, PermissionError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"myelin: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
