"""HTTP server security, Claude Code hooks, the API agent loop, MCP and the installer."""
import asyncio
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from myelin import agent as AG
from myelin import hooks
from myelin import proposals as P
from myelin.install import plan
from myelin.recall import RecallEngine
from myelin.server import App, make_handler
from myelin.tools import BrainTools


@pytest.fixture()
def server(brain):
    app = App(brain, "127.0.0.1", 0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    port = httpd.server_address[1]
    app.port = port
    app.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield app, f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def _req(url, token=None, method="GET", body=None, headers=None):
    h = dict(headers or {})
    if token:
        h["X-Myelin-Token"] = token
    data = json.dumps(body).encode() if body is not None else None
    if data is not None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_server_requires_token_and_checks_host(server):
    app, base = server
    assert _req(base + "/api/status")[0] == 401
    assert _req(base + "/api/status", "wrong")[0] == 401
    assert _req(base + "/api/status", app.token, headers={"Host": "evil.example:80"})[0] == 421
    status, headers, body = _req(base + "/api/status", app.token)
    assert status == 200 and json.loads(body)["notes"] > 0
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_server_page_carries_token_and_blocks_traversal(server):
    app, base = server
    status, _, body = _req(base + "/")
    assert status == 200 and app.token.encode() in body
    assert _req(base + "/../myelin/server.py")[0] == 404
    assert _req(base + "/%2e%2e/server.py")[0] == 404


def test_server_recall_feedback_and_decisions(server):
    app, base = server
    status, _, body = _req(base + "/api/recall?q=keel+harbour+mooring", app.token)
    res = json.loads(body)
    assert status == 200 and res["results"]
    note_id = res["results"][0]["note_id"]
    assert _req(base + "/api/feedback", app.token, "POST",
                {"recall_id": res["recall_id"], "note_id": note_id, "useful": True})[0] == 200
    BrainTools(app.brain, app.engine).propose_link("SAIL_2", "BAKE_2", "test")
    pid = P.listing(app.brain, "pending", "link")[0]["id"]
    refused = _req(base + "/api/decide", app.token, "POST", {"id": pid, "accept": True},
                   headers={"Origin": "http://evil.example"})
    assert refused[0] == 403
    status, _, body = _req(base + "/api/decide", app.token, "POST", {"id": pid, "accept": True})
    assert status == 200 and json.loads(body)["status"] == "applied"
    status, _, body = _req(base + "/api/graph", app.token)
    graph = json.loads(body)
    assert graph["nodes"] and graph["edges"]
    status, _, body = _req(base + "/api/note?id=SAIL_2", app.token)
    assert status == 200 and "[[BAKE_2]]" in json.loads(body)["content"]
    assert _req(base + "/api/note?id=NOPE", app.token)[0] == 404


def _run_hook(event, payload, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload) if not isinstance(payload, str) else payload))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert hooks.main(event) == 0
    return out.getvalue()


def test_prompt_hook_injects_only_when_confident(brain, monkeypatch):
    root = str(brain.cfg.root)
    out = _run_hook("prompt", {"prompt": "what do my notes say about sourdough levain hydration",
                               "cwd": root, "session_id": "h1"}, monkeypatch)
    assert out.startswith("<myelin_recall") and "notes/BAKE_" in out
    assert _run_hook("prompt", {"prompt": "zebra quantum lattice chromodynamics please",
                                "cwd": root, "session_id": "h1"}, monkeypatch) == ""
    assert _run_hook("prompt", {"prompt": "/clear", "cwd": root}, monkeypatch) == ""
    assert _run_hook("prompt", "not json", monkeypatch) == ""


def test_read_hook_records_access(brain, monkeypatch):
    path = str(brain.cfg.root / "notes" / "ASTRO_2.md")
    _run_hook("read", {"tool_name": "Read", "tool_input": {"file_path": path}, "session_id": "h2",
                       "cwd": str(brain.cfg.root)}, monkeypatch)
    assert brain.store.one("SELECT COUNT(*) FROM access WHERE session='h2'")[0] == 1


def _block(kind, **kw):
    return SimpleNamespace(type=kind, **kw)


class FakeClient:
    """Scripted stand-in for anthropic.Anthropic(): records requests, replays responses."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=lambda o: o.__dict__)))
        return self.script.pop(0)


def _resp(stop, content, i=1000, o=200, cr=0):
    usage = SimpleNamespace(input_tokens=i, output_tokens=o, cache_read_input_tokens=cr,
                            cache_creation_input_tokens=0)
    return SimpleNamespace(stop_reason=stop, content=content, usage=usage, model="claude-opus-5-5", stop_details=None)


def test_agent_loop_uses_tools_and_cites(brain):
    client = FakeClient([
        _resp("tool_use", [_block("tool_use", id="t1", name="recall", input={"query": "keel tack", "k": 3})]),
        _resp("tool_use", [_block("tool_use", id="t2", name="read_note", input={"note": "SAIL_0"}),
                           _block("tool_use", id="t3", name="nope", input={})]),
        _resp("end_turn", [_block("text", text="The keel notes are in [[SAIL_0]] and [[MISSING]].")], cr=800),
    ])
    out = AG.ask(brain, RecallEngine(brain), "what about keels?", client=client)
    assert out["stop_reason"] == "end_turn" and out["turns"] == 3
    assert out["cited"] == ["SAIL_0"]
    first = client.requests[0]
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    assert first["betas"] == [AG.FALLBACK_BETA]
    assert all(t["strict"] for t in first["tools"])
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    results = client.requests[2]["messages"][-1]["content"]
    assert any(r.get("is_error") for r in results)          # unknown tool reported, not crashed
    assert "<note_content>" in results[0]["content"]
    assert out["usage"]["cost_usd"] > 0


def test_agent_stops_on_refusal_and_budget(brain):
    refusal = _resp("refusal", [])
    refusal.stop_details = SimpleNamespace(category="cyber")
    out = AG.ask(brain, RecallEngine(brain), "q", client=FakeClient([refusal]))
    assert out["stop_reason"] == "refusal" and "cyber" in out["answer"]
    brain.cfg.agent_budget_usd = 0.0001
    loop = [_resp("tool_use", [_block("tool_use", id=f"t{i}", name="recall", input={"query": "x", "k": 1})],
                  i=100000) for i in range(5)]
    out = AG.ask(brain, RecallEngine(brain), "q", client=FakeClient(loop))
    assert out["stop_reason"] == "budget"


def test_review_records_verdicts_without_deciding(brain):
    BrainTools(brain, RecallEngine(brain)).propose_link("ASTRO_1", "ASTRO_4", "x")
    verdict = json.dumps({"verdict": "accept", "reason": "both cover parallax"})
    out = AG.review_proposals(brain, RecallEngine(brain), client=FakeClient([_resp("end_turn", [_block("text", text=verdict)])]))
    assert out["reviewed"][0]["verdict"] == "accept"
    prop = P.listing(brain, "pending", "link")[0]
    assert prop["status"] == "pending" and prop["payload"]["review"]["verdict"] == "accept"


def test_mcp_server_exposes_the_tool_set(brain):
    pytest.importorskip("mcp")   # the optional `myelin[mcp]` extra
    from myelin.mcp_server import build
    mcp = build(brain.cfg)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {"recall", "read_note", "neighbors", "suggest_links", "propose_link", "propose_note",
                     "critique", "feedback", "brain_status"}


def test_installer_merges_and_is_idempotent(brain, tmp_path):
    settings = brain.cfg.root / ".claude" / "settings.json"
    settings.parent.mkdir()
    existing = {"permissions": {"allow": ["Bash(ls)"]},
                "hooks": {"PostToolUse": [{"matcher": "Write", "hooks": [{"type": "command", "command": "x.sh"}]}]}}
    settings.write_text(json.dumps(existing))
    p = plan(brain.cfg)
    _, before, after = p["settings"]
    assert after["permissions"] == existing["permissions"]
    post = after["hooks"]["PostToolUse"]
    assert post[0]["hooks"][0]["command"] == "x.sh" and len(post) == 2
    assert set(after["hooks"]) == {"PostToolUse", "UserPromptSubmit", "Stop"}
    settings.write_text(json.dumps(after))
    (brain.cfg.root / ".mcp.json").write_text(json.dumps(p["mcp"][2]))
    again = plan(brain.cfg)
    assert again["settings"][1] == again["settings"][2] and again["mcp"][1] == again["mcp"][2]
    # The documented `claude install` and the bare `claude` both reach the installer.
    from myelin.cli import build_parser
    for argv in (["claude", "install", "--apply"], ["claude", "--apply"], ["claude", "install"]):
        args = build_parser().parse_args(argv)
        assert args.fn.__name__ == "cmd_claude" and args.apply is ("--apply" in argv)


def test_snapshot_is_static_scoped_and_safe(brain, tmp_path):
    from myelin.snapshot import build_snapshot
    out = tmp_path / "snap"
    info = build_snapshot(brain, RecallEngine(brain), out, ["sourdough levain proofing"])
    assert info["notes"] == brain.store.one("SELECT COUNT(*) FROM notes")[0] and info["queries"] == 1
    html = (out / "index.html").read_text()
    assert 'name="myelin-mode" content="snapshot"' in html and "%%MYELIN_TOKEN%%" not in html
    assert (out / "js" / "app.js").is_file() and (out / "app.css").is_file()
    data = json.loads((out / "snapshot.json").read_text())
    # Note bodies stay out; the vault is named, not located.
    assert all(n["content"] == "" and n["content_omitted"] for n in data["notes"].values())
    assert data["status"]["root"] == "vault" and data["status"]["snapshot"] is True
    assert data["recalls"]["sourdough levain proofing"]["results"]
    assert set(data["proposals"]) == {"pending", "applied", "rejected"}
    # Re-exporting over a snapshot is fine; over anything else it refuses.
    build_snapshot(brain, RecallEngine(brain), out, [])
    keep = tmp_path / "precious"
    keep.mkdir()
    (keep / "note.md").write_text("mine")
    with pytest.raises(ValueError):
        build_snapshot(brain, RecallEngine(brain), keep, [])
    assert (keep / "note.md").read_text() == "mine"
