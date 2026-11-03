"""Local web UI and JSON API. Standard library only.

Security posture, because a local server that can edit notes is an attack
surface for every web page the user has open:

* binds 127.0.0.1 by default;
* every `/api` call needs a per-process random token, delivered only inside the
  served page (cross-origin pages cannot read it, so they cannot forge calls);
* the `Host` header must name this server — defeats DNS rebinding;
* a state-changing request with a foreign `Origin` is refused;
* request bodies are capped; static files are confined to the UI directory;
* strict CSP: no inline script, no third-party origins. Note bodies are
  rendered by the client's own escaping markdown renderer, never as raw HTML.
"""
from __future__ import annotations

import contextlib
import json
import mimetypes
import os
import queue
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, urlparse

from . import locks
from . import proposals as P
from .brain import Brain
from .consolidate import load_reranker, structure
from .daemon import Scheduler
from .learning import record_feedback
from .recall import RecallEngine

UI_DIR = Path(__file__).resolve().parent / "ui"
MAX_BODY = 256 * 1024
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


class Hub:
    """In-process fan-out of live events to SSE clients."""

    def __init__(self) -> None:
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def publish(self, kind: str, data: dict[str, Any]) -> None:
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait({"kind": kind, "data": data, "ts": time.time()})
                except queue.Full:
                    pass


class App:
    def __init__(self, brain: Brain, host: str, port: int):
        self.brain = brain
        self.engine = RecallEngine(brain)
        load_reranker(brain, self.engine)
        self.host, self.port = host, port
        self.token = secrets.token_urlsafe(24)
        self.hub = Hub()
        self.scheduler: Scheduler | None = None
        self.allowed_hosts = {f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]", host)}

    # ── read endpoints ───────────────────────────────────────────────────
    def status(self, _: dict) -> dict[str, Any]:
        s = self.brain.store
        version, params = self.brain.champion()
        ok, msg, n = s.verify_ledger(max_age=5.0)   # polled; `ledger` and the CLI always re-hash
        heartbeat = float(s.get_meta("daemon_heartbeat", "0") or 0)
        sleeping = bool(self.scheduler and self.scheduler.sleeping.locked())
        busy = sleeping or bool(self.scheduler and self.scheduler.indexing.is_set())
        return {
            "root": str(self.brain.cfg.root), "notes": s.one("SELECT COUNT(*) FROM notes")[0],
            "passages": s.one("SELECT COUNT(*) FROM passages")[0],
            "edges": {r[0]: r[1] for r in s.q("SELECT kind, COUNT(*) FROM edges GROUP BY kind")},
            "pending": {r[0]: r[1] for r in s.q("SELECT kind, COUNT(*) FROM proposals WHERE status='pending' GROUP BY kind")},
            "champion": {"version": version, "params": params.to_dict()},
            "holdout_mrr": s.latest_metric("quiz.mrr.holdout"),
            "ledger": {"ok": ok, "message": msg, "events": n},
            "last_sleep": float(s.get_meta("last_sleep", "0") or 0),
            "last_ingest": float(s.get_meta("last_ingest", "0") or 0),
            "daemon_alive": busy or time.time() - heartbeat < max(30.0, 4 * self.brain.cfg.poll_seconds),
            "sleeping": sleeping,
            "agent_available": _agent_available(),
        }

    def graph(self, _: dict) -> dict[str, Any]:
        s = self.brain.store
        st = structure(self.brain)
        g = self.brain.graph()
        nodes = [{"id": r["id"], "title": r["title"], "stem": r["stem"], "path": r["path"], "type": r["type"],
                  "words": r["words"], "community": st["community"].get(r["id"], -1),
                  "rank": round(st["pagerank"].get(r["id"], 0.0), 6), "degree": g.degree(r["id"])}
                 for r in s.q("SELECT id,title,stem,path,type,words FROM notes ORDER BY id")]
        edges = []
        for r in s.q("SELECT src,dst,kind,weight FROM edges"):
            edges.append({"s": r[0], "t": r[1], "kind": r[2], "w": round(r[3], 3)})
        pending = []
        ids = {r["path"]: r["id"] for r in s.q("SELECT id,path FROM notes")}
        for p in P.listing(self.brain, "pending", "link", 200):
            a, b = ids.get(p["payload"]["src_path"]), ids.get(p["payload"]["dst_path"])
            if a and b:
                pending.append({"s": a, "t": b, "kind": "proposed", "w": p["score"], "id": p["id"]})
        return {"nodes": nodes, "edges": edges + pending,
                "communities": {str(k): v for k, v in st["labels"].items()}}

    def note(self, q: dict) -> dict[str, Any]:
        row = self.brain.store.note_row(_one(q, "id"))
        if row is None:
            raise KeyError("no such note")
        try:
            text = (self.brain.cfg.root / row["path"]).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            raise KeyError(f"{row['path']} was removed from the vault since it was indexed") from None
        ids = {r["id"]: (r["title"], r["stem"]) for r in self.brain.store.q("SELECT id,title,stem FROM notes")}
        nb = []
        for v, w, kind, fwd in self.brain.graph().neighbors(row["id"]):
            nb.append({"id": v, "title": ids[v][0], "kind": kind, "forward": fwd, "w": round(w, 3)})
        nb.sort(key=lambda x: -x["w"])
        props = [p for p in P.listing(self.brain, "pending", "link", 500)
                 if row["path"] in (p["payload"]["src_path"], p["payload"]["dst_path"])]
        broken = [r[0] for r in self.brain.store.q("SELECT DISTINCT target_raw FROM links WHERE src=? AND target_path IS NULL",
                                                   (row["id"],))]
        stems = {r["stem"]: r["id"] for r in self.brain.store.q("SELECT id,stem FROM notes")}
        return {"id": row["id"], "title": row["title"], "path": row["path"], "stem": row["stem"], "type": row["type"],
                "words": row["words"], "updated_at": row["updated_at"], "tags": json.loads(row["tags_json"]),
                "frontmatter_error": row["frontmatter_error"], "content": text[:200_000], "neighbors": nb[:40],
                "proposals": props, "broken_links": broken, "stems": stems,
                "protected": P.is_protected(self.brain.cfg, row["path"])}

    def recall(self, q: dict) -> dict[str, Any]:
        query = _one(q, "q").strip()
        if not query:
            raise ValueError("empty query")
        return self.engine.recall(query[:2000], k=_clamp(_one(q, "k", "10"), 1, 30), source="ui")

    def proposals(self, q: dict) -> dict[str, Any]:
        return {"items": P.listing(self.brain, _one(q, "status", "pending"), _one(q, "kind", "") or None,
                                   _clamp(_one(q, "limit", "100"), 1, 500))}

    def selfmodel(self, _: dict) -> dict[str, Any]:
        path = self.brain.cfg.state / "self_model.json"
        if not path.exists():
            return {"missing": True}
        return json.loads(path.read_text(encoding="utf-8"))

    def metrics(self, q: dict) -> dict[str, Any]:
        names = _one(q, "names", "quiz.mrr.holdout,quiz.mrr.train").split(",")
        return {n: self.brain.store.metric_series(n, 300) for n in names[:8]}

    def evolution(self, _: dict) -> dict[str, Any]:
        from .config import GENOME_BOUNDS, RecallParams
        rows = self.brain.store.q("SELECT version,parent,status,fitness_json,created_at,params_json FROM configs "
                                  "ORDER BY version DESC LIMIT 200")
        last = self.brain.store.one("SELECT payload_json, created_at FROM events WHERE kind='SLEEP' "
                                    "ORDER BY sequence DESC LIMIT 1")
        return {"configs": [{"version": r[0], "parent": r[1], "status": r[2], "fitness": json.loads(r[3]),
                             "created_at": r[4], "params": json.loads(r[5])} for r in rows],
                "bounds": {k: [lo, hi, integer] for k, (lo, hi, integer) in GENOME_BOUNDS.items()},
                "defaults": RecallParams().to_dict(),
                "last_sleep": {**json.loads(last[0]), "at": last[1]} if last else None}

    def ledger(self, q: dict) -> dict[str, Any]:
        ok, msg, n = self.brain.store.verify_ledger()
        return {"verified": {"ok": ok, "message": msg, "events": n},
                "events": self.brain.store.events(_clamp(_one(q, "limit", "200"), 1, 1000),
                                                  _clamp(_one(q, "after", "0"), 0, 2 ** 62))}

    # ── live events ──────────────────────────────────────────────────────
    def stream(self, stop: threading.Event | None = None, idle: float = 1.0) -> Iterator[tuple[str, Any]]:
        """Live events for one subscriber, shared by the server's event stream
        and the desktop window: the in-process hub (sleep progress, ingest) and
        the ledger, polled because its events can come from other processes (the
        CLI, the MCP server). Yields ("", None) when idle, so a consumer can send
        a keep-alive or notice that it should stop."""
        q = self.hub.subscribe()
        after = self.brain.store.one("SELECT COALESCE(MAX(sequence),0) FROM events")[0]
        try:
            while stop is None or not stop.is_set():
                for ev in self.brain.store.events(50, after):
                    after = ev["sequence"]
                    yield "ledger", ev
                try:
                    item = q.get(timeout=idle)
                    yield item["kind"], item["data"]
                except queue.Empty:
                    yield "", None
        finally:
            self.hub.unsubscribe(q)

    # ── write endpoints ──────────────────────────────────────────────────
    def feedback(self, body: dict) -> dict[str, Any]:
        out = record_feedback(self.brain, str(body["recall_id"]), int(body["note_id"]),
                              1.0 if body.get("useful") else 0.0, "ui")
        self.hub.publish("feedback", out)
        return out

    def decide(self, body: dict) -> dict[str, Any]:
        out = P.decide(self.brain, int(body["id"]), bool(body["accept"]), "ui", str(body.get("reason", ""))[:500])
        self.hub.publish("proposal", out)
        return out

    def revert(self, body: dict) -> dict[str, Any]:
        out = P.revert(self.brain, int(body["id"]), "ui")
        self.hub.publish("proposal", {"id": body["id"], "status": "reverted"})
        return out

    def critique(self, body: dict) -> dict[str, Any]:
        from .quality import critique, vault_baseline
        return critique(str(body.get("text", ""))[:100_000], vault_baseline(self.brain))

    def sleep(self, _: dict) -> dict[str, Any]:
        if self.scheduler is None:
            raise RuntimeError("this server does not run the scheduler (--no-daemon, or another Myelin process "
                               "owns it); run `myelin sleep` instead")
        if self.scheduler.sleeping.locked():
            return {"started": False, "reason": "a sleep cycle is already running"}
        threading.Thread(target=self.scheduler.run_sleep, daemon=True).start()
        return {"started": True}

    def ask(self, body: dict) -> dict[str, Any]:
        from .agent import ask
        question = str(body.get("question", "")).strip()[:4000]
        if not question:
            raise ValueError("empty question")
        return ask(self.brain, self.engine, question)

    def routes(self) -> dict[tuple[str, str], Callable[[dict], dict]]:
        return {
            ("GET", "/api/status"): self.status, ("GET", "/api/graph"): self.graph,
            ("GET", "/api/note"): self.note, ("GET", "/api/recall"): self.recall,
            ("GET", "/api/proposals"): self.proposals, ("GET", "/api/selfmodel"): self.selfmodel,
            ("GET", "/api/metrics"): self.metrics, ("GET", "/api/evolution"): self.evolution,
            ("GET", "/api/ledger"): self.ledger,
            ("POST", "/api/feedback"): self.feedback, ("POST", "/api/decide"): self.decide,
            ("POST", "/api/revert"): self.revert, ("POST", "/api/critique"): self.critique,
            ("POST", "/api/sleep"): self.sleep, ("POST", "/api/ask"): self.ask,
        }


def _one(q: dict, key: str, default: str | None = None) -> str:
    values = q.get(key)
    if not values:
        if default is None:
            raise ValueError(f"missing parameter {key!r}")
        return default
    return values[0]


def _clamp(value: str, lo: int, hi: int) -> int:
    return max(lo, min(int(value), hi))


def _agent_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or (Path.home() / ".config" / "anthropic").exists())


def make_handler(app: App):
    routes = app.routes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "myelin"
        sys_version = ""

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            if os.environ.get("MYELIN_HTTP_LOG"):
                super().log_message(fmt, *args)

        def _headers(self, status: int, ctype: str, length: int, extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def _json(self, status: int, obj: Any) -> None:
            body = json.dumps(obj, default=str).encode()
            self._headers(status, "application/json; charset=utf-8", len(body))
            self.wfile.write(body)

        def _guard(self, url) -> bool:
            if self.headers.get("Host", "") not in app.allowed_hosts:
                self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "unexpected Host header"})
                return False
            if url.path.startswith("/api/"):
                token = self.headers.get("X-Myelin-Token") or _one(parse_qs(url.query), "token", "")
                # Bytes, so a non-ASCII header is a wrong token, not a crashed handler.
                if not secrets.compare_digest(token.encode("utf-8", "replace"), app.token.encode()):
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "missing or wrong token"})
                    return False
            return True

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if not self._guard(url):
                return
            if url.path == "/api/events":
                return self._sse()
            handler = routes.get(("GET", url.path))
            if handler:
                return self._call(handler, parse_qs(url.query))
            return self._static(url.path)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            if not self._guard(url):
                return
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc not in app.allowed_hosts:
                return self._json(HTTPStatus.FORBIDDEN, {"error": "cross-origin request refused"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
            if length > MAX_BODY:
                return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "body must be a JSON object"})
            handler = routes.get(("POST", url.path))
            if not handler:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "no such endpoint"})
            self._call(handler, body)

        def _call(self, handler: Callable[[dict], dict], arg: dict) -> None:
            try:
                self._json(HTTPStatus.OK, handler(arg))
            except KeyError as exc:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(exc).strip("'\"")})
            except (ValueError, TypeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except (PermissionError, FileExistsError) as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except Exception as exc:  # surface the failure; never a blank 500
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

        def _static(self, path: str) -> None:
            if path in ("", "/"):
                html = (UI_DIR / "index.html").read_text(encoding="utf-8").replace("%%MYELIN_TOKEN%%", app.token)
                body = html.encode()
                self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(body))
                self.wfile.write(body)
                return
            target = (UI_DIR / path.lstrip("/")).resolve()
            if UI_DIR not in target.parents or not target.is_file():
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix == ".js":
                ctype = "text/javascript; charset=utf-8"
            data = target.read_bytes()
            self._headers(HTTPStatus.OK, ctype, len(data))
            self.wfile.write(data)

        def _sse(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                with contextlib.closing(app.stream()) as events:
                    for kind, data in events:
                        if kind:
                            self._send(kind, data)
                        else:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _send(self, kind: str, data: Any) -> None:
            self.wfile.write(f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n".encode())
            self.wfile.flush()

    return Handler


def attach_scheduler(app: App, stop: threading.Event) -> Any:
    """Run the scheduler in this process unless another Myelin process already
    schedules this vault (one per vault: `myelin daemon`, a server, or the desktop
    app). Returns the lock handle to release on exit, or None."""
    owner = locks.try_lock(app.brain.cfg.state / "daemon.lock")
    if owner is None:
        return None
    worker_brain = Brain(app.brain.cfg)   # its own SQLite connection for the worker thread
    app.scheduler = Scheduler(worker_brain, stop, notify=app.hub.publish)
    threading.Thread(target=app.scheduler.loop, daemon=True, name="myelin-scheduler").start()
    return owner


def serve(brain: Brain, host: str | None = None, port: int | None = None, background: bool = True) -> int:
    host = host or brain.cfg.host
    port = port or brain.cfg.port
    if brain.store.one("SELECT COUNT(*) FROM notes")[0] == 0:
        print("first run: indexing the vault…")
        brain.ingest()
    app = App(brain, host, port)
    stop = threading.Event()
    owner = attach_scheduler(app, stop) if background else None
    if background and owner is None:
        # One scheduler per vault. Another process (`myelin daemon` or a second
        # server) already has it; its work still shows up here through the ledger.
        print("note: another Myelin process schedules this vault; this server will not start a second scheduler")
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: listening on {host}, reachable beyond this machine; the token is the only gate")
    print(f"Myelin on http://{'localhost' if host == '127.0.0.1' else host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
        locks.release(owner)
    return 0
