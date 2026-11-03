"""The desktop app: one bundled document, the in-process bridge, live events, and
the launchers. The window itself needs a display and a web view; everything it
runs is tested here without one."""
import json
import os
import plistlib
import queue
import re
import shutil
import subprocess
import sys
import threading

import pytest

from myelin import native as N
from myelin.server import App


# ── the bundled document ───────────────────────────────────────────────
def test_bundle_keeps_every_module_and_leaves_no_module_syntax():
    js = N.bundle_js()
    for name in ("api", "util", "icons", "md", "atlas", "views", "evolution", "inspector", "tour", "app"):
        assert f"const __module_{name} = " in js
    assert not re.search(r"^\s*(import|export)\b", js, re.M)
    # Dependencies are defined before their importers.
    assert js.index("const __module_util") < js.index("const __module_views") < js.index("const __module_app")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_bundle_is_valid_javascript(tmp_path):
    path = tmp_path / "bundle.mjs"
    path.write_text(N.bundle_js())
    out = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_page_is_self_contained_and_shut_off_from_the_network():
    page = N.bundle_page()
    assert '<meta name="myelin-mode" content="native">' in page
    assert "connect-src 'none'" in page and "default-src 'none'" in page
    for outside in ('href="app.css"', 'src="js/app.js"', "%%MYELIN_TOKEN%%", 'url("fonts/'):
        assert outside not in page
    assert page.count('<script type="module">') == 1 and "<script src" not in page
    assert "data:font/woff2;base64," in page


def test_bundler_refuses_forms_it_does_not_understand(tmp_path):
    (tmp_path / "app.js").write_text('import x from "./y.js";\n')
    with pytest.raises(ValueError):
        N.bundle_js(tmp_path)


# ── the bridge ─────────────────────────────────────────────────────────
def test_bridge_answers_like_the_api(brain):
    bridge = N.Bridge(App(brain, "127.0.0.1", 0))
    status = bridge.call("status")
    assert status["data"]["notes"] > 0
    note = bridge.call("note", {"id": "ASTRO_0"})["data"]
    assert note["title"] and note["neighbors"]
    found = bridge.call("recall", {"q": "telescope orbit parallax", "k": 3})["data"]
    assert len(found["results"]) == 3 and found["recall_id"]
    assert bridge.call("note", {"id": "NO_SUCH_NOTE"})["error"] == "no such note"
    assert "no such call" in bridge.call("drop_tables")["error"]
    assert bridge.call("decide", {"id": 999999, "accept": True})["error"]
    # Only the two entry points are visible to the page.
    assert [m for m in dir(bridge) if not m.startswith("_")] == ["call", "subscribe"]


def _detail(script):
    start = script.index("detail: ") + len("detail: ")
    return json.loads(script[start:script.rindex(" }))")])


class _Window:
    def __init__(self):
        self.scripts = queue.Queue()

    def evaluate_js(self, script):
        self.scripts.put(script)


def test_events_reach_the_page(brain):
    app = App(brain, "127.0.0.1", 0)
    window, stop = _Window(), threading.Event()
    pump = N.EventPump(app, window, stop)
    bridge = N.Bridge(app)
    bridge._pump = pump
    assert bridge.subscribe() is True
    try:
        app.hub.publish("sleep_step", {"step": "ingest", "ok": True, "seconds": 0.1})
        brain.store.event("TEST", "native", {"ü": "ascii-safe"})
        seen = [window.scripts.get(timeout=5) for _ in range(2)]
    finally:
        stop.set()
    kinds = {_detail(s)["kind"] for s in seen}
    assert kinds == {"sleep_step", "ledger"}
    assert all(s.isascii() and s.startswith("window.dispatchEvent(new CustomEvent('myelin:event'") for s in seen)


def test_status_tells_a_first_run_from_an_empty_vault(vault, tmp_path, monkeypatch):
    from myelin.brain import Brain
    from myelin.config import load_config
    from myelin.daemon import Scheduler
    monkeypatch.setenv("MYELIN_STATE_DIR", str(tmp_path / "fresh"))
    fresh = Brain(load_config(vault))
    try:
        app = App(fresh, "127.0.0.1", 0)
        app.scheduler = Scheduler(fresh)
        first = app.status({})
        assert first["notes"] == 0 and first["last_ingest"] == 0 and not first["daemon_alive"]
        with app.scheduler.sleeping:          # a sleep longer than the heartbeat window is still a live scheduler
            assert app.status({})["daemon_alive"]
        app.scheduler.indexing.set()          # and so is a long first ingest
        assert app.status({})["daemon_alive"]
        app.scheduler.indexing.clear()
        fresh.ingest()
        assert app.status({})["last_ingest"] > 0
    finally:
        fresh.close()


def test_nvidia_hosts_get_the_fallback_renderer(monkeypatch):
    monkeypatch.delenv("WEBKIT_DISABLE_DMABUF_RENDERER", raising=False)
    monkeypatch.setattr(N.sys, "platform", "linux")
    monkeypatch.setattr(N.Path, "exists", lambda self: str(self) == "/proc/driver/nvidia/version")
    N._webkit_fallbacks()
    assert os.environ["WEBKIT_DISABLE_DMABUF_RENDERER"] == "1"
    monkeypatch.setenv("WEBKIT_DISABLE_DMABUF_RENDERER", "0")
    N._webkit_fallbacks()
    assert os.environ["WEBKIT_DISABLE_DMABUF_RENDERER"] == "0"      # a value already set is left alone


def test_readiness_names_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "webview", None)
    hint = N.readiness()
    assert hint and "pip install pywebview" in hint


# ── launchers ──────────────────────────────────────────────────────────
def test_linux_launcher_installs_entry_and_icons(brain, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    plan = N.plan_launcher(brain.cfg, python="/opt/py 3/bin/python3", system="Linux", home=tmp_path)
    plan.commands = []          # no icon-cache tools in a test
    plan.apply()
    entry = (tmp_path / "share" / "applications" / "myelin.desktop").read_text()
    assert 'Exec="/opt/py 3/bin/python3" -m myelin --root ' in entry and entry.rstrip().endswith("StartupWMClass=Myelin")
    # By file, not by theme name: a running desktop caches a failed name lookup.
    icon = tmp_path / "share" / "icons" / "hicolor" / "256x256" / "apps" / "myelin.png"
    assert f"Icon={icon}\n" in entry and icon.stat().st_size > 0 and f"Path={N.PACKAGE.parent}" in entry
    for size in (16, 32, 256, 512):
        assert (tmp_path / "share" / "icons" / "hicolor" / f"{size}x{size}" / "apps" / "myelin.png").stat().st_size > 0
    if shutil.which("desktop-file-validate"):
        out = subprocess.run(["desktop-file-validate", str(tmp_path / "share" / "applications" / "myelin.desktop")],
                             capture_output=True, text=True)
        assert out.returncode == 0 and not out.stdout.strip(), out.stdout


def test_exec_quoting_follows_the_desktop_entry_spec():
    assert N._desktop_exec(["/usr/bin/python3", "-m", "myelin"]) == "/usr/bin/python3 -m myelin"
    assert N._desktop_exec(['/a b/"c"$', "100%"]) == r'"/a b/\\"c\\"\\$" 100%%'


def test_macos_bundle_is_well_formed(brain, tmp_path):
    plan = N.plan_launcher(brain.cfg, python="/usr/local/bin/python3", system="Darwin", home=tmp_path)
    files = {path.relative_to(tmp_path).as_posix(): (content, mode) for path, content, mode in plan.files}
    info = plistlib.loads(files["Applications/Myelin.app/Contents/Info.plist"][0])
    assert info["CFBundleExecutable"] == "Myelin" and info["CFBundleIconFile"] == "myelin"
    script, mode = files["Applications/Myelin.app/Contents/MacOS/Myelin"]
    assert script.startswith(b"#!/bin/sh") and b" -m myelin --root " in script and mode == 0o755
    assert files["Applications/Myelin.app/Contents/Resources/myelin.icns"][0][:4] == b"icns"


def test_windows_shortcut_command_quotes_paths(brain, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "Ro'aming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    plan = N.plan_launcher(brain.cfg, python=str(tmp_path / "python.exe"), system="Windows", home=tmp_path,
                           desktop_shortcut=True)
    (ico_path, ico, _), = plan.files
    assert ico[:4] == b"\x00\x00\x01\x00" and ico_path.name == "myelin.ico"
    script = plan.commands[0][-1]
    assert "Ro''aming" in script and "Myelin.lnk" in script and "GetFolderPath('Desktop')" in script
    assert "-m myelin --root" in script and "IconLocation" in script


def test_linux_desktop_icon_is_executable_and_marked_trusted(brain, tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setattr(N, "_xdg_desktop_dir", lambda home: tmp_path / "Schreibtisch")   # localized names come from xdg-user-dir
    monkeypatch.setattr(N.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "gio" else None)
    plan = N.plan_launcher(brain.cfg, python="/usr/bin/python3", system="Linux", home=tmp_path, desktop_shortcut=True)
    icon = tmp_path / "Schreibtisch" / "myelin.desktop"
    (path, entry, mode), = [f for f in plan.files if f[0] == icon]
    menu_entry = dict((f[0], f[1]) for f in plan.files)[tmp_path / "share" / "applications" / "myelin.desktop"]
    assert mode == 0o755 and entry == menu_entry          # the same launcher, in both places
    assert ["gio", "set", str(icon), "metadata::trusted", "true"] in plan.commands          # GNOME
    assert ["gio", "set", "-t", "string", str(icon), "metadata::xfce-exe-checksum",
            hashlib.sha256(entry).hexdigest()] in plan.commands                              # Xfce
    plan.commands = []
    plan.apply()
    assert icon.stat().st_mode & 0o777 == 0o755


def test_macos_desktop_icon_links_to_the_bundle(brain, tmp_path):
    plan = N.plan_launcher(brain.cfg, python="/usr/bin/python3", system="Darwin", home=tmp_path, desktop_shortcut=True)
    assert ["ln", "-sfn", str(tmp_path / "Applications" / "Myelin.app"),
            str(tmp_path / "Desktop" / "Myelin.app")] in plan.commands
