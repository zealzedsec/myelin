"""Myelin as a desktop application: the interface in its own native window.

There is no browser and no listening socket. pywebview drives the operating
system's own web view (WebKitGTK on Linux, WKWebView on macOS, WebView2 on
Windows); the page reaches the brain through pywebview's in-process bridge, and
live events are pushed into the page from Python. The page is the interface the
local server serves, bundled at launch into one self-contained document
(styles, fonts and scripts inline) whose content policy forbids any network
access from inside the window.

    myelin app                 open the window
    myelin app --check         say whether this machine can, and what to install if not
    myelin install-app         put Myelin in the applications menu with its icon
                               (a dry run; --apply writes it)
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import platform
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import __version__
from . import locks
from .brain import Brain
from .config import Config
from .store import atomic_write

PACKAGE = Path(__file__).resolve().parent
UI_DIR = PACKAGE / "ui"
ASSETS = PACKAGE / "assets"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)

# pywebview builds its bridge functions with `new Function`, hence 'unsafe-eval'.
# Everything that could reach a network or pull in outside content is 'none'.
NATIVE_CSP = ("default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; "
              "font-src data:; img-src data:; connect-src 'none'; media-src 'none'; frame-src 'none'; "
              "worker-src 'none'; base-uri 'none'; form-action 'none'")


# ── one document ─────────────────────────────────────────────────────────
_IMPORT = re.compile(r'^import\s*\{([^}]*)\}\s*from\s*"\./([\w-]+)\.js";[ \t]*$', re.M)
_EXPORT = re.compile(r"^export\s+(async\s+function|function|const|let|class)\s+([A-Za-z_$][\w$]*)", re.M)
_LEFTOVER = re.compile(r"^\s*(?:import|export)\b", re.M)


def _destructure(names: str) -> str:
    """`a, b as c` from an import list becomes `a, b: c` for a destructuring."""
    out = []
    for item in (x.strip() for x in names.split(",")):
        if item:
            source, _, local = item.partition(" as ")
            out.append(f"{source.strip()}: {local.strip()}" if local else source.strip())
    return ", ".join(out)


def bundle_js(js_dir: Path = UI_DIR / "js", entry: str = "app") -> str:
    """The interface's ES modules as one script, each module in its own scope.

    The modules use one import form (`import { a, b as c } from "./x.js";`)
    and named exports, so the transform is exact rather than heuristic, and
    anything outside those forms is refused instead of guessed at. Modules run
    once, in dependency order, exactly as the browser would run them.
    """
    order: list[str] = []
    sources: dict[str, str] = {}

    def visit(name: str, trail: tuple[str, ...] = ()) -> None:
        if name in sources:
            return
        if name in trail:
            raise ValueError(f"import cycle: {' -> '.join(trail + (name,))}")
        src = (js_dir / f"{name}.js").read_text(encoding="utf-8")
        for m in _IMPORT.finditer(src):
            visit(m.group(2), trail + (name,))
        sources[name] = src
        order.append(name)

    visit(entry)
    ident = lambda name: "__module_" + re.sub(r"\W", "_", name)   # noqa: E731
    parts = []
    for name in order:
        src = sources[name]
        exports = [m.group(2) for m in _EXPORT.finditer(src)]
        body = _EXPORT.sub(lambda m: f"{m.group(1)} {m.group(2)}", src)
        body = _IMPORT.sub(lambda m: f"const {{ {_destructure(m.group(1))} }} = {ident(m.group(2))};", body)
        if _LEFTOVER.search(body):
            raise ValueError(f"{name}.js uses an import or export form the bundler does not handle")
        parts.append(f"// ── {name}.js\nconst {ident(name)} = (() => {{\n{body}\nreturn {{ {', '.join(exports)} }};\n}})();")
    script = "\n".join(parts)
    # Inside an inline script the HTML parser still reacts to two sequences:
    # `</script` ends it, and `<!--` can switch it into an escaped state.
    for seq in ("</script", "<!--"):
        if seq in script.lower():
            raise ValueError(f"a module contains {seq!r}, which the HTML parser would act on")
    return script


def inline_fonts(css: str, ui_dir: Path = UI_DIR) -> str:
    def data_uri(m: re.Match) -> str:
        blob = base64.b64encode((ui_dir / m.group(1)).read_bytes()).decode()
        return f'url("data:font/woff2;base64,{blob}")'
    return re.sub(r'url\("(fonts/[^"]+\.woff2)"\)', data_uri, css)


def bundle_page() -> str:
    """index.html with its styles, fonts and scripts inline, in native mode."""
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    css = inline_fonts((UI_DIR / "app.css").read_text(encoding="utf-8"))
    if "</style" in css.lower():
        raise ValueError("the stylesheet contains '</style'")

    def swap(text: str, old: str, new: str) -> str:
        if old not in text:   # loud, so a change to index.html cannot silently break the app
            raise ValueError(f"index.html no longer contains {old!r}")
        return text.replace(old, new, 1)

    html = swap(html, '<meta name="myelin-token" content="%%MYELIN_TOKEN%%">',
                f'<meta name="myelin-mode" content="native">\n'
                f'<meta http-equiv="Content-Security-Policy" content="{NATIVE_CSP}">')
    html = swap(html, '<link rel="icon" href="favicon.svg" type="image/svg+xml">\n', "")
    html = swap(html, '<link rel="stylesheet" href="app.css">', f"<style>\n{css}\n</style>")
    return swap(html, '<script type="module" src="js/app.js"></script>',
                f'<script type="module">\n{bundle_js()}\n</script>')


def write_page(directory: Path) -> Path:
    path = directory / "myelin.html"
    atomic_write(path, bundle_page())
    return path


# ── the bridge ───────────────────────────────────────────────────────────
class Bridge:
    """What the page may ask of the brain: the same handlers the HTTP API routes
    to, called in-process. pywebview exposes every public method of this object
    to the page, so the public surface is exactly `call` and `subscribe`."""

    def __init__(self, app) -> None:
        self._app = app
        self._pump: EventPump | None = None
        self._routes: dict[str, tuple[str, Callable[[dict], Any]]] = {
            path.rsplit("/", 1)[1]: (method, fn) for (method, path), fn in app.routes().items()}

    def call(self, name: str, args: dict | None = None) -> dict[str, Any]:
        route = self._routes.get(str(name))
        if route is None:
            return {"error": f"no such call: {name}"}
        method, fn = route
        args = args if isinstance(args, dict) else {}
        try:
            # GET handlers take query-string shaped input; POST handlers a JSON body.
            arg = {k: [str(v)] for k, v in args.items() if v is not None} if method == "GET" else args
            return {"data": json.loads(json.dumps(fn(arg), default=str))}
        except KeyError as exc:
            return {"error": str(exc).strip("'\"")}
        except (ValueError, TypeError, PermissionError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
            return {"error": str(exc)}
        except Exception as exc:  # shown to the person, never swallowed
            return {"error": f"{type(exc).__name__}: {exc}"}

    def subscribe(self) -> bool:
        if self._pump is not None:
            self._pump.start()
        return True


class EventPump:
    """Pushes live events into the page: the same stream the server sends as
    server-sent events, delivered by evaluating a one-line dispatch in the page."""

    def __init__(self, app, window, stop: threading.Event):
        self.app, self.window, self.stop = app, window, stop
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True, name="myelin-events")
                self._thread.start()

    def _run(self) -> None:
        with contextlib.closing(self.app.stream(self.stop)) as events:
            for kind, data in events:
                if kind and not self.push(kind, data):
                    return

    def push(self, kind: str, data: Any) -> bool:
        detail = json.dumps({"kind": kind, "data": data}, default=str)   # ASCII-only: safe inside a JS literal
        try:
            self.window.evaluate_js(f"window.dispatchEvent(new CustomEvent('myelin:event', {{ detail: {detail} }}))")
            return True
        except Exception:
            return False   # the window is closing


# ── readiness ────────────────────────────────────────────────────────────
def readiness() -> str | None:
    """None when this interpreter can open the window; otherwise what to install."""
    try:
        import webview  # noqa: F401
    except ImportError:
        return ("The desktop app needs pywebview:  pip install pywebview\n" + _backend_hint())
    if sys.platform.startswith("linux") and not (_gtk_ready() or _qt_ready()):
        return "pywebview is installed, but it has no window backend on this machine.\n" + _backend_hint()
    return None


def _gtk_ready() -> bool:
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        for version in ("4.1", "4.0"):
            with contextlib.suppress(ValueError):
                gi.require_version("WebKit2", version)
                from gi.repository import Gtk, WebKit2  # noqa: F401
                return True
    except (ImportError, ValueError):
        pass
    return False


def _qt_ready() -> bool:
    try:
        import qtpy  # noqa: F401
        from qtpy import QtWebEngineWidgets  # noqa: F401
        return True
    except Exception:
        return False


def _backend_hint() -> str:
    if sys.platform.startswith("linux"):
        return ("On Linux it uses the system's GTK web view. Debian/Ubuntu:\n"
                "    sudo apt install python3-gi gir1.2-gtk-3.0 gir1.2-webkit2-4.1\n"
                "Fedora:  sudo dnf install python3-gobject webkit2gtk4.1\n"
                "Arch:    sudo pacman -S --needed python-gobject webkit2gtk-4.1\n"
                "Or use Qt instead:  pip install 'pywebview[qt]'\n"
                "The GTK bindings belong to the system Python, so launch Myelin with that interpreter.")
    if sys.platform == "darwin":
        return "On macOS pywebview uses the system WebKit through pyobjc, which pip installs with it."
    return "On Windows pywebview uses Edge WebView2, which ships with Windows 11 and current Windows 10."


# ── the window ───────────────────────────────────────────────────────────
def icon_path(size: int = 256) -> Path:
    return ASSETS / f"myelin-{size}.png"


def run_app(brain: Brain, debug: bool = False, tour: bool = False) -> int:
    _claim_identity()   # before anything opens the display: GTK names its first window then
    _webkit_fallbacks()
    problem = readiness()
    if problem:
        print(problem, file=sys.stderr)
        return 2
    import webview

    from .server import App, attach_scheduler

    home = brain.cfg.state / "app"
    app = App(brain, "127.0.0.1", 0)
    stop = threading.Event()
    owner = attach_scheduler(app, stop)
    bridge = Bridge(app)
    try:
        page = write_page(home)
        window = webview.create_window(
            "Myelin", url=page.as_uri() + ("?tour=auto" if tour else ""), js_api=bridge, width=1440, height=900, min_size=(980, 640),
            background_color="#050404", text_select=True)
        bridge._pump = EventPump(app, window, stop)
        webview.start(func=_after_start, debug=debug, private_mode=False, storage_path=str(home / "webview"),
                      icon=str(icon_path(256)) if sys.platform.startswith("linux") else None)
    except Exception:
        # Started from a launcher there is no terminal to print to; leave the reason on disk.
        atomic_write(home / "last-error.txt", f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{traceback.format_exc()}")
        raise
    finally:
        stop.set()
        locks.release(owner)
    return 0


def _webkit_fallbacks() -> None:
    """Under NVIDIA's proprietary driver, WebKitGTK's DMA-BUF renderer leaves the
    window blank on many systems, and this is the documented switch to the renderer
    that works there. Only on such a system, and never over a value already set."""
    if sys.platform.startswith("linux") and Path("/proc/driver/nvidia/version").exists():
        os.environ.setdefault("WEBKIT_DISABLE_DMABUF_RENDERER", "1")


def _claim_identity() -> None:
    """Make the desktop recognise the window as Myelin rather than as a Python
    process, so the dock and the task switcher show Myelin's name and icon and
    group the window under its launcher."""
    sys.argv[0] = "myelin"   # Qt names the window class after argv[0]
    if sys.platform.startswith("linux"):
        with contextlib.suppress(Exception):
            import gi
            gi.require_version("Gdk", "3.0")
            from gi.repository import Gdk, GLib
            GLib.set_prgname("myelin")            # WM_CLASS instance and Wayland app_id: matches myelin.desktop
            GLib.set_application_name("Myelin")
            Gdk.set_program_class("Myelin")       # WM_CLASS class: matches StartupWMClass=Myelin
    elif sys.platform == "win32":
        with contextlib.suppress(Exception):
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Myelin.Desktop")
    elif sys.platform == "darwin":
        with contextlib.suppress(Exception):
            from Foundation import NSBundle
            NSBundle.mainBundle().infoDictionary()["CFBundleName"] = "Myelin"


def _after_start() -> None:
    if sys.platform == "darwin":   # the Dock shows the interpreter's icon unless told otherwise
        with contextlib.suppress(Exception):
            from AppKit import NSApplication, NSImage
            from PyObjCTools import AppHelper
            image = NSImage.alloc().initWithContentsOfFile_(str(ASSETS / "myelin.icns"))
            AppHelper.callAfter(NSApplication.sharedApplication().setApplicationIconImage_, image)


# ── launchers ────────────────────────────────────────────────────────────
@dataclass
class Launcher:
    system: str
    files: list[tuple[Path, bytes, int]] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"Myelin desktop launcher for {self.system}:"]
        lines += [f"  write {path}" for path, _, _ in self.files]
        for cmd in self.commands:
            shown = " ".join(shlex.quote(c) for c in cmd)
            lines.append(f"  run   {shown if len(shown) <= 160 else shown[:157] + '…'}")
        lines += [f"  note  {n}" for n in self.notes]
        return "\n".join(lines)

    def apply(self) -> None:
        for path, content, mode in self.files:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(path, content)
            path.chmod(mode)
        for cmd in self.commands:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run(cmd, check=False, capture_output=True, timeout=60)


def launch_command(cfg: Config, python: str) -> tuple[list[str], Path]:
    """The command a launcher runs, and the directory it runs in. Running from
    the directory that holds the `myelin` package makes `-m myelin` resolve
    whether or not Myelin was pip-installed."""
    return [python, "-m", "myelin", "--root", str(cfg.root), "app"], PACKAGE.parent


def plan_launcher(cfg: Config, python: str | None = None, system: str | None = None,
                  desktop_shortcut: bool = False, home: Path | None = None) -> Launcher:
    system = system or platform.system()
    python = python or sys.executable
    home = home or Path.home()
    cmd, workdir = launch_command(cfg, python)
    if system == "Linux":
        return _linux(cfg, cmd, workdir, home, desktop_shortcut)
    if system == "Darwin":
        return _macos(cmd, workdir, home, desktop_shortcut)
    if system == "Windows":
        return _windows(cmd, workdir, home, desktop_shortcut)
    raise ValueError(f"no launcher recipe for {system}")


def _desktop_exec(args: list[str]) -> str:
    """The Exec value of a desktop entry, per the freedesktop Desktop Entry spec:
    arguments with reserved characters are double-quoted with `"`, `` ` ``, `$`
    and `\\` backslash-escaped; then the string-level escape doubles every
    backslash again, and `%` is doubled because it introduces field codes."""
    def quote(arg: str) -> str:
        if re.fullmatch(r"[\w@%+=:,./-]+", arg):
            return arg
        return '"' + re.sub(r'(["`$\\])', r"\\\1", arg) + '"'
    return " ".join(quote(a) for a in args).replace("\\", "\\\\").replace("%", "%%")


def _linux(cfg: Config, cmd: list[str], workdir: Path, home: Path, desktop_shortcut: bool) -> Launcher:
    import os
    data = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    out = Launcher("Linux")
    for size in ICON_SIZES[:-1]:
        out.files.append((data / "icons" / "hicolor" / f"{size}x{size}" / "apps" / "myelin.png",
                          icon_path(size).read_bytes(), 0o644))
    out.files.append((data / "icons" / "hicolor" / "scalable" / "apps" / "myelin.svg",
                      (ASSETS / "myelin.svg").read_bytes(), 0o644))
    # The entry names its icon by absolute path, not by theme name. A desktop that is
    # already running looks a theme name up the moment the entry appears, before it
    # has rescanned the icon directories it was just given, and caches the generic
    # icon it fell back to; a file path needs no lookup. The theme copies above stay
    # for anything that asks for the `myelin` icon by name.
    icon_file = data / "icons" / "hicolor" / "256x256" / "apps" / "myelin.png"
    entry = "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.5",
        "Name=Myelin",
        "GenericName=Knowledge brain",
        f"Comment=Recall, connect and evolve the {cfg.root.name} vault",
        f"Exec={_desktop_exec(cmd)}",
        f"Path={workdir}",
        f"Icon={icon_file}",
        "Terminal=false",
        "Categories=Office;",
        "Keywords=notes;vault;obsidian;recall;knowledge;graph;",
        "StartupNotify=true",
        "StartupWMClass=Myelin",
        "",
    ]).encode()
    applications = data / "applications"
    out.files.append((applications / "myelin.desktop", entry, 0o644))
    if desktop_shortcut:
        icon = _xdg_desktop_dir(home) / "myelin.desktop"
        out.files.append((icon, entry, 0o755))   # executable: KDE and others will not launch it otherwise
        if shutil.which("gio"):
            # GNOME's desktop icons and Xfce each refuse to run a launcher on the desktop
            # until it is marked trusted, and each keeps that mark in its own attribute.
            out.commands.append(["gio", "set", str(icon), "metadata::trusted", "true"])
            out.commands.append(["gio", "set", "-t", "string", str(icon), "metadata::xfce-exe-checksum",
                                 hashlib.sha256(entry).hexdigest()])
        out.notes.append(f"desktop icon: {icon}")
    if shutil.which("update-desktop-database"):
        out.commands.append(["update-desktop-database", str(applications)])
    if shutil.which("gtk-update-icon-cache"):
        out.commands.append(["gtk-update-icon-cache", "-f", "-t", str(data / "icons" / "hicolor")])
    out.notes.append("appears in the applications menu as Myelin; the window groups under this icon")
    return out


def _xdg_desktop_dir(home: Path) -> Path:
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        found = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=5).stdout.strip()
        if found:
            return Path(found)
    return home / "Desktop"


def _macos(cmd: list[str], workdir: Path, home: Path, desktop_shortcut: bool = False) -> Launcher:
    app = home / "Applications" / "Myelin.app"
    bundle = app / "Contents"
    info = {
        "CFBundleName": "Myelin", "CFBundleDisplayName": "Myelin", "CFBundleIdentifier": "local.myelin.desktop",
        "CFBundleExecutable": "Myelin", "CFBundleIconFile": "myelin", "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": __version__, "CFBundleVersion": __version__,
        "LSMinimumSystemVersion": "10.15", "NSHighResolutionCapable": True,
    }
    script = "\n".join(["#!/bin/sh", f"cd {shlex.quote(str(workdir))} || exit 1",
                        "exec " + " ".join(shlex.quote(c) for c in cmd), ""]).encode()
    out = Launcher("macOS")
    out.files += [(bundle / "Info.plist", plistlib.dumps(info), 0o644),
                  (bundle / "MacOS" / "Myelin", script, 0o755),
                  (bundle / "Resources" / "myelin.icns", (ASSETS / "myelin.icns").read_bytes(), 0o644)]
    out.notes.append("Myelin.app in ~/Applications: open it from Launchpad or Spotlight, or drag it to the Dock")
    if desktop_shortcut:   # Finder shows a link to the bundle with the bundle's icon
        out.commands.append(["ln", "-sfn", str(app), str(home / "Desktop" / "Myelin.app")])
        out.notes.append(f"desktop icon: {home / 'Desktop' / 'Myelin.app'}")
    return out


def _windows(cmd: list[str], workdir: Path, home: Path, desktop_shortcut: bool) -> Launcher:
    import os
    exe = Path(cmd[0])
    windowless = exe.with_name("pythonw.exe")   # no console window behind the app
    target = str(windowless if windowless.exists() else exe)
    local = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local") / "Myelin"
    programs = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming") / \
        "Microsoft" / "Windows" / "Start Menu" / "Programs"
    ico = local / "myelin.ico"
    args = subprocess.list2cmdline(cmd[1:])
    q = lambda s: "'" + str(s).replace("'", "''") + "'"   # noqa: E731  PowerShell single-quoted literal
    links = [f"Join-Path {q(programs)} 'Myelin.lnk'"]
    if desktop_shortcut:
        links.append("Join-Path ([Environment]::GetFolderPath('Desktop')) 'Myelin.lnk'")
    script = "; ".join(
        ["$shell = New-Object -ComObject WScript.Shell"] +
        [f"$s = $shell.CreateShortcut(({link})); $s.TargetPath = {q(target)}; $s.Arguments = {q(args)}; "
         f"$s.WorkingDirectory = {q(workdir)}; $s.IconLocation = {q(str(ico) + ',0')}; "
         f"$s.Description = 'Myelin'; $s.Save()" for link in links])
    out = Launcher("Windows")
    out.files.append((ico, (ASSETS / "myelin.ico").read_bytes(), 0o644))
    out.commands.append(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script])
    out.notes.append("Start menu entry Myelin" + (" and a desktop shortcut" if desktop_shortcut else ""))
    return out
