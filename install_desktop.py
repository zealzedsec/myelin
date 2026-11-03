#!/usr/bin/env python3
"""Put Myelin on this computer's desktop and in its applications menu.

Run it on the computer where the vault lives, with the system Python:

    python3 myelin/install_desktop.py        Linux, macOS (from the vault folder)
    py myelin\\install_desktop.py             Windows

It gives Myelin a private Python environment (nothing is installed into the
system Python), installs pywebview, numpy and PyYAML into it, checks that a
native window can open, and then runs `myelin install-app --apply --desktop`
with that environment's Python, so the icon starts the right interpreter.

On Linux the window is the system's GTK web view, which pip cannot install. If
it is missing, the script prints the one package command to run, and stops;
run the script again afterwards. It uses only the standard library.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <vault>/myelin, which holds the `myelin` package
VAULT = HERE.parent
SYSTEM = platform.system()


def env_dir() -> Path:
    if SYSTEM == "Windows":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Myelin" / "env"
    if SYSTEM == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Myelin" / "env"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "myelin" / "env"


def env_python(env: Path) -> Path:
    return env / ("Scripts/python.exe" if SYSTEM == "Windows" else "bin/python")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def main() -> int:
    if sys.version_info < (3, 11):
        print(f"Myelin needs Python 3.11 or newer; this is {platform.python_version()}.")
        return 1
    if not (HERE / "myelin" / "native.py").exists():
        print(f"Run this from a Myelin checkout; {HERE / 'myelin'} is not one.")
        return 1
    env = env_dir()
    py = env_python(env)
    print(f"Myelin for {VAULT}\n")

    print("1. A private Python environment for Myelin")
    if not py.exists():
        # On Linux the GTK bindings come from the system Python's packages, so the
        # environment has to see them.
        venv.EnvBuilder(with_pip=True, system_site_packages=(SYSTEM == "Linux")).create(env)
    print(f"   {env}\n")

    print("2. pywebview, numpy and PyYAML")
    if run([py, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
            "pywebview>=5", "numpy>=1.26", "pyyaml>=6"]).returncode != 0:
        print("\npip could not install them; the messages above say why.")
        return 1
    print()

    print("3. Can a native window open here?")
    check = run([py, "-m", "myelin", "app", "--check"], cwd=HERE, capture_output=True, text=True)
    print("   " + (check.stdout + check.stderr).strip().replace("\n", "\n   ") + "\n")
    if check.returncode != 0:
        if SYSTEM == "Linux" and not sys.executable.startswith("/usr/bin/"):
            print(f"This was run with {sys.executable}. The GTK bindings belong to the system Python;\n"
                  f"delete {env} and run the script again with /usr/bin/python3.")
        print("Install what is named above, then run this script again.")
        return 1

    print("4. The launcher, with its icon, in the applications menu and on the desktop")
    done = run([py, "-m", "myelin", "--root", VAULT, "install-app", "--apply", "--desktop"], cwd=HERE)
    if done.returncode != 0:
        return done.returncode
    print("\nMyelin is on your desktop. Double-click it, or find it among your applications.")
    if SYSTEM == "Linux":
        print("On GNOME, if the icon shows a warning badge, right-click it and choose 'Allow Launching'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
