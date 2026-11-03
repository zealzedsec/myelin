"""The scheduler: watch the vault, ingest on change, sleep on schedule.

Designed against the way AGENT_BRAIN_VAULT's daemon died — blocked for weeks on
an `input()` approval prompt with no terminal attached. Nothing here ever reads
stdin. Anything that needs a person (accepting a proposal) is asynchronous: it
waits in the inbox, and the daemon keeps running.

Single instance per vault via a lock on `daemon.lock` in the state directory,
shared with `myelin serve`, so the two never schedule the same vault twice.
"""
from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable

from . import locks
from .brain import Brain
from .vault import scoped_notes


def _fingerprint(brain: Brain) -> dict[str, float]:
    out = {}
    for rel, full in scoped_notes(brain.cfg):
        try:
            out[rel] = full.stat().st_mtime
        except OSError:
            continue
    return out


class Scheduler:
    def __init__(self, brain: Brain, stop: threading.Event | None = None,
                 notify: Callable[[str, dict], None] | None = None):
        self.brain = brain
        self.stop = stop or threading.Event()
        self.notify = notify or (lambda kind, data: None)
        self.sleeping = threading.Lock()
        self.indexing = threading.Event()   # set while an ingest runs; a first one can take minutes

    def tick(self, seen: dict[str, float]) -> dict[str, float]:
        # Also at the start: a first tick (index the vault, then the first sleep) can
        # take a minute, and the scheduler is alive throughout it.
        self.brain.store.set_meta("daemon_heartbeat", str(time.time()))
        current = _fingerprint(self.brain)
        if current != seen:
            self.indexing.set()
            try:
                report = self.brain.ingest()
            finally:
                self.indexing.clear()
            if report.changed:
                self.notify("ingest", report.__dict__)
        due = time.time() - float(self.brain.store.get_meta("last_sleep", "0") or 0)
        if due >= self.brain.cfg.sleep_interval_minutes * 60:
            self.run_sleep()
        self.brain.store.set_meta("daemon_heartbeat", str(time.time()))
        return current

    def run_sleep(self, generations: int = 3) -> dict | None:
        if not self.sleeping.acquire(blocking=False):
            return None  # one consolidation at a time
        try:
            from .consolidate import SleepBusy, sleep

            def step(name: str, entry: dict) -> None:
                # A sleep can outlast the liveness window; each finished stage shows the
                # scheduler is still alive to a reader in another process.
                self.brain.store.set_meta("daemon_heartbeat", str(time.time()))
                self.notify("sleep_step", {"step": name, **{k: v for k, v in entry.items() if k in ("ok", "seconds", "error")}})

            self.notify("sleep_started", {})
            try:
                report = sleep(self.brain, generations=generations, on_step=step)
            except SleepBusy as exc:
                self.notify("sleep_finished", {"seconds": 0, "skipped": str(exc)})
                return None
            self.notify("sleep_finished", {"seconds": report["seconds"], "ledger": report["ledger"]})
            return report
        finally:
            self.sleeping.release()

    def loop(self) -> None:
        seen: dict[str, float] = {}
        while not self.stop.is_set():
            try:
                seen = self.tick(seen)
            except Exception as exc:  # a bad cycle is logged, never fatal
                self.notify("fault", {"error": f"{type(exc).__name__}: {exc}"})
            self.stop.wait(self.brain.cfg.poll_seconds)


def acquire_lock(path: Path):
    handle = locks.try_lock(path)
    if handle is None:
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def run_daemon(brain: Brain, once: bool = False) -> int:
    lock = acquire_lock(brain.cfg.state / "daemon.lock")
    if lock is None:
        print("myelin: another daemon already owns this vault")
        return 1
    log = lambda kind, data: print(f"[{time.strftime('%H:%M:%S')}] {kind} {data}", flush=True)
    sched = Scheduler(brain, notify=log)
    if once:
        brain.ingest()
        sched.run_sleep()
        return 0
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: sched.stop.set())
    log("started", {"root": str(brain.cfg.root), "sleep_every_min": brain.cfg.sleep_interval_minutes})
    sched.loop()
    log("stopped", {})
    return 0
