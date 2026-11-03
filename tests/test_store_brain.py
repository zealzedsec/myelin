import json

from myelin.store import Store
from myelin.vault import PARSER_VERSION


def test_ledger_detects_tampering_and_truncation(tmp_path):
    s = Store(tmp_path / "b.db")
    for i in range(5):
        s.event("TEST", f"s{i}", {"i": i})
    ok, msg, n = s.verify_ledger()
    assert ok and n == 5
    s.x("UPDATE events SET payload_json=? WHERE sequence=3", (json.dumps({"i": 99}),))
    assert s.verify_ledger()[0] is False
    s.x("UPDATE events SET payload_json=? WHERE sequence=3", ('{"i":2}',))
    assert s.verify_ledger()[0] is True
    s.x("DELETE FROM events WHERE sequence=5")
    ok, msg, _ = s.verify_ledger()
    assert not ok and "checkpoint" in msg


def test_nested_transactions_roll_back_together(tmp_path):
    s = Store(tmp_path / "b.db")
    try:
        with s.tx() as db:
            db.execute("INSERT INTO meta(key,value) VALUES('a','1')")
            with s.tx() as db2:
                db2.execute("INSERT INTO meta(key,value) VALUES('b','2')")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert s.get_meta("a") is None and s.get_meta("b") is None


def test_ingest_is_idempotent(brain):
    events = brain.store.one("SELECT COUNT(*) FROM events")[0]
    report = brain.ingest()
    assert not report.changed
    assert brain.store.one("SELECT COUNT(*) FROM events")[0] == events


def test_ingest_tracks_edits_and_deletions(brain):
    root = brain.cfg.root
    target = root / "notes" / "ASTRO_2.md"
    target.write_text(target.read_text() + "\nNow mentions [[BAKE_3]] too.\n")
    report = brain.ingest()
    assert report.updated == 1
    src = brain.store.note_row("ASTRO_2")["id"]
    dst = brain.store.note_row("BAKE_3")["id"]
    assert brain.store.one("SELECT 1 FROM edges WHERE src=? AND dst=? AND kind='explicit'", (src, dst))
    (root / "notes" / "SAIL_4.md").unlink()
    report = brain.ingest()
    assert report.removed == 1
    assert brain.store.note_row("notes/SAIL_4.md") is None


def test_links_resolve_and_code_links_are_ignored(brain):
    raws = {r[0] for r in brain.store.q("SELECT target_raw FROM links")}
    assert "NOT_A_REAL_LINK" not in raws and "ALSO_NOT_A_LINK" not in raws
    row = brain.store.one("SELECT target_path FROM links WHERE target_raw='runner.py'")
    assert row[0] == "system/runner.py"
    assert brain.store.one("SELECT target_path FROM links WHERE target_raw='GHOST_NOTE'")[0] is None


def test_parser_version_change_forces_reparse(brain):
    brain.store.set_meta("parser_version", "0")
    report = brain.ingest()
    assert report.updated == brain.store.one("SELECT COUNT(*) FROM notes")[0]
    assert brain.store.get_meta("parser_version") == PARSER_VERSION


def test_semantic_edges_and_vectors(brain):
    d = brain.dense()
    assert d["passages"].shape[0] == brain.store.one("SELECT COUNT(*) FROM passages")[0]
    assert brain.store.one("SELECT COUNT(*) FROM edges WHERE kind='semantic'")[0] > 0
