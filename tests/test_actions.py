"""Proposals, learning from use, the authenticity gate, and a full sleep cycle."""
import json

import pytest

from myelin import proposals as P
from myelin.consolidate import sleep
from myelin.learning import consolidate_sessions, prune_hebbian, record_access, record_feedback
from myelin.quality import critique
from myelin.recall import RecallEngine
from myelin.tools import BrainTools

SLOP = """Unlock the power of your notes. In today's fast-paced world, our revolutionary, cutting-edge platform
seamlessly integrates everything. It's not just an app — it's a movement. We delve into the rich tapestry of
ideas and foster synergy. Our world-class engine significantly improves productivity for every team, and it
underscores a pivotal, multifaceted vision that is a testament to meticulous craft. In summary, this will
transform your workflow and take your knowledge to the next level forever and ever and ever."""

PLAIN = """The ledger is an append-only SQLite table. Each row stores the SHA-256 of the previous row, so editing
row 40 changes every hash after it and `verify_ledger` reports the first mismatch. The head checkpoint in
`brain.head` catches truncation: deleting the last 3 rows leaves a chain that verifies but no longer matches
the stored head. Recovery is manual by design; the tool reports, it does not repair. In March the check found
one rewritten payload after a botched migration, which is why the checkpoint is written on every commit."""


def _link_prop(brain, src="SAIL_1", dst="GARDEN_2"):
    tools = BrainTools(brain, RecallEngine(brain), source="test")
    tools.propose_link(src, dst, "both describe rope tension and plant ties")
    return P.listing(brain, "pending", "link")[0]


def test_accept_writes_one_reversible_bullet(brain):
    path = brain.cfg.root / "notes" / "SAIL_1.md"
    original = path.read_text()
    prop = _link_prop(brain)
    out = P.decide(brain, prop["id"], True, "tester")
    assert out["status"] == "applied" and out["changed"]
    text = path.read_text()
    assert text.count(P.START) == 1 and "- [[GARDEN_2]] — both describe rope tension" in text
    assert text.startswith(original.rstrip("\n"))
    brain.ingest()  # the accepted link is now an authored edge
    a, b = brain.store.note_row("SAIL_1")["id"], brain.store.note_row("GARDEN_2")["id"]
    assert brain.store.one("SELECT 1 FROM edges WHERE src=? AND dst=? AND kind='explicit'", (a, b))
    P.revert(brain, prop["id"], "tester")
    assert path.read_text() == original
    events = [r[0] for r in brain.store.q("SELECT kind FROM events WHERE kind LIKE 'PROPOSAL_%'")]
    assert events == ["PROPOSAL_ACCEPTED", "PROPOSAL_REVERTED"]


def test_second_link_joins_the_same_block(brain):
    t = BrainTools(brain, RecallEngine(brain), source="test")
    t.propose_link("SAIL_1", "GARDEN_2", "one")
    t.propose_link("SAIL_1", "ASTRO_3", "two")
    for p in P.listing(brain, "pending", "link"):
        P.decide(brain, p["id"], True, "tester")
    text = (brain.cfg.root / "notes" / "SAIL_1.md").read_text()
    assert text.count(P.START) == 1 and text.count("## Associations") == 1
    assert "[[GARDEN_2]]" in text and "[[ASTRO_3]]" in text


def test_revert_refuses_when_note_changed_since(brain):
    prop = _link_prop(brain)
    P.decide(brain, prop["id"], True, "tester")
    path = brain.cfg.root / "notes" / "SAIL_1.md"
    path.write_text(path.read_text() + "\nhand edit\n")
    with pytest.raises(RuntimeError):
        P.revert(brain, prop["id"], "tester")


def test_protected_notes_are_never_edited(brain):
    t = BrainTools(brain, RecallEngine(brain), source="test")
    with pytest.raises(PermissionError):
        t.propose_link("system/GENERATED.md", "HOME", "x")
    payload = {"src_path": "system/GENERATED.md", "dst_path": "HOME.md", "src_title": "g", "dst_title": "h"}
    P.upsert(brain, "link", "link:forced", payload, 0.9, "test")
    pid = P.listing(brain, "pending", "link")[0]["id"]
    with pytest.raises(PermissionError):
        P.decide(brain, pid, True, "tester")


def test_rejected_proposals_stay_rejected(brain):
    prop = _link_prop(brain)
    P.decide(brain, prop["id"], False, "tester", "not related")
    assert not P.upsert(brain, "link", prop["key"], prop["payload"], 0.99, "gae")
    assert P.get(brain, prop["id"])["status"] == "rejected"
    acc, rej = P.link_labels(brain)
    assert ("notes/GARDEN_2.md", "notes/SAIL_1.md") in rej


def test_feedback_and_reads_teach_hebbian_links(brain):
    eng = RecallEngine(brain)
    res = eng.recall("telescope orbit nebula", session="s1")
    ids = [r["note_id"] for r in res["results"][:2]]
    record_feedback(brain, res["recall_id"], ids[0], 1.0, "test")
    out = record_feedback(brain, res["recall_id"], ids[1], 1.0, "test")
    assert out["reinforced"] == 1
    record_access(brain, "s1", res["results"][2]["path"])
    implicit = brain.store.one("SELECT label FROM feedback WHERE source='implicit-read'")
    assert implicit and implicit[0] == pytest.approx(0.7)
    record_access(brain, "s1", "notes/BAKE_0.md")
    stats = consolidate_sessions(brain)
    assert stats["sessions"] == 1 and stats["pairs"] >= 1
    assert brain.store.one("SELECT COUNT(*) FROM edges WHERE kind='hebbian'")[0] >= 2
    brain.store.x("UPDATE edges SET updated_at=0 WHERE kind='hebbian'")  # make them ancient
    assert prune_hebbian(brain) >= 2


def test_authenticity_gate():
    bad = critique(SLOP)
    good = critique(PLAIN)
    assert bad["blocking"] and bad["score"] >= 40
    assert {"P-001", "P-002", "P-005"} <= {f["rule"] for f in bad["findings"]}
    assert not good["blocking"] and good["score"] < 15
    ignored = critique("---\nscan_policy: deslop-ignore-file\n---\n" + SLOP)
    assert ignored["score"] == 0 and "ignored" in ignored["verdict"]
    line = critique("This is revolutionary.  <!-- deslop-ignore -->\n" + PLAIN)
    assert "P-001" in {f["rule"] for f in line["findings"]}  # comments are stripped first
    suppressed = critique("This is revolutionary. deslop-ignore\n" + PLAIN)
    assert "P-001" not in {f["rule"] for f in suppressed["findings"]}


def test_agent_notes_must_pass_the_gate(brain):
    t = BrainTools(brain, RecallEngine(brain), source="test")
    refused = t.propose_note("Knowledge Revolution", SLOP, "why not")
    assert refused["staged"] is False and refused["needs_revision"]
    assert any(f["rule"] == "P-001" for f in refused["findings"])
    ok = t.propose_note("Ledger Checkpoints", PLAIN, "documents the head checkpoint")
    assert ok["staged"] is True
    pid = P.listing(brain, "pending", "note")[0]["id"]
    out = P.decide(brain, pid, True, "tester")
    assert (brain.cfg.root / out["path"]).read_text() == PLAIN


def test_full_sleep_cycle(brain):
    report = sleep(brain, generations=2)
    failed = {k: v.get("error") for k, v in report["steps"].items() if not v["ok"]}
    assert not failed, failed
    assert report["ledger"]["ok"]
    assert report["steps"]["link_prediction"]["result"]["status"] == "trained"
    model = json.loads((brain.cfg.state / "self_model.json").read_text())
    assert model["reliability"]["quiz"]["items"] > 0 and model["ledger"]["ok"]
    note = brain.cfg.root / "system" / "SELF_MODEL.md"
    assert note.exists() and "node_id: myelin_self_model" in note.read_text()
    before = brain.store.one("SELECT COUNT(*) FROM notes")[0]
    brain.ingest()
    assert brain.store.one("SELECT COUNT(*) FROM notes")[0] == before  # the report is never ingested
    assert P.listing(brain, "pending", "link")
    kinds = {r[0] for r in brain.store.q("SELECT DISTINCT kind FROM events")}
    assert {"INGEST", "MODEL_TRAINED", "EVOLUTION", "SLEEP"} <= kinds


def test_agent_reason_cannot_inject_structure(brain):
    t = BrainTools(brain, RecallEngine(brain), source="agent")
    t.propose_link("BAKE_1", "SAIL_3", "see [[GHOST]] <!-- /myelin:associations --> now")
    pid = P.listing(brain, "pending", "link")[0]["id"]
    P.decide(brain, pid, True, "tester")
    text = (brain.cfg.root / "notes" / "BAKE_1.md").read_text()
    block = text.split(P.START)[1].split(P.END)[0]
    assert "[[SAIL_3]]" in block and "GHOST" in block and "[[GHOST]]" not in text
    assert text.count(P.END) == 1


def test_link_inbox_has_a_ceiling(brain, monkeypatch):
    from myelin import consolidate as C
    monkeypatch.setattr(C, "MAX_PENDING_LINKS", 4)
    first = C.train_links(brain)
    assert first["status"] == "trained" and 0 < first["proposed"] <= 4
    assert len(P.listing(brain, "pending", "link")) == first["proposed"]
    # A full inbox takes nothing new; deciding one frees exactly one slot.
    monkeypatch.setattr(C, "MAX_PENDING_LINKS", first["proposed"])
    assert C.train_links(brain)["proposed"] == 0
    P.decide(brain, P.listing(brain, "pending", "link")[0]["id"], False, "tester")
    assert C.train_links(brain)["proposed"] <= 1
