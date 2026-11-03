"""The `Brain` facade: ingestion, the semantic space, and cached views of the graph.

Ingestion is incremental and idempotent: a note is re-parsed only when its
content hash changes, and running `ingest()` twice in a row writes nothing the
second time. That property is tested, because the evolution engine and the
ledger both assume that "nothing changed" produces no events.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, RecallParams
from .encoders import CommandEncoder, Encoder, LSAEncoder, l2n
from .graph import Graph
from .store import Store, canonical_json, pack, unpack, utc_now
from .text import chunk
from .vault import PARSER_VERSION, Note, Resolver, parse_note, scoped_notes


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    refit: bool = False
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed or self.refit)


class Brain:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = Store(cfg.db_path)
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        self._encoder: Encoder | None = None
        self._graph: tuple[str, Graph] | None = None
        self._dense: tuple[str, dict[str, Any]] | None = None

    def close(self) -> None:
        self.store.close()

    # ── encoder ──────────────────────────────────────────────────────────
    @property
    def encoder_path(self) -> Path:
        return self.cfg.models_dir / "lsa.npz"

    def encoder(self) -> Encoder | None:
        if self._encoder is not None:
            current = self.store.get_meta("encoder_id")
            if current is None or current == self._encoder.id:
                return self._encoder
        if self.cfg.encoder_kind == "command":
            self._encoder = CommandEncoder(self.cfg.encoder_command)
        elif self.encoder_path.exists():
            self._encoder = LSAEncoder.load(self.encoder_path)
        return self._encoder

    def _fit_encoder(self) -> Encoder:
        if self.cfg.encoder_kind == "command":
            enc = CommandEncoder(self.cfg.encoder_command)
            self.store.set_meta("encoder_id", enc.id)
            # Recorded for the drift check; without it every ingest looked like a
            # fully drifted corpus and re-embedded the whole vault.
            self.store.set_meta("encoder_fit_passages", str(self.store.one("SELECT COUNT(*) FROM passages")[0]))
            self._encoder = enc
            return enc
        texts = [r["heading"] + "\n" + r["text"] for r in self.store.q("SELECT heading,text FROM passages")]
        started = time.time()
        enc = LSAEncoder.fit(texts, dim=self.cfg.encoder_dim, max_features=self.cfg.max_features,
                             seed=self.cfg.seed)
        digest = enc.save(self.encoder_path)
        version = (self.store.one("SELECT COALESCE(MAX(version),0) FROM models WHERE name='lsa'")[0]) + 1
        metrics = {"passages": len(texts), "vocab": len(enc.vocab), "dim": enc.dim,
                   "seconds": round(time.time() - started, 3)}
        with self.store.tx() as db:
            db.execute("UPDATE models SET active=0 WHERE name='lsa'")
            db.execute("INSERT INTO models(name,version,path,sha256,metrics_json,created_at,active) "
                       "VALUES('lsa',?,?,?,?,?,1)",
                       (version, str(self.encoder_path), digest, canonical_json(metrics), time.time()))
        self.store.set_meta("encoder_id", enc.id)
        self.store.set_meta("encoder_fit_passages", str(len(texts)))
        self.store.event("MODEL_TRAINED", "lsa", {"version": version, "sha256": digest, **metrics})
        self._encoder = enc
        return enc

    # ── ingestion ────────────────────────────────────────────────────────
    def ingest(self, refit: bool = False) -> IngestReport:
        started = time.time()
        report = IngestReport()
        on_disk = dict(scoped_notes(self.cfg))
        known = {r["path"]: (r["id"], r["raw_hash"], r["mtime"]) for r in
                 self.store.q("SELECT id,path,raw_hash,mtime FROM notes")}
        changed_ids: list[int] = []
        reparse = self.store.get_meta("parser_version") != PARSER_VERSION

        for rel in sorted(set(known) - set(on_disk)):
            nid = known[rel][0]
            with self.store.tx() as db:
                db.execute("DELETE FROM vectors WHERE owner='p' AND owner_id IN (SELECT id FROM passages WHERE note_id=?)",
                           (nid,))
                db.execute("DELETE FROM vectors WHERE owner='n' AND owner_id=?", (nid,))
                db.execute("DELETE FROM passages_fts WHERE rowid IN (SELECT id FROM passages WHERE note_id=?)", (nid,))
                db.execute("DELETE FROM notes WHERE id=?", (nid,))
            report.removed += 1
        if report.removed:
            self._bump("vectors_version")

        parsed: list[Note] = []
        for rel, full in sorted(on_disk.items()):
            prior = known.get(rel)
            try:
                if prior and not reparse and abs(prior[2] - full.stat().st_mtime) < 1e-6:
                    report.unchanged += 1
                    continue
                note = parse_note(rel, full)
            except OSError as exc:
                report.errors.append(f"{rel}: {exc}")
                continue
            if prior and not reparse and prior[1] == note.raw_hash:
                self.store.x("UPDATE notes SET mtime=? WHERE id=?", (note.mtime, prior[0]))
                report.unchanged += 1
                continue
            parsed.append(note)

        with self.store.tx() as db:
            for note in parsed:
                note_id = self._upsert_note(db, note, known.get(note.path))
                changed_ids.append(note_id)
                if note.path in known:
                    report.updated += 1
                else:
                    report.added += 1
        if report.removed or parsed:
            self._resolve_links()
            self._rebuild_authored_edges()

        self.store.set_meta("parser_version", PARSER_VERSION)
        total = self.store.one("SELECT COUNT(*) FROM passages")[0]
        fit_at = int(self.store.get_meta("encoder_fit_passages", "0") or 0)
        drift = abs(total - fit_at) / max(fit_at, 1)
        if total and (refit or self.encoder() is None or drift > 0.25):
            self._fit_encoder()
            self._embed(all_notes=True)
            report.refit = True
        elif changed_ids:
            self._embed(note_ids=changed_ids)
        if report.changed:
            self.rebuild_semantic_edges()
        # Whether or not anything changed, and before the INGEST event, so whoever sees
        # the event sees this too: a window that opened before the first ingest tells
        # "still reading the vault" from "the vault is empty" by it.
        self.store.set_meta("last_ingest", str(time.time()))
        if report.changed:
            self.store.event("INGEST", "vault", {
                "added": report.added, "updated": report.updated, "removed": report.removed,
                "refit": report.refit, "notes": len(on_disk),
            })
        report.seconds = round(time.time() - started, 3)
        return report

    def _upsert_note(self, db, note: Note, prior: tuple | None) -> int:
        now = utc_now()
        words = len(note.body.split())
        values = (note.stem, note.title, note.note_type, json.dumps(note.tags),
                  canonical_json(note.frontmatter), note.frontmatter_error, note.raw_hash, note.mtime, words, now)
        if prior:
            note_id = prior[0]
            db.execute("UPDATE notes SET stem=?,title=?,type=?,tags_json=?,frontmatter_json=?,frontmatter_error=?,"
                       "raw_hash=?,mtime=?,words=?,updated_at=? WHERE id=?", (*values, note_id))
            # The old passages' vectors go with them; otherwise they outlive their
            # rows and a reused passage id could briefly carry another note's vector.
            db.execute("DELETE FROM vectors WHERE owner='p' AND owner_id IN (SELECT id FROM passages WHERE note_id=?)",
                       (note_id,))
            db.execute("DELETE FROM passages_fts WHERE rowid IN (SELECT id FROM passages WHERE note_id=?)",
                       (note_id,))
            db.execute("DELETE FROM passages WHERE note_id=?", (note_id,))
            db.execute("DELETE FROM links WHERE src=?", (note_id,))
        else:
            cur = db.execute("INSERT INTO notes(path,stem,title,type,tags_json,frontmatter_json,frontmatter_error,"
                             "raw_hash,mtime,words,updated_at,first_seen) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                             (note.path, *values, now))
            note_id = int(cur.lastrowid)
        for p in chunk(note.body, note.title):
            cur = db.execute("INSERT INTO passages(note_id,ordinal,heading,text) VALUES(?,?,?,?)",
                             (note_id, p.ordinal, p.heading, p.text))
            db.execute("INSERT INTO passages_fts(rowid,heading,text) VALUES(?,?,?)",
                       (cur.lastrowid, p.heading, p.text))
        for link in note.links:
            db.execute("INSERT INTO links(src,target_raw,context,from_frontmatter) VALUES(?,?,?,?)",
                       (note_id, link.target, link.context, int(link.from_frontmatter)))
        return note_id

    def _resolve_links(self) -> None:
        ids = self.store.note_ids()
        resolver = Resolver(self.cfg, list(ids))
        rows = self.store.q("SELECT l.id, l.target_raw, n.path FROM links l JOIN notes n ON n.id=l.src")
        with self.store.tx() as db:
            for row in rows:
                target = resolver.resolve(row["target_raw"], row["path"])
                db.execute("UPDATE links SET target_path=?, dst=? WHERE id=?",
                           (target, ids.get(target) if target else None, row["id"]))
        self.store.set_meta("duplicate_stems", json.dumps(resolver.duplicate_stems()))

    def _rebuild_authored_edges(self) -> None:
        now = time.time()
        rows = self.store.q("SELECT src,dst,from_frontmatter,COUNT(*) c FROM links WHERE dst IS NOT NULL "
                            "AND dst!=src GROUP BY src,dst,from_frontmatter")
        with self.store.tx() as db:
            db.execute("DELETE FROM edges WHERE kind IN ('explicit','related')")
            for r in rows:
                kind = "related" if r["from_frontmatter"] else "explicit"
                # Repeated mentions strengthen a link, with diminishing returns.
                weight = min(1.0, 0.6 + 0.2 * np.log1p(r["c"]))
                db.execute("INSERT INTO edges(src,dst,kind,weight,count,updated_at) VALUES(?,?,?,?,?,?) "
                           "ON CONFLICT(src,dst,kind) DO UPDATE SET weight=excluded.weight,count=excluded.count",
                           (r["src"], r["dst"], kind, float(weight), r["c"], now))
        self._bump("graph_version")

    def _embed(self, note_ids: list[int] | None = None, all_notes: bool = False) -> None:
        enc = self.encoder()
        if enc is None:
            return
        if all_notes:
            self.store.x("DELETE FROM vectors")
            notes = self.store.q("SELECT id,title FROM notes")
        else:
            marks = ",".join("?" * len(note_ids or []))
            notes = self.store.q(f"SELECT id,title FROM notes WHERE id IN ({marks})", note_ids or [])
        for start in range(0, len(notes), 256):
            batch = notes[start:start + 256]
            ids = [r["id"] for r in batch]
            marks = ",".join("?" * len(ids))
            passages = self.store.q(f"SELECT id,note_id,heading,text FROM passages WHERE note_id IN ({marks}) "
                                    "ORDER BY note_id,ordinal", ids)
            if not passages:
                continue
            pvecs = enc.encode([p["heading"] + "\n" + p["text"] for p in passages])
            by_note: dict[int, list[tuple[np.ndarray, int]]] = {}
            for p, v in zip(passages, pvecs):
                by_note.setdefault(p["note_id"], []).append((v, len(p["text"])))
            with self.store.tx() as db:
                db.execute(f"DELETE FROM vectors WHERE owner='p' AND owner_id IN "
                           f"(SELECT id FROM passages WHERE note_id IN ({marks}))", ids)
                db.execute(f"DELETE FROM vectors WHERE owner='n' AND owner_id IN ({marks})", ids)
                for p, v in zip(passages, pvecs):
                    db.execute("INSERT INTO vectors(owner,owner_id,encoder,vec) VALUES('p',?,?,?)",
                               (p["id"], enc.id, pack(v)))
                for nid, items in by_note.items():
                    weights = np.array([np.sqrt(max(n, 1)) for _, n in items])
                    centroid = l2n((np.stack([v for v, _ in items]) * weights[:, None]).sum(axis=0))
                    db.execute("INSERT INTO vectors(owner,owner_id,encoder,vec) VALUES('n',?,?,?)",
                               (nid, enc.id, pack(centroid)))
        self._bump("vectors_version")

    def rebuild_semantic_edges(self) -> int:
        notes = self.note_matrix()
        ids, m = notes["ids"], notes["matrix"]
        now = time.time()
        rows: list[tuple[int, int, float]] = []
        if len(ids) > 1:
            sims = m @ m.T
            np.fill_diagonal(sims, -1.0)
            k = min(self.cfg.semantic_k, len(ids) - 1)
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k > 0 else np.zeros((len(ids), 0), int)
            seen: set[tuple[int, int]] = set()
            for i, row in enumerate(top):
                for j in row:
                    s = float(sims[i, j])
                    a, b = sorted((ids[i], ids[int(j)]))
                    if s >= self.cfg.semantic_min_sim and (a, b) not in seen:
                        seen.add((a, b))
                        rows.append((a, b, s))
        with self.store.tx() as db:
            db.execute("DELETE FROM edges WHERE kind='semantic'")
            db.executemany("INSERT INTO edges(src,dst,kind,weight,count,updated_at) VALUES(?,?,'semantic',?,1,?)",
                           [(a, b, s, now) for a, b, s in rows])
        self._bump("graph_version")
        return len(rows)

    # ── cached views ─────────────────────────────────────────────────────
    def _bump(self, key: str) -> None:
        # One statement, so two processes bumping at once both count. A lost
        # bump would leave a third process's cache believing nothing changed.
        self.store.x("INSERT INTO meta(key,value) VALUES(?, '1') ON CONFLICT(key) "
                     "DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)", (key,))

    def graph(self) -> Graph:
        version = self.store.get_meta("graph_version", "0") or "0"
        if self._graph and self._graph[0] == version:
            return self._graph[1]
        node_ids = [r[0] for r in self.store.q("SELECT id FROM notes ORDER BY id")]
        rows = [tuple(r) for r in self.store.q("SELECT src,dst,kind,weight,updated_at FROM edges")]
        g = Graph.from_rows(node_ids, rows, self.cfg.hebbian_half_life_days)
        self._graph = (version, g)
        return g

    def invalidate(self) -> None:
        self._graph = None
        self._dense = None

    def dense(self) -> dict[str, Any]:
        version = (self.store.get_meta("vectors_version", "0") or "0") + ":" + (self.store.get_meta("encoder_id", "") or "")
        if self._dense and self._dense[0] == version:
            return self._dense[1]
        # Only vectors from the current encoder. While a refit is re-embedding, the
        # old basis's vectors are not comparable with a query in the new one; for
        # those few seconds dense recall sees fewer passages, never wrong ones.
        enc_id = self.store.get_meta("encoder_id", "") or ""
        rows = self.store.q("SELECT v.owner_id, v.vec, p.note_id FROM vectors v JOIN passages p ON p.id=v.owner_id "
                            "WHERE v.owner='p' AND v.encoder=? ORDER BY v.owner_id", (enc_id,))
        notes = self.store.q("SELECT v.owner_id, v.vec FROM vectors v JOIN notes n ON n.id=v.owner_id "
                             "WHERE v.owner='n' AND v.encoder=? ORDER BY v.owner_id", (enc_id,))
        data = {
            "passage_ids": np.array([r[0] for r in rows], dtype=np.int64),
            "passage_note": np.array([r[2] for r in rows], dtype=np.int64),
            "passages": np.stack([unpack(r[1]) for r in rows]) if rows else np.zeros((0, 1), np.float32),
            "ids": [r[0] for r in notes],
            "matrix": np.stack([unpack(r[1]) for r in notes]) if notes else np.zeros((0, 1), np.float32),
        }
        self._dense = (version, data)
        return data

    def note_matrix(self) -> dict[str, Any]:
        d = self.dense()
        return {"ids": d["ids"], "matrix": d["matrix"]}

    # ── recall genome ────────────────────────────────────────────────────
    def champion(self) -> tuple[int, RecallParams]:
        row = self.store.one("SELECT version,params_json FROM configs WHERE status='champion' "
                             "ORDER BY version DESC LIMIT 1")
        if row is None:
            return 0, self.cfg.params
        return int(row[0]), RecallParams.from_dict(json.loads(row[1]))
