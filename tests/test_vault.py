from pathlib import Path

from myelin.config import Config, load_config
from myelin.vault import Resolver, in_scope, parse_note, scoped_notes, strip_code


def test_code_is_not_links_but_stays_searchable():
    text = "real [[A]]\n```\n[[FAKE]] in a fence\n```\nand `[[INLINE]] first_party_md_files` here"
    out = strip_code(text)
    assert "[[FAKE]]" not in out and "[[INLINE]]" not in out
    assert "[[A]]" in out
    assert "first_party_md_files" in out and "FAKE" in out


def test_parse_links_aliases_headings_frontmatter(tmp_path: Path):
    p = tmp_path / "n.md"
    p.write_text('---\nrelated: ["[[R]]"]\ntags: [x]\n---\n# Title Here\n'
                 "See [[B#Section|alias]] and ![[C^block]].\n\n- item about [[D]] — with context words here\n")
    note = parse_note("n.md", p)
    targets = [(l.target, l.from_frontmatter) for l in note.links]
    assert ("B", False) in targets and ("C", False) in targets and ("D", False) in targets
    assert ("R", True) in targets
    assert note.title == "Title Here"
    assert "alias" in note.body and "[[" not in note.body
    item = next(l for l in note.links if l.target == "D")
    # list items carry their own line, not the whole list
    assert "[...]" in item.context and "with context words" in item.context


def test_scope_rules(tmp_path: Path):
    cfg = Config(root=tmp_path, include=["notes/**", "ROOT.md"], exclude_dirs=["vendor"],
                 exclude_files=["SKIP.md"], report_note="notes/REPORT.md")
    assert in_scope("notes/a.md", cfg)
    assert in_scope("ROOT.md", cfg)
    assert not in_scope("other/a.md", cfg)
    assert not in_scope("notes/vendor/a.md", cfg)
    assert not in_scope("notes/SKIP.md", cfg)
    assert not in_scope("notes/REPORT.md", cfg)  # never ingest our own report
    cfg2 = Config(root=tmp_path, include=["**/*.md"])
    assert in_scope("top.md", cfg2) and in_scope("a/b/c.md", cfg2)


def test_resolver_matches_obsidian_rules(vault: Path):
    cfg = load_config(vault)
    notes = [rel for rel, _ in scoped_notes(cfg)]
    r = Resolver(cfg, notes)
    assert r.resolve("ASTRO_1") == "notes/ASTRO_1.md"
    assert r.resolve("runner.py") == "system/runner.py"          # non-note targets resolve
    assert r.resolve("notes/ASTRO_1") == "notes/ASTRO_1.md"      # path-qualified
    assert r.resolve("astro_1") == "notes/ASTRO_1.md"            # case-insensitive fallback
    assert r.resolve("GHOST_NOTE") is None


def test_scoped_notes_respects_exclusions(vault: Path):
    (vault / "node_modules").mkdir()
    (vault / "node_modules" / "x.md").write_text("# vendored")
    cfg = load_config(vault)
    rels = {rel for rel, _ in scoped_notes(cfg)}
    assert "node_modules/x.md" not in rels
    assert "HOME.md" in rels
