"""Reading the vault: scope rules, markdown parsing, and Obsidian-style link resolution.

Three lessons from AGENT_BRAIN_VAULT are encoded here because each one produced a
badly wrong number before it was understood:

1. **Scope.** 98% of the `.md` files under that vault were vendored repos. An
   unscoped scan reported 27,126 broken links. Scope is configuration, applied
   before anything is counted.
2. **Code is not prose.** `[[TARGET_NOTE]]` inside a fenced block or an inline code
   span is an example, not a link. Both are stripped before links are parsed.
3. **Links resolve to any file**, not only notes: `[[runner.py]]` is valid.
   A target set built from `.md` stems alone reports such links as broken.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from .config import Config

# Bump whenever parsing output changes for an unchanged file. Ingest compares it
# with the stored value and re-parses everything on a mismatch, so an upgrade
# never leaves an index built by older rules.
PARSER_VERSION = "4"   # 4: every link to the masked target is masked in its context

_FENCE = re.compile(r"^(\s*)(```+|~~~+)[^\n]*\n.*?^\1\2[ \t]*$", re.DOTALL | re.MULTILINE)
_INLINE_CODE = re.compile(r"(`+)(?!`).+?(?<!`)\1(?!`)", re.DOTALL)
_COMMENT = re.compile(r"%%.*?%%|<!--.*?-->", re.DOTALL)
_WIKILINK = re.compile(r"(!?)\[\[([^\[\]\n]+?)\]\]")
_TAG = re.compile(r"(?<![\w/#&])#([A-Za-z][\w/\-]*)")
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class LinkRef:
    target: str          # as written, minus alias/heading/block parts
    context: str         # surrounding paragraph with the link itself masked out
    embed: bool = False
    from_frontmatter: bool = False


@dataclass
class Note:
    path: str                    # posix path relative to the vault root
    stem: str
    title: str
    body: str                    # prose with code removed, used for indexing
    raw_hash: str
    mtime: float
    frontmatter: dict[str, Any] = field(default_factory=dict)
    frontmatter_error: str | None = None
    tags: list[str] = field(default_factory=list)
    links: list[LinkRef] = field(default_factory=list)

    @property
    def note_type(self) -> str:
        value = self.frontmatter.get("type")
        return str(value) if value else "note"


def in_scope(rel: str, cfg: Config) -> bool:
    if cfg.report_note and rel == cfg.report_note:
        return False  # never learn from our own generated report
    parts = rel.split("/")
    if any(p in cfg.exclude_dirs for p in parts[:-1]):
        return False
    if parts[-1] in cfg.exclude_files:
        return False
    if any(fnmatch.fnmatchcase(rel, pat) for pat in cfg.exclude):
        return False
    return any(_glob_match(rel, pat) for pat in cfg.include)


def _glob_match(rel: str, pattern: str) -> bool:
    # fnmatch's `*` already crosses `/`, so `dir/**` works as a prefix match; the
    # special case is `**/*.md`, which must also match files at the root.
    if pattern.startswith("**/") and fnmatch.fnmatchcase(rel, pattern[3:]):
        return True
    return fnmatch.fnmatchcase(rel, pattern)


def walk_files(cfg: Config) -> Iterator[tuple[str, Path]]:
    """Every file under the root outside excluded directories (for resolution)."""
    root = cfg.root
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in cfg.exclude_dirs and not d.startswith("."))
        for name in sorted(filenames):
            full = Path(dirpath) / name
            yield full.relative_to(root).as_posix(), full


def scoped_notes(cfg: Config) -> list[tuple[str, Path]]:
    return [(rel, full) for rel, full in walk_files(cfg)
            if rel.endswith(".md") and in_scope(rel, cfg)]


def _defang(code: str) -> str:
    return code.replace("[[", "").replace("]]", "")


def strip_code(text: str) -> str:
    """Neutralise code for link parsing while keeping its words for search.

    `[[TARGET_NOTE]]` inside a code span is an example, not a link, so the
    brackets go. The identifiers stay: `runner.py` and `first_party_md_files`
    are exactly what someone searches for. Fence lines themselves are dropped.
    """
    def fence(m: re.Match) -> str:
        lines = m.group(0).split("\n")
        return "\n".join(["", *(_defang(ln) for ln in lines[1:-1]), ""])
    text = _FENCE.sub(fence, text)
    text = _COMMENT.sub(" ", text)
    return _INLINE_CODE.sub(lambda m: _defang(m.group(0).strip("`")), text)


def split_frontmatter(raw: str) -> tuple[dict[str, Any], str, str | None]:
    if not raw.startswith("---"):
        return {}, raw, None
    lines = raw.split("\n")
    if lines[0].strip() != "---":
        return {}, raw, None
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            block = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1:])
            try:
                data = yaml.safe_load(block) or {}
            except yaml.YAMLError as exc:
                return {}, body, f"YAML: {str(exc).splitlines()[0]}"
            if not isinstance(data, dict):
                return {}, body, "frontmatter is not a mapping"
            return data, body, None
    return {}, raw, "unterminated frontmatter"


def _link_target(inner: str) -> str:
    target = inner.split("|", 1)[0]
    target = re.split(r"[#^]", target, maxsplit=1)[0]
    return target.strip()


_LIST_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)]|\|)")


def _paragraph_around(text: str, start: int, end: int, target: str = "") -> str:
    """The context a reader sees around a link, with the link masked. Any other
    link to the same note in that context is masked too: the self-quiz asks
    "which note belongs at the gap?", and a second mention would answer it.

    In prose that is the paragraph. In a list or table it is the item's own line
    plus its section heading: the whole list as one "paragraph" is a bag of
    sibling names that says nothing about which one belongs at the gap."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    if _LIST_LINE.match(text[line_start:line_end]):
        heading = ""
        for m in re.finditer(r"^#{1,6}\s+(.+)$", text[:line_start], re.MULTILINE):
            heading = m.group(1)
        para = heading + ": " + text[line_start:start] + " [...] " + text[end:line_end]
    else:
        left = text.rfind("\n\n", 0, start)
        right = text.find("\n\n", end)
        left = 0 if left == -1 else left + 2
        right = len(text) if right == -1 else right
        para = text[left:start] + " [...] " + text[end:right]
    same = target.strip().lower()
    para = _WIKILINK.sub(lambda m: "[...]" if same and _link_target(m.group(2)).lower() == same
                         else _display(m.group(2)), para)
    para = re.sub(r"^\s*[-*+|]\s*", "", para)
    return " ".join(para.split())[:600]


def _display(inner: str) -> str:
    if "|" in inner:
        return inner.split("|", 1)[1]
    return _link_target(inner)


def _frontmatter_links(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        for m in _WIKILINK.finditer(value):
            yield _link_target(m.group(2))
    elif isinstance(value, list):
        for item in value:
            yield from _frontmatter_links(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _frontmatter_links(item)


def parse_note(rel: str, full: Path) -> Note:
    raw = full.read_text(encoding="utf-8", errors="replace")
    fm, body, fm_error = split_frontmatter(raw)
    prose = strip_code(body)
    stem = Path(rel).stem

    links: list[LinkRef] = []
    for m in _WIKILINK.finditer(prose):
        target = _link_target(m.group(2))
        if not target:
            continue
        links.append(LinkRef(target, _paragraph_around(prose, m.start(), m.end(), target), bool(m.group(1))))
    for target in _frontmatter_links(fm):
        if target:
            links.append(LinkRef(target, "", from_frontmatter=True))

    tags: list[str] = []
    fm_tags = fm.get("tags")
    if isinstance(fm_tags, str):
        fm_tags = [t.strip() for t in fm_tags.split(",")]
    if isinstance(fm_tags, list):
        tags.extend(str(t).lstrip("#") for t in fm_tags if t)
    tags.extend(m.group(1) for m in _TAG.finditer(prose))
    tags = sorted(dict.fromkeys(t.lower() for t in tags))

    title = fm.get("title") if isinstance(fm.get("title"), str) else None
    if not title:
        h1 = _H1.search(prose)
        title = _clean_heading(h1.group(1)) if h1 else stem
    readable = _WIKILINK.sub(lambda m: _display(m.group(2)), prose)

    return Note(
        path=rel, stem=stem, title=title or stem, body=readable,
        raw_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        mtime=full.stat().st_mtime, frontmatter=_jsonable(fm),
        frontmatter_error=fm_error, tags=tags, links=links,
    )


def _clean_heading(text: str) -> str:
    # Drop leading emoji/pictographs that many vault notes put in H1s.
    cleaned = re.sub(r"^[^\w\[(]+", "", text).strip()
    return cleaned or text.strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Resolver:
    """Obsidian's link resolution, close enough to agree with its metadataCache:
    match by note stem, by full filename for non-notes, by path suffix for
    path-qualified links, and case-insensitively as a last resort."""

    def __init__(self, cfg: Config, note_paths: list[str]):
        self.by_stem: dict[str, list[str]] = {}
        self.by_name: dict[str, list[str]] = {}
        self.lower: dict[str, list[str]] = {}
        self.notes = set(note_paths)
        for rel, _ in walk_files(cfg):
            p = Path(rel)
            key = p.stem if p.suffix == ".md" else p.name
            bucket = self.by_stem if p.suffix == ".md" else self.by_name
            bucket.setdefault(key, []).append(rel)
            self.lower.setdefault(key.lower(), []).append(rel)
        for rel in note_paths:  # scoped notes always resolvable even if hidden dirs
            stem = Path(rel).stem
            if rel not in self.by_stem.get(stem, []):
                self.by_stem.setdefault(stem, []).append(rel)
                self.lower.setdefault(stem.lower(), []).append(rel)

    def resolve(self, target: str, source: str | None = None) -> str | None:
        target = target.strip()
        if target.endswith(".md"):
            target = target[:-3]
        if "/" in target:
            suffix = target if Path(target).suffix else target + ".md"
            hits = [p for p in self.notes if p.endswith(suffix)] or \
                   [p for paths in self.by_name.values() for p in paths if p.endswith(target)]
            return self._pick(hits, source)
        hits = self.by_stem.get(target) or self.by_name.get(target) or self.lower.get(target.lower())
        return self._pick(hits or [], source)

    def _pick(self, hits: list[str], source: str | None) -> str | None:
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]
        # Ambiguous stem: prefer an in-scope note, then the closest folder.
        scoped = [h for h in hits if h in self.notes] or hits
        if source:
            src_dir = str(Path(source).parent)
            same = [h for h in scoped if str(Path(h).parent) == src_dir]
            if same:
                return same[0]
        return sorted(scoped, key=lambda h: (h.count("/"), h))[0]

    def duplicate_stems(self) -> dict[str, list[str]]:
        return {k: v for k, v in self.by_stem.items() if len(v) > 1 and any(p in self.notes for p in v)}
