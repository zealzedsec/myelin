"""Tokenization and passage chunking.

Kept deliberately plain: the same function tokenizes documents and queries, so
any quirk is at least symmetric. Identifiers are split as well as kept whole —
`first_party_md_files` yields `first_party_md_files`, `first`, `party`, `md`,
`files` — because vault notes talk about code by name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor
not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves
also via per etc using use used may might must shall vs within without
""".split())

_WORD = re.compile(r"[a-z0-9]+(?:[._\-'][a-z0-9]+)*")
_SPLIT = re.compile(r"[._\-']")


def _normalize(word: str) -> str:
    if word.endswith("'s"):
        word = word[:-2]
    # Plural folding, conservative: only long words, never "-ss" ("process").
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss") and not word.endswith("us"):
        if word.endswith("ies") and len(word) > 5:
            return word[:-3] + "y"
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercased content tokens, stopwords removed, compounds also split."""
    out: list[str] = []
    for match in _WORD.finditer(text.lower()):
        word = match.group(0)
        parts = _SPLIT.split(word)
        if len(parts) > 1:
            if word not in STOPWORDS:
                out.append(word)
            for part in parts:
                if len(part) > 1 and part not in STOPWORDS:
                    out.append(_normalize(part))
        elif len(word) > 1 and word not in STOPWORDS:
            out.append(_normalize(word))
    return out


def features(text: str) -> list[str]:
    """Unigrams plus adjacent-pair bigrams; the LSA encoder's input vocabulary."""
    toks = tokenize(text)
    grams = list(toks)
    grams.extend(f"{a}__{b}" for a, b in zip(toks, toks[1:]) if a != b)
    return grams


def fts_query(text: str, limit: int = 24) -> str:
    """Build an FTS5 OR-query. AND-joining every word (what the vault's old
    semantic_router did) returns nothing for any natural-language question that
    contains one word absent from the target passage."""
    seen: list[str] = []
    for tok in tokenize(text):
        tok = tok.replace('"', "")
        if tok and tok not in seen and re.fullmatch(r"[a-z0-9]+", tok):
            seen.append(tok)
        if len(seen) >= limit:
            break
    return " OR ".join(f'"{t}"' for t in seen)


@dataclass
class Passage:
    ordinal: int
    heading: str
    text: str


_HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")


def chunk(body: str, title: str, max_words: int = 180, overlap: int = 40) -> list[Passage]:
    """Split by headings, then window long sections. Each passage carries its
    heading path so a hit can be explained ("CLAUDE › Scope rule")."""
    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    current: list[str] = []
    heading = title
    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            if current:
                sections.append((heading, current))
            level = len(m.group(1))
            # Leading emoji and symbols are decoration in many vault headings.
            label = re.sub(r"^[^\w\[(]+", "", m.group(2).strip("# ")).strip() or m.group(2).strip("# ")
            path = path[: level - 1] + [label]
            heading = " › ".join([title, *[p for p in path if p and p != title]])
            current = []
            continue
        current.append(line)
    if current:
        sections.append((heading, current))

    passages: list[Passage] = []
    for head, lines in sections:
        words = " ".join(lines).split()
        if not words:
            continue
        step = max(1, max_words - overlap)
        for start in range(0, len(words), step):
            window = words[start:start + max_words]
            if len(window) < 8 and start > 0:
                break
            passages.append(Passage(len(passages), head, " ".join(window)))
            if start + max_words >= len(words):
                break
    if not passages:
        passages.append(Passage(0, title, title))
    return passages
