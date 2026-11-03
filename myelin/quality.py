"""Authenticity gate for agent-written prose.

The vault's ANTI_SLOP_DOCTRINE states the principle this module enforces:
*a tell is an unspecified default, not a banned value.* Two kinds of check
follow from it.

**Default patterns** (rules `P-0xx`) are phrasings that appear because nothing
else was chosen: marketing openers, the "it's not just X — it's Y" frame, the
rhetorical-question opener, the tidy summary closer, model-favourite vocabulary,
unquantified superlatives. Like `deslop.py`, each rule has a repetition
threshold: one occurrence can be a decision, several cannot.

**Voice distance** is relative, not absolute. Em-dashes are not slop in this
vault — its author uses them constantly. So style is measured against a
baseline built from the vault's own hand-written notes (sentence length,
dash and question rates, numbers and identifiers per 100 words, hedging,
intensifiers, vocabulary richness), and text is flagged only where it sits far
from how this author actually writes.

What this does not do: it cannot prove who or what wrote a text, and it makes
no promise that output is undetectable. It measures whether choices were made,
and it tells the writer — human or agent — exactly which ones were not.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from .vault import split_frontmatter

SEVERITY = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str
    threshold: int
    title: str
    pattern: re.Pattern
    why: str
    fix: str


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE | re.MULTILINE)


RULES: list[Rule] = [
    Rule("P-001", "high", 1, "Marketing-cliché opener",
         _rx(r"\b(supercharge|unleash|elevate your|transform your|unlock the (?:full )?power|take \w+(?: \w+)? to the next level"
             r"|effortless(?:ly)?|seamless(?:ly)? integrat\w*|revolutionary|game[- ]chang\w+|cutting[- ]edge|robust and scalable"
             r"|built for modern teams|empower your|reimagined)\b"),
         "Interchangeable copy: it could describe any product, so it describes none.",
         "Replace with one concrete claim, ideally with a number: what it does, for whom, how much."),
    Rule("P-002", "high", 1, "“Not just X — it's Y” frame",
         _rx(r"\b(?:it'?s|this is|isn'?t|is) not (?:just|only|merely) [^.\n]{1,60}?(?:—|–|-{1,2}|,|;)\s*(?:it'?s|but)\b"
             r"|\bnot only\b[^.\n]{1,80}\bbut also\b"),
         "A contrast set up only to be knocked down; one of the most recognisable generated rhythms.",
         "State the actual claim directly."),
    Rule("P-003", "medium", 1, "Rhetorical question, answered at once",
         _rx(r"\b(?:why|what|how|who)\b[^?\n]{3,80}\?\s+(?:because|the answer|simple|it'?s simple|here'?s (?:why|how|the)"
             r"|the short answer|the reason|well,)"),
         "Asking the reader a question you are about to answer delays the answer.",
         "Lead with the answer; the question is implied by the heading."),
    Rule("P-004", "medium", 1, "Tidy summary closer",
         _rx(r"(?:^|\n|\. )(?:In (?:summary|conclusion|short|essence)|Ultimately|Overall|All in all|The bottom line(?: is)?|"
             r"At the end of the day)\s*[,:]"),
         "Restating what was just said; closing every section with a bow is a template.",
         "End when the content ends. Cut the closer or replace it with the next action."),
    Rule("P-005", "medium", 3, "Model-favourite vocabulary",
         _rx(r"\b(delv(?:e|es|ing)|tapestry|testament to|ever[- ]evolving|realm of|multifaceted|pivotal|intricate|embark"
             r"|navigat(?:e|ing) the complexit\w+|harness(?:ing)? the|foster(?:s|ing)?|underscore[sd]?|paramount|meticulous(?:ly)?"
             r"|bustling|vibrant|beacon|synerg\w+|holistic|in today'?s (?:fast[- ]paced|digital)|landscape of|a myriad of)\b"),
         "Words that are rare in human technical writing and common in model output.",
         "Use the plain word: 'look at', 'important', 'complex', 'use', 'show'."),
    Rule("P-006", "low", 2, "Throat-clearing filler",
         _rx(r"\b(it'?s worth noting(?: that)?|it is important to note(?: that)?|needless to say|it goes without saying"
             r"|when it comes to|in order to|the fact that|as (?:we|you) can see)\b"),
         "Words that announce a point instead of making it.",
         "Delete the phrase; the sentence usually stands without it."),
    Rule("P-007", "high", 1, "Unquantified superlative",
         _rx(r"\b(world[- ]class|best[- ]in[- ]class|unparalleled|state[- ]of[- ]the[- ]art|industry[- ]leading|second to none"
             r"|unmatched|next[- ]generation|blazing(?:ly)? fast)\b"),
         "A claim of superiority with no measurement behind it.",
         "Give the measurement, or drop the claim."),
    Rule("P-008", "medium", 2, "Vague intensity claim",
         _rx(r"\b(significantly|dramatically|massively|vastly|greatly|substantially) (?:improv|increas|reduc|enhanc|boost|speed)\w*\b"
             r"(?![^.\n]*\d)"),
         "'Significantly better' with no number in the sentence is unfalsifiable.",
         "Put the number in the same sentence: 'cut p95 latency from 410 ms to 95 ms'."),
    Rule("P-009", "high", 1, "Assistant chatter",
         _rx(r"^(?:great question|certainly!|absolutely!|of course!|sure!|i hope this helps|let me know if you"
             r"|as an ai\b|i'?d be happy to)"),
         "Conversational residue from a chat model, out of place in a document.",
         "Delete it."),
    Rule("P-010", "low", 3, "Adjective triad",
         _rx(r"\b(\w+(?:ly)?), (\w+), and (\w+) (?:solution|platform|system|experience|approach|way|tool|design)s?\b"),
         "Three stacked adjectives before a generic noun ('fast, reliable, and secure platform').",
         "Keep the one adjective you can prove; show the rest."),
]

FEATURES = ["sentence_len", "sentence_len_sd", "dash_rate", "question_rate", "number_rate", "identifier_rate",
            "first_person_rate", "hedge_rate", "intensifier_rate", "word_len", "ttr", "bullet_ratio"]
FEATURE_NAMES = {
    "sentence_len": "words per sentence", "sentence_len_sd": "sentence-length variation",
    "dash_rate": "em-dashes per 100 words", "question_rate": "questions per 100 words",
    "number_rate": "numbers per 100 words", "identifier_rate": "code identifiers/paths per 100 words",
    "first_person_rate": "first-person words per 100", "hedge_rate": "hedges per 100 words",
    "intensifier_rate": "intensifiers per 100 words", "word_len": "characters per word",
    "ttr": "vocabulary richness (type/token, first 300 words)", "bullet_ratio": "share of lines that are bullets",
}

_FENCE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_NUM = re.compile(r"\b\d[\d,.]*%?")
_IDENT = re.compile(r"`[^`\n]+`|\b\w+\.(?:py|md|json|sh|ts|js|toml|yaml|db)\b|\b[a-z]+_[a-z_]+\b|\b\w+/\w[\w/.-]*")
_FIRST = re.compile(r"\b(i|we|my|our|me|us)\b", re.IGNORECASE)
_HEDGE = re.compile(r"\b(might|may|could|perhaps|possibly|likely|arguably|somewhat|generally|typically)\b", re.IGNORECASE)
_INTENS = re.compile(r"\b(very|really|extremely|highly|incredibly|truly|deeply|absolutely|super)\b", re.IGNORECASE)


def prose(text: str) -> str:
    _, body, _ = split_frontmatter(text)
    body = _FENCE.sub(" ", body)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    return body


def style_features(text: str) -> dict[str, float] | None:
    body = prose(text)
    lines = [ln for ln in body.splitlines() if ln.strip() and not ln.lstrip().startswith(("|", "#", ">"))]
    items = [re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", ln) for ln in lines]
    flat = " ".join(items)
    words = _WORD.findall(flat)
    n = len(words)
    if n < 60:
        return None
    # A line break ends a sentence too: bullet items rarely carry a full stop,
    # and joining them first produced 80-word "sentences".
    sentences = [s for item in items for s in _SENT.split(item) if len(_WORD.findall(s)) >= 3]
    lens = [len(_WORD.findall(s)) for s in sentences] or [n]
    mean = sum(lens) / len(lens)
    sd = math.sqrt(sum((x - mean) ** 2 for x in lens) / len(lens))
    per100 = lambda count: 100.0 * count / n
    head = [w.lower() for w in words[:300]]
    bullets = sum(1 for ln in lines if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", ln))
    return {
        "sentence_len": mean, "sentence_len_sd": sd,
        "dash_rate": per100(flat.count("—") + flat.count(" – ")),
        "question_rate": per100(flat.count("?")),
        "number_rate": per100(len(_NUM.findall(flat))),
        "identifier_rate": per100(len(_IDENT.findall(body))),
        "first_person_rate": per100(len(_FIRST.findall(flat))),
        "hedge_rate": per100(len(_HEDGE.findall(flat))),
        "intensifier_rate": per100(len(_INTENS.findall(flat))),
        "word_len": sum(len(w) for w in words) / n,
        "ttr": len(set(head)) / max(1, len(head)),
        "bullet_ratio": bullets / max(1, len(lines)),
    }


def build_baseline(texts: list[str]) -> dict[str, Any]:
    rows = [f for f in (style_features(t) for t in texts) if f]
    if len(rows) < 5:
        return {"n": len(rows)}
    out: dict[str, Any] = {"n": len(rows), "mean": {}, "sd": {}}
    for k in FEATURES:
        values = sorted(r[k] for r in rows)
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        # Robust floor: a feature the author never varies would otherwise make
        # every tiny deviation look like a huge z-score.
        out["mean"][k] = mean
        out["sd"][k] = max(sd, 0.15 * abs(mean), 0.05)
    return out


def critique(text: str, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    fm, _, _ = split_frontmatter(text)
    if str(fm.get("scan_policy", "")).endswith("ignore-file"):
        # A note that documents tells has to quote them (ANTI_SLOP_DOCTRINE does).
        return {"score": 0.0, "verdict": "ignored by scan_policy", "words": 0, "findings": [],
                "voice": {"available": False}, "specificity_per_100w": None, "blocking": False}
    body = "\n".join("" if "deslop-ignore" in ln else ln for ln in prose(text).split("\n"))
    words = max(1, len(_WORD.findall(body)))
    findings = []
    for rule in RULES:
        hits = list(rule.pattern.finditer(body))
        if len(hits) < rule.threshold:
            continue
        examples = []
        for m in hits[:3]:
            line = body.count("\n", 0, m.start()) + 1
            examples.append({"line": line, "text": " ".join(m.group(0).split())[:90]})
        findings.append({"rule": rule.id, "title": rule.title, "severity": rule.severity, "count": len(hits),
                         "examples": examples, "why": rule.why, "fix": rule.fix})
    weight = sum(SEVERITY[f["severity"]] * min(f["count"], 5) for f in findings)
    # Density-normalised: five tells in 3,000 words is not five tells in 150.
    pattern_score = min(100.0, 100.0 * weight / (6.0 + words / 120.0))

    voice: dict[str, Any] = {"available": False}
    feats = style_features(text)
    if feats and baseline and baseline.get("mean"):
        deviations = []
        for k in FEATURES:
            z = (feats[k] - baseline["mean"][k]) / baseline["sd"][k]
            if abs(z) >= 2.0:
                deviations.append({"feature": FEATURE_NAMES[k], "this_text": round(feats[k], 2),
                                   "your_notes": round(baseline["mean"][k], 2), "z": round(z, 1)})
        zs = [min(4.0, abs((feats[k] - baseline["mean"][k]) / baseline["sd"][k])) for k in FEATURES]
        distance = math.sqrt(sum(z * z for z in zs) / len(zs))
        voice = {"available": True, "distance": round(distance, 2), "baseline_notes": baseline["n"],
                 "deviations": sorted(deviations, key=lambda d: -abs(d["z"]))}
    elif feats is None:
        voice = {"available": False, "reason": "under 60 words of prose"}

    voice_penalty = max(0.0, (voice.get("distance", 0.0) - 1.2) * 20.0) if voice.get("available") else 0.0
    score = round(min(100.0, pattern_score + voice_penalty), 1)
    verdict = "authored" if score < 15 else ("some defaults showing" if score < 40 else "reads as generated")
    return {"score": score, "verdict": verdict, "words": words, "findings": findings, "voice": voice,
            "specificity_per_100w": round(feats["number_rate"] + feats["identifier_rate"], 2) if feats else None,
            "blocking": any(f["severity"] == "high" for f in findings)}


def vault_baseline(brain) -> dict[str, Any]:
    """Baseline from hand-written notes only: protected (generated or log) notes
    describe the generator's voice, not the author's."""
    from .proposals import is_protected
    cached = brain.store.get_meta("style_baseline")
    version = brain.store.get_meta("graph_version", "0")
    if cached:
        data = json.loads(cached)
        if data.get("graph_version") == version:
            return data
    texts = []
    for r in brain.store.q("SELECT path FROM notes"):
        if is_protected(brain.cfg, r["path"]):
            continue
        try:
            texts.append((brain.cfg.root / r["path"]).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    base = build_baseline(texts)
    base["graph_version"] = version
    brain.store.set_meta("style_baseline", json.dumps(base))
    return base


def rules_table() -> list[dict[str, Any]]:
    return [{k: v for k, v in asdict(r).items() if k != "pattern"} for r in RULES]
