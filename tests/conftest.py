"""A synthetic vault rich enough to exercise every subsystem: five topic clusters
of six notes, intra-cluster links with real prose around them, a few bridges,
code blocks containing fake links, a protected generated note, and a non-note
link target."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myelin.brain import Brain  # noqa: E402
from myelin.config import load_config  # noqa: E402

TOPICS = {
    "astro": ("telescope", "orbit", "nebula", "spectrograph", "parallax", "redshift", "aperture", "eclipse"),
    "bake": ("sourdough", "levain", "crumb", "proofing", "hydration", "gluten", "oven", "starter"),
    "sec": ("firewall", "exploit", "ciphertext", "phishing", "sandbox", "malware", "token", "audit"),
    "sail": ("mainsail", "tack", "keel", "harbour", "mooring", "jib", "regatta", "rigging"),
    "garden": ("compost", "seedling", "mulch", "pruning", "trellis", "aphid", "loam", "irrigation"),
}


def _sentence(rng: random.Random, words: tuple[str, ...], n: int = 14) -> str:
    filler = ["the", "notes", "describe", "how", "careful", "work", "with", "each", "measured", "step", "matters"]
    picks = [rng.choice(words) if rng.random() < 0.45 else rng.choice(filler) for _ in range(n)]
    return " ".join(picks).capitalize() + "."


def build_vault(root: Path, seed: int = 3) -> Path:
    rng = random.Random(seed)
    (root / "notes").mkdir(parents=True)
    (root / "system").mkdir()
    (root / "system" / "runner.py").write_text("print('hi')\n")
    names = {t: [f"{t.upper()}_{i}" for i in range(6)] for t in TOPICS}
    for topic, words in TOPICS.items():
        for i, name in enumerate(names[topic]):
            peers = [n for n in names[topic] if n != name]
            linked = rng.sample(peers, 3)
            paras = [
                f"# {name.replace('_', ' ').title()}\n",
                _sentence(rng, words) + " " + _sentence(rng, words),
                f"This connects to [[{linked[0]}]] because the {words[i % 8]} and {words[(i + 3) % 8]} overlap "
                f"in practice. " + _sentence(rng, words),
                "## Details\n",
                f"- {words[(i + 1) % 8]} relates to [[{linked[1]}]] through {words[(i + 2) % 8]} handling",
                f"- see [[{linked[2]}|the {words[(i + 4) % 8]} note]] for {words[(i + 5) % 8]} specifics",
                "\n```\nexample: [[NOT_A_REAL_LINK]] inside code\n```\n",
                f"Inline `[[ALSO_NOT_A_LINK]]` and `{words[i % 8]}_config` stay as text.",
            ]
            if i == 0:
                other = rng.choice([t for t in TOPICS if t != topic])
                paras.append(f"A bridge to [[{names[other][1]}]] where {words[2]} meets {TOPICS[other][2]}.")
            if i == 1:
                paras.append("The runner lives at [[runner.py]] and a missing target is [[GHOST_NOTE]].")
            fm = f"---\ntype: note\ntags: [{topic}]\nrelated: [\"[[HOME]]\"]\n---\n"
            (root / "notes" / f"{name}.md").write_text(fm + "\n\n".join(paras) + "\n")
    home = ["---\ntype: moc\n---\n# Home\n"] + [f"- [[{names[t][0]}]] — the {t} cluster" for t in TOPICS]
    (root / "HOME.md").write_text("\n".join(home) + "\n")
    (root / "system" / "GENERATED.md").write_text("---\ntype: analytics\n---\n# Generated\nRewritten by a job. "
                                                   "Links to [[HOME]].\n")
    (root / "myelin.toml").write_text(
        '[vault]\ninclude = ["**/*.md"]\nprotected = ["system/GENERATED.md"]\n'
        'report_note = "system/SELF_MODEL.md"\ninbox_dir = "inbox"\n'
        '[daemon]\nevolution_lambda = 4\n')
    return root


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    return build_vault(tmp_path / "vault")


@pytest.fixture()
def brain(vault: Path, tmp_path: Path, monkeypatch) -> Brain:
    monkeypatch.setenv("MYELIN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("MYELIN_ROOT", raising=False)
    b = Brain(load_config(vault))
    b.ingest()
    yield b
    b.close()
