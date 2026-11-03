# Myelin — expanded brief, design, and roadmap

This document has three parts. **Part 1** is the original request rewritten as a brief
that any engineer or agent can execute and be held to: every adjective in the request
becomes a measurement. **Part 2** is the design as built in v0.1, with the evidence
for each claim. **Part 3** is the roadmap, each step with an acceptance test.

---

## Part 1 — The brief, expanded

### The request as given

> Analyze the brain vault and how it connects nodes and what makes it intelligent and
> self-evolving; then build new software that unifies the concept into something
> smarter, with a high-quality UI, Claude Code and agent integration, evolved agentic
> automation, and neural network integration. Raise the quality and authenticity of
> everything the agents produce so nothing reads as generic generated boilerplate.

### What it left unstated, and the decisions made

| Missing detail | Decision | Why |
|---|---|---|
| What "intelligent" means | Recall that finds the right note from imperfect context, measured on a held-out benchmark | An adjective nobody can test is a wish |
| What "self-aware" means | A self-model built only from measurements, regenerated every cycle, including what it does badly | Self-description that is never re-measured goes stale (see the vault's BLACKBOARD) |
| What "self-evolving" means | Parameters and structure change only when a change beats the incumbent on data it was not tuned on | Unconstrained self-modification is how systems drift into nonsense |
| Where learning signal comes from | Authored links (free labels), human decisions on proposals, and which recalled notes Claude actually reads | Each is a real signal already produced by normal use |
| Who may write to notes | Nobody but a person, through an inbox | The vault's invariant: model output is untrusted; no direct memory writes |
| Neural network scope | Models small enough to train in seconds on a laptop, compared against classical baselines on held-out data | A neural net that loses to Adamic–Adar should not ship, and the system should say so |
| Runtime dependencies | numpy and PyYAML required; MCP and Anthropic SDK optional | Must run where the vault runs, offline, without a GPU |
| UI quality bar | The vault's own web-craft rules and `deslop.py`, passed and reported | The user already wrote the standard; use it |
| Authenticity | A gate on agent prose using the vault's anti-slop rules plus distance from the author's measured voice | "Undetectable" is not a promise anyone can keep; "made choices, here is the evidence" is |
| Security | Local by default; token, Host and Origin checks; strict CSP; no network calls without a configured key | A local server that edits notes is attackable from any open browser tab |

### The brief (reusable)

> **Role.** You are building the cognitive layer for a markdown knowledge vault used by
> one person and several AI agents.
>
> **Goal.** Given a question or a half-written paragraph, return the notes that answer
> it, explain why each was returned, and get measurably better at this over time
> without anyone editing configuration.
>
> **Constraints.**
> 1. Notes stay plain markdown. The tool never rewrites a note except to append an
>    accepted link inside a delimited block, and every such edit is reversible.
> 2. Scope is configuration: vendored and generated files are excluded before anything
>    is counted. Code spans are never parsed as links.
> 3. Every durable state change is an event in an append-only, hash-chained ledger.
> 4. Nothing blocks on a terminal prompt. Decisions that need a person wait in an inbox.
> 5. Runs offline. Any model provider is optional and budget-capped in dollars.
> 6. Agent-written prose passes an authenticity gate before a person sees it.
>
> **Definitions that make the goal testable.**
> - *Recall quality*: MRR@10 on cloze items built from authored links (mask one link,
>   find its target from the surrounding paragraph), reported on a held-out split with
>   a bootstrap confidence interval, next to the score of a random ranking.
> - *Self-evolution*: a configuration or model replaces the incumbent only if it wins
>   on held-out items by a stated margin without losing recall@10.
> - *Self-awareness*: a self-model that reports topics, accuracy with intervals,
>   per-signal ablations, link-prediction scores against baselines, unanswered
>   questions, and ledger integrity, all recomputed each cycle.
> - *Neural integration*: at least one trained network whose held-out performance is
>   reported against classical baselines, with automatic fallback when it loses.
> - *Authenticity*: no high-severity default pattern from the vault's doctrine, and
>   style within two standard deviations of the author's own notes on the measured
>   features, or a finding that says which feature is off and by how much.
>
> **Deliverables.** A review of the existing vault with measurements; the software with
> tests; a web UI; an MCP server and hooks for Claude Code; documentation; this brief.
>
> **Acceptance.** All tests pass and the count is reported. The UI scores under 15 on
> `deslop.py` and shows no console errors in light and dark themes and at 390 px wide.
> The review cites a file, a symbol, or a measurement for every claim.

---

## Part 2 — Design as built (v0.1)

```text
 vault/*.md ──scope──► parse ─► passages ─► LSA space ─► dense index ┐
       │                 │                                          ├─► seeds ─► spreading ─► rerank ─► results
       │                 └─► [[links]] ─► typed weighted graph ─────┘   (BM25 +    activation   (learned,     + "why"
       │                                     ▲         │                 cosine)                 gated)
       │                                     │         ▼
       │                    Hebbian (usage)  │   GCN autoencoder ─► link proposals ─► inbox ─► person ─► note
       │                                     │                                         │
       └── self-quiz (cloze from links) ─► evolution (champion/challenger) ─► self-model ─► ledger (hash chain)
```

### Learning loops, by timescale

| Loop | Timescale | Signal | What changes | Guard |
|---|---|---|---|---|
| Spreading activation | per query | seed scores | which neighbours surface | fan normalisation on both endpoints |
| Hebbian | per session | notes read together | `hebbian` edge weights, w ← w + η(1−w) | half-life decay, pruning floor |
| Reranker | per sleep | feedback + quiz preference pairs | a 9→16→1 MLP | starts at weight 0; evolution must justify any weight |
| Link prediction | per sleep | authored links, rejected proposals | GCN autoencoder weights; proposals | chosen only if it beats Adamic–Adar and cosine on validation |
| Evolution | per sleep | self-quiz | 13 recall parameters | holdout gate, minimum holdout size, rollback |
| Proposals | per decision | accept / reject | notes (via inbox); training labels | protected notes, reversible edits, never re-proposed after rejection |

### Components

| Module | Responsibility |
|---|---|
| `vault.py` | scope rules, frontmatter, code-safe link parsing, Obsidian-compatible resolution |
| `store.py` | SQLite schema, WAL, nested transactions, hash-chained ledger with head checkpoint |
| `encoders.py`, `sparse.py` | LSA via randomized SVD on a numpy CSR matrix; pluggable command encoder |
| `graph.py` | typed adjacency, spreading activation, PageRank, Louvain communities, Hebbian rule |
| `recall.py` | candidate generation, ranking, explanation, gap logging |
| `neural/` | reverse-mode autograd (gradient-checked), AdamW, GCN layers, link predictor, reranker |
| `selfeval.py`, `evolve.py` | cloze benchmark with leakage controls; (1+λ) evolution strategy |
| `proposals.py`, `learning.py` | inbox with reversible note edits; feedback, reads, Hebbian consolidation |
| `quality.py` | authenticity gate: doctrine rules + stylometric distance from the author |
| `consolidate.py`, `selfmodel.py`, `daemon.py` | sleep cycle, self-model, scheduler without stdin |
| `server.py`, `ui/` | local API with token/Host/Origin checks, SSE, strict CSP; the web UI |
| `mcp_server.py`, `hooks.py`, `install.py`, `agent.py` | Claude Code tools and hooks; Claude API agent |

### Results on AGENT_BRAIN_VAULT (2026-10-04)

| Measure | Value |
|---|---|
| Notes / passages / links | 55 / 1,185 / 477 |
| Self-quiz | 216 items, 45 held out |
| Holdout MRR@10 | 0.175 at seed → 0.247 after 10 generations (CI 0.161–0.342); random 0.054 |
| Link prediction, test AUC | GCN autoencoder 0.80–0.83; Adamic–Adar 0.79; cosine 0.76–0.77 |
| Topics found | 6 (Louvain); naive label propagation found 1 |
| Sleep cycle | 8–14 s end to end |
| Query latency | 2–6 ms warm |
| UI | `deslop.py` 0/100; no console errors; no horizontal overflow at 390 px |
| Tests | 50 passed |

### Known limits

- The holdout is small; differences under about 0.09 MRR are inside the noise.
- Evolution consults the holdout once per generation; repeated reuse slowly overfits
  it. The count is reported so this can be judged, and Part 3 replaces it.
- LSA captures topical similarity, not paraphrase. A stronger encoder plugs in through
  `encoder.command`; nothing is assumed about its quality until the quiz measures it.
- The authenticity gate measures choices, not authorship. It will pass careful
  generated prose and can flag unusual human prose; it says which feature fired.
- The Claude agent path is tested with a scripted client, not against the live API,
  in this environment (no key was available).

---

## Part 3 — Roadmap

Each step names what it adds and the test that says it worked. Order is by expected
value per effort.

### v0.2 — Harder benchmark, honest holdout

1. **Rotating holdout with a sealed test split.** Three-way split by path hash: train
   (selection), validation (promotion), sealed test (reported only, read once per
   week). *Accept when* the self-model reports all three and the sealed score is never
   used in a decision (asserted in a test).
2. **Question items from Claude.** `myelin agent quiz` asks Claude for two natural
   questions per note, cached and versioned. Paraphrased questions are the case LSA
   handles worst, so this is where encoder upgrades will show. *Accept when* the quiz
   has a `kind` column and ablations are reported per kind.
3. **Encoder bake-off.** Run the quiz with LSA and each configured command encoder; keep
   the winner by holdout MRR, record the decision in the ledger. *Accept when*
   switching encoders is a ledger event with both scores.

### v0.3 — Memory with time

4. **Version history as episodic memory.** Read `git log --follow` per note; store
   versions; let recall answer "what did this note say in August". *Accept when* a test
   recalls a passage that exists only in an earlier commit.
5. **Ordered co-activation.** Replace symmetric Hebbian updates with a timing-dependent
   rule: reading A shortly before B strengthens A→B more than B→A. *Accept when* the
   quiz ablation shows directed learned edges at least matching symmetric ones.
6. **Staleness.** Flag notes whose claims contradict newer notes on the same topic
   (Claude extracts claims; a small NLI check compares them). Output is proposals,
   never edits. *Accept when* a seeded contradiction in the fixture vault is found.

### v0.4 — Stronger learning

7. **GraphSAGE with neighbour sampling** for vaults beyond ~5,000 notes, replacing
   full-graph GCN passes. *Accept when* a 20,000-note synthetic vault trains in under
   two minutes with test AUC within 0.02 of the full-graph model.
8. **CMA-ES** in place of the (1+λ) strategy once the genome grows past ~15 parameters.
   *Accept when* it reaches the (1+λ) result in fewer generations on the fixture.
9. **Calibrated proposals.** Isotonic calibration of link probabilities against human
   decisions, so "p 0.9" means 90% get accepted. *Accept when* ECE on decided proposals
   drops below 0.1 with at least 50 decisions.

### v0.5 — Agents that improve their own output

10. **Draft–critique–revise loop.** When `propose_note` returns findings, the agent
    revises and retries up to twice; the inbox shows the revision history. *Accept
    when* the share of first drafts needing revision is tracked per agent over time.
11. **Learned authenticity model.** Train a classifier on accepted versus rejected
    agent drafts (features: rule hits, stylometry, specificity). It ranks the inbox;
    it does not block. *Accept when* it beats the rule score at predicting acceptance
    on held-out decisions.
12. **Code and UI gate in CI.** Run `deslop.py --strict` and `myelin critique --strict`
    on changed files in pull requests. *Accept when* a seeded tell fails the check.

### v0.6 — Reach

13. **Obsidian sidebar.** Show recall and suggested links for the open note through
    the vault's existing Local REST API bridge. *Accept when* accepting a suggestion in
    Obsidian produces the same ledger event as the web UI.
14. **Multi-vault federation.** Query several vaults with per-vault scores calibrated
    on each vault's own quiz. *Accept when* results from two fixture vaults interleave
    by calibrated score, not raw score.
15. **Ledger anchoring.** Write the ledger head hash into each git commit trailer, so
    tampering with the database is detectable from repository history alone.

### Non-goals

- Autonomous editing of notes. A person decides; the system proposes and learns.
- A hosted service. Local first; anything networked is opt-in and budgeted.
- Claims of general intelligence. Every capability is the size of its measurement.
