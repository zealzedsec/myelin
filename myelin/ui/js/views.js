import { MODE, api } from "./api.js";
import { axonArrow, icon, loader, viaGlyph } from "./icons.js";
import { renderMarkdown } from "./md.js";
import { clear, fmt, h, marked, plate, queryTerms, toast } from "./util.js";

const emptyState = (title, body) => h("div", { class: "empty" }, h("strong", null, title), body);

function presetList(presets, ctx) {
  return h("div", { class: "presets" }, presets.map((q) => h("button", { type: "button", onclick: () => ctx.ask(q) }, q)));
}

// ── Plate 02 · Recall ───────────────────────────────────────────────────
export async function viewRecall(root, ctx) {
  const q = ctx.query || "";
  const page = h("div", { class: "page" });
  root.appendChild(page);
  const presets = await api.presets();
  if (!q) {
    page.append(plate({ no: "02", name: "Recall", title: "Ask the vault",
      lede: "Keyword and meaning-based matching find the first notes; activation then spreads along links. Every result says where its score came from." }),
      presets.length ? [h("h2", { class: "sect" }, "Questions recorded in this snapshot"), presetList(presets, ctx)]
        : emptyState("Type a question or paste a paragraph in the bar above.", " Press / to jump there."));
    return;
  }
  page.append(plate({ no: "02", name: "Recall", title: q, display: true }), loader("Spreading activation"));
  let res;
  try { res = await api.recall(q, 12); }
  catch (e) { clear(page).append(plate({ no: "02", name: "Recall", title: q, display: true }), h("p", { class: "err" }, e.message)); return; }
  clear(page);
  if (res.snapshotMiss) {
    page.append(plate({ no: "02", name: "Recall", title: q, display: true,
      lede: "This is a snapshot, so only the questions below have recorded answers. The local app answers anything." }),
      presetList(res.presets, ctx));
    return;
  }
  ctx.lastRecall = res;
  const p = ctx.status?.champion?.params || {};
  const terms = queryTerms(q);
  page.append(
    plate({ no: "02", name: "Recall", title: q, display: true,
      meta: [`${res.ms} ms`, `best similarity ${fmt.n(res.best_similarity, 2)}`, `config v${res.config_version}`, res.recall_id],
      lede: res.confident ? null : "Low confidence: nothing in the vault is close, so this question was logged as a knowledge gap.",
      actions: [h("span", { class: "key" },
        h("span", null, h("i", { class: "k-dense" }), "meaning"), h("span", null, h("i", { class: "k-sparse" }), "words"),
        h("span", null, h("i", { class: "k-act" }), "graph"), h("span", null, h("i", { class: "k-rerank" }), "learned"))] }));

  if (ctx.status?.agent_available) {
    const out = h("div");
    const btn = h("button", { class: "btn", type: "button", onclick: async () => {
      btn.disabled = true;
      btn.textContent = "Claude is reading the vault…";
      try {
        const a = await api.ask(q);
        clear(out).append(h("div", { class: "synth" },
          renderMarkdown(a.answer, { stems: ctx.stems, onLink: ctx.openNote }),
          h("div", { class: "usage" }, `${a.model} · ${a.turns} turns · ${fmt.int(a.usage.input_tokens)} in / ${fmt.int(a.usage.output_tokens)} out` +
            ` · cache read ${fmt.int(a.usage.cache_read_input_tokens)} · $${a.usage.cost_usd.toFixed(4)} · cited ${a.cited.length}`)));
        btn.remove();
      } catch (e) { toast(e.message, true); btn.disabled = false; btn.textContent = "Answer with Claude"; }
    } }, "Answer with Claude");
    page.append(out, h("p", null, btn));
  }

  if (!res.results.length) { page.append(emptyState("Nothing matched.", " The question was logged as a gap.")); return; }
  const top = Math.max(...res.results.map((r) => r.score), 1e-9);
  const list = h("ol", { class: "results" });
  const rows = res.results.map((r, i) => {
    const parts = [
      ["k-dense", "meaning", (p.w_dense ?? 0.5) * r.parts.dense], ["k-sparse", "words", (p.w_sparse ?? 0.5) * r.parts.sparse],
      ["k-act", "graph", (p.w_activation ?? 1) * r.parts.activation], ["k-rerank", "learned", (p.w_rerank ?? 0) * (r.parts.rerank ?? 0)],
    ];
    const total = parts.reduce((s, [, , v]) => s + v, 0) || 1;
    const bar = h("span", { class: "bar", role: "img", "aria-label": parts.map(([, n, v]) => `${n} ${v.toFixed(2)}`).join(", ") },
      parts.map(([k, , v]) => { const s = h("span", { class: k }); s.style.width = "0%"; requestAnimationFrame(() => { s.style.width = `${(100 * v) / total * (r.score / top)}%`; }); return s; }));
    const fb = (useful) => h("button", { class: useful ? "btn" : "btn neg", type: "button", "aria-pressed": "false",
      onclick: async (e) => {
        const b = e.currentTarget;
        try {
          await api.feedback(res.recall_id, r.note_id, useful);
          b.parentElement.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
          b.setAttribute("aria-pressed", "true");
          toast(MODE === "snapshot" ? "Noted in this page only; the local app trains on it" : useful
            ? "Marked useful; the reranker and learned links train on it" : "Marked not useful");
        } catch (err) { toast(err.message, true); }
      } }, useful ? "Useful" : "Not useful");
    const section = r.heading && r.heading !== r.title ? r.heading.split(" › ").slice(1).join(" › ") : "";
    return h("li", { class: "hit", dataset: { id: r.note_id } },
      h("span", { class: "rank" }, String(i + 1).padStart(2, "0")),
      h("div", null,
        h("h3", null, h("button", { type: "button", onclick: () => ctx.openNote(r.note_id) }, r.title)),
        h("div", { class: "path" }, r.path, section ? ` · ${section}` : ""),
        r.snippet ? h("p", { class: "snip" }, marked(r.snippet, terms)) : null),
      h("div", { class: "meter" }, h("span", { class: "score" }, fmt.n(r.score)), bar,
        h("dl", { class: "parts" }, parts.filter(([, , v]) => v > 0.005).map(([k, n, v]) =>
          [h("dt", null, h("i", { class: k }), n), h("dd", null, v.toFixed(2))]))),
      h("div", { class: "foot" },
        r.via ? h("span", { class: "via" }, viaGlyph(), h("span", null, "reached via ", h("em", null, `${r.via.kind} link`), ` from ${r.via.seed_title}`))
              : h("span", { class: "via" }, "direct match on the text"),
        h("span", { class: "fb" }, fb(true), fb(false))));
  });
  list.append(...rows);
  page.append(list, h("div", { class: "hints" }, h("span", null, h("kbd", null, "j"), " ", h("kbd", null, "k"), " move"),
    h("span", null, h("kbd", null, "↵"), " open note"), h("span", null, h("kbd", null, "u"), " useful")));

  let cur = -1;
  const focusRow = (i) => {
    cur = Math.max(0, Math.min(rows.length - 1, i));
    rows.forEach((r, j) => r.classList.toggle("cur", j === cur));
    rows[cur].scrollIntoView({ block: "nearest", behavior: "smooth" });
  };
  ctx.keys = (e) => {
    if (e.key === "j" || e.key === "k") { focusRow(cur + (e.key === "j" ? 1 : -1)); return true; }
    if (cur >= 0 && e.key === "Enter") { ctx.openNote(res.results[cur].note_id); return true; }
    if (cur >= 0 && e.key === "u") { rows[cur].querySelector(".fb .btn:not(.neg)").click(); return true; }
    return false;
  };
}

// ── Plate 03 · Inbox ────────────────────────────────────────────────────
const KIND_LABEL = { link: "Links", note: "Notes", merge: "Overlaps", gap: "Gaps" };

function synapseDiagram(pl) {
  const shared = (pl.evidence?.common_titles || []).slice(0, 3);
  const W = 560, y = 30;
  const svg = h("svg:svg", { class: "synapse", viewBox: `0 0 ${W} 78`, role: "img",
    "aria-label": shared.length ? `${pl.src_title} and ${pl.dst_title} share ${shared.join(", ")}` : "No shared neighbours; similar content" });
  const sx = 14, dx = W - 14;
  const mids = shared.map((t, i) => [sx + ((dx - sx) * (i + 1)) / (shared.length + 1), i % 2 ? y + 22 : y - 14, t]);
  for (const [mx, my] of mids) {
    svg.append(h("svg:path", { class: "ax", d: `M${sx} ${y} Q ${(sx + mx) / 2} ${my} ${mx} ${my}` }),
               h("svg:path", { class: "ax", d: `M${mx} ${my} Q ${(mx + dx) / 2} ${my} ${dx} ${y}` }));
  }
  svg.append(h("svg:path", { class: "prop", d: `M${sx + 8} ${y} L ${dx - 8} ${y}` }));
  for (const [mx, my, t] of mids) {
    svg.append(h("svg:circle", { class: "mid-n", cx: mx, cy: my, r: 4.5 }),
               h("svg:text", { x: mx, y: my + (my > y ? 18 : -9), "text-anchor": "middle" }, t.length > 34 ? t.slice(0, 32) + "…" : t));
  }
  for (const x of [sx, dx]) svg.append(h("svg:circle", { class: "end", cx: x, cy: y, r: 7 }), h("svg:circle", { class: "nuc", cx: x - 1, cy: y - 1, r: 2.5 }));
  const cut = (t) => (t.length > 30 ? t.slice(0, 28) + "…" : t);
  svg.append(h("svg:text", { class: "end-t", x: sx - 6, y: y + 24 }, cut(pl.src_title)),
             h("svg:text", { class: "end-t", x: dx + 6, y: y + 24, "text-anchor": "end" }, cut(pl.dst_title)));
  if (!shared.length) svg.append(h("svg:text", { x: W / 2, y: y - 10, "text-anchor": "middle" }, `no shared neighbours · cosine ${fmt.n(pl.evidence?.cosine, 2)}`));
  return svg;
}

function gauge(p) {
  const fill = h("span", { class: "fill" });
  fill.style.width = `${Math.round(100 * p)}%`;
  return h("div", { class: "gauge", title: "Link probability from the chosen predictor; the tick marks 0.5" },
    h("div", { class: "lbl" }, h("span", null, "probability"), h("b", null, fmt.n(p, 2))),
    h("div", { class: "trk" }, fill, h("span", { class: "mid" })));
}

export async function viewInbox(root, ctx) {
  const page = h("div", { class: "page" });
  root.appendChild(page);
  const state = ctx.inbox || (ctx.inbox = { kind: "link", status: "pending", cursor: 0 });
  let items = [];
  const listEl = h("ul", { class: "cards" });
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Proposal kind" });
  const statusSeg = h("div", { class: "seg", role: "group", "aria-label": "Status" });
  const head = h("div");

  async function load() {
    const [pending, all] = await Promise.all([api.proposals("pending"), api.proposals(state.status)]);
    const counts = {};
    for (const p of pending.items) counts[p.kind] = (counts[p.kind] || 0) + 1;
    clear(seg).append(...Object.entries(KIND_LABEL).map(([k, label]) => h("button", { type: "button",
      "aria-pressed": String(state.kind === k), onclick: () => { state.kind = k; state.cursor = 0; load(); } },
      label, h("span", { class: "n" }, counts[k] || 0))));
    clear(statusSeg).append(...[["pending", "Pending"], ["applied", "Applied"], ["rejected", "Rejected"]].map(([s, label]) =>
      h("button", { type: "button", "aria-pressed": String(state.status === s), onclick: () => { state.status = s; load(); } }, label)));
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    clear(head).append(plate({ no: "03", name: "Inbox", meta: [`${total} pending`, MODE === "snapshot" ? "decisions stay in this page" : "decisions train the link predictor"],
      title: "Changes waiting for you",
      lede: "Myelin never edits a note on its own. Accepting a link appends one bullet to the source note's Associations block, reversibly; a rejected pair is never proposed again.",
      actions: [statusSeg] }), h("p", null, seg));
    items = all.items.filter((p) => p.kind === state.kind);
    render();
  }

  function render() {
    clear(listEl);
    if (!items.length) {
      listEl.appendChild(h("li", null, emptyState(`No ${state.status} ${KIND_LABEL[state.kind].toLowerCase()}.`,
        state.status === "pending" ? " The next sleep cycle may stage more." : "")));
      return;
    }
    state.cursor = Math.min(state.cursor, items.length - 1);
    items.forEach((p, i) => listEl.appendChild(card(p, i === state.cursor)));
  }

  function decide(p, el, accept, reason = "") {
    return api.decide(p.id, accept, reason).then((out) => {
      toast(out.local ? (accept ? "Accepted in this page; the local app writes the link" : "Rejected in this page")
        : accept ? (out.path ? `Wrote the link into ${out.path}` : "Accepted") : "Rejected; it will not be proposed again");
      el?.classList.add("leaving");
      setTimeout(() => { items = items.filter((x) => x.id !== p.id); render(); ctx.refreshStatus(); }, 260);
    }).catch((e) => toast(e.message, true));
  }

  function card(p, current) {
    const pl = p.payload;
    const reasonInput = h("input", { class: "text", type: "text", placeholder: "Reason (optional)", "aria-label": "Rejection reason" });
    let el;
    const acts = p.status === "pending" ? h("div", { class: "acts" },
      h("button", { class: "btn primary", type: "button", onclick: () => decide(p, el, true) },
        p.kind === "link" ? "Accept and add link" : p.kind === "note" ? "Accept and write note" : "Acknowledge"),
      h("button", { class: "btn", type: "button", onclick: () => decide(p, el, false, reasonInput.value) }, p.kind === "gap" ? "Dismiss" : "Reject"),
      reasonInput) :
      p.status === "applied" && p.kind === "link" ? h("div", { class: "acts" },
        h("button", { class: "btn", type: "button", onclick: () => api.revert(p.id).then(() => { toast("Reverted"); load(); })
          .catch((e) => toast(e.message, true)) }, "Revert")) : null;
    let body;
    if (p.kind === "link") {
      body = [
        h("div", { class: "head" },
          h("div", { class: "pair" },
            h("button", { type: "button", onclick: () => ctx.openNote(pl.src_path) }, pl.src_title), axonArrow(),
            h("button", { type: "button", onclick: () => ctx.openNote(pl.dst_path) }, pl.dst_title)),
          gauge(pl.probability ?? p.score)),
        synapseDiagram(pl),
        pl.why ? h("div", { class: "why" }, pl.why) : null,
        h("div", { class: "meta" },
          pl.utility !== undefined ? h("span", { class: "chip", title: "Probability divided by how ordinary the target's popularity is" }, `utility ${fmt.n(pl.utility, 2)}`) : null,
          h("span", { class: "chip myelin" }, p.method === "gae" ? "graph autoencoder" : p.method),
          pl.evidence?.cosine !== undefined ? h("span", { class: "chip" }, `cosine ${fmt.n(pl.evidence.cosine, 2)}`) : null),
        pl.review ? h("div", { class: `review ${pl.review.verdict}` }, h("strong", null, `Claude review: ${pl.review.verdict}. `), pl.review.reason) : null,
      ];
    } else if (p.kind === "note") {
      body = [h("h3", null, pl.title), pl.why ? h("div", { class: "why" }, pl.why) : null,
        h("div", { class: "meta" }, pl.quality ? h("span", { class: `chip ${pl.quality.score < 15 ? "ok" : "warn"}` },
          `authenticity ${pl.quality.score} · ${pl.quality.verdict}`) : null, h("span", { class: "chip" }, p.method)),
        h("pre", { class: "draft" }, pl.content)];
    } else if (p.kind === "merge") {
      body = [h("div", { class: "pair" },
        h("button", { type: "button", onclick: () => ctx.openNote(pl.a_path) }, pl.a_title), h("span", { class: "chip nissl" }, `≈ ${fmt.n(pl.similarity, 2)}`),
        h("button", { type: "button", onclick: () => ctx.openNote(pl.b_path) }, pl.b_title)),
        h("div", { class: "why" }, "Near-duplicate content. Merging is a judgement call, so Myelin only flags it.")];
    } else {
      body = [h("h3", null, `“${pl.example}”`), h("div", { class: "why" },
        `Asked ${pl.count}× with best similarity ${fmt.n(pl.best_similarity)}; the vault has nothing close.`)];
    }
    el = h("li", { class: p.status === "pending" ? "card" : "card decided", "aria-current": String(current), dataset: { id: p.id } }, body, acts);
    return el;
  }

  page.append(head, listEl, h("div", { class: "hints" },
    h("span", null, h("kbd", null, "j"), " ", h("kbd", null, "k"), " move"), h("span", null, h("kbd", null, "a"), " accept"), h("span", null, h("kbd", null, "r"), " reject")));
  page.prepend(loader("Loading proposals"));
  await load().catch((e) => page.append(h("p", { class: "err" }, e.message)));
  page.querySelector(".loader")?.remove();

  ctx.keys = (e) => {
    const cards = [...listEl.querySelectorAll(".card")];
    if (!cards.length) return false;
    if (e.key === "j" || e.key === "k") {
      state.cursor = Math.max(0, Math.min(items.length - 1, state.cursor + (e.key === "j" ? 1 : -1)));
      render();
      listEl.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      return true;
    }
    const p = items[state.cursor];
    if (p && p.status === "pending" && (e.key === "a" || e.key === "r")) {
      decide(p, listEl.querySelector('[aria-current="true"]'), e.key === "a");
      return true;
    }
    return false;
  };
}

// ── Plate 05 · Health ───────────────────────────────────────────────────
export async function viewHealth(root, ctx) {
  const page = h("div", { class: "page" });
  root.appendChild(page);
  page.append(loader("Reading the self-model"));
  const m = await api.selfmodel();
  clear(page);
  if (m.missing) { page.append(plate({ no: "05", name: "Health", title: "No self-model yet" }), emptyState("Run a sleep cycle to build one.", "")); return; }
  const hl = m.health;
  const noteBtn = (title) => h("button", { class: "btn ghost", type: "button", onclick: () => ctx.openNote(title) }, title);
  const maxRef = Math.max(...hl.broken_targets.map((b) => b.count), 1);
  const topics = m.identity.topics;
  const totalTopic = topics.reduce((s, t) => s + t.size, 0) || 1;
  const compo = h("div", { class: "compo", role: "img", "aria-label": topics.map((t) => `${t.label}: ${t.size} notes`).join("; ") },
    topics.map((t, i) => { const s = h("span", { class: `tn${i % 6}`, title: `${t.label} · ${t.size} notes` }, t.size); s.style.flex = `${t.size} 1 0`; return s; }));
  page.append(
    plate({ no: "05", name: "Health", meta: [`measured ${fmt.time(m.generated_at)}`, m.identity.encoder], title: "Structure and integrity",
      lede: "Counted under the vault's scope rule, with code spans removed before links are parsed." }),
    h("div", { class: "readings" },
      reading("Unresolved links", `${hl.broken_links}`, `of ${hl.links} links`),
      reading("Without authored links", `${hl.orphan_count}`, `of ${m.identity.notes} notes`),
      reading("Duplicate stems", `${Object.keys(hl.duplicate_stems).length}`, "ambiguous wikilinks"),
      reading("Ledger", m.ledger.ok ? "intact" : "broken", m.ledger.message)),
    h("div", { class: "grid-2" },
      h("section", { class: "panel" }, h("h2", { class: "sect" }, "Missing link targets", h("span", { class: "aside" }, "refs")),
        hl.broken_targets.length ? hl.broken_targets.map((b) => {
          const fill = h("span", { class: "fill" }); fill.style.width = `${(100 * b.count) / maxRef}%`;
          return h("div", { class: "hbar" }, h("span", { class: "mono" }, b.target), h("span", { class: "track" }, fill), h("span", { class: "v" }, b.count), h("span"));
        }) : h("p", { class: "muted" }, "Every link resolves.")),
      h("section", { class: "panel" }, h("h2", { class: "sect" }, "Topics", h("span", { class: "aside" }, `${topics.length} communities · Louvain`)),
        compo,
        h("table", { class: "data" }, h("thead", null, h("tr", null, h("th", null, "Topic"), h("th", { class: "num" }, "Notes"), h("th", null, "Core notes"))),
          h("tbody", null, topics.map((t, ti) => h("tr", null,
            h("td", { class: "mono" }, h("span", { class: "topic" }, h("i", { class: `sw tn${ti % 6}`, "aria-hidden": "true" }), h("span", null, t.label))),
            h("td", { class: "num mono" }, t.size),
            h("td", { class: "core" }, t.core.map((c, i) => h("button", { class: "wl-inline", type: "button", onclick: () => ctx.openNote(t.core_stems[i]) }, c))))))))),
    h("section", { class: "panel" }, h("h2", { class: "sect" }, "Asked and not answered"),
      m.gaps.length ? h("table", { class: "data" }, h("thead", null, h("tr", null, h("th", null, "Question"), h("th", { class: "num" }, "Times"), h("th", { class: "num" }, "Best sim."))),
        h("tbody", null, m.gaps.map((g) => h("tr", null, h("td", null, g.query), h("td", { class: "num mono" }, g.count), h("td", { class: "num mono" }, fmt.n(g.best_similarity))))))
        : h("p", { class: "muted" }, "No question has come back empty-handed yet.")),
    hl.orphans.length ? h("section", { class: "panel" }, h("h2", { class: "sect" }, "Notes without authored links"), h("div", null, hl.orphans.map(noteBtn))) : null);
  page.classList.add("stack");
}

function reading(k, v, s) {
  return h("div", { class: "reading" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), h("div", { class: "s" }, s));
}

// ── Plate 06 · Ledger ───────────────────────────────────────────────────
const LEDGER_KINDS = [["", "All"], ["EVOLUTION", "Evolution"], ["MODEL_TRAINED", "Models"], ["SLEEP", "Sleep"], ["INGEST", "Ingest"], ["PROPOSAL", "Decisions"]];

export async function viewLedger(root, ctx) {
  const page = h("div", { class: "page" });
  root.appendChild(page);
  page.append(loader("Verifying the chain"));
  const data = await api.ledger(400);
  clear(page);
  const v = data.verified;
  let filter = "";
  const tbody = h("tbody");
  const summarise = (ev) => {
    const p = ev.payload;
    switch (ev.kind) {
      case "EVOLUTION": return `generation ${p.generation}: ${p.verdict}`;
      case "MODEL_TRAINED": return p.chosen ? `chose ${p.chosen === "gae" ? "graph autoencoder" : p.chosen}; test AUC ${Object.entries(p.scores || {}).map(([k, s]) => `${k} ${fmt.n(s.test_auc)}`).join(", ")}`
        : `${ev.subject} v${p.version ?? ""} · sha256 ${p.sha256 ? p.sha256.slice(0, 12) : ""}`;
      case "INGEST": return `+${p.added} ~${p.updated} −${p.removed}${p.refit ? " · semantic space refit" : ""} (${p.notes} notes)`;
      case "SLEEP": return `${p.seconds}s · ${Object.entries(p.steps).filter(([, s]) => !s.ok).map(([k]) => k).join(", ") || "all 8 steps ok"}`;
      case "CONFIG_SEEDED": return `champion v${p.version} · train MRR ${fmt.n(p.fitness?.train?.mrr)} · holdout MRR ${fmt.n(p.fitness?.holdout?.mrr)}`;
      case "CONFIG_ROLLBACK": return `v${p.from} → v${p.to}`;
      case "PROPOSAL_ACCEPTED": case "PROPOSAL_REJECTED":
        return `${p.kind} #${p.id} by ${p.actor}${p.reason ? `: ${p.reason}` : ""}${p.after_sha256 ? ` · ${p.after_sha256.slice(0, 12)}` : ""}`;
      case "PROPOSAL_REVERTED": return `#${p.id} by ${p.actor}`;
      case "AGENT_REVIEW": return `${p.count} reviewed by ${p.model} · $${p.cost_usd}`;
      default: return JSON.stringify(p).slice(0, 200);
    }
  };
  const row = (ev, fresh = false) => h("tr", { class: fresh ? "fresh" : null, dataset: { kind: ev.kind } },
    h("td", { class: "link", "aria-hidden": "true" }),
    h("td", { class: "num mono" }, ev.sequence), h("td", { class: "mono" }, fmt.time(ev.created_at)),
    h("td", { class: "mono" }, ev.kind), h("td", { class: "mono" }, ev.subject.length > 40 ? ev.subject.slice(0, 38) + "…" : ev.subject),
    h("td", { class: "payload" }, summarise(ev)), h("td", { class: "mono muted" }, ev.hash));
  const events = [...data.events].reverse();
  const render = () => {
    clear(tbody).append(...events.filter((ev) => !filter || ev.kind.startsWith(filter)).map((ev) => row(ev)));
  };
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Event kind" }, LEDGER_KINDS.map(([k, label]) =>
    h("button", { type: "button", "aria-pressed": String(k === filter), onclick: (e) => {
      filter = k;
      seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
      e.currentTarget.setAttribute("aria-pressed", "true");
      render();
    } }, label)));
  page.append(
    plate({ no: "06", name: "Ledger", meta: [`${v.events} events`], title: "Every durable change, hash-chained",
      lede: "Each event's SHA-256 covers the one before it, and the head is checkpointed outside the database, so editing or truncating history is detectable." }),
    h("div", { class: `verify ${v.ok ? "ok" : "bad"}` }, icon(v.ok ? "check" : "cross"), v.ok ? `Chain verified · ${v.message}` : `Verification failed · ${v.message}`,
      events[0] ? h("span", { class: "head" }, `head ${events[0].hash}`) : null),
    h("p", null, seg),
    h("table", { class: "data chain" }, h("thead", null, h("tr", null, ["", "#", "When", "Kind", "Subject", "Detail", "Hash"].map((c, i) =>
      h("th", { class: i === 1 ? "num" : null }, c)))), tbody));
  render();
  ctx.onLedger = (ev) => { events.unshift(ev); if (!filter || ev.kind.startsWith(filter)) tbody.insertBefore(row(ev, true), tbody.firstChild); };
}
