// A guided tour of this brain. Every number in the captions is read from the
// brain at the moment the tour starts; nothing here is canned copy about a
// vault that is not this one. `?tour=auto` advances on its own (used to record
// the demo); otherwise ← and → step, Esc leaves.
import { MODE, api } from "./api.js";
import { fmt, h, reducedMotion, toast } from "./util.js";

const DEFAULT_QUERY = "why did the daemon stop running";
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export async function startTour(ctx, { auto = false } = {}) {
  document.querySelector(".tour")?.remove();
  const [status, m, pending] = await Promise.all([api.status(), api.selfmodel(), api.proposals("pending", "link")]);
  const presets = await api.presets();
  const query = presets.find((p) => /daemon/i.test(p)) || presets[0] || DEFAULT_QUERY;
  const q = m.reliability?.quiz || {};
  const lp = m.reliability?.link_prediction || {};
  const edges = status.edges || {};
  const authored = (edges.explicit || 0) + (edges.related || 0);
  const topics = m.identity?.topics?.length ?? 0;
  const gae = lp.scores?.gae;

  const steps = [
    { view: "#atlas", title: "Your vault, drawn as tissue", ms: 9000,
      text: `${status.notes} notes and ${authored} authored links. Each note is a stained cell body sized by PageRank; the faint regions are ${topics} topics found by Louvain community detection. Hover a note for its topic and links.` },
    { view: "#atlas", title: "Ask, and watch activation spread", ms: 11000,
      text: `“${query}”. The closest text matches light up first, then activation travels along links to related notes, hopping between sheath gaps the way an impulse crosses a myelinated axon.`,
      run: async () => { await wait(reducedMotion() ? 200 : 1600); await ctx.askInAtlas(query); } },
    { view: `#recall/${encodeURIComponent(query)}`, title: "Every score explains itself", ms: 10000,
      text: "The same recall as a list. Each score splits into meaning, words, graph and learned parts, and a graph hit names the link that led to it. Marking a result useful trains the reranker." },
    { view: null, title: "Open any note", ms: 9000,
      text: "Its neighbourhood (outgoing links on the right, backlinks on the left), the links Myelin suggests for it, and the note itself.",
      run: async () => { const top = ctx.lastRecall?.results?.[0]; if (top) await ctx.openNote(top.note_id); } },
    { view: "#inbox", title: "Nothing is edited behind your back", ms: 10000,
      text: `${pending.items.length} link suggestions wait for a decision. Each shows the shared neighbours that justify it and the probability from the ${lp.chosen === "gae" ? "graph autoencoder" : "chosen predictor"}${gae ? `, which scores ${fmt.n(gae.test_auc, 2)} AUC on links it never saw` : ""}.`,
      run: async () => ctx.closeNote() },
    { view: "#evolution", title: "It measures itself", ms: 11000,
      text: q.holdout ? `${q.items} quiz items built from your own links. Held-out MRR is ${fmt.n(q.holdout.mrr)} (95% CI ${fmt.n(q.holdout.mrr_ci[0], 2)}–${fmt.n(q.holdout.mrr_ci[1], 2)}) against ${fmt.n(q.random_mrr)} for random ranking. A settings change is kept only if it wins on items it was not tuned on.`
        : "Run a sleep cycle to build the self-quiz." },
    { view: null, title: MODE === "snapshot" ? "Sleep: the last cycle, replayed" : "Sleep: consolidation, live", ms: 14000,
      text: MODE === "snapshot" ? "Replaying the last recorded cycle at its real step timings: ingest, learned links, topics, link prediction, reranker, evolution, review, self-model."
        : "A real cycle, running now: ingest, learned links, topics, link prediction, reranker, evolution, review, self-model. Each stage is drawn at its measured duration.",
      run: async () => { document.querySelector(".plate .actions .btn.primary")?.scrollIntoView({ block: "center" }); await wait(400);
        document.querySelector(".plate .actions .btn.primary")?.click(); await wait(600);
        document.querySelector(".hypno")?.closest(".panel")?.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" }); } },
    { view: "#ledger", title: "And every change is on the record", ms: 9000,
      // Read when the step opens: the sleep step above has just added events.
      text: async () => { const l = (await api.status()).ledger;
        const when = MODE === "snapshot" ? "verified when this snapshot was taken" : "and the chain re-verified just now";
        return `${l.events} events, each hash covering the one before it, ${when}: ${l.ok ? "intact" : "broken"}.`; } },
  ];

  let i = -1;
  let timer = null;
  const card = h("div", { class: "tour", role: "dialog", "aria-label": "Guided tour" });
  document.body.appendChild(card);
  const root = document.documentElement;
  root.classList.add("touring");
  const dock = () => root.style.setProperty("--overlay-bottom", `${card.offsetHeight + 28}px`);
  const close = () => {
    clearTimeout(timer); card.remove(); document.removeEventListener("keydown", keys);
    root.classList.remove("touring"); root.style.removeProperty("--overlay-bottom");
  };
  const go = async (j) => {
    if (j < 0 || j >= steps.length) { close(); if (j >= steps.length) toast("Tour finished. Ask the vault anything."); return; }
    clearTimeout(timer);
    i = j;
    const s = steps[i];
    const text = typeof s.text === "function" ? await s.text().catch((e) => e.message) : s.text;
    if (i !== j) return;   // a quicker key press moved on while the text loaded
    const prog = h("i");
    prog.style.width = `${(100 * (i + 1)) / steps.length}%`;
    card.replaceChildren(
      h("div", { class: "id" }, h("span", null, `Tour · ${i + 1} / ${steps.length}`), h("span", null, MODE === "snapshot" ? "snapshot" : "live")),
      h("h3", null, s.title), h("p", null, text),
      h("div", { class: "ctrl" },
        h("button", { class: "btn", type: "button", onclick: () => go(i - 1), disabled: i === 0 }, "Back"),
        h("span", { class: "prog" }, prog),
        h("button", { class: "btn", type: "button", onclick: close }, "Close"),
        h("button", { class: "btn next", type: "button", onclick: () => go(i + 1) }, i === steps.length - 1 ? "Finish" : "Next")));
    dock();
    try {
      if (s.view) await ctx.navigate(s.view);
      if (s.run) await s.run();
    } catch (e) { toast(e.message, true); }
    if (auto) timer = setTimeout(() => go(i + 1), s.ms);
  };
  const keys = (e) => {
    if (e.target.closest("input, textarea")) return;
    if (e.key === "ArrowRight") go(i + 1);
    else if (e.key === "ArrowLeft") go(i - 1);
    else if (e.key === "Escape") close();
  };
  document.addEventListener("keydown", keys);
  await go(0);
}
