// Plate 04 — Evolution: the evidence behind every claim the brain makes about itself.
import { MODE, api } from "./api.js";
import { loader } from "./icons.js";
import { clear, fmt, h, plate, toast } from "./util.js";

const GENE_NOTE = {
  w_dense: "meaning (LSA cosine)", w_sparse: "words (BM25)", w_activation: "graph spread", w_rerank: "learned reranker",
  seed_k: "seeds that spread", steps: "hops", decay: "activation kept per hop", fan_exponent: "hub dilution",
  k_explicit: "authored links", k_related: "frontmatter links", k_semantic: "semantic links", k_hebbian: "learned links",
  k_backlink: "backlinks",
};
const STEPS = [
  ["ingest", "Ingest", "parse changed notes; refit the semantic space on drift"],
  ["hebbian", "Hebbian", "fold co-used notes into learned links; prune decayed ones"],
  ["structure", "Structure", "PageRank and Louvain topics"],
  ["link_prediction", "Link prediction", "train the graph autoencoder; compare to baselines; stage proposals"],
  ["reranker", "Reranker", "train on feedback and quiz preference pairs"],
  ["evolution", "Evolution", "mutate the recall genome; promote only on a holdout gain"],
  ["review", "Review", "stage near-duplicates and repeated unanswered questions"],
  ["self_model", "Self-model", "re-measure everything; publish"],
];
const LESION = {
  full: "Full system", dense_only: "Meaning only", sparse_only: "Words only", no_graph: "Without the graph", no_learned_edges: "Without learned links",
};

function pathLength(points) {
  let L = 0;
  for (let i = 1; i < points.length; i++) L += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  return Math.ceil(L) + 2;
}

function accuracyChart(hold, train, { randomLevel, ci }) {
  const W = 680, H = 232, L = 42, R = 104, T = 14, B = 28;
  const all = [...hold, ...train].map((p) => p[1]).concat(randomLevel ? [randomLevel] : [], ci || []);
  const n = Math.max(hold.length, train.length, 2);
  // Round axis: ticks land on 0.05 or 0.1 steps rather than wherever the data ends.
  const span = Math.max(1e-6, Math.max(...all) - Math.min(...all));
  const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25].find((s) => span / s <= 5) || 0.25;
  const lo = Math.max(0, Math.floor((Math.min(...all) - step * 0.2) / step) * step);
  const hi = Math.min(1, Math.ceil((Math.max(...all) + step * 0.2) / step) * step);
  const ticks = Math.round((hi - lo) / step);
  const right = [];   // end-of-line labels, de-collided before drawing
  const x = (i) => L + ((W - L - R) * i) / Math.max(1, n - 1);
  const y = (v) => T + (H - T - B) * (1 - (v - lo) / Math.max(1e-6, hi - lo));
  const svg = h("svg:svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Holdout MRR over ${hold.length} generations, from ${fmt.n(hold[0]?.[1])} to ${fmt.n(hold.at(-1)?.[1])}` });
  const grid = h("svg:g", { class: "grid" });
  for (let k = 0; k <= ticks; k++) {
    const v = lo + step * k;
    grid.append(h("svg:line", { x1: L, x2: W - R, y1: y(v), y2: y(v) }), h("svg:text", { x: L - 8, y: y(v) + 3, "text-anchor": "end" }, v.toFixed(2)));
  }
  svg.append(grid, h("svg:text", { x: L, y: H - 6 }, "generation 1"), h("svg:text", { x: W - R, y: H - 6, "text-anchor": "end" }, `generation ${n}`));
  if (ci && hold.length) {
    const xe = x(hold.length - 1);
    svg.append(h("svg:rect", { class: "ci", x: xe - 5, y: y(ci[1]), width: 10, height: Math.max(1, y(ci[0]) - y(ci[1])), rx: 2 }),
);
    right.push({ y: y(ci[1]) + 4, text: `95% CI ${ci[0].toFixed(2)}–${ci[1].toFixed(2)}` });
  }
  if (randomLevel) svg.append(h("svg:line", { class: "s-random", x1: L, x2: W - R, y1: y(randomLevel), y2: y(randomLevel) }),
    h("svg:text", { x: W - R, y: y(randomLevel) - 5, "text-anchor": "end" }, `random ranking ${randomLevel.toFixed(3)}`));
  const line = (pts, cls) => {
    const xy = pts.map((p, i) => [x(i), y(p[1])]);
    const d = xy.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join("");
    const path = h("svg:path", { class: `${cls} draw`, d });
    const len = pathLength(xy);
    path.style.strokeDasharray = cls === "s-train" ? "3 3" : `${len}`;
    if (cls !== "s-train") path.style.setProperty("--len", `${len}`);
    else path.classList.remove("draw");
    return [path, xy];
  };
  if (train.length) svg.append(line(train, "s-train")[0]);
  if (hold.length) {
    const [path, xy] = line(hold, "s-hold");
    const area = `M${xy[0][0]},${y(lo)} ` + xy.map(([px, py]) => `L${px.toFixed(1)},${py.toFixed(1)}`).join(" ") + ` L${xy.at(-1)[0]},${y(lo)} Z`;
    svg.append(h("svg:path", { class: "area", d: area }), path);
    let lastTag = -Infinity;
    hold.forEach((p, i) => {
      const rose = i && p[1] > hold[i - 1][1] + 1e-9;
      const dot = h("svg:circle", { class: rose ? "pt promo" : "pt", cx: x(i), cy: y(p[1]), r: rose ? 4 : 2.5 });
      dot.append(h("svg:title", null, `generation ${i + 1}: holdout MRR ${p[1].toFixed(3)}${rose ? " (rose)" : ""}`));
      svg.append(dot);
      if (rose && x(i) - lastTag > 40) {   // close promotions keep their dots; only the first is labelled
        svg.append(h("svg:text", { class: "tag", x: x(i), y: y(p[1]) - 10, "text-anchor": "middle" }, `+${(p[1] - hold[i - 1][1]).toFixed(3)}`));
        lastTag = x(i);
      }
    });
    right.push({ y: y(hold.at(-1)[1]) + 4, text: `holdout ${hold.at(-1)[1].toFixed(3)}`, cls: "tag" });
  }
  if (train.length) right.push({ y: y(train.at(-1)[1]) + 4, text: `train ${train.at(-1)[1].toFixed(3)}` });
  right.sort((a, b) => a.y - b.y);
  for (let i = 1; i < right.length; i++) right[i].y = Math.max(right[i].y, right[i - 1].y + 13);
  for (const r of right) svg.append(h("svg:text", { x: W - R + 14, y: r.y, class: r.cls || null }, r.text));
  return svg;
}

function genome(params, defaults, bounds) {
  return h("div", { class: "genome" }, Object.entries(params).map(([k, v]) => {
    const [lo, hi, integer] = bounds[k] || [0, 1, false];
    const pos = (val) => `${(100 * (val - lo)) / Math.max(1e-9, hi - lo)}%`;
    const d = defaults[k];
    const moved = Math.abs(v - d) > 1e-6;
    const def = h("span", { class: "def", title: `default ${d}` }); def.style.left = pos(d);
    const cur = h("span", { class: "cur", title: `now ${v}` }); cur.style.left = pos(v);
    const span = h("span", { class: "span" });
    span.style.left = pos(Math.min(v, d)); span.style.width = `${(100 * Math.abs(v - d)) / Math.max(1e-9, hi - lo)}%`;
    const show = (x) => (integer ? String(Math.round(x)) : x.toFixed(2));
    return h("div", { class: moved ? "gene moved" : "gene" },
      h("div", { class: "nm" }, k), h("div", { class: "ds" }, GENE_NOTE[k] || ""),
      h("div", { class: "val" }, show(v), h("small", null, moved ? `default ${show(d)}` : "default")),
      h("div", { class: "trk", role: "img", "aria-label": `${k} ${show(v)} in range ${lo} to ${hi}; default ${show(d)}` }, span, def, cur));
  }));
}

function lesions(ab) {
  const full = ab.full;
  const max = Math.max(...Object.values(ab), 1e-6);
  return Object.entries(ab).map(([k, v]) => {
    const fill = h("span", { class: "fill" }); fill.style.width = `${(100 * v) / max}%`;
    const ref = h("span", { class: "ref" }); ref.style.left = `${(100 * full) / max}%`;
    const delta = v - full;
    const cls = k === "full" ? "hbar full" : delta > 0.002 ? "hbar costly" : "hbar";
    return h("div", { class: cls }, h("span", null, LESION[k] || k), h("span", { class: "track" }, fill, ref), h("span", { class: "v" }, fmt.n(v)),
      h("span", { class: `d ${delta > 0.002 ? "pos" : delta < -0.002 ? "neg" : ""}` }, k === "full" ? "" : `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`));
  });
}

function hypnogram(steps, { live = false } = {}) {
  const W = 540, H = 214, L = 112, R = 58, T = 10, rowH = 24;
  const total = STEPS.reduce((s, [k]) => s + (steps[k]?.seconds ?? 0), 0) || 1;
  const svg = h("svg:svg", { class: "hypno", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": STEPS.filter(([k]) => steps[k]).map(([k, n]) => `${n} ${steps[k].seconds}s`).join(", ") });
  let t = 0, prev = null;
  STEPS.forEach(([k, name, desc], i) => {
    const yy = T + i * rowH + rowH / 2;
    svg.append(h("svg:line", { class: "rail-l", x1: L, x2: W - R, y1: yy, y2: yy }),
      h("svg:text", { class: "stage", x: L - 12, y: yy + 3, "text-anchor": "end" }, name));
    const s = steps[k];
    if (!s) return;
    const x0 = L + ((W - L - R) * t) / total;
    const width = Math.max(4, ((W - L - R) * s.seconds) / total);
    if (prev) svg.append(h("svg:line", { class: "step", x1: x0, x2: x0, y1: prev, y2: yy }));
    const seg = h("svg:line", { class: s.ok ? "seg" : "seg bad", x1: x0 + 2, x2: x0 + width - 2, y1: yy, y2: yy });
    seg.append(h("svg:title", null, `${name}: ${desc} · ${s.seconds}s${s.error ? " · " + s.error : ""}`));
    svg.append(seg, h("svg:text", { class: "dur", x: W - R + 10, y: yy + 3 }, `${s.seconds.toFixed(2)}s`));
    t += s.seconds;
    prev = yy;
  });
  const yb = T + STEPS.length * rowH + 10;
  svg.append(h("svg:text", { x: L, y: yb }, "0s"), h("svg:text", { x: W - R, y: yb, "text-anchor": "end" }, `${total.toFixed(1)}s${live ? " so far" : ""}`));
  return svg;
}

export async function viewEvolution(root, ctx) {
  const page = h("div", { class: "page stack" });
  root.appendChild(page);
  page.append(loader("Reading the evolution record"));
  const [m, series, evo] = await Promise.all([api.selfmodel(), api.metrics(["quiz.mrr.holdout", "quiz.mrr.train"]), api.evolution()]);
  clear(page);

  const live = {};
  const hypnoBox = h("div");
  const hypnoMeta = h("span", { class: "aside" });
  const drawHypno = (steps, isLive) => { clear(hypnoBox).append(hypnogram(steps, { live: isLive })); };
  const sleepBtn = h("button", { class: "btn primary", type: "button", onclick: async () => {
    try {
      const r = await api.sleep();
      if (!r.started) { toast(r.reason); return; }
      sleepBtn.disabled = true;
      for (const k of Object.keys(live)) delete live[k];
      drawHypno(live, true);
      hypnoMeta.textContent = r.replay ? "replaying the recorded cycle" : "running now";
    } catch (e) { toast(e.message, true); }
  } }, MODE === "snapshot" ? "Replay sleep cycle" : "Run sleep cycle");
  ctx.onSleep = (kind, data) => {
    if (kind === "sleep_started") { sleepBtn.disabled = true; }
    if (kind === "sleep_step") { live[data.step] = { ok: data.ok, seconds: data.seconds, error: data.error }; drawHypno(live, true); }
    if (kind === "sleep_finished") {
      sleepBtn.disabled = false;
      hypnoMeta.textContent = `finished in ${Number(data.seconds).toFixed(1)}s`;
      toast(data.replay ? "Replayed the recorded sleep cycle" : `Sleep cycle finished in ${data.seconds}s`);
      if (!data.replay) setTimeout(() => ctx.rerender(), 600);
    }
  };
  if (ctx.status?.sleeping) sleepBtn.disabled = true;

  if (m.missing) {
    page.append(plate({ no: "04", name: "Evolution", title: "No measurements yet", actions: [sleepBtn],
      lede: "The first sleep cycle builds the self-quiz, trains the models and seeds the champion configuration." }),
      h("section", { class: "panel" }, h("h2", { class: "sect" }, "Sleep cycle", hypnoMeta), hypnoBox));
    drawHypno({}, false);
    return;
  }
  const q = m.reliability.quiz;
  const ev = m.evolution;
  const lp = m.reliability.link_prediction;
  const rr = m.reliability.reranker;
  const hold = series["quiz.mrr.holdout"] || [];
  const train = series["quiz.mrr.train"] || [];
  const gae = lp.scores?.gae;

  page.append(
    plate({ no: "04", name: "Evolution", meta: [`${ev.generations} generations`, `${ev.promotions} promotions`, `holdout consulted ${ev.holdout_uses}×`],
      title: "It quizzes itself, and keeps only what wins",
      lede: `${q.items} cloze items built from your own links: mask one, find its target from the surrounding paragraph. ${q.holdout_items} are held out; a change is promoted only if it wins on those.`,
      actions: [sleepBtn] }),
    h("div", { class: "readings" },
      reading("Holdout MRR@10", fmt.n(q.holdout?.mrr), q.holdout ? `95% CI ${fmt.n(q.holdout.mrr_ci[0], 2)}–${fmt.n(q.holdout.mrr_ci[1], 2)}` : ""),
      reading("Recall@10", fmt.n(q.holdout?.r10, 2), `random MRR ${fmt.n(q.random_mrr)}`),
      reading("Link prediction AUC", gae ? fmt.n(gae.test_auc) : "—", lp.chosen ? `chosen: ${lp.chosen === "gae" ? "graph autoencoder" : lp.chosen}` : lp.status || ""),
      reading("Champion", `v${ev.champion_version}`, `σ ${ev.sigma.toFixed(3)} mutation scale`)),
    h("div", { class: "grid-3" },
      h("section", { class: "panel" }, h("h2", { class: "sect" }, "Recall accuracy by generation", h("span", { class: "aside" }, "MRR@10")),
        accuracyChart(hold, train, { randomLevel: q.random_mrr, ci: q.holdout?.mrr_ci }),
        h("p", { class: "note" }, "Solid: held-out items. Dashed: training items. Filled points mark generations where the held-out score rose.",
          q.holdout && q.holdout.n < 30 ? ` With ${q.holdout.n} held-out items, differences smaller than the interval are noise.` : "")),
      h("section", { class: "panel" }, h("h2", { class: "sect" }, "Lesion study", h("span", { class: "aside" }, "MRR, one signal removed")),
        q.ablations ? lesions(q.ablations) : h("p", { class: "muted" }, "No quiz yet."),
        h("p", { class: "note" }, "The line marks the full system. A bar that reaches past it means that signal is currently costing accuracy."))),
    h("section", { class: "panel" }, h("h2", { class: "sect" }, `Recall genome · champion v${ev.champion_version}`, h("span", { class: "aside" }, "dot: now · tick: default · range: allowed")),
      genome(ev.params, evo.defaults || {}, evo.bounds || {})),
    h("div", { class: "grid-2" },
      h("section", { class: "panel" }, h("h2", { class: "sect" }, "Sleep cycle", hypnoMeta), hypnoBox,
        h("p", { class: "note" }, "Each stage of consolidation on its own row, drawn at its measured duration.")),
      h("section", { class: "panel" }, h("h2", { class: "sect" }, "Link prediction", h("span", { class: "aside" }, `${lp.edges ?? 0} links · ${lp.nodes ?? 0} notes`)),
        lp.status === "trained" ? h("table", { class: "data" },
          h("thead", null, h("tr", null, h("th", null, "Method"), h("th", { class: "num" }, "val AUC"), h("th", { class: "num" }, "test AUC"), h("th", { class: "num", title: "Held-out links against non-links that share a neighbour" }, "2-hop AUC"), h("th", { class: "num" }, "test AP"))),
          h("tbody", null, Object.entries(lp.scores).map(([k, s]) => h("tr", null,
            h("td", null, k === lp.chosen ? h("span", { class: "chip myelin" }, k === "gae" ? "graph autoencoder" : k.replace("_", "–")) : (k === "gae" ? "graph autoencoder" : k.replace("_", "–"))),
            h("td", { class: "num mono" }, fmt.n(s.val_auc)), h("td", { class: "num mono" }, fmt.n(s.test_auc)), h("td", { class: "num mono" }, fmt.n(s.test_auc_2hop)), h("td", { class: "num mono" }, fmt.n(s.test_ap))))))
          : h("p", { class: "muted" }, `Not trained: ${lp.status || "no data"}.`),
        h("p", { class: "note" }, "A quarter of the links are hidden before training; the predictor with the best validation score is used. 2-hop AUC tests the same hidden links against non-links that share a neighbour, the case a suggestion has to get right."),
        h("h2", { class: "sect" }, "Learned reranker"),
        rr.status === "trained" ? h("dl", { class: "kv" },
          h("dt", null, "Training pairs"), h("dd", null, `${rr.pairs} (${rr.sources?.quiz ?? 0} quiz, ${rr.sources?.feedback ?? 0} feedback)`),
          h("dt", null, "Pairs ordered right"), h("dd", null, `${fmt.n(rr.val_pairwise_acc)} on validation`), h("dt", null, "Blend weight"), h("dd", null, fmt.n(ev.params.w_rerank)))
          : h("p", { class: "muted" }, `Not trained: ${rr.status}. It needs 12 preference pairs.`))),
    h("section", { class: "panel" }, h("h2", { class: "sect" }, "Recent generations"),
      h("table", { class: "data" }, h("thead", null, h("tr", null, h("th", { class: "num" }, "Config"), h("th", null, "Status"), h("th", { class: "num" }, "Train MRR"), h("th", { class: "num" }, "Holdout MRR"), h("th", null, "When"))),
        h("tbody", null, evo.configs.slice(0, 14).map((c) => h("tr", null, h("td", { class: "num mono" }, `v${c.version}`),
          h("td", null, h("span", { class: `chip ${c.status === "champion" ? "myelin" : c.status === "retired" ? "nissl" : ""}` }, c.status)),
          h("td", { class: "num mono" }, fmt.n(c.fitness?.train?.mrr)), h("td", { class: "num mono" }, fmt.n(c.fitness?.holdout?.mrr)),
          h("td", { class: "mono muted" }, fmt.ago(c.created_at))))))));
  const last = evo.last_sleep;
  if (last) { drawHypno(last.steps, false); hypnoMeta.textContent = `last run ${last.seconds}s`; }
  else drawHypno({}, false);
}

function reading(k, v, s) {
  return h("div", { class: "reading" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), h("div", { class: "s" }, s));
}
