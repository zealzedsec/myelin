// Plate 01 — the Atlas.
//
// The graph drawn as tissue. Notes are stained somata (cresyl violet) with a
// nucleus once they are large enough on screen; ordinary links are bare axons;
// links learned from use carry a myelin sheath whose thickness is their weight.
// Topics found by Louvain are drawn as faint membranes around their members.
//
// Recall plays as saltatory conduction: from each seed note a pulse hops between
// the gaps in the sheath (the nodes of Ranvier) to the note it activated, which
// is how a real impulse travels a myelinated axon. Under reduced motion the
// paths are drawn complete and still.
//
// Layout: a velocity-Verlet force simulation. Springs only along authored,
// semantic and learned links; springs that cross topics are weak and hub springs
// are divided by degree; each topic is anchored to a point on a ring. Positions
// persist per note path, so the map is the same map tomorrow.
import { edgeSwatch, somaGlyph } from "./icons.js";
import { cssVar, h, reducedMotion } from "./util.js";

const KINDS = [
  ["explicit", "Authored links"],
  ["related", "Frontmatter links"],
  ["semantic", "Semantic neighbours"],
  ["hebbian", "Learned from use"],
  ["proposed", "Proposed links"],
];
const POS_KEY = "myelin.atlas.positions.v2";
const HOP_MS = 75;

function loadPositions() {
  try { return JSON.parse(localStorage.getItem(POS_KEY) || "{}"); } catch { return {}; }
}
function savePositions(nodes) {
  try {
    const out = {};
    for (const n of nodes) out[n.path] = [Math.round(n.x * 10) / 10, Math.round(n.y * 10) / 10];
    localStorage.setItem(POS_KEY, JSON.stringify(out));
  } catch { /* storage unavailable: the map simply re-settles next time */ }
}
const hash = (n) => (((n * 2654435761) >>> 0) % 10000) / 10000;

function convexHull(points) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), q) <= 0) lower.pop(); lower.push(q); }
  for (const q of [...p].reverse()) { while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

export function mountAtlas(container, { graph, onSelect, selected }) {
  const show = { explicit: true, related: false, semantic: true, hebbian: true, proposed: false };
  const counts = {};
  for (const e of graph.edges) counts[e.kind] = (counts[e.kind] || 0) + 1;

  const canvas = h("canvas", { tabindex: "0", role: "img",
    "aria-label": `Map of ${graph.nodes.length} notes. Drag to pan, scroll or plus and minus to zoom, F to fit. The Recall view lists the same results as text.` });
  const card = h("div", { class: "float card", hidden: true });
  const trace = h("div", { class: "float trace", hidden: true, "aria-live": "polite" });
  const zoomRead = h("span", null, "×1.00");
  const legend = h("div", { class: "float legend" },
    h("h2", { class: "sect" }, "Plate 01 · Atlas", h("span", { class: "aside mono" }, `${graph.nodes.length} notes`)),
    KINDS.map(([kind, label]) => h("label", null,
      h("input", { type: "checkbox", checked: show[kind], onchange: (e) => { show[kind] = e.target.checked; draw(); } }),
      edgeSwatch(kind), label, h("span", { class: "n" }, counts[kind] || 0))),
    h("div", { class: "key-nodes" }, somaGlyph(), "Notes sized by PageRank; ring marks the selection"));
  const zoom = h("div", { class: "float zoom" },
    h("button", { type: "button", "aria-label": "Zoom out", onclick: () => { touched = true; zoomBy(0.8); } }, "−"),
    h("button", { type: "button", "aria-label": "Zoom in", onclick: () => { touched = true; zoomBy(1.25); } }, "+"),
    h("button", { type: "button", "aria-label": "Fit to screen", onclick: () => fit() }, "⌗"),
    h("span", { class: "scale" }, h("i"), zoomRead));
  const wrap = h("div", { class: "atlas" }, canvas, legend, zoom, trace, card);
  container.appendChild(wrap);

  // ── model ──────────────────────────────────────────────────────────────
  const stored = loadPositions();
  const commSize = new Map();
  for (const n of graph.nodes) commSize.set(n.community, (commSize.get(n.community) || 0) + 1);
  const ringOrder = [...commSize.keys()].filter((c) => commSize.get(c) >= 3).sort((a, b) => commSize.get(b) - commSize.get(a));
  const anchors = new Map();
  const ring = 120 + 34 * Math.sqrt(graph.nodes.length);
  ringOrder.forEach((c, i) => {
    const a = (i / Math.max(1, ringOrder.length)) * Math.PI * 2 - Math.PI / 2;
    anchors.set(c, [Math.cos(a) * ring, Math.sin(a) * ring]);
  });
  const byId = new Map();
  for (const n of graph.nodes) {
    const p = stored[n.path];
    const anchor = anchors.get(n.community) || [0, 0];
    byId.set(n.id, { ...n, x: p ? p[0] : anchor[0] + (hash(n.id) - 0.5) * 140, y: p ? p[1] : anchor[1] + (hash(n.id + 7) - 0.5) * 140,
                     vx: 0, vy: 0, fixed: false });
  }
  const nodes = [...byId.values()];
  const ranks = nodes.map((n) => n.rank).sort((a, b) => a - b);
  const maxRank = Math.max(...ranks, 1e-9);
  for (const n of nodes) {
    n.r = 3 + 9 * Math.sqrt(n.rank / maxRank);
    n.pct = Math.round(100 * (1 - ranks.findIndex((r) => r >= n.rank) / Math.max(1, ranks.length)));
  }
  const edges = graph.edges.filter((e) => byId.has(e.s) && byId.has(e.t)).map((e) => ({ ...e, a: byId.get(e.s), b: byId.get(e.t) }));
  const adjacent = new Map(nodes.map((n) => [n.id, new Set()]));
  for (const e of edges) { adjacent.get(e.s).add(e.t); adjacent.get(e.t).add(e.s); }

  let view = { k: 1, x: 0, y: 0 };
  let hover = null;
  let sel = selected ?? null;
  let hits = new Map();
  let paths = [];
  let pulseStart = 0;
  let alpha = Object.keys(stored).length >= nodes.length * 0.9 ? 0.2 : 1;
  let raf = 0;
  let touched = false;
  let settledOnce = false;
  let tickCount = 0;
  let colors = readColors();

  function readColors() {
    const c = (n) => cssVar(n);
    return { bg: c("--canvas-bg"), ink: c("--ink"), ink2: c("--ink-2"), ink3: c("--ink-3"), line: c("--line"), lineStrong: c("--line-strong"),
             myelin: c("--myelin"), myelinInk: c("--myelin-ink"), myelinWash: c("--myelin-wash"), nissl: c("--nissl"), nisslInk: c("--nissl-ink"),
             soma: c("--soma"), nucleus: c("--soma-nucleus"), hit: c("--hit"), hitNucleus: c("--hit-nucleus"),
             glow: c("--glow"), glowBlur: parseFloat(c("--glow-blur")) || 0, ambient: c("--ambient"),
             region: c("--region"), grid: c("--grid"), font: c("--font-data") };
  }

  // ── simulation ─────────────────────────────────────────────────────────
  function tick() {
    const n = nodes.length;
    for (const node of nodes) {
      const anchor = anchors.get(node.community);
      const pull = anchor ? 0.03 : 0.004;
      const [ax, ay] = anchor || [0, 0];
      node.vx += (ax - node.x) * pull * alpha;
      node.vy += (ay - node.y) * pull * alpha;
    }
    const step = n > 1500 ? Math.ceil(n / 1500) : 1;
    for (let i = 0; i < n; i++) {
      const p = nodes[i];
      for (let j = i + 1 + ((i + tickCount) % step); j < n; j += step) {
        const q = nodes[j];
        let dx = p.x - q.x, dy = p.y - q.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) { dx = hash(i + j) - 0.5; dy = hash(i * j + 3) - 0.5; d2 = 1; }
        if (d2 > 160000) continue;
        const f = (520 * step * alpha) / d2;
        dx *= f; dy *= f;
        p.vx += dx; p.vy += dy; q.vx -= dx; q.vy -= dy;
      }
    }
    for (const e of edges) {
      if (e.kind === "proposed") continue;
      const rest = e.kind === "semantic" ? 110 : e.kind === "hebbian" ? 70 : 64;
      const base = e.kind === "semantic" ? 0.02 : e.kind === "related" ? 0.004 : 0.04;
      const across = e.a.community !== e.b.community ? 0.2 : 1;
      const k = across * base * Math.min(1, e.w + 0.2) / Math.pow(Math.max(1, e.a.degree, e.b.degree), 0.75);
      const dx = e.b.x - e.a.x, dy = e.b.y - e.a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = ((d - rest) / d) * k * alpha;
      e.a.vx += dx * f; e.a.vy += dy * f; e.b.vx -= dx * f; e.b.vy -= dy * f;
    }
    for (const node of nodes) {
      if (node.fixed) { node.vx = node.vy = 0; continue; }
      node.vx *= 0.55; node.vy *= 0.55;
      node.x += node.vx; node.y += node.vy;
    }
    tickCount++;
  }

  function loop(now) {
    raf = 0;
    if (alpha > 0.004) {
      for (let i = 0; i < (alpha > 0.5 ? 3 : 1); i++) tick();
      alpha *= 0.985;
      if (!touched && !hits.size && (alpha <= 0.004 || (!settledOnce && alpha < 0.3))) fit();
      if (alpha <= 0.004) { savePositions(nodes); settledOnce = true; }
    }
    draw(now);
    const animating = paths.length && !reducedMotion() && now - pulseStart < paths.length * 140 + 2600;
    if (alpha > 0.004 || animating) raf = requestAnimationFrame(loop);
  }
  const kick = (a = 0.3) => { alpha = Math.max(alpha, a); if (!raf) raf = requestAnimationFrame(loop); };

  // ── drawing ────────────────────────────────────────────────────────────
  function draw(now = performance.now()) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, ht = canvas.clientHeight;
    if (!w || !ht) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(ht * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(ht * dpr);
    }
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = colors.bg;
    g.fillRect(0, 0, w, ht);
    if (colors.ambient && colors.ambient !== "transparent") {
      // A faint red light from above the field: the only colour on the ground.
      const amb = g.createRadialGradient(w * 0.5, -ht * 0.2, 0, w * 0.5, -ht * 0.2, Math.max(w, ht) * 0.9);
      amb.addColorStop(0, colors.ambient); amb.addColorStop(1, "transparent");
      g.fillStyle = amb; g.fillRect(0, 0, w, ht);
    }
    zoomRead.textContent = `×${view.k.toFixed(2)}`;
    drawGraticule(g, w, ht);
    g.translate(w / 2 + view.x, ht / 2 + view.y);
    g.scale(view.k, view.k);
    const lw = 1 / view.k;
    const focus = hover ?? sel;
    const near = focus !== null ? adjacent.get(focus) : null;

    drawRegions(g, lw);

    for (const e of edges) {
      if (!show[e.kind]) continue;
      const incident = focus !== null && (e.s === focus || e.t === focus);
      let a = focus === null ? 1 : incident ? 1 : 0.16;
      if (hits.size && !(hits.has(e.s) && hits.has(e.t))) a *= 0.4;
      g.globalAlpha = a;
      g.setLineDash([]);
      g.lineCap = "butt";
      if (e.kind === "hebbian") {
        g.strokeStyle = colors.myelinWash;
        g.lineWidth = lw * (3 + 6 * e.w);
        g.setLineDash([9 * lw, 3 * lw]);
        g.beginPath(); g.moveTo(e.a.x, e.a.y); g.lineTo(e.b.x, e.b.y); g.stroke();
        g.setLineDash([]);
        g.strokeStyle = colors.myelin; g.lineWidth = lw * 1.2;
      } else if (e.kind === "explicit") {
        g.strokeStyle = colors.lineStrong; g.lineWidth = lw * (incident ? 1.6 : 0.9);
      } else if (e.kind === "related") {
        g.strokeStyle = colors.lineStrong; g.lineWidth = lw; g.setLineDash([1.5 * lw, 2.5 * lw]);
      } else if (e.kind === "semantic") {
        g.strokeStyle = colors.myelin; g.globalAlpha = a * 0.42; g.lineWidth = lw; g.setLineDash([3 * lw, 3 * lw]);
      } else if (e.kind === "proposed") {
        g.strokeStyle = colors.nissl; g.lineWidth = lw * 1.2; g.setLineDash([5 * lw, 3 * lw]);
      }
      g.beginPath(); g.moveTo(e.a.x, e.a.y); g.lineTo(e.b.x, e.b.y); g.stroke();
    }
    g.setLineDash([]);
    g.globalAlpha = 1;

    const arrived = drawConduction(g, lw, now);

    for (const n of nodes) {
      const dim = (focus !== null && n.id !== focus && !near.has(n.id)) || (hits.size && !hits.has(n.id));
      g.globalAlpha = dim ? 0.28 : 1;
      const hit = hits.get(n.id);
      const r = n.r + (hit ? 1.5 + 3 * hit : 0);
      g.beginPath(); g.arc(n.x, n.y, r, 0, Math.PI * 2);
      g.fillStyle = hit ? colors.hit : colors.soma;
      if (hit && !dim) { g.shadowColor = colors.glow; g.shadowBlur = colors.glowBlur * (0.5 + hit); }
      g.fill();
      g.shadowBlur = 0;
      if (r * view.k >= 4.5) {
        g.beginPath(); g.arc(n.x - r * 0.12, n.y - r * 0.1, r * 0.38, 0, Math.PI * 2);
        g.fillStyle = hit ? colors.hitNucleus : colors.nucleus;
        g.globalAlpha = (dim ? 0.28 : 1) * (hit ? 0.9 : 0.75);
        g.fill();
        g.globalAlpha = dim ? 0.28 : 1;
      }
      const bloom = arrived.get(n.id);
      if (bloom !== undefined && bloom < 1) {
        g.strokeStyle = colors.myelin; g.lineWidth = lw * 1.5; g.globalAlpha = 1 - bloom;
        g.shadowColor = colors.glow; g.shadowBlur = colors.glowBlur;
        g.beginPath(); g.arc(n.x, n.y, r + (4 + 10 * bloom) * lw, 0, Math.PI * 2); g.stroke();
        g.shadowBlur = 0; g.globalAlpha = 1;
      }
      if (n.id === sel) {
        g.lineWidth = 2 * lw; g.strokeStyle = colors.nisslInk;
        g.beginPath(); g.arc(n.x, n.y, r + 4 * lw, 0, Math.PI * 2); g.stroke();
      }
    }
    g.globalAlpha = 1;
    drawLabels(g, focus, near);
  }

  function drawGraticule(g, w, ht) {
    let spacing = 50;
    while (spacing * view.k < 70) spacing *= 2;
    while (spacing * view.k > 150) spacing /= 2;
    const ox = w / 2 + view.x, oy = ht / 2 + view.y;
    const step = spacing * view.k;
    g.strokeStyle = colors.grid; g.lineWidth = 1;
    g.beginPath();
    for (let x = ((ox % step) + step) % step; x < w; x += step) { g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, ht); }
    for (let y = ((oy % step) + step) % step; y < ht; y += step) { g.moveTo(0, Math.round(y) + 0.5); g.lineTo(w, Math.round(y) + 0.5); }
    g.stroke();
  }

  function drawRegions(g, lw) {
    const groups = new Map();
    for (const n of nodes) {
      if (!anchors.has(n.community)) continue;
      if (!groups.has(n.community)) groups.set(n.community, []);
      groups.get(n.community).push([n.x, n.y]);
    }
    const pad = 26;
    for (const pts of groups.values()) {
      const hull = convexHull(pts);
      if (hull.length < 3) continue;
      const cx = hull.reduce((s, p) => s + p[0], 0) / hull.length, cy = hull.reduce((s, p) => s + p[1], 0) / hull.length;
      const grown = hull.map(([x, y]) => { const d = Math.hypot(x - cx, y - cy) || 1; return [x + ((x - cx) / d) * pad, y + ((y - cy) / d) * pad]; });
      const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      g.beginPath();
      const start = mid(grown.at(-1), grown[0]);
      g.moveTo(start[0], start[1]);
      for (let i = 0; i < grown.length; i++) {
        const m = mid(grown[i], grown[(i + 1) % grown.length]);
        g.quadraticCurveTo(grown[i][0], grown[i][1], m[0], m[1]);
      }
      g.closePath();
      g.fillStyle = colors.region; g.fill();
      g.strokeStyle = colors.line; g.lineWidth = lw; g.globalAlpha = 0.7; g.stroke(); g.globalAlpha = 1;
    }
  }

  // Saltatory conduction along each recall path. Returns, per target note, how
  // far its arrival "bloom" has progressed (0..1).
  function drawConduction(g, lw, now) {
    const arrived = new Map();
    if (!paths.length) return arrived;
    const still = reducedMotion();
    const seg = 14 * lw, gap = 4 * lw;
    paths.forEach((p, i) => {
      const dx = p.b.x - p.a.x, dy = p.b.y - p.a.y;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len;
      const gaps = [];
      for (let s = p.a.r + seg; s < len - p.b.r; s += seg + gap) gaps.push(s);
      const t0 = pulseStart + i * 140;
      const hop = still ? gaps.length : Math.floor((now - t0) / HOP_MS);
      const reach = hop >= gaps.length ? len : hop < 0 ? 0 : gaps[hop];
      g.lineCap = "round";
      g.strokeStyle = colors.myelinWash; g.lineWidth = lw * 6; g.globalAlpha = 0.9;
      g.setLineDash([seg, gap]); g.lineDashOffset = -p.a.r;
      g.beginPath(); g.moveTo(p.a.x, p.a.y); g.lineTo(p.b.x, p.b.y); g.stroke();
      g.setLineDash([]); g.lineDashOffset = 0;
      g.strokeStyle = colors.myelin; g.lineWidth = lw * 1.8; g.globalAlpha = 1;
      g.shadowColor = colors.glow; g.shadowBlur = colors.glowBlur;
      g.beginPath(); g.moveTo(p.a.x, p.a.y); g.lineTo(p.a.x + ux * reach, p.a.y + uy * reach); g.stroke();
      if (!still && hop >= 0 && hop < gaps.length) {
        const s = gaps[hop];
        g.fillStyle = colors.myelinInk; g.shadowBlur = colors.glowBlur * 1.6;
        g.beginPath(); g.arc(p.a.x + ux * s, p.a.y + uy * s, 3.2 * lw, 0, Math.PI * 2); g.fill();
      }
      g.shadowBlur = 0;
      if (hop >= gaps.length) {
        const since = still ? 1 : (now - (t0 + gaps.length * HOP_MS)) / 600;
        arrived.set(p.b.id, Math.min(1, Math.max(0, since)));
      }
    });
    g.lineCap = "butt";
    return arrived;
  }

  function drawLabels(g, focus, near) {
    g.textBaseline = "middle";
    g.lineJoin = "round";
    const label = (text, x, y, size, color, weight = 400, align = "left") => {
      g.textAlign = align;
      g.font = `${weight} ${size / view.k}px ${colors.font}`;
      g.lineWidth = 3.5 / view.k; g.strokeStyle = colors.bg; g.strokeText(text, x, y);
      g.fillStyle = color; g.fillText(text, x, y);
    };
    const placed = [];
    const fits = (x, y, wpx, centred = false) => {
      const sx = x * view.k - (centred ? wpx / 2 : 0), sy = y * view.k, box = [sx, sy - 7, sx + wpx, sy + 7];
      if (placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) return false;
      placed.push(box);
      return true;
    };
    if (!hits.size) {
      const tops = new Map();
      for (const n of nodes) {
        if (!anchors.has(n.community)) continue;
        const t = tops.get(n.community) || { x: 0, n: 0, minY: Infinity };
        t.x += n.x; t.n++; t.minY = Math.min(t.minY, n.y);
        tops.set(n.community, t);
      }
      for (const [c, t] of tops) {
        const name = graph.communities[String(c)];
        if (!name) continue;
        const text = name.toUpperCase();
        g.font = `500 ${10 / view.k}px ${colors.font}`;
        const y = t.minY - 30 / view.k;
        if (fits(t.x / t.n, y, g.measureText(text).width * view.k + 10, true)) label(text, t.x / t.n, y, 10, colors.ink3, 500, "center");
      }
    }
    const rankCut = [...nodes].sort((a, b) => b.rank - a.rank)[Math.min(nodes.length - 1, 11)]?.rank ?? 0;
    const order = [...nodes].sort((a, b) => (b.id === focus) - (a.id === focus) || (hits.get(b.id) || 0) - (hits.get(a.id) || 0) || b.rank - a.rank);
    g.font = `400 ${11 / view.k}px ${colors.font}`;
    for (const n of order) {
      const must = n.id === focus || n.id === sel;
      const want = must || hits.has(n.id) || (near && near.has(n.id) && view.k > 0.9)
        || (n.rank >= rankCut && !hits.size && focus === null) || view.k > 2.2;
      if (!want) continue;
      const text = n.title.length > 42 ? n.title.slice(0, 40) + "…" : n.title;
      const wpx = g.measureText(text).width * view.k + 6;
      const gap = n.r + 5 / view.k;
      // Right of the cell first; a recalled note that is crowded out there
      // tries the left, then above and below, before its label is dropped.
      const spots = [[n.x + gap, n.y, "left", false]];
      if (must || hits.has(n.id)) {
        spots.push([n.x - gap - wpx / view.k, n.y, "left", false],
                   [n.x, n.y - gap - 7 / view.k, "center", true], [n.x, n.y + gap + 7 / view.k, "center", true]);
      }
      const spot = spots.find(([x, y, , c]) => fits(x, y, wpx, c)) || (must ? spots[0] : null);
      if (!spot) continue;
      const strong = n.id === focus || hits.has(n.id);
      label(text, spot[0], spot[1], 11, strong ? colors.ink : colors.ink2, strong ? 500 : 400, spot[2]);
    }
  }

  // ── interaction ────────────────────────────────────────────────────────
  const toWorld = (sx, sy) => {
    const rect = canvas.getBoundingClientRect();
    return [(sx - rect.left - rect.width / 2 - view.x) / view.k, (sy - rect.top - rect.height / 2 - view.y) / view.k];
  };
  const nodeAt = (sx, sy) => {
    const [x, y] = toWorld(sx, sy);
    let best = null, bestD = Infinity;
    for (const n of nodes) {
      const d = Math.hypot(n.x - x, n.y - y);
      if (d < n.r + 6 / view.k && d < bestD) { best = n; bestD = d; }
    }
    return best;
  };
  function zoomBy(f, sx, sy) {
    const rect = canvas.getBoundingClientRect();
    const cx = (sx ?? rect.left + rect.width / 2) - rect.left - rect.width / 2;
    const cy = (sy ?? rect.top + rect.height / 2) - rect.top - rect.height / 2;
    const k = Math.min(6, Math.max(0.15, view.k * f));
    view.x = cx - ((cx - view.x) * k) / view.k;
    view.y = cy - ((cy - view.y) * k) / view.k;
    view.k = k;
    draw();
  }
  // padLeft / padRight keep the fitted set clear of the legend and the recall
  // trace, which float over the canvas.
  function fit(only, { maxK = 3, padLeft = 0, padRight = 0 } = {}) {
    const set = only?.length ? only : nodes;
    if (!set.length) return;
    const xs = set.map((n) => n.x), ys = set.map((n) => n.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    // Anything docked over the bottom of the screen (the tour card) declares its
    // height in --overlay-bottom, and the fit keeps clear of it.
    const padBottom = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--overlay-bottom")) || 0;
    const w = (canvas.clientWidth || 800) - padLeft - padRight, ht = (canvas.clientHeight || 600) - padBottom;
    view.k = Math.min(maxK, Math.max(0.2, 0.8 * Math.min(w / Math.max(80, x1 - x0 + 220), ht / Math.max(80, y1 - y0 + 90))));
    view.x = -((x0 + x1) / 2) * view.k + (padLeft - padRight) / 2 - 60 * view.k;
    view.y = -((y0 + y1) / 2) * view.k + 10 - padBottom / 2;
    draw();
  }

  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    const n = nodeAt(e.clientX, e.clientY);
    drag = { n, sx: e.clientX, sy: e.clientY, vx: view.x, vy: view.y, moved: false };
    if (n) n.fixed = true;
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (e) => {
    if (drag) {
      const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.n) { const [x, y] = toWorld(e.clientX, e.clientY); drag.n.x = x; drag.n.y = y; kick(0.08); }
      else { touched = true; view.x = drag.vx + dx; view.y = drag.vy + dy; draw(); }
      card.hidden = true;
      return;
    }
    const n = nodeAt(e.clientX, e.clientY);
    if ((n?.id ?? null) !== hover) { hover = n?.id ?? null; draw(); }
    if (n) {
      const rect = wrap.getBoundingClientRect();
      const topic = graph.communities[String(n.community)];
      card.hidden = false;
      card.replaceChildren(h("strong", null, n.title), h("span", { class: "mono" }, n.path),
        h("dl", null,
          topic ? [h("dt", null, "topic"), h("dd", null, topic)] : null,
          h("dt", null, "links"), h("dd", null, String(n.degree)),
          h("dt", null, "PageRank"), h("dd", null, `top ${Math.max(1, 100 - n.pct)}%`),
          h("dt", null, "words"), h("dd", null, n.words.toLocaleString("en-US"))));
      const x = Math.min(e.clientX - rect.left, rect.width - 320);
      card.style.left = `${x}px`;
      card.style.top = `${e.clientY - rect.top}px`;
      canvas.style.cursor = "pointer";
    } else { card.hidden = true; canvas.style.cursor = ""; }
  });
  canvas.addEventListener("pointerup", (e) => {
    canvas.classList.remove("dragging");
    if (drag?.n) { drag.n.fixed = false; savePositions(nodes); }
    if (drag && !drag.moved) {
      const n = nodeAt(e.clientX, e.clientY);
      if (n) { sel = n.id; onSelect(n.id); }
      else if (sel !== null) { sel = null; }
      draw();
    }
    drag = null;
  });
  canvas.addEventListener("pointerleave", () => { hover = null; card.hidden = true; draw(); });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); touched = true; zoomBy(Math.exp(-e.deltaY * 0.0015), e.clientX, e.clientY); },
                          { passive: false });
  canvas.addEventListener("keydown", (e) => {
    if (e.key === "+" || e.key === "=") { touched = true; zoomBy(1.25); }
    else if (e.key === "-") { touched = true; zoomBy(0.8); }
    else if (e.key === "f" || e.key === "0") fit();
    else if (e.key === "Escape" && hits.size) api.clearRecall();
    else return;
    e.preventDefault();
  });
  const ro = new ResizeObserver(() => draw());
  ro.observe(canvas);
  requestAnimationFrame(() => { fit(); kick(alpha); });

  const api = {
    select(id) { sel = id; draw(); },
    themeChanged() { colors = readColors(); draw(); },
    fit: () => fit(),
    showRecall(result, onPick, onOpenList) {
      hits = new Map();
      paths = [];
      const top = result.results;
      const max = Math.max(...top.map((r) => r.score), 1e-9);
      for (const r of top) hits.set(r.note_id, r.score / max);
      top.forEach((r) => {
        if (r.via && byId.has(r.via.seed_id) && byId.has(r.note_id)) {
          hits.set(r.via.seed_id, Math.max(hits.get(r.via.seed_id) || 0, 0.3));
          paths.push({ a: byId.get(r.via.seed_id), b: byId.get(r.note_id), kind: r.via.kind });
        }
      });
      pulseStart = performance.now();
      trace.hidden = false;
      trace.replaceChildren(
        h("h2", { class: "sect" }, "Recall", h("span", { class: "aside mono" }, `${result.ms} ms · sim ${Number(result.best_similarity).toFixed(2)}`)),
        h("div", { class: "q" }, result.query),
        h("ol", null, top.map((r) => h("li", null,
          h("button", { class: "t", type: "button", onclick: () => onPick(r.note_id) }, r.title),
          r.via ? h("span", { class: "from" }, "via ", h("em", null, `${r.via.kind} link`), ` from ${r.via.seed_title}`)
                : h("span", { class: "from" }, "direct text match")))),
        h("div", { class: "foot" },
          h("button", { class: "btn ghost", type: "button", onclick: () => api.clearRecall() }, "Clear"),
          onOpenList ? h("button", { class: "btn", type: "button", onclick: onOpenList }, "Open in Recall") : null));
      const set = top.map((r) => byId.get(r.note_id)).filter(Boolean);
      for (const p of paths) set.push(p.a);
      const wide = canvas.clientWidth > 900;
      fit(set, { maxK: 1.6, padLeft: wide ? 260 : 0, padRight: wide ? 340 : 0 });
      if (!raf) raf = requestAnimationFrame(loop);
    },
    clearRecall() { hits = new Map(); paths = []; trace.hidden = true; fit(); },
    destroy() { cancelAnimationFrame(raf); ro.disconnect(); savePositions(nodes); },
  };
  return api;
}
