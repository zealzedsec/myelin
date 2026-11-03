// Three transports behind one interface.
//
// native    the desktop app (`myelin app`). Calls go to Python in-process through
//           pywebview's bridge and live events are pushed into the page, so there
//           is no HTTP, no port and no token.
// live      the local server. A per-process token arrives in a meta tag and goes
//           on every call; cross-origin pages cannot read it, so cannot forge calls.
// snapshot  a static export (`myelin snapshot`). Reads snapshot.json, answers the
//           preset queries with the results Myelin actually produced, replays the
//           last sleep cycle at its recorded step timings, and keeps decisions in
//           the page. Nothing is written anywhere.
const TOKEN = document.querySelector('meta[name="myelin-token"]')?.content || "";
const DECLARED = document.querySelector('meta[name="myelin-mode"]')?.content;
export const MODE = DECLARED === "snapshot" || DECLARED === "native" ? DECLARED : "live";

async function call(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: { "X-Myelin-Token": TOKEN, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty or non-JSON body */ }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

const live = {
  status: () => call("GET", "/api/status"),
  graph: () => call("GET", "/api/graph"),
  note: (id) => call("GET", `/api/note?id=${encodeURIComponent(id)}`),
  recall: (q, k = 10) => call("GET", `/api/recall?q=${encodeURIComponent(q)}&k=${k}`),
  proposals: (status = "pending", kind = "") =>
    call("GET", `/api/proposals?status=${status}&kind=${encodeURIComponent(kind)}&limit=300`),
  selfmodel: () => call("GET", "/api/selfmodel"),
  metrics: (names) => call("GET", `/api/metrics?names=${encodeURIComponent(names.join(","))}`),
  evolution: () => call("GET", "/api/evolution"),
  ledger: (limit = 300) => call("GET", `/api/ledger?limit=${limit}`),
  feedback: (recall_id, note_id, useful) => call("POST", "/api/feedback", { recall_id, note_id, useful }),
  decide: (id, accept, reason = "") => call("POST", "/api/decide", { id, accept, reason }),
  revert: (id) => call("POST", "/api/revert", { id }),
  sleep: () => call("POST", "/api/sleep", {}),
  ask: (question) => call("POST", "/api/ask", { question }),
  critique: (text) => call("POST", "/api/critique", { text }),
};

// ── native transport ─────────────────────────────────────────────────────
// pywebview announces its bridge with `pywebviewready`; calls wait for it.
const bridgeReady = new Promise((resolve) => {
  if (window.pywebview?.api?.call) resolve();
  else window.addEventListener("pywebviewready", () => resolve(), { once: true });
});
async function bridge(name, args = {}) {
  await bridgeReady;
  const out = await window.pywebview.api.call(name, args);
  if (!out || out.error) throw new Error(out ? out.error : `no answer to ${name}`);
  return out.data;
}
const native = {
  status: () => bridge("status"),
  graph: () => bridge("graph"),
  note: (id) => bridge("note", { id }),
  recall: (q, k = 10) => bridge("recall", { q, k }),
  proposals: (status = "pending", kind = "") => bridge("proposals", { status, kind, limit: 300 }),
  selfmodel: () => bridge("selfmodel"),
  metrics: (names) => bridge("metrics", { names: names.join(",") }),
  evolution: () => bridge("evolution"),
  ledger: (limit = 300) => bridge("ledger", { limit }),
  feedback: (recall_id, note_id, useful) => bridge("feedback", { recall_id, note_id, useful }),
  decide: (id, accept, reason = "") => bridge("decide", { id, accept, reason }),
  revert: (id) => bridge("revert", { id }),
  sleep: () => bridge("sleep"),
  ask: (question) => bridge("ask", { question }),
  critique: (text) => bridge("critique", { text }),
  presets: async () => [],
};

// ── snapshot transport ───────────────────────────────────────────────────
let snap = null;
const emitter = { handlers: null };
const decisions = new Map();     // proposal id -> "applied" | "rejected"
const norm = (q) => q.toLowerCase().replace(/\s+/g, " ").trim().replace(/[?.!]+$/, "");
const loadSnap = () => snap ??= fetch("snapshot.json").then((r) => {
  if (!r.ok) throw new Error("snapshot.json is missing next to index.html");
  return r.json();
});
const clone = (x) => JSON.parse(JSON.stringify(x));
export const STEP_ORDER = ["ingest", "hebbian", "structure", "link_prediction", "reranker", "evolution", "review", "self_model"];

const snapshot = {
  status: async () => {
    const d = await loadSnap();
    const pending = {};
    for (const p of d.proposals.pending) if (!decisions.has(p.id)) pending[p.kind] = (pending[p.kind] || 0) + 1;
    return { ...d.status, pending, generated_at: d.generated_at };
  },
  graph: async () => (await loadSnap()).graph,
  note: async (ref) => {
    const d = await loadSnap();
    const key = String(ref);
    const all = Object.values(d.notes);
    const n = d.notes[key] || all.find((x) => x.path === key || x.stem === key)
      || all.find((x) => x.title.toLowerCase() === key.toLowerCase() || x.stem.toLowerCase() === key.toLowerCase());
    if (!n) throw new Error("no such note in this snapshot");
    const out = clone(n);
    out.stems = d.stems;
    out.proposals = out.proposals.filter((p) => !decisions.has(p.id));
    return out;
  },
  recall: async (q) => {
    const d = await loadSnap();
    const hit = Object.entries(d.recalls).find(([k]) => norm(k) === norm(q));
    if (hit) return clone(hit[1]);
    return { snapshotMiss: true, query: q, presets: Object.keys(d.recalls), results: [] };
  },
  presets: async () => Object.keys((await loadSnap()).recalls),
  proposals: async (status = "pending", kind = "") => {
    const d = await loadSnap();
    const pool = [...d.proposals.pending, ...d.proposals.applied, ...d.proposals.rejected].map((p) =>
      decisions.has(p.id) ? { ...p, status: decisions.get(p.id) } : p);
    return { items: pool.filter((p) => p.status === status && (!kind || p.kind === kind)) };
  },
  selfmodel: async () => (await loadSnap()).selfmodel,
  metrics: async () => (await loadSnap()).metrics,
  evolution: async () => (await loadSnap()).evolution,
  ledger: async () => (await loadSnap()).ledger,
  feedback: async (recall_id, note_id, useful) => ({ recall_id, note_id, label: useful ? 1 : 0, local: true }),
  decide: async (id, accept) => {
    decisions.set(id, accept ? "applied" : "rejected");
    return { id, accepted: accept, status: accept ? "applied" : "rejected", local: true };
  },
  revert: async (id) => { decisions.delete(id); return { reverted: true, local: true }; },
  sleep: async () => {
    const d = await loadSnap();
    const last = d.evolution.last_sleep;
    if (!last) return { started: false, reason: "this snapshot has no recorded sleep cycle" };
    const h = emitter.handlers || {};
    h.sleep_started?.({});
    let t = 300;
    for (const step of STEP_ORDER) {
      const rec = last.steps[step];
      if (!rec) continue;
      // Real recorded durations; very short steps get a floor so they stay visible.
      t += Math.max(260, rec.seconds * 1000);
      setTimeout(() => h.sleep_step?.({ step, ok: rec.ok, seconds: rec.seconds, error: rec.error }), t);
    }
    setTimeout(() => h.sleep_finished?.({ seconds: last.seconds, replay: true }), t + 200);
    return { started: true, replay: true };
  },
  ask: async () => { throw new Error("Claude answers run in the local app with an API key, not in a snapshot"); },
  critique: async () => { throw new Error("the authenticity check runs in the local app"); },
};

export const api = MODE === "snapshot" ? snapshot : MODE === "native" ? native : { ...live, presets: async () => [] };

export function events(handlers) {
  if (MODE === "native") {
    window.addEventListener("myelin:event", (e) => handlers[e.detail.kind]?.(e.detail.data));
    bridgeReady.then(() => window.pywebview.api.subscribe()).then(() => handlers.connection?.(true),
      () => handlers.connection?.(false));
    return;
  }
  if (MODE === "snapshot") {
    emitter.handlers = handlers;
    handlers.connection?.(true);
    return;
  }
  let source = null;
  let retry = 1000;
  const open = () => {
    source = new EventSource(`/api/events?token=${encodeURIComponent(TOKEN)}`);
    source.onopen = () => { retry = 1000; handlers.connection?.(true); };
    source.onerror = () => {
      handlers.connection?.(false);
      source.close();
      setTimeout(open, retry);
      retry = Math.min(retry * 2, 15000);
    };
    for (const kind of ["ledger", "ingest", "sleep_started", "sleep_step", "sleep_finished", "proposal", "feedback", "fault"]) {
      source.addEventListener(kind, (e) => handlers[kind]?.(JSON.parse(e.data)));
    }
  };
  open();
}
