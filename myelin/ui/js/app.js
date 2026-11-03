// Shell: routing, the status strip, the ask bar, the command palette, keyboard,
// and the hooks the guided tour drives.
import { MODE, STEP_ORDER, api, events } from "./api.js";
import { mountAtlas } from "./atlas.js";
import { viewEvolution } from "./evolution.js";
import { icon, loader } from "./icons.js";
import { showNote } from "./inspector.js";
import { startTour } from "./tour.js";
import { clear, fmt, h, toast, announce } from "./util.js";
import { viewHealth, viewInbox, viewLedger, viewRecall } from "./views.js";

const $ = (s) => document.querySelector(s);
const viewEl = $("#view");
const inspector = $("#inspector");
const askInput = $("#q");
const VIEWS = ["atlas", "recall", "inbox", "evolution", "health", "ledger"];

const ctx = {
  status: null, stems: {}, query: "", atlas: null, graph: null, keys: null, lastRecall: null, awaitingIngest: false,
  openNote: (id) => showNote(inspector, id, ctx),
  closeNote: () => { inspector.hidden = true; clear(inspector); ctx.atlas?.select(null); },
  onNoteShown: (id) => ctx.atlas?.select(id),
  refreshStatus: () => refreshStatus(),
  rerender: () => route(),
  // Go to a view and resolve once it has rendered. A new history entry only
  // when the address actually changes, so Back still walks the views.
  navigate: async (hash) => {
    if (location.hash !== hash) history.pushState(null, "", hash);
    await route();
  },
  // Recall drawn on the atlas: seeds light up, then activation travels the links.
  askInAtlas: async (q) => {
    ctx.query = q;
    askInput.value = q;
    if (!ctx.atlas) await ctx.navigate("#atlas");
    const res = await api.recall(q, 10);
    if (res.snapshotMiss) return ctx.navigate(`#recall/${encodeURIComponent(q)}`);
    ctx.lastRecall = res;
    ctx.atlas?.showRecall(res, (id) => ctx.openNote(id), () => ctx.navigate(`#recall/${encodeURIComponent(q)}`));
    announce(`${res.results.length} notes recalled`);
    return res;
  },
  ask: (q) => {
    ctx.query = q;
    askInput.value = q;
    return ctx.navigate(`#recall/${encodeURIComponent(q)}`);
  },
};

// ── theme ────────────────────────────────────────────────────────────────
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("myelin.theme", t); } catch { /* per-viewer convenience only */ }
  ctx.atlas?.themeChanged();
}
// Vantablack is the identity, so it is the default whatever the OS or an
// embedding host prefers; paper is there for whoever chooses it in this page.
let stored = null;
try { stored = localStorage.getItem("myelin.theme"); } catch { /* ignore */ }
applyTheme(stored === "light" ? "light" : "dark");
$("#theme").append(icon("theme"));
$("#theme").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

// ── chrome ───────────────────────────────────────────────────────────────
document.querySelectorAll("[data-icon]").forEach((el) => el.append(icon(el.dataset.icon)));
$("#lens").replaceWith(icon("lens", "lens"));
$("#tour-btn").prepend(icon("tour"));
$("#tour-btn").addEventListener("click", () => startTour(ctx).catch((e) => toast(e.message, true)));
$("#sleep").append(icon("sleep"), MODE === "snapshot" ? "Replay sleep" : "Sleep");
$("#sleep").addEventListener("click", async () => {
  try {
    const r = await api.sleep();
    toast(r.replay ? "Replaying the last recorded sleep cycle" : r.started ? "Sleep cycle started: consolidating, training, evolving" : r.reason);
  } catch (e) { toast(e.message, true); }
});
if (MODE === "snapshot") {
  $("#mode-chip").hidden = false;
  $("#mode-chip").title = "A static export: recorded answers, no server, nothing is written";
}

// ── first run ────────────────────────────────────────────────────────────
// The desktop app opens its window before it has read the vault: the scheduler
// indexes it, then runs the first sleep, which builds the self-quiz and measures
// recall. The atlas waits for the first, the tour for the second.
const scheduling = (s) => !!s && (s.daemon_alive || s.sleeping);
const indexing = (s) => scheduling(s) && !s.last_ingest && !s.last_sleep;   // a brain that has slept was read before
const measured = (s) => !!s && s.notes > 0 && s.last_sleep > 0;
function indexed(statusIsFresh = false) {   // the first ingest finished: draw what it found
  if (!ctx.awaitingIngest) return;
  ctx.awaitingIngest = false;
  (statusIsFresh ? Promise.resolve() : refreshStatus()).then(route);
}

// ── status strip ─────────────────────────────────────────────────────────
const readouts = $("#readouts");
let statusTimer = null;
const pollStatus = (ms) => { clearTimeout(statusTimer); statusTimer = setTimeout(refreshStatus, ms); };
const soon = () => pollStatus(400);
let live = false;
let eventsDot = null;
function readout(label, ...value) { return h("div", { class: "readout" }, h("span", null, label), ...value); }
async function refreshStatus() {
  try { ctx.status = await api.status(); }
  catch (e) { clear(readouts).append(readout("server", h("b", { class: "err" }, e.message))); return; }
  const s = ctx.status;
  const edges = s.edges || {};
  const total = Object.values(edges).reduce((a, b) => a + b, 0);
  const pending = Object.values(s.pending || {}).reduce((a, b) => a + b, 0);
  const count = $("#inbox-count");
  count.hidden = !pending;
  count.textContent = pending > 99 ? "99+" : String(pending);
  $("#root").textContent = s.root;
  $("#root").title = s.root;
  eventsDot = h("span", { class: live ? "dot ok" : "dot warn" });
  const tail = MODE === "snapshot"
    ? [readout("snapshot", h("span", { class: "dot ok" }), h("b", null, fmt.time(s.generated_at)))]
    : [readout("scheduler", h("span", { class: s.daemon_alive ? "dot ok" : "dot" }),
         h("b", null, s.sleeping ? "sleeping" : indexing(s) ? "indexing" : s.daemon_alive ? "watching" : "off")),
       readout("last sleep", h("b", null, fmt.ago(s.last_sleep))),
       readout("events", eventsDot, h("b", null, live ? "live" : "reconnecting"))];
  clear(readouts).append(
    readout("notes", h("b", null, fmt.int(s.notes))),
    readout("edges", h("b", null, fmt.int(total)), h("span", { class: "muted" },
      `auth ${(edges.explicit || 0) + (edges.related || 0)} · sem ${edges.semantic || 0} · learned ${edges.hebbian || 0}`)),
    readout("recall mrr", h("b", null, fmt.n(s.holdout_mrr)), h("span", { class: "muted" }, "holdout")),
    readout("config", h("b", null, `v${s.champion.version}`)),
    readout("ledger", h("span", { class: s.ledger.ok ? "dot ok" : "dot bad" }), h("b", null, `${fmt.int(s.ledger.events)} events`)),
    ...tail);
  // Polled as well as pushed: the event that ends the first ingest can fire
  // before this window has subscribed.
  if (ctx.awaitingIngest) {
    if (indexing(s)) pollStatus(2000);
    else indexed(true);
  }
}
function blip() {
  if (!eventsDot) return;
  eventsDot.classList.remove("blip");
  void eventsDot.getBoundingClientRect();   // restart the animation
  eventsDot.classList.add("blip");
}

// ── routing ──────────────────────────────────────────────────────────────
// Renders are serialized: the tour can navigate while a slow view is still
// loading, and two renders interleaving into #view would duplicate it.
let routing = Promise.resolve();
function route() {
  routing = routing.then(render, render);
  return routing;
}
async function render() {
  const [name, ...rest] = (location.hash.slice(1) || "atlas").split("/");
  const view = VIEWS.includes(name) ? name : "atlas";
  if (view === "recall" && rest.length) ctx.query = decodeURIComponent(rest.join("/"));
  if (view === "recall" && ctx.query) askInput.value = ctx.query;
  document.querySelectorAll(".rail a").forEach((a) => {
    if (a.dataset.view === view) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  ctx.atlas?.destroy();
  ctx.atlas = null;
  ctx.keys = null;
  ctx.onLedger = null;
  ctx.onSleep = null;
  ctx.awaitingIngest = false;
  clear(viewEl);
  document.title = `Myelin · ${view[0].toUpperCase()}${view.slice(1)}`;
  try {
    if (view === "atlas") {
      // Notes arrive early in an ingest, links and the semantic space later: on a
      // first run, wait for all of it rather than draw half a graph.
      if (indexing(ctx.status)) {
        ctx.awaitingIngest = true;
        viewEl.append(h("div", { class: "page" }, h("div", { class: "empty" }, h("strong", null, "Reading the vault for the first time."),
          " The atlas draws itself as soon as every note is indexed."), loader("Indexing notes")));
        pollStatus(2000);
        return;
      }
      const pending = h("div", { class: "page" }, loader("Drawing the atlas"));
      viewEl.append(pending);
      try { ctx.graph = await api.graph(); } finally { pending.remove(); }
      ctx.stems = Object.fromEntries(ctx.graph.nodes.map((n) => [n.stem, n.id]));
      if (!ctx.graph.nodes.length) {
        viewEl.append(h("div", { class: "page" }, h("div", { class: "empty" }, h("strong", null, "The vault is empty or out of scope."),
          " Check the include rules in myelin.toml.")));
        return;
      }
      ctx.atlas = mountAtlas(viewEl, { graph: ctx.graph, onSelect: (id) => ctx.openNote(id), selected: null });
    } else if (view === "recall") await viewRecall(viewEl, ctx);
    else if (view === "inbox") await viewInbox(viewEl, ctx);
    else if (view === "evolution") await viewEvolution(viewEl, ctx);
    else if (view === "health") await viewHealth(viewEl, ctx);
    else if (view === "ledger") await viewLedger(viewEl, ctx);
  } catch (e) {
    viewEl.append(h("div", { class: "page" }, h("p", { class: "err" }, e.message)));
  }
}
window.addEventListener("hashchange", () => route());

// ── ask bar ──────────────────────────────────────────────────────────────
$("#ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = askInput.value.trim();
  if (!q) return;
  try {
    if (ctx.atlas) await ctx.askInAtlas(q);
    else await ctx.ask(q);
  } catch (err) { toast(err.message, true); }
});

// ── command palette ──────────────────────────────────────────────────────
const palette = $("#palette");
function openPalette(mode = "commands") {
  palette.hidden = false;
  const input = h("input", { type: "text", placeholder: mode === "help" ? "" : "Jump to a note or run a command", "aria-label": "Command", spellcheck: "false" });
  const list = h("ul", { role: "listbox" });
  const box = h("div", { class: "box" });
  clear(palette).append(box);
  if (mode === "help") {
    box.append(h("dl", null,
      [["/", "Focus the ask bar"], ["⌘K or Ctrl K", "Command palette: jump to any note"], ["g then a·r·i·e·h·l", "Go to Atlas, Recall, Inbox, Evolution, Health, Ledger"],
       ["+ / − / f", "Atlas: zoom in, out, fit"], ["j / k", "Next / previous result or card"], ["a / r", "Accept / reject the current card"],
       ["t", "Start the guided tour"], ["Esc", "Close the inspector or this panel"]]
        .map(([k, v]) => [h("dt", null, h("kbd", null, k)), h("dd", null, v)])));
    return;
  }
  const commands = [
    ...VIEWS.map((v) => ({ label: `Go to ${v[0].toUpperCase()}${v.slice(1)}`, hint: "view", run: () => { location.hash = `#${v}`; } })),
    { label: "Take the guided tour", hint: "tour", run: () => $("#tour-btn").click() },
    { label: MODE === "snapshot" ? "Replay the last sleep cycle" : "Run a sleep cycle", hint: "train · evolve", run: () => $("#sleep").click() },
    { label: "Switch vantablack / paper", hint: "theme", run: () => $("#theme").click() },
  ];
  let selected = 0;
  let current = [];
  const render = () => {
    const q = input.value.toLowerCase().trim();
    const notes = (ctx.graph?.nodes || []).map((n) => ({ label: n.title, hint: n.path, run: () => ctx.openNote(n.id) }));
    const score = (label) => {
      const l = label.toLowerCase();
      if (!q) return 1;
      if (l.includes(q)) return 2 - l.indexOf(q) / 100;
      let i = 0;
      for (const ch of l) if (ch === q[i]) i++;
      return i === q.length ? 0.5 : 0;
    };
    current = [...commands, ...notes].map((c) => ({ ...c, s: score(c.label) })).filter((c) => c.s > 0)
      .sort((a, b) => b.s - a.s).slice(0, 40);
    selected = Math.min(selected, Math.max(0, current.length - 1));
    clear(list).append(...current.map((c, i) => h("li", { role: "option", "aria-selected": String(i === selected),
      onclick: () => { closePalette(); c.run(); } }, c.label, h("span", { class: "hint" }, c.hint))));
  };
  input.addEventListener("input", () => { selected = 0; render(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      selected = (selected + (e.key === "ArrowDown" ? 1 : -1) + current.length) % Math.max(1, current.length);
      render();
      list.children[selected]?.scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" && current[selected]) { closePalette(); current[selected].run(); }
  });
  box.append(input, list);
  render();
  input.focus();
}
function closePalette() { palette.hidden = true; clear(palette); }
palette.addEventListener("click", (e) => { if (e.target === palette) closePalette(); });
$("#help").addEventListener("click", () => openPalette("help"));

// ── keyboard ─────────────────────────────────────────────────────────────
let goPending = 0;
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openPalette(); return; }
  if (e.key === "Escape") {
    if (!palette.hidden) return closePalette();
    if (document.querySelector(".tour")) return;   // the tour handles its own Esc
    if (!inspector.hidden) return ctx.closeNote();
    if (document.activeElement === askInput) askInput.blur();
    return;
  }
  const typing = e.target.closest("input, textarea, [contenteditable]");
  if (typing || e.metaKey || e.ctrlKey || e.altKey || !palette.hidden) return;
  if (e.key === "/") { e.preventDefault(); askInput.focus(); askInput.select(); return; }
  if (e.key === "?") { openPalette("help"); return; }
  if (e.key === "g") { goPending = Date.now(); return; }
  if (Date.now() - goPending < 1200) {
    const map = { a: "atlas", r: "recall", i: "inbox", e: "evolution", h: "health", l: "ledger" };
    if (map[e.key]) { location.hash = `#${map[e.key]}`; goPending = 0; return; }
  }
  if (e.key === "t" && !document.querySelector(".tour")) { $("#tour-btn").click(); return; }
  if (ctx.keys?.(e)) e.preventDefault();
});

// ── live events ──────────────────────────────────────────────────────────
let firstRunStage = null;   // set while the first-run card is showing
events({
  connection: (ok) => { live = ok; soon(); },
  ledger: (ev) => { blip(); ctx.onLedger?.(ev); if (ev.kind === "INGEST") indexed(); soon(); },
  ingest: (d) => {
    blip();
    toast(ctx.awaitingIngest ? `Indexed ${fmt.int(d.added)} notes` : `Vault changed: +${d.added} ~${d.updated} −${d.removed}`);
    indexed(); soon();
  },
  sleep_started: () => { blip(); soon(); ctx.onSleep?.("sleep_started", {}); firstRunStage?.(null); },
  sleep_step: (d) => { blip(); ctx.onSleep?.("sleep_step", d); firstRunStage?.(d.step); },
  sleep_finished: (d) => {
    blip(); soon();
    if (ctx.onSleep) ctx.onSleep("sleep_finished", d);
    else toast(d.replay ? "Replay finished" : `Sleep finished in ${d.seconds}s`);
  },
  proposal: () => { blip(); soon(); },
  feedback: () => blip(),
  fault: (d) => toast(d.error, true),
});

// ?tour=auto plays the tour (used to record the demo). The desktop app and a
// snapshot also play it once, on first opening, or whenever opened at #tour.
let tourMode = new URLSearchParams(location.search).get("tour");
if (!tourMode && location.hash === "#tour") { tourMode = "auto"; history.replaceState(null, "", "#atlas"); }
if (!tourMode && MODE !== "live" && !location.hash) {   // the desktop app and a snapshot: once per viewer
  let seen = false;
  try { seen = localStorage.getItem("myelin.tour.seen") === "1"; } catch { /* nowhere to remember it: play it */ }
  if (!seen) tourMode = "auto";
}
const tourSeen = () => { try { localStorage.setItem("myelin.tour.seen", "1"); } catch { /* per-viewer only */ } };

// The tour quotes this brain's own measurements, so on a first run it waits for
// the first sleep to make them, and says meanwhile what is happening. Resolves
// true to play the tour now, false to leave it.
function firstMeasurement() {
  if (measured(ctx.status) || !scheduling(ctx.status)) return Promise.resolve(true);
  return new Promise((resolve) => {
    const root = document.documentElement;
    const line = loader("Indexing notes");
    const stages = (n) => { line.lastChild.textContent = `First sleep · ${n} of ${STEP_ORDER.length} stages`; };
    const started = Date.now();
    let done = 0;
    let timer = null;
    const finish = (play, ownDock = true) => {
      clearTimeout(timer);
      firstRunStage = null;
      if (ownDock) { card.remove(); root.classList.remove("touring"); root.style.removeProperty("--overlay-bottom"); }
      resolve(play);
    };
    const card = h("div", { class: "tour", role: "status", "aria-live": "polite" },
      h("div", { class: "id" }, h("span", null, "First run"), h("span", null, "live")),
      h("h3", null, "Reading your vault"),
      h("p", null, "Myelin is indexing your notes, then running its first sleep: it builds a quiz from your own links " +
        "and measures how well it recalls them. The tour starts once that first measurement is in."),
      line,
      h("div", { class: "ctrl" }, h("button", { class: "btn", type: "button", onclick: () => { tourSeen(); finish(false); } }, "Skip the tour")));
    document.body.append(card);
    root.classList.add("touring");
    root.style.setProperty("--overlay-bottom", `${card.offsetHeight + 28}px`);
    firstRunStage = (step) => { done = step ? Math.max(done, STEP_ORDER.indexOf(step) + 1) : 0; stages(done); };
    const check = async () => {
      if (!card.isConnected) { tourSeen(); return finish(false, false); }   // started by hand; that tour owns the dock now
      await refreshStatus();
      const s = ctx.status;
      if (s?.last_ingest && !s.notes) return finish(false);   // nothing to tour; it plays once there is
      if (measured(s) || !scheduling(s) || Date.now() - started > 600_000) return finish(true);
      if (s?.sleeping && !done) stages(0);
      timer = setTimeout(check, 2000);
    };
    timer = setTimeout(check, 2000);
  });
}

refreshStatus().then(route).then(async () => {
  if (!tourMode) return;
  if (tourMode === "auto" && !(await firstMeasurement())) return;
  tourSeen();
  startTour(ctx, { auto: tourMode === "auto" }).catch((e) => toast(e.message, true));
});
setInterval(refreshStatus, 30000);
