// DOM construction without innerHTML. Strings become text nodes, so vault content
// can never become markup; the CSP forbids inline script and inline style anyway.

export function h(tag, attrs, ...children) {
  const svg = tag.startsWith("svg:");
  const el = svg ? document.createElementNS("http://www.w3.org/2000/svg", tag.slice(4))
                 : document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "class") el.setAttribute("class", v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k === "text") el.textContent = v;
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, String(v));
    }
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

export const fmt = {
  n: (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d),
  int: (x) => (x ?? 0).toLocaleString("en-US"),
  ago(ts) {
    if (!ts) return "never";
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 60) return `${Math.round(s)} s ago`;
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) return `${(s / 3600).toFixed(1)} h ago`;
    return `${Math.round(s / 86400)} d ago`;
  },
  time(iso) {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  },
};

let toastTimer = null;
export function toast(message, isError = false) {
  document.querySelector(".toast")?.remove();
  const el = h("div", { class: isError ? "toast err" : "toast", role: isError ? "alert" : "status" }, message);
  document.body.appendChild(el);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), isError ? 7000 : 3200);
}

export function announce(text) {
  const live = document.getElementById("announce");
  live.textContent = "";
  requestAnimationFrame(() => { live.textContent = text; });
}

export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Highlight query terms in a snippet, returning nodes (never HTML).
export function marked(text, terms) {
  if (!terms.length) return [text];
  const re = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  const out = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(h("mark", null, m[0]));
    last = m.index + m[0].length;
  }
  out.push(text.slice(last));
  return out;
}

export function queryTerms(q) {
  const stop = new Set("a an the and or of to in on for with is are was what which how why does do my our this that from by at as be it".split(" "));
  return [...new Set(q.toLowerCase().match(/[a-z0-9][a-z0-9_.-]{2,}/g) || [])].filter((t) => !stop.has(t));
}

// Plate header: every view is a numbered plate whose label line carries real
// measurements, the way a slide label carries specimen, stain and date.
export function plate({ no, name, meta = [], title, display = false, lede, actions = [] }) {
  return h("header", { class: "plate" },
    h("div", { class: "plate-id" }, h("b", null, `Plate ${no}`), h("span", null, name), meta.filter(Boolean).map((m) => h("span", null, m))),
    h("div", { class: "plate-row" },
      h("div", null, h("h1", { class: display ? "display" : null }, title), lede ? h("p", { class: "lede" }, lede) : null),
      actions.length ? h("div", { class: "actions" }, actions) : null));
}
