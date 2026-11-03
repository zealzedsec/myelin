// Drawn on a 24-unit grid, 1.5 stroke, round joins. Each depicts its view's
// actual object rather than a stock metaphor.
import { h } from "./util.js";

const P = {
  atlas: ["M6 7.5a2 2 0 1 0 0-.01", "M18 6a2 2 0 1 0 0-.01", "M12 17.5a2.5 2.5 0 1 0 0-.01",
          "M7.8 8.6 10.6 15.2", "M16.6 7.6 13.4 15.3", "M8 7.1 16 6.3"],                 // three somata joined by axons
  recall: ["M12 12a2 2 0 1 0 0-.01", "M7.8 7.8a6 6 0 0 1 8.4 0", "M5 5a10 10 0 0 1 14 0", "M7.8 16.2a6 6 0 0 0 8.4 0"],  // activation spreading
  inbox: ["M4 13.5 6.5 6h11l2.5 7.5", "M4 13.5V18h16v-4.5h-5a3 3 0 0 1-6 0Z", "M12 3v6.5", "M9.5 7 12 9.5 14.5 7"],     // a staged item
  evolution: ["M4 19c4 0 6-3 8-7s4-7 8-7", "M12 12c2 1.5 4 3.5 8 3.5", "M17.5 13.7 20 15.5 17.6 17.3"],                 // champion and challenger
  health: ["M12 12a1.5 1.5 0 1 0 0-.01", "M12 12m-4.5 0a4.5 4.5 0 1 0 9 0a4.5 4.5 0 1 0-9 0", "M12 12m-8 0a8 8 0 1 0 16 0a8 8 0 1 0-16 0"], // axon cross-section
  ledger: ["M4 5h6v5H4Z", "M14 14h6v5h-6Z", "M4 14h6v5H4Z", "M7 10v4", "M10 16.5h4", "M14 5h6v5h-6Z", "M10 7.5h4"],    // chained blocks
  sleep: ["M19 14.5A7.5 7.5 0 1 1 9.5 5a6 6 0 0 0 9.5 9.5Z"],
  theme: ["M12 12m-8 0a8 8 0 1 0 16 0a8 8 0 1 0-16 0", "M12 4v16"],                       // a half-lit field of view
  tour: ["M12 12m-8.5 0a8.5 8.5 0 1 0 17 0a8.5 8.5 0 1 0-17 0", "M10.2 8.6v6.8l5.4-3.4Z"],
  lens: ["M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13Z", "M15.3 15.3 20 20"],
  check: ["M5 12.5 10 17 19 7.5"],
  cross: ["M6 6l12 12", "M18 6 6 18"],
  close: ["M6 6l12 12", "M18 6 6 18"],
  open: ["M14 4h6v6", "M20 4l-9 9", "M18 14v6H4V6h6"],
};

export function icon(name, cls = "") {
  const svg = h("svg:svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "1.5",
                             "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", class: cls || null });
  for (const d of P[name] || []) svg.appendChild(h("svg:path", { d }));
  if (name === "theme") svg.appendChild(h("svg:path", { d: "M12 4a8 8 0 0 1 0 16Z", fill: "currentColor", stroke: "none" }));
  if (name === "tour") svg.lastChild.setAttribute("fill", "currentColor");
  return svg;
}

// Legend swatch for an edge kind. Learned links carry a myelin sheath.
export function edgeSwatch(kind) {
  const svg = h("svg:svg", { viewBox: "0 0 30 10", "aria-hidden": "true" });
  if (kind === "hebbian") svg.appendChild(h("svg:line", { x1: 2, y1: 5, x2: 28, y2: 5, class: "sw-hebbian-sheath" }));
  svg.appendChild(h("svg:line", { x1: 1, y1: 5, x2: 29, y2: 5, class: `sw-${kind}` }));
  return svg;
}

// Legend glyph for notes: a stained soma with its nucleus, and the selection ring.
export function somaGlyph() {
  return h("svg:svg", { viewBox: "0 0 34 16", "aria-hidden": "true" },
    h("svg:circle", { cx: 6, cy: 8, r: 3, class: "sw-soma" }),
    h("svg:circle", { cx: 18, cy: 8, r: 6, class: "sw-soma" }),
    h("svg:circle", { cx: 18, cy: 8, r: 2.2, class: "sw-nucleus" }),
    h("svg:circle", { cx: 30, cy: 8, r: 3, class: "sw-soma" }),
    h("svg:circle", { cx: 30, cy: 8, r: 5, class: "sw-ring" }));
}

// The "axon" connector between two titles in a link proposal.
export function axonArrow() {
  return h("svg:svg", { viewBox: "0 0 64 12", "aria-hidden": "true" },
    h("svg:line", { x1: 0, y1: 6, x2: 58, y2: 6 }),
    h("svg:rect", { x: 6, y: 3.5, width: 14, height: 5, rx: 2.5 }),
    h("svg:rect", { x: 24, y: 3.5, width: 14, height: 5, rx: 2.5 }),
    h("svg:path", { d: "M56 2 63 6 56 10Z" }));
}

// Saltatory-conduction loader.
export function loader(text = "Reading the vault") {
  return h("div", { class: "loader", role: "status" },
    h("svg:svg", { viewBox: "0 0 72 12", "aria-hidden": "true" },
      h("svg:line", { x1: 0, y1: 6, x2: 72, y2: 6 }),
      [4, 26, 48].map((x) => h("svg:rect", { x: x + 2, y: 2.5, width: 16, height: 7, rx: 3.5 })),
      h("svg:circle", { cx: 3, cy: 6, r: 2.4 })),
    text);
}

// A two-node "reached via" glyph: seed soma, myelinated axon, target soma.
export function viaGlyph() {
  return h("svg:svg", { viewBox: "0 0 52 10", "aria-hidden": "true" },
    h("svg:line", { x1: 4, y1: 5, x2: 48, y2: 5 }),
    h("svg:rect", { x: 11, y: 2.5, width: 11, height: 5, rx: 2.5 }),
    h("svg:rect", { x: 26, y: 2.5, width: 11, height: 5, rx: 2.5 }),
    h("svg:circle", { cx: 4, cy: 5, r: 3.2 }),
    h("svg:circle", { cx: 48, cy: 5, r: 3.6, class: "end" }));
}
