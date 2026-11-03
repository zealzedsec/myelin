// The note inspector: one note, its neighbourhood drawn as a small ego map, the
// links Myelin suggests for it, and the note itself.
import { MODE, api } from "./api.js";
import { icon, loader } from "./icons.js";
import { renderMarkdown } from "./md.js";
import { clear, fmt, h, toast } from "./util.js";

const KIND_LABEL = { explicit: "authored", related: "frontmatter", semantic: "semantic", hebbian: "learned" };

function dirGlyph(forward) {
  return h("svg:svg", { viewBox: "0 0 16 8", "aria-hidden": "true" },
    h("svg:line", { x1: 1, y1: 4, x2: 13, y2: 4, stroke: "currentColor", "stroke-width": 1.2 }),
    forward ? h("svg:path", { d: "M11 1.5 15 4 11 6.5", fill: "none", stroke: "currentColor", "stroke-width": 1.2 })
            : h("svg:path", { d: "M5 1.5 1 4 5 6.5", fill: "none", stroke: "currentColor", "stroke-width": 1.2 }));
}

// Backlinks stack on the left, outgoing links on the right, each a curved
// dendrite into the note. Similar and learned neighbours, which have no
// direction, join the shorter column as hollow cells. Columns never overlap,
// so no two labels can collide.
function egoMap(n, open) {
  const W = 440, cx = W / 2, colL = 150, colR = W - 150, ROW = 19, PER = 9;
  const best = new Map();
  for (const x of n.neighbors) if (!best.has(x.id) || x.w > best.get(x.id).w) best.set(x.id, x);
  const items = [...best.values()].sort((a, b) => b.w - a.w);
  const authored = (x) => x.kind === "explicit" || x.kind === "related";
  const left = items.filter((x) => authored(x) && !x.forward);
  const right = items.filter((x) => authored(x) && x.forward);
  for (const x of items.filter((y) => !authored(y))) (left.length <= right.length ? left : right).push(x);
  const shownL = left.slice(0, PER), shownR = right.slice(0, PER);
  const rows = Math.max(shownL.length, shownR.length, 1);
  const H = 34 + rows * ROW + 10, cy = 30 + (rows * ROW) / 2;
  const svg = h("svg:svg", { class: "ego", viewBox: `0 0 ${W} ${H}`, role: "group",
    "aria-label": `${shownL.length + shownR.length} strongest associations` });
  svg.append(h("svg:text", { class: "dir", x: W - 4, y: 12, "text-anchor": "end" }, "LINKS TO →"),
             h("svg:text", { class: "dir", x: 4, y: 12 }, "← LINKED FROM"));
  const column = (list, x, side) => list.forEach((nb, i) => {
    const y = 30 + ROW * (i + 0.5) + ((rows - list.length) * ROW) / 2;
    const bend = (x - cx) * 0.55;
    svg.insertBefore(h("svg:path", { class: `k-${nb.kind}`, d: `M${cx} ${cy} C ${cx + bend} ${cy}, ${x - bend} ${y}, ${x} ${y}` }), svg.firstChild);
    const label = nb.title.length > 24 ? nb.title.slice(0, 23) + "…" : nb.title;
    const g = h("svg:g", { class: authored(nb) ? "nb" : "nb assoc", tabindex: "0", role: "button", "aria-label": `${nb.title}, ${KIND_LABEL[nb.kind] || nb.kind} link` },
      h("svg:circle", { cx: x, cy: y, r: 3 + 3 * Math.min(1, nb.w) }),
      h("svg:text", { x: x + side * 10, y: y + 3.5, "text-anchor": side > 0 ? "start" : "end" }, label));
    g.append(h("svg:title", null, `${nb.title} · ${KIND_LABEL[nb.kind] || nb.kind} · weight ${nb.w}`));
    g.addEventListener("click", () => open(nb.id));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(nb.id); } });
    svg.append(g);
  });
  column(shownL, colL, -1);
  column(shownR, colR, 1);
  const hidden = left.length - shownL.length + right.length - shownR.length;
  if (hidden > 0) svg.append(h("svg:text", { class: "more", x: cx, y: H - 2, "text-anchor": "middle" }, `+${hidden} more below`));
  svg.append(h("svg:circle", { class: "center", cx, cy, r: 11 }), h("svg:circle", { class: "center-n", cx: cx - 1.5, cy: cy - 1.5, r: 4 }));
  return svg;
}

export async function showNote(panel, id, ctx) {
  panel.hidden = false;
  clear(panel).append(h("div", { class: "insp-body" }, loader("Opening note")));
  let n;
  try { n = await api.note(id); }
  catch (e) { clear(panel).append(h("div", { class: "insp-body err" }, e.message)); return; }
  ctx.stems = n.stems;
  const open = (ref) => showNote(panel, ref, ctx);
  const groups = {};
  for (const x of n.neighbors) (groups[x.kind] ||= []).push(x);
  const decide = async (p, accept) => {
    try {
      const out = await api.decide(p.id, accept);
      toast(out.local ? "Recorded in this page only" : accept ? "Link written into the note" : "Rejected");
      open(n.id);
      ctx.refreshStatus();
    } catch (e) { toast(e.message, true); }
  };

  clear(panel).append(
    h("div", { class: "insp-head" },
      h("div", { class: "row" },
        h("div", { class: "plate-id" }, h("b", null, "Specimen"), h("span", null, `#${n.id}`), h("span", null, n.type)),
        h("button", { class: "btn icon", type: "button", "aria-label": "Close inspector", onclick: () => ctx.closeNote() }, icon("close"))),
      h("h2", null, n.title),
      h("div", { class: "path" }, n.path),
      h("div", { class: "chips" },
        h("span", { class: "chip" }, `${fmt.int(n.words)} words`),
        n.tags.slice(0, 6).map((t) => h("span", { class: "chip" }, `#${t}`)),
        n.protected ? h("span", { class: "chip warn", title: "Rewritten by another program; Myelin never edits it" }, "generated · read-only") : null,
        n.frontmatter_error ? h("span", { class: "chip bad", title: n.frontmatter_error }, "frontmatter error") : null,
        n.broken_links.length ? h("span", { class: "chip bad", title: n.broken_links.join(", ") }, `${n.broken_links.length} unresolved`) : null)),
    h("div", { class: "insp-body" },
      n.neighbors.length ? h("section", null, h("h2", { class: "sect" }, "Neighbourhood", h("span", { class: "aside" }, `${new Set(n.neighbors.map((x) => x.id)).size} notes`)),
        egoMap(n, open)) : null,
      n.proposals.length ? h("section", null, h("h2", { class: "sect" }, "Suggested links"),
        h("ul", { class: "assoc" }, n.proposals.map((p) => {
          const other = p.payload.src_path === n.path ? p.payload.dst_title : p.payload.src_title;
          return h("li", null, dirGlyph(p.payload.src_path === n.path), h("span", { class: "nb", title: p.payload.why }, other),
            h("span", { class: "acts" },
              h("button", { class: "btn", type: "button", onclick: () => decide(p, true) }, "Add"),
              h("button", { class: "btn ghost", type: "button", onclick: () => decide(p, false) }, "Reject")));
        }))) : null,
      Object.keys(groups).length ? h("section", null, h("h2", { class: "sect" }, "Associations"),
        Object.entries(groups).map(([kind, items]) => [
          h("div", { class: "assoc-kind" }, `${KIND_LABEL[kind] || kind} · ${items.length}`),
          h("ul", { class: "assoc" }, items.slice(0, 10).map((x) => h("li", null, dirGlyph(x.forward),
            h("button", { class: "nb", type: "button", onclick: () => open(x.id) }, x.title), h("span", { class: "w" }, fmt.n(x.w, 2))))),
        ])) : null,
      h("section", null, h("h2", { class: "sect" }, "Note"),
        n.content_omitted ? h("p", { class: "omitted" }, MODE === "snapshot"
          ? "This snapshot leaves note text out on purpose; it shows structure, scores and short recall excerpts. The local app shows the full note."
          : "Note text omitted.")
          : renderMarkdown(n.content, { stems: n.stems, onLink: open }))));
  panel.scrollTop = 0;
  ctx.onNoteShown?.(n.id);
}
