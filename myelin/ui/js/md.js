// A small markdown renderer that builds DOM nodes directly. It never parses HTML,
// so raw `<script>` or `<img onerror>` in a note renders as visible text.
// Covers what vault notes use: frontmatter, headings, lists, tables, quotes,
// fences, inline code, emphasis, links and [[wikilinks]].
import { h } from "./util.js";

const SAFE_URL = /^(https?:|mailto:)/i;

export function renderMarkdown(src, { stems = {}, onLink } = {}) {
  const root = h("div", { class: "md" });
  let text = src.replace(/\r\n?/g, "\n");
  const fm = text.match(/^---\n([\s\S]*?)\n---\n?/);
  if (fm) {
    root.appendChild(h("div", { class: "fm" }, fm[1]));
    text = text.slice(fm[0].length);
  }
  const lines = text.split("\n");
  let i = 0;
  const inline = (s) => renderInline(s, stems, onLink);

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    const fence = line.match(/^\s*(```+|~~~+)(.*)$/);
    if (fence) {
      const body = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence[1])) body.push(lines[i++]);
      i++;
      root.appendChild(h("pre", null, h("code", null, body.join("\n"))));
      continue;
    }
    const head = line.match(/^(#{1,6})\s+(.*)$/);
    if (head) {
      const level = Math.min(4, head[1].length);
      root.appendChild(h(`h${level}`, null, inline(head[2])));
      i++;
      continue;
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { root.appendChild(h("hr")); i++; continue; }
    if (line.startsWith(">")) {
      const body = [];
      while (i < lines.length && lines[i].startsWith(">")) body.push(lines[i++].replace(/^>\s?/, ""));
      root.appendChild(h("blockquote", null, inline(body.join(" "))));
      continue;
    }
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const header = cells(line);
      i += 2;
      const body = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) body.push(cells(lines[i++]));
      root.appendChild(h("table", null,
        h("thead", null, h("tr", null, header.map((c) => h("th", null, inline(c))))),
        h("tbody", null, body.map((r) => h("tr", null, r.map((c) => h("td", null, inline(c))))))));
      continue;
    }
    const item = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (item) {
      const ordered = /\d/.test(item[2]);
      const list = h(ordered ? "ol" : "ul");
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
        if (!m) {
          if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && list.lastChild) {
            list.lastChild.append(" ", ...inline(lines[i].trim()));
            i++;
            continue;
          }
          break;
        }
        const content = m[3].replace(/^\[( |x)\]\s+/i, (_, x) => (x.trim() ? "☑ " : "☐ "));
        const li = h("li", null, inline(content));
        if (m[1].length >= 2 && list.lastChild) {
          let sub = list.lastChild.querySelector(":scope > ul, :scope > ol");
          if (!sub) { sub = h(ordered ? "ol" : "ul"); list.lastChild.appendChild(sub); }
          sub.appendChild(li);
        } else {
          list.appendChild(li);
        }
        i++;
      }
      root.appendChild(list);
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|>|\s*```|\s*~~~|\s*([-*+]|\d+[.)])\s|\s*\|)/.test(lines[i])) {
      para.push(lines[i++]);
    }
    if (!para.length) para.push(lines[i++]);
    root.appendChild(h("p", null, inline(para.join(" "))));
  }
  return root;
}

const TOKEN = /(`+)([^`]+?)\1|!?\[\[([^\]]+?)\]\]|\[([^\]]+)\]\(([^)\s]+)\)|\*\*([^*]+)\*\*|__([^_]+)__|\*([^*\s][^*]*?)\*|~~([^~]+)~~/g;

export function renderInline(text, stems = {}, onLink) {
  const out = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1]) out.push(h("code", null, m[2]));
    else if (m[3]) {
      const [target, alias] = m[3].split("|");
      const stem = target.split(/[#^]/)[0].trim();
      const name = stem.split("/").pop().replace(/\.md$/, "");
      // Own keys only: a note may well link to [[constructor]] or [[__proto__]].
      const id = Object.hasOwn(stems, name) ? stems[name] : Object.hasOwn(stems, stem) ? stems[stem] : undefined;
      const label = (alias || target).trim();
      if (id !== undefined) out.push(h("button", { class: "wl", type: "button", onclick: () => onLink?.(id) }, label));
      else out.push(h("span", { class: "wl broken", title: `No note named “${stem}” in scope` }, label));
    } else if (m[4]) {
      out.push(SAFE_URL.test(m[5]) ? h("a", { href: m[5], target: "_blank", rel: "noopener noreferrer" }, m[4]) : m[4]);
    } else if (m[6] || m[7]) out.push(h("strong", null, m[6] || m[7]));
    else if (m[8]) out.push(h("em", null, m[8]));
    else if (m[9]) out.push(h("s", null, m[9]));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
