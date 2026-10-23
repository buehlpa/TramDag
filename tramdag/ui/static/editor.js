// TRAM-DAG visual configuration editor.
// Runs as an anywidget front-end (Jupyter / VS Code / Colab) and in the standalone
// browser app (tramdag/ui/server.py), which provides a model object with the same API.
// The model holds three values: "spec" (graph, edited here), "report" (validation result
// computed in Python) and "options" (data types, terms, networks, paths).

const NODE_W = 112;
const NODE_H = 42;
const TERM_LABEL = { ls: "LS", cs: "CS", ci: "CI" };
const TERM_NAME = { ls: "Linear shift", cs: "Complex shift", ci: "Complex intercept" };
const TERM_HELP = {
  ls: "β·x — one interpretable coefficient (log-odds ratio scale).",
  cs: "f(x) — a neural network shift; use a group to model interactions f(x1, x2).",
  ci: "The whole transformation function depends on x (non-additive). One per node.",
};
const DEFAULT_MODEL = { ls: "LinearShift", cs: "ComplexShiftDefaultTabular", ci: "ComplexInterceptDefaultTabular" };
const SVGNS = "http://www.w3.org/2000/svg";

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c) el.append(c);
  return el;
}

const clone = (x) => JSON.parse(JSON.stringify(x));
const needsLevels = (dt) => /yo|xn/i.test(dt || "");
const isOrdinal = (dt) => /ordinal/i.test(dt || "");

function render({ model, el }) {
  const uid = Math.random().toString(36).slice(2, 8);
  let spec = clone(model.get("spec") || { nodes: [], edges: [], training: {} });
  let report = model.get("report") || {};
  const options = model.get("options") || {};
  let selection = null; // {kind: "node"|"edge", id}
  let drag = null; // node drag or link drag state
  let lastPushed = JSON.stringify(spec);

  el.classList.add("tde");
  el.tabIndex = 0;

  // ------------------------------------------------------------------ layout
  const status = h("span", { class: "tde-status" });
  const saveInfo = h("span", { class: "tde-saveinfo" });
  const header = h(
    "div",
    { class: "tde-header" },
    h("div", { class: "tde-title" }, h("strong", {}, "TRAM-DAG"), h("span", { class: "tde-exp" }, options.experiment_name || "")),
    status,
    h("div", { class: "tde-spacer" }),
    saveInfo,
    h("button", { class: "tde-btn", title: "Arrange nodes by causal order", onclick: () => send({ type: "auto_layout", width: canvasWidth(), height: canvasHeight() }) }, "Auto layout"),
    h("button", { class: "tde-btn", title: "Discard unsaved changes and reload configuration.json", onclick: () => send({ type: "reload" }) }, "Reload"),
    h("button", { class: "tde-btn tde-primary", title: "Validate and write configuration.json", onclick: () => send({ type: "save" }) }, "Save")
  );
  const svg = s("svg", { class: "tde-canvas", role: "img", "aria-label": "DAG canvas" });
  const defs = s(
    "defs",
    {},
    ...["ls", "cs", "ci", "tmp"].map((t) =>
      s("marker", { id: `tde-arrow-${t}-${uid}`, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" },
        s("path", { d: "M 0 0 L 10 5 L 0 10 z", class: `tde-arrowhead tde-${t}` }))
    )
  );
  const gEdges = s("g");
  const gChips = s("g"); // labels above all edge lines so every label stays clickable
  const gNodes = s("g");
  const gTmp = s("g");
  svg.append(defs, gEdges, gChips, gNodes, gTmp);
  const hint = h("div", { class: "tde-hint" }, "Drag from ", h("span", { class: "tde-dot" }), " onto a node to add an edge · click an edge or node to edit it · Delete removes the selection");
  const canvasWrap = h("div", { class: "tde-canvas-wrap" }, svg, hint);
  const inspector = h("div", { class: "tde-inspector" });
  const footer = h("div", { class: "tde-footer" });
  el.replaceChildren(header, h("div", { class: "tde-main" }, canvasWrap, inspector), footer);
  const height = model.get("height") || 520;
  canvasWrap.style.height = `${height}px`;

  function canvasWidth() { return canvasWrap.clientWidth || 640; }
  function canvasHeight() { return canvasWrap.clientHeight || height; }

  // ------------------------------------------------------------------ sync
  function push() {
    const txt = JSON.stringify(spec);
    if (txt === lastPushed) return;
    lastPushed = txt;
    model.set("spec", clone(spec));
    model.save_changes();
  }
  function send(msg) { model.send(msg); }

  model.on("change:spec", () => {
    const incoming = model.get("spec");
    const txt = JSON.stringify(incoming);
    if (txt === JSON.stringify(spec)) return;
    spec = clone(incoming);
    lastPushed = txt;
    if (selection && !findSel()) selection = null;
    draw();
  });
  model.on("change:report", () => {
    report = model.get("report") || {};
    if (!drag) drawGraph();
    drawStatus();
    drawFooter();
    drawInspector();
  });

  // ------------------------------------------------------------------ helpers
  const nodeByName = (n) => spec.nodes.find((x) => x.name === n);
  const edgeId = (e) => `${e.source}→${e.target}`;
  function findSel() {
    if (!selection) return null;
    if (selection.kind === "node") return nodeByName(selection.id);
    return spec.edges.find((e) => edgeId(e) === selection.id);
  }
  function toLocal(evt) {
    const r = svg.getBoundingClientRect();
    return { x: evt.clientX - r.left, y: evt.clientY - r.top };
  }
  function nodeAt(p) {
    return spec.nodes.find((n) => Math.abs(p.x - n.x) <= NODE_W / 2 && Math.abs(p.y - n.y) <= NODE_H / 2);
  }
  function clip(from, to) {
    const dx = to.x - from.x, dy = to.y - from.y;
    if (!dx && !dy) return { x: to.x, y: to.y };
    const t = Math.min((NODE_W / 2 + 3) / Math.abs(dx || 1e-9), (NODE_H / 2 + 3) / Math.abs(dy || 1e-9));
    return { x: to.x - dx * Math.min(t, 1), y: to.y - dy * Math.min(t, 1) };
  }
  function segmentHitsNode(a, b, o) {
    for (let i = 1; i < 24; i++) {
      const t = i / 24, x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t;
      if (Math.abs(x - o.x) <= NODE_W / 2 + 4 && Math.abs(y - o.y) <= NODE_H / 2 + 4) return true;
    }
    return false;
  }
  function groupsOf(target, term) {
    const g = {};
    for (const e of spec.edges) if (e.target === target && e.term === term && e.group) (g[e.group] ||= []).push(e.source);
    return g;
  }
  function select(sel) { selection = sel; draw(); el.focus({ preventScroll: true }); }

  // ------------------------------------------------------------------ drawing
  function draw() {
    drawGraph();
    drawInspector();
    drawStatus();
    drawFooter();
  }

  function drawGraph() {
    gEdges.replaceChildren();
    gChips.replaceChildren();
    gNodes.replaceChildren();
    hint.style.display = spec.edges.length > 1 ? "none" : "";
    const errNodes = new Set();
    for (const msg of report.errors || []) for (const n of spec.nodes) if (msg.includes(`'${n.name}'`) || msg.includes(`${n.name} ->`) || msg.includes(`-> ${n.name}`)) errNodes.add(n.name);

    for (const e of spec.edges) {
      const a = nodeByName(e.source), b = nodeByName(e.target);
      if (!a || !b) continue;
      const reverse = spec.edges.some((o) => o.source === e.target && o.target === e.source);
      const blocked = spec.nodes.some((o) => o !== a && o !== b && segmentHitsNode(a, b, o));
      let start = clip(b, a), end = clip(a, b);
      let d, mid;
      if (reverse || blocked) {
        const bend = blocked ? Math.max(48, NODE_H * 1.4) : 26;
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        const nx = -(b.y - a.y), ny = b.x - a.x, len = Math.hypot(nx, ny) || 1;
        const cx = mx + (nx / len) * bend * 2, cy = my + (ny / len) * bend * 2;
        start = clip({ x: cx, y: cy }, a);
        end = clip({ x: cx, y: cy }, b);
        d = `M ${start.x} ${start.y} Q ${cx} ${cy} ${end.x} ${end.y}`;
        mid = { x: (start.x + 2 * cx + end.x) / 4, y: (start.y + 2 * cy + end.y) / 4 };
      } else {
        d = `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
        mid = { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 };
      }
      const selected = selection && selection.kind === "edge" && selection.id === edgeId(e);
      const label = TERM_LABEL[e.term] + (e.group ? `·${e.group}` : "");
      const lw = 12 + label.length * 7.2;
      const pick = (evt) => { evt.stopPropagation(); select({ kind: "edge", id: edgeId(e) }); };
      const cls = `tde-edge tde-${e.term}${selected ? " tde-selected" : ""}`;
      const tip = `${e.source} → ${e.target}: ${TERM_NAME[e.term]}${e.group ? ` (group ${e.group})` : ""}${e.model ? ` · ${e.model}` : ""}`;
      gEdges.append(s(
        "g",
        { class: cls, onpointerdown: pick },
        s("path", { d, class: "tde-edge-hit" }),
        s("path", { d, class: "tde-edge-line", "marker-end": `url(#tde-arrow-${e.term}-${uid})` }),
        s("title", {}, document.createTextNode(tip))
      ));
      gChips.append(s(
        "g",
        { class: cls, onpointerdown: pick },
        s("rect", { x: mid.x - lw / 2, y: mid.y - 10, width: lw, height: 20, rx: 10, class: "tde-chip" }),
        s("text", { x: mid.x, y: mid.y + 4, "text-anchor": "middle", class: "tde-chip-text" }, document.createTextNode(label)),
        s("title", {}, document.createTextNode(tip))
      ));
    }

    for (const n of spec.nodes) {
      const selected = selection && selection.kind === "node" && selection.id === n.name;
      const ord = isOrdinal(n.data_type);
      const badge = ord ? `ord${n.levels ? "·" + n.levels : ""}` : "cont";
      const g = s(
        "g",
        { class: `tde-node${ord ? " tde-ordinal" : ""}${selected ? " tde-selected" : ""}${errNodes.has(n.name) ? " tde-error" : ""}`, transform: `translate(${n.x - NODE_W / 2}, ${n.y - NODE_H / 2})` },
        s("rect", { width: NODE_W, height: NODE_H, rx: 10, class: "tde-node-box" }),
        s("text", { x: 12, y: 18, class: "tde-node-name" }, document.createTextNode(n.name.length > 13 ? n.name.slice(0, 12) + "…" : n.name)),
        s("text", { x: 12, y: 33, class: "tde-node-type" }, document.createTextNode(badge)),
        s("circle", { cx: NODE_W, cy: NODE_H / 2, r: 6, class: "tde-port" })
      );
      g.append(s("title", {}, document.createTextNode(`${n.name} — ${(options.data_types || {})[n.data_type] || n.data_type}`)));
      g.addEventListener("pointerdown", (evt) => {
        evt.stopPropagation();
        const p = toLocal(evt);
        if (evt.target.classList.contains("tde-port")) {
          drag = { kind: "link", from: n.name, x: p.x, y: p.y };
        } else {
          drag = { kind: "move", name: n.name, dx: p.x - n.x, dy: p.y - n.y, moved: false };
          selection = { kind: "node", id: n.name };
        }
        svg.setPointerCapture(evt.pointerId);
        draw();
      });
      gNodes.append(g);
    }
  }

  svg.addEventListener("pointermove", (evt) => {
    if (!drag) return;
    const p = toLocal(evt);
    if (drag.kind === "move") {
      const n = nodeByName(drag.name);
      n.x = Math.max(NODE_W / 2 + 2, Math.min(canvasWidth() - NODE_W / 2 - 2, p.x - drag.dx));
      n.y = Math.max(NODE_H / 2 + 2, Math.min(canvasHeight() - NODE_H / 2 - 2, p.y - drag.dy));
      drag.moved = true;
      drawGraph();
    } else if (drag.kind === "link") {
      const a = nodeByName(drag.from);
      const over = nodeAt(p);
      gTmp.replaceChildren(
        s("line", { x1: a.x + NODE_W / 2, y1: a.y, x2: p.x, y2: p.y, class: `tde-temp${over && over.name !== drag.from ? " tde-temp-ok" : ""}`, "marker-end": `url(#tde-arrow-tmp-${uid})` })
      );
    }
  });
  svg.addEventListener("pointerup", (evt) => {
    if (!drag) return;
    const p = toLocal(evt);
    if (drag.kind === "link") {
      gTmp.replaceChildren();
      const target = nodeAt(p);
      if (target && target.name !== drag.from) {
        let e = spec.edges.find((x) => x.source === drag.from && x.target === target.name);
        if (!e) {
          e = { source: drag.from, target: target.name, term: "ls", group: null, model: null };
          spec.edges.push(e);
        }
        selection = { kind: "edge", id: edgeId(e) };
        push();
      }
    } else if (drag.kind === "move" && drag.moved) {
      push();
    }
    drag = null;
    draw();
  });
  svg.addEventListener("pointerdown", () => select(null));

  el.addEventListener("keydown", (evt) => {
    if (!selection) return;
    const tag = (evt.target.tagName || "").toLowerCase();
    if (["input", "select", "textarea"].includes(tag)) return;
    if (evt.key === "Delete" || evt.key === "Backspace") {
      evt.preventDefault();
      deleteSelection();
    }
  });

  function deleteSelection() {
    if (!selection) return;
    if (selection.kind === "edge") spec.edges = spec.edges.filter((e) => edgeId(e) !== selection.id);
    else {
      spec.nodes = spec.nodes.filter((n) => n.name !== selection.id);
      spec.edges = spec.edges.filter((e) => e.source !== selection.id && e.target !== selection.id);
    }
    selection = null;
    push();
    draw();
  }

  // ------------------------------------------------------------------ inspector
  function field(label, control, help) {
    return h("label", { class: "tde-field" }, h("span", { class: "tde-label" }, label), control, help ? h("span", { class: "tde-help" }, help) : null);
  }
  function selectEl(value, choices, onchange) {
    const sel = h("select", { onchange: (e) => onchange(e.target.value) });
    for (const [v, label] of choices) sel.append(h("option", { value: v, selected: String(v) === String(value) }, label));
    return sel;
  }

  function modelChoices(role, current) {
    const models = (options.models || {})[role] || {};
    const list = Object.entries(models).map(([k, d]) => [k, d ? `${k} — ${d}` : k]);
    if (current && !models[current]) list.push([current, current]);
    list.push(["__custom__", "Custom… (e.g. MLP(32,32,tanh))"]);
    return list;
  }

  function drawInspector() {
    const item = findSel();
    inspector.replaceChildren();
    if (selection && selection.kind === "edge" && item) return edgeInspector(item);
    if (selection && selection.kind === "node" && item) return nodeInspector(item);
    graphInspector();
  }

  function edgeInspector(e) {
    const termRadios = h("div", { class: "tde-terms" });
    for (const t of ["ls", "cs", "ci"]) {
      termRadios.append(
        h("label", { class: `tde-term tde-${t}${e.term === t ? " tde-on" : ""}` },
          h("input", { type: "radio", name: `term-${uid}`, value: t, checked: e.term === t, onchange: () => { e.term = t; if (t === "ls") e.group = null; e.model = null; push(); draw(); } }),
          h("span", { class: "tde-term-name" }, TERM_NAME[t]),
          h("span", { class: "tde-help" }, TERM_HELP[t]))
      );
    }
    const parts = [
      h("div", { class: "tde-insp-title" }, `${e.source} → ${e.target}`),
      field("Term", termRadios),
    ];
    if (e.term !== "ls") {
      const groups = groupsOf(e.target, e.term);
      const choices = [["", "none (own network)"]];
      const ids = Object.keys(groups).map(Number).sort((a, b) => a - b);
      for (const g of ids) choices.push([g, `group ${g}: ${groups[g].join(", ")}`]);
      const next = [1, 2, 3, 4, 5, 6, 7, 8, 9].find((g) => !ids.includes(g));
      if (next) choices.push([next, `new group ${next}`]);
      parts.push(field("Interaction", selectEl(e.group || "", choices, (v) => { e.group = v ? Number(v) : null; push(); draw(); }),
        e.term === "cs" ? "Inputs in the same group share one network f(x1, x2, …)." : "All inputs of a complex intercept must be in one group."));
    }
    if (e.term === "ls") {
      parts.push(field("Network", h("div", { class: "tde-mono" }, "LinearShift")));
    } else {
      const role = e.term === "ci" ? "intercept" : "shift";
      const current = e.model || DEFAULT_MODEL[e.term];
      const custom = h("input", { type: "text", placeholder: "MLP(64,64,relu)", value: e.model && !((options.models || {})[role] || {})[e.model] ? e.model : "", onchange: (ev) => { e.model = ev.target.value.trim() || null; push(); } });
      custom.style.display = "none";
      const sel = selectEl(current, modelChoices(role, e.model), (v) => {
        if (v === "__custom__") { custom.style.display = ""; custom.focus(); return; }
        e.model = v === DEFAULT_MODEL[e.term] ? null : v;
        push();
        draw();
      });
      parts.push(field("Network", h("div", {}, sel, custom), e.group ? "The first input's network is used for the whole group." : null));
    }
    parts.push(
      h("div", { class: "tde-row" },
        h("button", { class: "tde-btn", onclick: () => { const r = { ...e, source: e.target, target: e.source }; spec.edges = spec.edges.filter((x) => x !== e); spec.edges.push(r); selection = { kind: "edge", id: edgeId(r) }; push(); draw(); } }, "Reverse"),
        h("button", { class: "tde-btn tde-danger", onclick: deleteSelection }, "Delete edge"))
    );
    inspector.append(...parts);
  }

  function nodeInspector(n) {
    const dtChoices = Object.entries(options.data_types || { continous: "continuous" });
    const parts = [
      h("div", { class: "tde-insp-title" }, n.name),
      field("Data type", selectEl(n.data_type, dtChoices, (v) => { n.data_type = v; if (!needsLevels(v)) n.levels = isOrdinal(v) ? n.levels : null; push(); draw(); }),
        "'one-hot as input' = parents see this variable as categories; 'ordinal as target' = ordinal likelihood."),
    ];
    if (isOrdinal(n.data_type)) {
      parts.push(field("Levels", h("input", { type: "number", min: 2, step: 1, value: n.levels ?? "", onchange: (ev) => { n.levels = ev.target.value ? Number(ev.target.value) : null; push(); draw(); } }), "Values must be coded 0 … levels-1."));
    }
    const f = (report.formulas || {})[n.name];
    if (f) parts.push(field("Transformation function", h("div", { class: "tde-mono" }, f)));
    const parents = spec.edges.filter((e) => e.target === n.name).map((e) => e.source);
    const children = spec.edges.filter((e) => e.source === n.name).map((e) => e.target);
    parts.push(field("Parents / children", h("div", {}, `${parents.join(", ") || "—"}  /  ${children.join(", ") || "—"}`)));
    parts.push(h("div", { class: "tde-row" }, h("button", { class: "tde-btn tde-danger", onclick: deleteSelection }, "Delete node")));
    inspector.append(...parts);
  }

  function graphInspector() {
    const tr = (spec.training ||= {});
    const num = (key, step) => h("input", { type: "number", step, value: tr[key] ?? "", onchange: (ev) => { tr[key] = ev.target.value === "" ? null : Number(ev.target.value); push(); } });
    const nameIn = h("input", { type: "text", placeholder: "new node name" });
    const typeSel = selectEl("continous", Object.entries(options.data_types || {}), () => {});
    const text = h("textarea", { rows: 4, placeholder: "x1 -> x2\nx1, x2 -> x3 [cs]\n(x1, x2) -> x4 [cs]" });
    inspector.append(
      h("div", { class: "tde-insp-title" }, "Experiment"),
      field("Configuration file", h("div", { class: "tde-mono tde-path" }, options.config_path || "(not written yet)")),
      h("div", { class: "tde-insp-title" }, "Training defaults"),
      h("div", { class: "tde-grid3" }, field("Epochs", num("epochs", 1)), field("Learning rate", num("learning_rate", "any")), field("Batch size", num("batch_size", 1))),
      h("div", { class: "tde-insp-title" }, "Add node"),
      h("div", { class: "tde-row" }, nameIn, typeSel,
        h("button", { class: "tde-btn", onclick: () => {
          const name = nameIn.value.trim();
          if (!name || nodeByName(name)) return;
          spec.nodes.push({ name, data_type: typeSel.value, levels: needsLevels(typeSel.value) ? 2 : null, x: 70 + Math.random() * 60, y: 50 + Math.random() * 60 });
          selection = { kind: "node", id: name };
          push();
          draw();
        } }, "Add")),
      h("div", { class: "tde-insp-title" }, "Type edges"),
      text,
      h("div", { class: "tde-row" }, h("button", { class: "tde-btn", onclick: () => { if (text.value.trim()) send({ type: "edges_text", text: text.value }); } }, "Apply"),
        h("span", { class: "tde-help" }, "Same syntax as TramDagConfig.from_dataframe(edges=…)."))
    );
  }

  // ------------------------------------------------------------------ status / footer
  function drawStatus() {
    const ne = (report.errors || []).length, nw = (report.warnings || []).length;
    status.className = `tde-status ${ne ? "tde-bad" : "tde-good"}`;
    status.textContent = ne ? `${ne} error${ne > 1 ? "s" : ""}` : nw ? `valid · ${nw} warning${nw > 1 ? "s" : ""}` : "valid";
    if (report.dirty) saveInfo.textContent = "unsaved changes";
    else if (report.saved_at) saveInfo.textContent = `saved ${report.saved_at}`;
    else saveInfo.textContent = "";
    saveInfo.title = report.saved_path || options.config_path || "";
  }

  function drawFooter() {
    footer.replaceChildren();
    const formulas = report.formulas || {};
    const fl = h("div", { class: "tde-formulas" });
    for (const n of spec.nodes) if (formulas[n.name]) fl.append(h("div", { class: "tde-mono", onclick: () => select({ kind: "node", id: n.name }) }, formulas[n.name]));
    footer.append(fl);
    const msgs = h("div", { class: "tde-msgs" });
    for (const m of report.errors || []) msgs.append(h("div", { class: "tde-msg tde-msg-error" }, m));
    for (const m of report.warnings || []) msgs.append(h("div", { class: "tde-msg tde-msg-warn" }, m));
    if (msgs.childElementCount) footer.append(msgs);
  }

  draw();
  return () => { el.replaceChildren(); };
}

export default { render };
