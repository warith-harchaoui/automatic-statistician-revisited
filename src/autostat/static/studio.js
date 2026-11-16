// The autostat studio: load a series, say what you know, search, read.
// Every visible string comes from locales/i18n.yaml through /i18n.json.
// Charts are hand-drawn SVG: no chart library, nothing loaded from elsewhere.
//
// Author: Warith Harchaoui <warithmetics@deraison.ai>

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";

/** Colours of the components, in order (deraison.ai palette). */
const PART_COLOURS = ["#0088FF", "#FF8D28", "#00C3D0", "#CB30E0", "#34C759", "#FF2D55", "#AC7F5E"];

const state = {
  strings: {},
  lang: "en",
  source: "file",
  dataset: null,
  series: null,
  run: null,
  result: null,
  timer: null,
};

// ---------------------------------------------------------------- strings

/**
 * Translate a key, filling {name} placeholders.
 *
 * @param {string} key Message id of locales/i18n.yaml.
 * @param {Object<string, (string|number)>} [params] Placeholder values.
 * @returns {string} The text, or the key when it is missing.
 */
function t(key, params = {}) {
  const text = state.strings[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, name) => String(params[name] ?? `{${name}}`));
}

/**
 * Load the strings of one language and apply them to the page.
 *
 * @param {string} lang Two-letter code.
 */
async function loadStrings(lang) {
  const response = await fetch(`/i18n.json?lang=${encodeURIComponent(lang)}`);
  state.strings = await response.json();
  state.lang = lang;
  document.documentElement.lang = lang;
  for (const el of document.querySelectorAll("[data-i18n]")) {
    el.textContent = t(el.dataset.i18n);
  }
  for (const el of document.querySelectorAll("[data-i18n-label]")) {
    el.setAttribute("aria-label", t(el.dataset.i18nLabel));
  }
  if (state.result) showResult(state.result);
  else if (state.series) {
    showTitle();
    drawMain();
  }
}

// ---------------------------------------------------------------- numbers

/**
 * Format a number for a label, with at most four significant digits.
 *
 * @param {number} v The number.
 * @returns {string} Its text in the page's language.
 */
function fmt(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "n/a";
  if (Math.abs(v) < 5e-4) v = 0;
  const digits = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 1 ? 2 : 3;
  return new Intl.NumberFormat(state.lang, { maximumFractionDigits: digits }).format(v + 0);
}

/**
 * Turn a decimal year into a Date, on the 365-day year of the reports.
 *
 * @param {number} x Decimal year.
 * @returns {Date} The moment.
 */
function toDate(x) {
  return new Date(Date.UTC(1970, 0, 1) + (x - 1970) * 365 * 86400000);
}

/**
 * Read a date or a number typed by the reader as a position on the time axis.
 *
 * @param {string} text E.g. "2024-11-05" or "1973.5".
 * @returns {number|null} The decimal year, or null when unreadable.
 */
function toAxis(text) {
  if (/^-?\d+(\.\d+)?$/.test(text.trim())) return Number(text);
  const d = Date.parse(text);
  return Number.isNaN(d) ? null : 1970 + (d - Date.UTC(1970, 0, 1)) / (365 * 86400000);
}

/**
 * Choose about n round tick values between lo and hi.
 *
 * @param {number} lo Lowest value.
 * @param {number} hi Highest value.
 * @param {number} n Wanted number of ticks.
 * @returns {number[]} Tick values.
 */
function ticks(lo, hi, n) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

/**
 * Label a tick of the time axis in the unit the span calls for.
 *
 * @param {number} x Position (decimal year when unit is "year").
 * @param {number} span Width of the axis.
 * @param {string} unit "year" or "".
 * @returns {string} The label.
 */
function timeLabel(x, span, unit) {
  if (unit !== "year") return fmt(x);
  if (span >= 4) return String(Math.round(x));
  const d = toDate(x);
  const options = span >= 0.4 ? { month: "short", year: "numeric" } : { month: "short", day: "numeric" };
  if (span < 0.02) Object.assign(options, { hour: "2-digit" });
  return new Intl.DateTimeFormat(state.lang, { ...options, timeZone: "UTC" }).format(d);
}

// ---------------------------------------------------------------- charts

/**
 * Create an SVG element with attributes.
 *
 * @param {string} tag Element name.
 * @param {Object<string, (string|number)>} attrs Attributes.
 * @returns {SVGElement} The element.
 */
function el(tag, attrs = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

/**
 * Draw a time-series chart: optional band, lines, points and markers.
 *
 * @param {HTMLElement} host Where the SVG goes (its size sets the chart's).
 * @param {Object} spec What to draw.
 * @param {number[]} spec.x Positions of points and lines unless a line has its own.
 * @param {Array<{x:number[], y:number[], colour:string, width?:number}>} [spec.lines]
 * @param {Array<{x:number[], lo:number[], hi:number[], colour:string}>} [spec.bands]
 * @param {{x:number[], y:number[]}} [spec.points]
 * @param {number} [spec.shadeFrom] Shade the region right of this position.
 * @param {Array<{x:number, label:string}>} [spec.markers] Vertical dashed lines.
 * @param {string} spec.unit "year" or "".
 * @param {string} spec.label Accessible description.
 * @param {boolean} [spec.small] Fewer ticks, smaller margins.
 * @param {boolean} [spec.integerX] One tick per whole number on the x axis.
 */
function chart(host, spec) {
  const width = Math.max(host.clientWidth, 240);
  const height = Math.max(host.clientHeight, 120);
  const m = spec.small ? { l: 44, r: 8, t: 8, b: 22 } : { l: 56, r: 12, t: 12, b: 28 };
  const xs = [...(spec.points?.x ?? []), ...(spec.lines ?? []).flatMap((l) => l.x), ...(spec.bands ?? []).flatMap((b) => b.x)];
  const ys = [
    ...(spec.points?.y ?? []),
    ...(spec.lines ?? []).flatMap((l) => l.y),
    ...(spec.bands ?? []).flatMap((b) => [...b.lo, ...b.hi]),
  ].filter(Number.isFinite);
  if (!xs.length || !ys.length) return;
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const pad = (y1 - y0 || 1) * 0.06;
  y0 -= pad;
  y1 += pad;
  if (x1 === x0) x1 = x0 + 1;
  const sx = (v) => m.l + ((v - x0) / (x1 - x0)) * (width - m.l - m.r);
  const sy = (v) => height - m.b - ((v - y0) / (y1 - y0)) * (height - m.t - m.b);
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: "100%", role: "img", class: "chart" });
  const title = el("title");
  title.textContent = spec.label;
  svg.append(title);

  if (spec.shadeFrom !== undefined && spec.shadeFrom < x1) {
    svg.append(el("rect", { x: sx(spec.shadeFrom), y: m.t, width: sx(x1) - sx(spec.shadeFrom), height: height - m.t - m.b, class: "fill-fill dark:fill-fill-dark" }));
  }
  for (const v of ticks(y0, y1, spec.small ? 3 : 5)) {
    svg.append(el("line", { x1: m.l, x2: width - m.r, y1: sy(v), y2: sy(v), class: "axis", "stroke-width": 1 }));
    const label = el("text", { x: m.l - 6, y: sy(v) + 4, "text-anchor": "end" });
    label.textContent = fmt(v);
    svg.append(label);
  }
  const span = x1 - x0;
  const xTicks = spec.integerX
    ? Array.from({ length: Math.round(x1 - x0) + 1 }, (_, i) => x0 + i)
    : ticks(x0, x1, spec.small ? 3 : Math.max(3, Math.floor(width / 110)));
  for (const v of xTicks) {
    const label = el("text", { x: sx(v), y: height - 8, "text-anchor": "middle" });
    label.textContent = timeLabel(v, span, spec.unit);
    svg.append(label);
  }
  for (const band of spec.bands ?? []) {
    const up = band.x.map((x, i) => [x, band.hi[i]]).filter(([, y]) => Number.isFinite(y));
    const down = band.x.map((x, i) => [x, band.lo[i]]).filter(([, y]) => Number.isFinite(y)).reverse();
    const d = [...up, ...down].map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("") + "Z";
    svg.append(el("path", { d, fill: band.colour, "fill-opacity": 0.18, stroke: "none" }));
  }
  if (spec.points) {
    const dots = el("g", { class: "fill-label dark:fill-label-dark" });
    const r = spec.points.x.length > 400 ? 1.2 : 2;
    spec.points.x.forEach((x, i) => {
      if (Number.isFinite(spec.points.y[i])) dots.append(el("circle", { cx: sx(x).toFixed(1), cy: sy(spec.points.y[i]).toFixed(1), r }));
    });
    svg.append(dots);
  }
  for (const line of spec.lines ?? []) {
    const d = line.x
      .map((x, i) => (Number.isFinite(line.y[i]) ? `${sx(x).toFixed(1)},${sy(line.y[i]).toFixed(1)}` : null))
      .filter(Boolean)
      .map((p, i) => `${i ? "L" : "M"}${p}`)
      .join("");
    svg.append(el("path", { d, fill: "none", stroke: line.colour, "stroke-width": line.width ?? 2, "stroke-linejoin": "round" }));
  }
  for (const marker of spec.markers ?? []) {
    if (marker.x < x0 || marker.x > x1) continue;
    svg.append(el("line", { x1: sx(marker.x), x2: sx(marker.x), y1: m.t, y2: height - m.b, stroke: "#FF8D28", "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
    const label = el("text", { x: sx(marker.x) + 4, y: m.t + 12 });
    label.textContent = marker.label;
    svg.append(label);
  }
  host.replaceChildren(svg);
}

/**
 * Markers of the dated events typed in the context box.
 *
 * @returns {Array<{x:number, label:string}>} One per readable date.
 */
function eventMarkers() {
  return readRows("events")
    .map((e) => ({ x: toAxis(e.at), label: e.name || "" }))
    .filter((e) => e.x !== null);
}

/** Draw the main chart: the data alone, or the data with the fit. */
function drawMain() {
  const r = state.result;
  const s = r ?? state.series;
  if (!s) return;
  const spec = {
    unit: s.unit,
    points: { x: s.x, y: s.y },
    markers: eventMarkers(),
    label: t("studio.chartLabel", { name: $("name").value || "data" }),
  };
  if (r) {
    const last = Math.max(...r.x);
    Object.assign(spec, {
      bands: [{ x: r.grid, lo: r.lower, hi: r.upper, colour: "#0088FF" }],
      lines: [{ x: r.grid, y: r.mean, colour: "#0088FF" }],
      shadeFrom: last,
    });
  }
  chart($("chart"), spec);
  $("legend").hidden = !r;
}

new ResizeObserver(() => {
  if (state.result) showResult(state.result);
  else drawMain();
}).observe($("chart"));

// ---------------------------------------------------------------- data

/**
 * Show a message under the data step.
 *
 * @param {string} text The message.
 */
function dataStatus(text) {
  $("data-status").textContent = text;
}

/**
 * Fill a select with options.
 *
 * @param {HTMLSelectElement} select The control.
 * @param {Array<{value:string, label:string}>} options Its options.
 * @param {string|null} chosen The value to select.
 */
function fill(select, options, chosen) {
  select.replaceChildren(
    ...options.map((o) => {
      const option = new Option(o.label, o.value);
      option.selected = o.value === chosen;
      return option;
    }),
  );
}

/**
 * Upload a file and describe its columns.
 *
 * @param {File} file The file the reader chose or dropped.
 */
async function upload(file) {
  $("file-name").textContent = file.name;
  dataStatus(t("studio.reading"));
  const response = await fetch(`/api/datasets?filename=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
  const body = await response.json();
  if (!response.ok) {
    dataStatus(body.error);
    return;
  }
  state.dataset = body;
  const numbers = body.columns.filter((c) => c.kind === "number").map((c) => ({ value: c.name, label: c.name }));
  const times = body.columns.filter((c) => c.kind !== "text").map((c) => ({ value: c.name, label: c.name }));
  fill($("time"), [{ value: "", label: t("studio.rowNumber") }, ...times], body.time ?? "");
  fill($("value"), numbers, body.value);
  $("columns").hidden = false;
  if (!$("name").value) $("name").value = file.name.replace(/\.[^.]+$/, "");
  dataStatus(t("studio.rows", { count: body.rows.toLocaleString(state.lang) }));
  await previewColumns();
}

/** Draw the series the chosen columns make. */
async function previewColumns() {
  if (!state.dataset) return;
  const response = await fetch(`/api/datasets/${state.dataset.id}/series`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ time: $("time").value, value: $("value").value, max_points: Number($("points").value) }),
  });
  const body = await response.json();
  if (!response.ok) {
    dataStatus(body.error);
    state.series = null;
  } else {
    state.series = body;
    state.result = null;
    showTitle();
    drawMain();
  }
  ready();
}

/** Load the bundled series the select names. */
async function loadExample() {
  const name = $("example").value;
  const response = await fetch(`/api/examples/${encodeURIComponent(name)}`);
  const body = await response.json();
  state.series = { ...body, unit: "year" };
  state.result = null;
  $("name").value = name.replace(/^\d+-/, "").replace(/-/g, " ");
  showTitle();
  drawMain();
  ready();
}

/** Put the series' name and size above the chart. */
function showTitle() {
  const s = state.result ?? state.series;
  $("chart-title").textContent = $("name").value || t("studio.chartEmpty");
  $("chart-sub").textContent = s ? t("studio.points_n", { count: s.x.length.toLocaleString(state.lang) }) : "";
}

/** Enable the run button when there is something to search. */
function ready() {
  $("run").disabled = !state.series || state.run !== null;
}

/**
 * Switch between the file and example tabs.
 *
 * @param {string} source "file" or "example".
 */
function selectSource(source) {
  state.source = source;
  for (const [tab, panel, name] of [["tab-file", "panel-file", "file"], ["tab-example", "panel-example", "example"]]) {
    $(tab).setAttribute("aria-selected", String(name === source));
    $(panel).hidden = name !== source;
  }
  $("columns").hidden = source !== "file" || !state.dataset;
  state.series = null;
  state.result = null;
  if (source === "example") loadExample();
  else if (state.dataset) previewColumns();
  else ready();
}

// ---------------------------------------------------------------- context

/**
 * Add one editable row to the cycles or events list.
 *
 * @param {string} list "cycles" or "events".
 * @param {Object<string, string>} [values] Initial values by data-key.
 */
function addRow(list, values = {}) {
  const template = $(list === "cycles" ? "cycle-row" : "event-row");
  const row = template.content.firstElementChild.cloneNode(true);
  for (const input of row.querySelectorAll("input")) input.value = values[input.dataset.key] ?? "";
  for (const node of row.querySelectorAll("[data-i18n-label]")) node.setAttribute("aria-label", t(node.dataset.i18nLabel));
  row.querySelector(".remove").addEventListener("click", () => {
    row.remove();
    drawMain();
  });
  row.addEventListener("input", () => drawMain());
  $(list).append(row);
  row.querySelector("input").focus();
}

/**
 * Read the filled rows of a list.
 *
 * @param {string} list "cycles" or "events".
 * @returns {Array<Object<string, string>>} One object per row with a value.
 */
function readRows(list) {
  return [...$(list).children]
    .map((row) => Object.fromEntries([...row.querySelectorAll("input")].map((i) => [i.dataset.key, i.value.trim()])))
    .filter((r) => (list === "cycles" ? r.period : r.at));
}

/**
 * The context as the host reads it.
 *
 * @returns {Object} positive, cycles and events.
 */
function readContext() {
  return {
    positive: $("positive").checked,
    cycles: readRows("cycles").map((c, i) => ({ name: c.name || `cycle ${i + 1}`, period: /^\d+(\.\d+)?$/.test(c.period) ? Number(c.period) : c.period })),
    events: readRows("events").map((e, i) => ({ name: e.name || `event ${i + 1}`, at: /^-?\d+(\.\d+)?$/.test(e.at) ? Number(e.at) : e.at })),
  };
}

// ---------------------------------------------------------------- run

/** Start a search and poll it. */
async function startRun() {
  const fields = {
    name: $("name").value || "data",
    context: readContext(),
    preset: $("preset").value,
    max_depth: Number($("depth").value),
    max_points: Number($("points").value),
    n_jobs: Number($("jobs").value),
    seed: Number($("seed").value),
    check: $("check").checked,
    propose: $("propose").value.trim(),
  };
  if (state.source === "example") fields.example = $("example").value;
  else Object.assign(fields, { dataset: state.dataset.id, time: $("time").value, value: $("value").value });
  $("error").hidden = true;
  const response = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(fields) });
  const body = await response.json();
  if (!response.ok) {
    showError(body.error);
    return;
  }
  state.run = body.id;
  state.result = null;
  $("results").hidden = true;
  $("progress").hidden = false;
  $("levels").replaceChildren();
  $("progress").scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  ready();
  poll();
}

/** Ask the host how the run is going, until it is done or failed. */
async function poll() {
  let view;
  try {
    view = await (await fetch(`/api/runs/${state.run}`)).json();
  } catch (error) {
    showError(t("studio.network"));
    state.run = null;
    ready();
    return;
  }
  showProgress(view);
  if (view.status === "done") {
    state.run = null;
    state.result = view.result;
    showResult(view.result);
    ready();
  } else if (view.status === "failed") {
    state.run = null;
    showError(view.error);
    ready();
  } else {
    state.timer = setTimeout(poll, 1000);
  }
}

/**
 * Show the levels finished so far and the elapsed time.
 *
 * @param {Object} view What GET /api/runs/{id} returned.
 */
function showProgress(view) {
  const done = view.levels.filter((l) => l.depth !== "event").length;
  const share = view.status === "done" ? 1 : Math.min(done / view.max_depth, 0.97);
  $("bar-fill").style.width = `${(share * 100).toFixed(0)}%`;
  $("bar").setAttribute("aria-valuemax", "100");
  $("bar").setAttribute("aria-valuenow", (share * 100).toFixed(0));
  const minutes = Math.floor(view.elapsed / 60);
  const seconds = Math.floor(view.elapsed % 60);
  const clock = `${minutes}:${String(seconds).padStart(2, "0")}`;
  const key = { queued: "studio.queued", searching: "studio.searching", describing: "studio.describing", done: "studio.done", failed: "studio.failed" }[view.status];
  $("progress-text").textContent = t(key, { level: done + 1, depth: view.max_depth, time: clock });
  $("levels").replaceChildren(
    ...view.levels.map((l) => {
      const item = document.createElement("li");
      item.className = "flex gap-2";
      const depth = document.createElement("span");
      depth.className = "w-10 shrink-0 text-label-2 dark:text-label-2-dark";
      depth.textContent = l.depth === "event" ? t("studio.eventStage") : `L${l.depth}`;
      const bic = document.createElement("span");
      bic.className = "w-14 shrink-0 text-right";
      bic.textContent = fmt(l.bic);
      const structure = document.createElement("span");
      structure.className = "min-w-0 truncate";
      structure.textContent = l.structure;
      structure.title = l.structure;
      item.append(depth, bic, structure);
      return item;
    }),
  );
}

/**
 * Show an error where the results go.
 *
 * @param {string} text The message.
 */
function showError(text) {
  $("error").textContent = text;
  $("error").hidden = false;
}

// ---------------------------------------------------------------- results

/**
 * Create an HTML element with a class and text.
 *
 * @param {string} tag Element name.
 * @param {string} [className] Classes.
 * @param {string} [text] Text content.
 * @returns {HTMLElement} The element.
 */
function h(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

/**
 * Say in words how much of the variance a component carries.
 *
 * @param {number|null} v Share in per cent.
 * @returns {string} E.g. "42 %".
 */
function pct(v) {
  if (v === null || v === undefined) return "n/a";
  if (v > 0 && v < 0.1) return "< 0.1 %";
  if (v > 99.9 && v < 100) return "> 99.9 %";
  return `${fmt(v)} %`;
}

/**
 * Draw everything the search found.
 *
 * @param {Object} r The payload of autostat.studio.payload.
 */
function showResult(r) {
  $("welcome").hidden = true;
  $("results").hidden = false;
  showTitle();
  drawMain();
  $("summary-model").textContent = t("studio.modelLine", { structure: r.model.structure, bic: fmt(r.model.bic), n: r.model.n });
  $("summary-list").replaceChildren(
    ...r.components.map((p, i) => {
      const li = h("li", "flex gap-3");
      const dot = h("span", "mt-1.5 h-3 w-3 shrink-0 rounded-full");
      dot.style.backgroundColor = PART_COLOURS[i % PART_COLOURS.length];
      const text = h("span", "", p.summary);
      li.append(dot, text);
      return li;
    }),
  );

  $("parts").replaceChildren(...r.components.map((p, i) => partCard(r, p, i)));
  showContext(r);
  showProbabilities(r);
  chart($("bic-chart"), {
    unit: "",
    small: true,
    integerX: true,
    lines: [{ x: r.levels.map((l) => l.depth), y: r.levels.map((l) => l.bic), colour: "#0088FF" }],
    points: { x: r.levels.map((l) => l.depth), y: r.levels.map((l) => l.bic) },
    label: t("studio.bicTitle"),
  });
  $("formula").textContent = r.formula ? [r.formula.model, ...r.formula.laws].join("\n") : t("studio.noFormula");
  showProposals(r);
}

/**
 * List what the language model proposed at each level, and whether the kept
 * structure was among its proposals.
 *
 * @param {Object} r The run's result.
 */
function showProposals(r) {
  const history = r.proposals || [];
  $("proposals").hidden = history.length === 0;
  if (!history.length) return;
  $("proposals-verdict").textContent = t(r.best_was_proposed ? "studio.proposedKept" : "studio.proposedNotKept");
  $("proposals-list").replaceChildren(
    ...history.map((h) => {
      const item = document.createElement("li");
      const what = h.error ? t("studio.proposalError", { error: h.error }) : (h.proposed.join(" · ") || t("studio.proposalNone"));
      item.textContent = `${t("studio.proposalLevel", { level: h.level + 1 })} ${what}`;
      return item;
    }),
  );
}

/**
 * One card per component: sentence, chart, numbers and checks.
 *
 * @param {Object} r The payload.
 * @param {Object} p The component.
 * @param {number} i Its rank.
 * @returns {HTMLElement} The card.
 */
function partCard(r, p, i) {
  const colour = PART_COLOURS[i % PART_COLOURS.length];
  const card = h("article", "rounded-2xl border border-line bg-card p-4 dark:border-line-dark dark:bg-card-dark");
  const head = h("div", "flex items-start gap-3");
  const n = h("span", "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-sm font-semibold text-white", String(i + 1));
  n.style.backgroundColor = colour;
  const title = h("h3", "font-medium leading-snug", p.summary);
  head.append(n, title);
  const plot = h("div", "mt-3 aspect-[5/2] w-full");
  const body = h("p", "mt-3 text-sm text-label-2 dark:text-label-2-dark", `${p.details.join(". ")}.`);
  const extra = h("p", "mt-2 text-sm text-label-2 dark:text-label-2-dark");
  extra.append(h("span", "font-medium text-label dark:text-label-dark", `${t("studio.extrapolation")} `), document.createTextNode(`${p.extrapolation.join(". ")}.`));
  const stats = h("dl", "mt-3 grid grid-cols-2 gap-2 text-sm");
  for (const [label, value] of [
    [t("studio.varExplained"), pct(p.stats.var_explained)],
    [t("studio.maeChange"), pct(-p.mae_reduction)],
  ]) {
    const box = h("div", "rounded-lg bg-fill px-3 py-2 dark:bg-fill-dark");
    box.append(h("dt", "text-xs text-label-2 dark:text-label-2-dark", label), h("dd", "font-mono", value));
    stats.append(box);
  }
  card.append(head, plot, body, extra, stats);
  if (p.check) card.append(checkBadges(p.check));
  const structure = h("p", "mt-3 truncate font-mono text-xs text-label-2 dark:text-label-2-dark", p.structure);
  structure.title = p.structure;
  card.append(structure);
  requestAnimationFrame(() =>
    chart(plot, {
      unit: r.unit,
      small: true,
      bands: [{ x: r.grid, lo: p.lower, hi: p.upper, colour }],
      lines: [{ x: r.grid, y: p.mean, colour }],
      shadeFrom: Math.max(...r.x),
      label: p.summary,
    }),
  );
  return card;
}

/**
 * Badges for a component's posterior predictive checks.
 *
 * @param {Object<string, number>} check p-values by statistic.
 * @returns {HTMLElement} The badge row.
 */
function checkBadges(check) {
  const row = h("div", "mt-3 flex flex-wrap items-center gap-2");
  row.append(h("span", "text-xs text-label-2 dark:text-label-2-dark", t("studio.checks")));
  const names = { qq_max: "QQ", acf_min: "ACF", pxx_max: "PSD", mmd: "MMD" };
  for (const [key, label] of Object.entries(names)) {
    const p = check[key];
    if (p === null || p === undefined) continue;
    const bad = p < 0.05;
    const badge = h("span", `badge ${bad ? "bg-brand-red/15 text-brand-red" : "bg-brand-green/15 text-label dark:text-label-dark"}`, `${label} p=${fmt(p)}`);
    badge.title = bad ? t("studio.checkBad") : t("studio.checkGood");
    row.append(badge);
  }
  return row;
}

/**
 * Show each expected cycle and dated event against what was found.
 *
 * @param {Object} r The payload.
 */
function showContext(r) {
  const f = r.features;
  const rows = [];
  for (const c of r.context.cycles) {
    const found = f[`cycle_${c.slug}_present`];
    const period = f[`cycle_${c.slug}_period`];
    const share = f[`cycle_${c.slug}_share`];
    const p = r.probabilities[`p_cycle_${c.slug}`];
    rows.push({
      ok: found,
      title: c.name,
      text: found ? t("studio.cycleFound", { period: periodText(period), share: pct(share) }) : t("studio.cycleMissing"),
      p,
    });
  }
  for (const e of r.context.events) {
    const found = f[`event_${e.slug}_changepoint`];
    const effect = f[`event_${e.slug}_effect`];
    const low = f[`event_${e.slug}_low`];
    const high = f[`event_${e.slug}_high`];
    const p = r.probabilities[`p_event_${e.slug}`];
    const effectText = effect === null || effect === undefined ? "" : t("studio.effect", { effect: fmt(effect), low: fmt(low), high: fmt(high) });
    rows.push({ ok: found, title: e.name, text: `${found ? t("studio.eventFound") : t("studio.eventMissing")} ${effectText}`, p });
  }
  if (f.unexpected_cycles) rows.push({ ok: null, title: t("studio.unexpectedCycle"), text: t("studio.unexpectedCycleText", { count: f.unexpected_cycles, period: periodText(f.unexpected_main_period) }) });
  if (f.unexpected_changepoints) rows.push({ ok: null, title: t("studio.unexpectedBreak"), text: t("studio.unexpectedBreakText", { count: f.unexpected_changepoints }) });
  if (f.positive_violated) rows.push({ ok: false, title: t("studio.positiveTitle"), text: t("studio.positiveViolated") });
  $("context-results").hidden = rows.length === 0;
  $("ctx-body").replaceChildren(
    ...rows.map((row) => {
      const item = h("div", "flex gap-3 rounded-xl bg-fill p-3 dark:bg-fill-dark");
      const icon = h("span", `mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-sm font-bold text-white ${row.ok === true ? "bg-brand-green" : row.ok === false ? "bg-brand-red" : "bg-brand-orange"}`, row.ok === true ? "✓" : row.ok === false ? "✕" : "!");
      icon.setAttribute("aria-hidden", "true");
      const text = h("div", "min-w-0");
      text.append(h("p", "font-medium", row.title), h("p", "text-sm text-label-2 dark:text-label-2-dark", row.text));
      if (row.p !== undefined && row.p !== null) text.append(h("p", "mt-1 text-xs text-label-2 dark:text-label-2-dark", t("studio.probabilityOf", { p: pct(100 * row.p) })));
      item.append(icon, text);
      return item;
    }),
  );
}

/**
 * Say a period in the unit a reader uses.
 *
 * @param {number|null} years Period in years.
 * @returns {string} E.g. "7.0 days".
 */
function periodText(years) {
  if (years === null || years === undefined) return "n/a";
  const days = years * 365;
  if (days < 2) return t("studio.hours", { n: fmt(days * 24) });
  if (days < 60) return t("studio.days", { n: fmt(days) });
  if (years < 2) return t("studio.months", { n: fmt(years * 12) });
  return t("studio.years", { n: fmt(years) });
}

/**
 * Bars for the probability of each property over the scored structures.
 *
 * @param {Object} r The payload.
 */
function showProbabilities(r) {
  const p = r.probabilities;
  const items = [
    ["studio.pPeriodic", p.p_periodic],
    ["studio.pLinear", p.p_linear],
    ["studio.pSmooth", p.p_smooth],
    ["studio.pChange", p.p_changepoint],
  ].filter(([, v]) => v !== undefined && v !== null);
  $("probs").replaceChildren(
    ...items.map(([key, v]) => {
      const li = h("li");
      const top = h("div", "flex justify-between text-sm");
      top.append(h("span", "", t(key)), h("span", "font-mono", pct(100 * v)));
      const track = h("div", "mt-1 h-2 overflow-hidden rounded-full bg-fill dark:bg-fill-dark");
      const bar = h("div", "h-full rounded-full bg-brand-blue");
      bar.style.width = `${(100 * v).toFixed(1)}%`;
      track.append(bar);
      li.append(top, track);
      return li;
    }),
  );
}

// ---------------------------------------------------------------- downloads

/**
 * Hand the reader a file.
 *
 * @param {string} name File name.
 * @param {string} text Content.
 * @param {string} type Media type.
 */
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Write the context as YAML, the format Context.from_yaml reads.
 *
 * @param {Object} c What readContext returned.
 * @returns {string} The YAML text.
 */
function contextYaml(c) {
  const q = (s) => JSON.stringify(String(s));
  const lines = [`name: ${q($("name").value || "data")}`, `positive: ${c.positive}`];
  lines.push("cycles:", ...(c.cycles.length ? c.cycles.map((x) => `  - {name: ${q(x.name)}, period: ${q(x.period)}}`) : ["  []"]));
  lines.push("events:", ...(c.events.length ? c.events.map((x) => `  - {name: ${q(x.name)}, at: ${q(x.at)}}`) : ["  []"]));
  return `${lines.join("\n").replace(/:\n {2}\[\]/g, ": []")}\n`;
}

// ---------------------------------------------------------------- wiring

/** Connect every control, then load the strings and the examples. */
async function main() {
  let lang = "en";
  try {
    lang = localStorage.getItem("autostat.lang") || (navigator.language || "en").slice(0, 2);
  } catch (error) { /* private window */ }
  if (!["en", "fr"].includes(lang)) lang = "en";
  $("lang").value = lang;
  await loadStrings(lang);

  $("lang").addEventListener("change", () => {
    try { localStorage.setItem("autostat.lang", $("lang").value); } catch (error) { /* ignore */ }
    loadStrings($("lang").value);
  });
  $("scheme").addEventListener("click", () => {
    const dark = document.documentElement.classList.toggle("dark");
    try { localStorage.setItem("autostat.scheme", dark ? "dark" : "light"); } catch (error) { /* ignore */ }
    if (state.result) showResult(state.result);
  });
  $("tab-file").addEventListener("click", () => selectSource("file"));
  $("tab-example").addEventListener("click", () => selectSource("example"));
  for (const tab of [$("tab-file"), $("tab-example")]) {
    tab.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        const next = tab.id === "tab-file" ? "example" : "file";
        selectSource(next);
        $(`tab-${next}`).focus();
      }
    });
  }
  $("file").addEventListener("change", () => $("file").files[0] && upload($("file").files[0]));
  const drop = $("drop");
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("border-brand-blue");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("border-brand-blue"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("border-brand-blue");
    if (event.dataTransfer.files[0]) upload(event.dataTransfer.files[0]);
  });
  $("time").addEventListener("change", previewColumns);
  $("value").addEventListener("change", previewColumns);
  $("example").addEventListener("change", loadExample);
  $("name").addEventListener("input", showTitle);
  $("depth").addEventListener("input", () => { $("depth-out").value = $("depth").value; });
  $("add-cycle").addEventListener("click", () => addRow("cycles"));
  $("add-event").addEventListener("click", () => addRow("events"));
  $("positive").addEventListener("change", () => {});
  $("run").addEventListener("click", startRun);
  $("dl-report").addEventListener("click", () => download("report.md", state.result.report, "text/markdown"));
  $("dl-json").addEventListener("click", () => download("autostat.json", JSON.stringify(state.result, null, 1), "application/json"));
  $("dl-context").addEventListener("click", () => download("context.yaml", contextYaml(readContext()), "text/yaml"));
  $("copy-formula").addEventListener("click", () => navigator.clipboard?.writeText($("formula").textContent));

  const examples = await (await fetch("/api/examples")).json();
  fill($("example"), examples.map((e) => ({ value: e.name, label: `${e.name.replace(/^\d+-/, "")} (${e.n})` })), "01-airline");
}

main();
