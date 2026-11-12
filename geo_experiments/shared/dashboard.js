/* Shared static viewer/report behavior. Comparisons use semantic metadata, never DOM positions. */
(() => {
  "use strict";
  const identity = x => JSON.stringify([x.section, x.table || "", x.row, x.column, x.unit, x.version]);
  const title = x => [x.experiment_title || x.name, x.run_name && `Run: ${x.run_name}`].filter(Boolean).join(" · ");
  function reportCard(element, report, compact = false) {
    element.replaceChildren();
    element.style.setProperty("--card-accent", report.accent || "#68b9f2");
    element.style.setProperty("--card-background", report.card_background || "#142138");
    const heading = document.createElement("strong");
    heading.textContent = report.experiment_title || report.name;
    element.append(heading);
    if (report.run_name) {
      const run = document.createElement("span");
      run.className = "card-run";
      run.textContent = `${compact ? " · " : ""}Run: ${report.run_name}`;
      element.append(run);
    }
    const detail = document.createElement("small");
    detail.textContent = report.subtitle || report.folder || "";
    if (!compact && detail.textContent) element.append(detail);
    if (compact) element.title = title(report);
  }
  const catalogNode = document.getElementById("geo-report-catalog");
  if (catalogNode) {
    const catalog = JSON.parse(catalogNode.textContent);
    const frame = document.getElementById("report-frame");
    const nav = document.getElementById("reports");
    let current = null;
    const params = () => new URLSearchParams(location.hash.slice(1));
    function sendCatalog() {
      if (current) frame.contentWindow.postMessage({type: "geo-report-catalog", report_id: current.id, document_id: current.document_id,
        reports: catalog, compare_id: params().get("compare") || ""}, "*");
    }
    function navigate() {
      const selected = catalog.find(x => x.id === params().get("report")) || catalog[0];
      if (!selected) {
        frame.hidden = true;
        document.getElementById("empty").hidden = false;
        return;
      }
      nav.querySelectorAll("button").forEach(b => b.setAttribute("aria-current", String(b.dataset.report === selected.id)));
      if (!current || selected.id !== current.id) {
        current = selected;
        frame.src = selected.url;
      } else sendCatalog();
    }
    for (const report of catalog) {
      const button = document.createElement("button");
      button.dataset.report = report.id;
      reportCard(button, report);
      button.onclick = () => {
        const p = params(); p.set("report", report.id);
        if (p.get("compare") === report.id) p.delete("compare");
        location.hash = p.toString();
      };
      nav.append(button);
    }
    window.addEventListener("message", event => {
      if (event.source !== frame.contentWindow || !current || !event.data) return;
      if (event.data.type === "geo-report-ready" && event.data.document_id === current.document_id) sendCatalog();
      if (event.data.type === "geo-comparison-changed") {
        if (event.data.report_id !== current.id) return;
        const p = params(); p.set("report", current.id);
        const id = event.data.compare_id;
        if (id && id !== current.id && catalog.some(r => r.id === id)) p.set("compare", id);
        else p.delete("compare");
        location.hash = p.toString();
      }
    });
    frame.addEventListener("load", sendCatalog);
    window.addEventListener("hashchange", navigate);
    navigate();
    return;
  }
  const dataNode = document.getElementById("geo-report-data");
  if (!dataNode) return;
  const own = JSON.parse(dataNode.textContent);
  const selector = document.getElementById("compare-report");
  const note = document.getElementById("compare-note");
  if (!selector || !note) return;
  let catalog = [];
  let activeId = own.id;
  // Keep the select value/change-event contract with styled one-line choices.
  const style = document.createElement("style");
  style.textContent = `
    .geo-compare-picker{display:inline-block;position:relative;margin-left:7px;vertical-align:middle;max-width:100%}
    .geo-compare-card{box-sizing:border-box;font:inherit;line-height:20px;text-align:left;color:#e4edf7;background:var(--card-background,#142138);border:1px solid #425571;border-left:3px solid var(--card-accent,#68b9f2);border-radius:6px;padding:5px 10px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .geo-compare-card strong{display:inline;font-size:13px;color:var(--card-accent,#68b9f2)}
    .geo-compare-card .card-run{display:inline;color:#c1d1e3;font-size:12px}
    .geo-compare-card:hover,.geo-compare-card:focus-visible,.geo-compare-card[aria-selected=true]{border-color:var(--card-accent,#68b9f2);outline:1px solid var(--card-accent,#68b9f2);outline-offset:1px}
    .geo-compare-trigger{width:max-content;max-width:min(700px,calc(100vw - 100px));padding-right:30px;position:relative}
    .geo-compare-trigger:after{content:'▾';position:absolute;right:10px;top:5px;color:#9aacc3}
    .geo-compare-trigger:disabled{opacity:.6;cursor:default}
    .geo-compare-menu{position:absolute;z-index:100;top:calc(100% + 4px);left:0;width:max-content;min-width:100%;max-width:calc(100vw - 100px);max-height:60vh;overflow:auto;display:grid;gap:4px;padding:6px;background:#0b1220;border:1px solid #425571;border-radius:7px;box-shadow:0 12px 32px #0008}
    .geo-compare-menu[hidden]{display:none!important}
    .geo-compare-menu .geo-compare-card{display:block;width:100%}
  `;
  document.head.append(style);
  selector.hidden = true;
  const picker = document.createElement("span");
  picker.className = "geo-compare-picker";
  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "geo-compare-card geo-compare-trigger";
  trigger.setAttribute("aria-label", "Compare to");
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");
  const menu = document.createElement("span");
  menu.id = "geo-compare-menu";
  menu.className = "geo-compare-menu";
  menu.setAttribute("role", "listbox");
  menu.setAttribute("aria-label", "Comparison reports");
  menu.hidden = true;
  trigger.setAttribute("aria-controls", menu.id);
  picker.append(trigger, menu);
  selector.after(picker);
  const noComparison = {id: "", experiment_title: "No comparison"};
  function closePicker(focus = false) {
    menu.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
    if (focus) trigger.focus();
  }
  function updatePicker() {
    const selected = catalog.find(report => report.id === selector.value) || noComparison;
    reportCard(trigger, selected, true);
    trigger.disabled = selector.disabled;
    for (const option of menu.children) option.setAttribute("aria-selected", String(option.dataset.value === selector.value));
  }
  function renderPicker() {
    menu.replaceChildren();
    for (const report of [noComparison, ...catalog.filter(report => report.id !== activeId)]) {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "geo-compare-card";
      option.setAttribute("role", "option");
      option.dataset.value = report.id;
      reportCard(option, report, true);
      option.onclick = () => {
        selector.value = report.id;
        closePicker(true);
        selector.dispatchEvent(new Event("change", {bubbles: true}));
      };
      menu.append(option);
    }
    updatePicker();
  }
  function openPicker() {
    if (trigger.disabled) return;
    menu.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
    (menu.querySelector('[aria-selected="true"]') || menu.firstElementChild)?.focus();
  }
  trigger.onclick = () => menu.hidden ? openPicker() : closePicker();
  trigger.addEventListener("keydown", event => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") { event.preventDefault(); openPicker(); }
  });
  menu.addEventListener("keydown", event => {
    const options = [...menu.children];
    const index = options.indexOf(document.activeElement);
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      options[(index + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length]?.focus();
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      options[event.key === "Home" ? 0 : options.length - 1]?.focus();
    }
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !menu.hidden) { event.preventDefault(); closePicker(true); }
  });
  document.addEventListener("click", event => { if (!picker.contains(event.target)) closePicker(); });
  document.addEventListener("focusin", event => { if (!picker.contains(event.target)) closePicker(); });
  renderPicker();
  function metricFor(cell) {
    const section = cell.closest("section")?.querySelector("h2")?.textContent.trim().replace(/\s+/g, " ");
    let table = cell.closest("table");
    while (table && !table.hasAttribute("data-compare-table")) table = table.parentElement?.closest("table");
    return {section, table: table?.dataset.compareTable || "", row: cell.dataset.compareRow,
      column: cell.dataset.compareColumn, unit: cell.dataset.unit, version: cell.dataset.version};
  }
  function compare() {
    updatePicker();
    document.querySelectorAll(".metric-delta").forEach(x => x.remove());
    const base = catalog.find(x => x.id === selector.value);
    if (!base) { note.textContent = "No comparison selected."; return; }
    const values = new Map();
    const duplicates = new Set();
    for (const metric of base.metrics || []) {
      const key = identity(metric);
      if (values.has(key)) duplicates.add(key);
      if (Number.isFinite(metric.value)) values.set(key, metric.value);
    }
    let matched = 0;
    document.querySelectorAll("table td[data-value][data-compare-row]").forEach(cell => {
      const key = identity(metricFor(cell));
      const value = Number(cell.dataset.value);
      if (!values.has(key) || duplicates.has(key) || !Number.isFinite(value)) return;
      const difference = value - values.get(key);
      const decimals = Number(cell.dataset.decimals || 0);
      const scientific = cell.dataset.numberFormat === "scientific";
      const formatted = scientific ? Math.abs(difference).toPrecision(3) : Math.abs(difference).toLocaleString(undefined, {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
      const rounded = scientific ? Math.abs(difference) : Number(Math.abs(difference).toFixed(decimals));
      const sign = rounded === 0 ? "" : difference > 0 ? "+" : "−";
      const span = document.createElement("span");
      span.className = "metric-delta" + (rounded === 0 ? "" : difference > 0 ? " positive" : " negative");
      span.textContent = ` (${sign}${formatted})`;
      span.title = `Difference from ${base.name}: ${difference} ${cell.dataset.unit === "percent" ? "percentage points" : cell.dataset.unit}`;
      cell.append(span);
      matched++;
    });
    note.textContent = `${matched} matching table values compared with ${base.name}. Percentage differences are percentage points.`;
  }
  window.addEventListener("message", event => {
    if (window.parent === window || event.source !== window.parent || event.data?.type !== "geo-report-catalog" || event.data.document_id !== own.id) return;
    activeId = event.data.report_id;
    catalog = Array.isArray(event.data.reports) ? event.data.reports : [];
    selector.replaceChildren(new Option("No comparison", ""));
    for (const report of catalog) if (report.id !== activeId) selector.add(new Option(title(report), report.id));
    selector.disabled = !catalog.some(report => report.id !== activeId);
    selector.value = event.data.compare_id || "";
    renderPicker();
    document.getElementById("report-browser-link").hidden = true;
    compare();
  });
  selector.addEventListener("change", () => {
    compare();
    window.parent.postMessage({type: "geo-comparison-changed", report_id: activeId, compare_id: selector.value}, "*");
  });
  if (window.parent !== window) window.parent.postMessage({type: "geo-report-ready", document_id: own.id}, "*");
})();
