/* Shared static viewer/report behavior. Comparisons use semantic metadata, never DOM positions. */
(() => {
  "use strict";
  const identity = x => JSON.stringify([x.section, x.table || "", x.row, x.column, x.unit, x.version]);
  const title = x => [x.name, x.subtitle].filter(Boolean).join(" · ");
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
      button.textContent = report.name;
      const detail = document.createElement("small");
      detail.textContent = report.subtitle || report.folder;
      button.append(detail);
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
  function metricFor(cell) {
    const section = cell.closest("section")?.querySelector("h2")?.textContent.trim().replace(/\s+/g, " ");
    let table = cell.closest("table");
    while (table && !table.hasAttribute("data-compare-table")) table = table.parentElement?.closest("table");
    return {section, table: table?.dataset.compareTable || "", row: cell.dataset.compareRow,
      column: cell.dataset.compareColumn, unit: cell.dataset.unit, version: cell.dataset.version};
  }
  function compare() {
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
    catalog = (Array.isArray(event.data.reports) ? event.data.reports : []).filter(report =>
      own.comparison_family !== "target_multimap" || report.comparison_family === "target_multimap");
    selector.replaceChildren(new Option("No comparison", ""));
    for (const report of catalog) if (report.id !== activeId) selector.add(new Option(title(report), report.id));
    selector.disabled = !catalog.some(report => report.id !== activeId);
    selector.value = event.data.compare_id || "";
    document.getElementById("report-browser-link").hidden = true;
    compare();
  });
  selector.addEventListener("change", () => {
    compare();
    window.parent.postMessage({type: "geo-comparison-changed", report_id: activeId, compare_id: selector.value}, "*");
  });
  if (window.parent !== window) window.parent.postMessage({type: "geo-report-ready", document_id: own.id}, "*");
})();
