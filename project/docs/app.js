// Page logic: tables, charts and interactions. No external libraries.
"use strict";
const SRC = ["A", "B", "C", "D"];
const MODELS = ["mean", "lookup", "linear", "forest"];
const NAME = { mean: "Mean baseline", lookup: "Model-median lookup", linear: "Linear (ridge)", forest: "Random forest" };
const SHORT = { mean: "mean", lookup: "lookup", linear: "linear", forest: "forest" };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COL = {}; [...SRC, ...MODELS].forEach((k) => (COL[k] = css("--" + k)));
const $ = (id) => document.getElementById(id);
const L = (x, d = 2) => "₹" + (x / 1e5).toFixed(d) + " L";
const pct = (x, d = 0, sign = false) => (sign && x > 0 ? "+" : "") + x.toFixed(d) + "%";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const SVGNS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, s, attrs = {}) {
  const t = svgEl("text", { x, y, "font-size": 12, fill: "#5d6576", ...attrs }, parent);
  t.textContent = s;
  return t;
}
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

let R, P, MOD;

Promise.all([fetch("data/results.json").then((r) => r.json()), fetch("data/points.json").then((r) => r.json())])
  .then(([r, p]) => {
    R = r; P = p;
    kpis(); sourceTable(); sameModel(); cleaning(); losoDiagram();
    initBars(); mainTable(); initExplorer(); ablation(); bandTable(); worstTable(); initRecal();
    fetch("data/models.json").then((x) => x.json()).then((m) => { MOD = m; initPredictor(); });
  })
  .catch((e) => {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<p class="note" style="padding:20px 0">Could not load data files (${esc(e.message)}). If you opened index.html directly from disk, serve the folder instead: <code>python -m http.server -d docs</code>.</p>`);
  });

// ---------------------------------------------------------------- header + data
function kpis() {
  const rs = R.metrics.random_split, lo = R.metrics.loso;
  const ho = SRC.map((s) => lo[s].overall.forest.mae);
  const biases = SRC.map((s) => lo[s].overall.forest.bias_pct);
  const beats = SRC.filter((s) => lo[s].diffs.forest_minus_lookup.ci[1] < 0).length;
  const k = [
    [L(rs.overall.forest.mae), "Forest MAE, random split"],
    [`${L(Math.min(...ho))}–${(Math.max(...ho) / 1e5).toFixed(2)} L`, "Forest MAE on a held-out source"],
    [`${pct(Math.max(...biases), 0, true)} / ${pct(Math.min(...biases), 0, true)}`, "Worst over- / under-pricing bias (held-out A / C)"],
    [`${beats} of 4`, "Held-out sources where the forest clearly beats the model-median lookup"],
  ];
  $("kpis").innerHTML = k.map(([v, l]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
}

function sourceTable() {
  const S = R.eda.sources;
  const rec = (s, c) => (S[s].recorded_pct[c] > 50 ? '<span class="yes">✓</span>' : '<span class="no">—</span>');
  const rows = [
    ["Listings (after cleaning)", (s) => S[s].n.toLocaleString()],
    ["Median price", (s) => L(S[s].price_median)],
    ["Mean price", (s) => L(S[s].price_mean)],
    ["Median recorded age (yrs)", (s) => S[s].age_median],
    ["Median km", (s) => Math.round(S[s].km_median).toLocaleString()],
    ["Automatic", (s) => pct(S[s].automatic_pct)],
    ["Diesel", (s) => pct(S[s].diesel_pct)],
    ["Luxury brands", (s) => pct(S[s].luxury_pct, 1)],
    ["Distinct brand+models", (s) => S[s].n_models],
    ["Records engine", (s) => rec(s, "engine")], ["Records owner count", (s) => rec(s, "owner")],
    ["Records mileage / power", (s) => rec(s, "mileage")], ["Records seats", (s) => rec(s, "seats")],
    ["Records location", (s) => rec(s, "location")], ["Records body type", (s) => rec(s, "type")],
  ];
  $("source-table").innerHTML = `<thead><tr><th></th>${SRC.map((s) => `<th><span class="sw" style="background:${COL[s]}"></span>Source ${s}</th>`).join("")}</tr></thead><tbody>` +
    rows.map(([n, f]) => `<tr><td>${n}</td>${SRC.map((s) => `<td>${f(s)}</td>`).join("")}</tr>`).join("") + "</tbody>";
  const lo = R.metrics.loso;
  $("unseen-text").textContent = `When a source is held out, ${SRC.map((s) => `${s}: ${lo[s].unseen_model_share.toFixed(1)}%`).join(", ")} of its listings are brand+model names that never appear in the other three sources.`;
}

function sameModel() {
  const t = R.eda.same_model_median_lakh_age5to8;
  const f = (v) => (v == null ? '<span class="no">—</span>' : v.toFixed(2));
  $("same-model").innerHTML = `<thead><tr><th>Model (age 5–8)</th>${SRC.map((s) => `<th>${s}</th>`).join("")}</tr></thead><tbody>` +
    Object.entries(t).map(([m, v]) => {
      const vals = SRC.map((s) => v[s]).filter((x) => x != null);
      const mx = Math.max(...vals), mn = Math.min(...vals);
      return `<tr><td>${esc(m)}</td>${SRC.map((s) => `<td style="${v[s] === mx ? "color:#b5452b;font-weight:700" : v[s] === mn ? "color:#2a6fdb;font-weight:700" : ""}">${f(v[s])}</td>`).join("")}</tr>`;
    }).join("") + `</tbody><caption class="small" style="caption-side:bottom;text-align:left;padding-top:6px">Median price in ₹ lakh. Blue = cheapest source for that model, red = most expensive.</caption>`;
}

function cleaning() {
  const c = R.cleaning;
  $("clean-list").innerHTML = [
    `Dropped ${c.dropped_missing_identity} row with no brand/model/age.`,
    `Assigned each row to source A–D from its missing-column pattern (0 rows unassigned).`,
    `Removed ${c.dropped_exact_duplicates} exact duplicate rows (all within one source), so no copy sits on both sides of a split.`,
    `Removed ${c.dropped_km_over_500k} rows with km &gt; 500,000.`,
    `Recoded 0 as missing for mileage (${c.zero_to_missing_mileage}) and seats (${c.zero_to_missing_seats}).`,
    `Final: <strong>${c.final_rows.toLocaleString()}</strong> listings.`,
  ].map((s) => `<li>${s}</li>`).join("");
}

function losoDiagram() {
  const n = R.eda.sources;
  let h = `<div class="loso-row"><span></span>${SRC.map((s) => `<span class="small" style="text-align:center">Source ${s}</span>`).join("")}</div>`;
  SRC.forEach((t, i) => {
    h += `<div class="loso-row"><span class="small">Fold ${i + 1}: test ${t}</span>${SRC.map((s) => `<span class="c ${s === t ? "te" : "tr"}">${s === t ? "test" : "train"}</span>`).join("")}</div>`;
  });
  h += `<p class="small">Grey: training sources (an inner leave-one-source-out loop over these three picks hyperparameters). Red: scored once.</p>`;
  $("loso-diagram").innerHTML = h;
}

// ---------------------------------------------------------------- results
function initBars() {
  document.querySelectorAll('input[name="barmetric"]').forEach((r) => r.addEventListener("change", drawBars));
  drawBars();
}
function drawBars() {
  const metric = document.querySelector('input[name="barmetric"]:checked').value;
  const scale = metric === "mae" || metric === "rmse" ? 1e5 : 1;
  const val = (d) => d[metric] / scale;
  const rs = R.metrics.random_split.by_source, lo = R.metrics.loso;
  const W = 960, H = 330, m = { l: 56, r: 10, t: 34, b: 44 };
  const all = [];
  SRC.forEach((s) => MODELS.forEach((k) => all.push(val(rs[s][k]), val(lo[s].overall[k]))));
  let lo_ = Math.min(0, ...all), hi = Math.max(0, ...all);
  const ticks = niceTicks(lo_, hi, 6);
  lo_ = Math.min(lo_, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const y = (v) => m.t + (H - m.t - m.b) * (1 - (v - lo_) / (hi - lo_));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Grouped bar chart of error by source and model" });
  ticks.forEach((t) => {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: t === 0 ? "#888" : "#ebe8e0" }, svg);
    text(svg, m.l - 6, y(t) + 4, metric === "bias_pct" || metric === "medape" ? t + "%" : t, { "text-anchor": "end" });
  });
  const gw = (W - m.l - m.r) / 4, slot = (gw - 30) / 4;
  SRC.forEach((s, i) => {
    const gx = m.l + i * gw + 15;
    text(svg, gx + (gw - 30) / 2, H - 14, `Source ${s} held out`, { "text-anchor": "middle", "font-weight": 600, fill: "#1d2330", "font-size": 13 });
    MODELS.forEach((k, j) => {
      const x0 = gx + j * slot, bw = slot * 0.36;
      [[rs[s][k], "#dcd8ce", x0 + slot * 0.1, "random split"], [lo[s].overall[k], COL[k], x0 + slot * 0.1 + bw, "held-out source"]].forEach(([d, c, x, lab]) => {
        const v = val(d);
        const r = svgEl("rect", { x, width: bw - 1, y: Math.min(y(v), y(0)), height: Math.max(1, Math.abs(y(v) - y(0))), fill: c, rx: 2 }, svg);
        svgEl("title", {}, r).textContent = `${NAME[k]}, source ${s}, ${lab}: ${metric === "mae" || metric === "rmse" ? L(d[metric]) : pct(d[metric], 1, metric === "bias_pct")}`;
      });
    });
  });
  // legend
  [["#dcd8ce", "Random split (source seen in training)"], ...MODELS.map((k) => [COL[k], NAME[k] + ", held out"])].forEach(([c, l], i) => {
    const lx = m.l + i * 178;
    svgEl("rect", { x: lx, y: 8, width: 11, height: 11, fill: c, rx: 2 }, svg);
    text(svg, lx + 15, 18, l, { "font-size": 11.5 });
  });
  $("bars").replaceChildren(svg);
}

function mainTable() {
  const rs = R.metrics.random_split, lo = R.metrics.loso;
  const bestR = MODELS.reduce((a, b) => (rs.overall[a].mae < rs.overall[b].mae ? a : b));
  const bestS = Object.fromEntries(SRC.map((s) => [s, MODELS.reduce((a, b) => (lo[s].overall[a].mae < lo[s].overall[b].mae ? a : b))]));
  let h = `<thead><tr><th>Model</th><th>Random split<br>MAE [95% CI]</th><th>Random split<br>med. % err</th>${SRC.map((s) => `<th>Held-out ${s}<br>MAE · bias</th>`).join("")}</tr></thead><tbody>`;
  MODELS.forEach((k) => {
    const ci = rs.ci[k];
    h += `<tr><td><span class="sw" style="background:${COL[k]}"></span>${NAME[k]}</td>
      <td style="${k === bestR ? "font-weight:700" : ""}">${(rs.overall[k].mae / 1e5).toFixed(2)} <span class="small">[${(ci[0] / 1e5).toFixed(2)}, ${(ci[1] / 1e5).toFixed(2)}]</span></td>
      <td>${pct(rs.overall[k].medape, 1)}</td>
      ${SRC.map((s) => { const o = lo[s].overall[k]; return `<td style="${k === bestS[s] ? "font-weight:700" : ""}">${(o.mae / 1e5).toFixed(2)} <span class="small">· ${pct(o.bias_pct, 0, true)}</span></td>`; }).join("")}</tr>`;
  });
  h += `<tr><td class="small">Forest − lookup (paired 95% CI)</td><td class="small" colspan="2">${dci(rs.diffs.forest_minus_lookup)}</td>${SRC.map((s) => `<td class="small">${dci(lo[s].diffs.forest_minus_lookup)}</td>`).join("")}</tr>`;
  h += `<tr><td class="small">Forest − linear (paired 95% CI)</td><td class="small" colspan="2">${dci(rs.diffs.forest_minus_linear)}</td>${SRC.map((s) => `<td class="small">${dci(lo[s].diffs.forest_minus_linear)}</td>`).join("")}</tr>`;
  $("main-table").innerHTML = h + `</tbody><caption class="small" style="caption-side:bottom;text-align:left;padding-top:6px">MAE in ₹ lakh. Bias = median of predicted/actual − 1. Bold = lowest MAE in the column. A negative difference means the first model is better; differences are in ₹ lakh.</caption>`;
  const tn = SRC.map((s) => { const t = lo[s].tuning; return `held-out ${s}: α=${t.linear.best.alpha}, leaf=${t.forest.best.min_samples_leaf}, features=${t.forest.best.max_features}`; });
  $("tuning-note").textContent = `Hyperparameters chosen without the test source. Random split: α=${rs.tuning.linear.best.alpha}, leaf=${rs.tuning.forest.best.min_samples_leaf}, features=${rs.tuning.forest.best.max_features}; ${tn.join("; ")}.`;
}
function dci(d) {
  const f = (v) => (v / 1e5).toFixed(2);
  const sig = d.ci[0] > 0 || d.ci[1] < 0;
  return `<span style="${sig ? "font-weight:700;color:#1d2330" : ""}">${f(d.est)}</span> [${f(d.ci[0])}, ${f(d.ci[1])}]`;
}

// ---------------------------------------------------------------- explorer
let EX = {};
function initExplorer() {
  ["ex-design", "ex-source", "ex-model", "ex-unseen"].forEach((id) => $(id).addEventListener("change", drawExplorer));
  const cv = $("scatter");
  cv.addEventListener("mousemove", hover);
  cv.addEventListener("mouseleave", () => ($("tip").hidden = true));
  window.addEventListener("resize", drawExplorer);
  drawExplorer();
}
function drawExplorer() {
  const design = $("ex-design").value, s = $("ex-source").value, k = $("ex-model").value;
  const unseenOn = $("ex-unseen").checked && design === "loso";
  $("ex-unseen").disabled = design !== "loso";
  const cols = P[design + "_cols"], ix = Object.fromEntries(cols.map((c, i) => [c, i]));
  const rows = P[design].filter((r) => r[ix.source] === s);
  const cv = $("scatter"), dpr = window.devicePixelRatio || 1;
  const Wc = cv.clientWidth || 560, Hc = Math.round(Wc * 0.82);
  cv.width = Wc * dpr; cv.height = Hc * dpr; cv.style.height = Hc + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, Wc, Hc);
  const m = { l: 52, r: 12, t: 12, b: 40 }, lo = Math.log10(3e4), hi = Math.log10(8e6);
  const X = (v) => m.l + (Wc - m.l - m.r) * (Math.log10(Math.min(Math.max(v, 3e4), 8e6)) - lo) / (hi - lo);
  const Y = (v) => Hc - m.b - (Hc - m.t - m.b) * (Math.log10(Math.min(Math.max(v, 3e4), 8e6)) - lo) / (hi - lo);
  ctx.font = "11.5px system-ui, sans-serif"; ctx.fillStyle = "#5d6576"; ctx.strokeStyle = "#ebe8e0";
  [5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6].forEach((t) => {
    ctx.beginPath(); ctx.moveTo(X(t), m.t); ctx.lineTo(X(t), Hc - m.b); ctx.moveTo(m.l, Y(t)); ctx.lineTo(Wc - m.r, Y(t)); ctx.stroke();
    const lab = t / 1e5 + " L";
    ctx.textAlign = "center"; ctx.fillText(lab, X(t), Hc - m.b + 15);
    ctx.textAlign = "right"; ctx.fillText(lab, m.l - 5, Y(t) + 4);
  });
  ctx.textAlign = "center"; ctx.fillText("Actual price (log scale)", (m.l + Wc) / 2, Hc - 6);
  ctx.save(); ctx.translate(12, (Hc - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Predicted price", 0, 0); ctx.restore();
  // ±20% band and diagonal
  ctx.fillStyle = "rgba(47,143,91,0.07)"; ctx.beginPath();
  ctx.moveTo(X(3e4), Y(3e4 * 1.2)); ctx.lineTo(X(8e6 / 1.2), Y(8e6)); ctx.lineTo(X(8e6), Y(8e6)); ctx.lineTo(X(8e6), Y(8e6 * 0.8)); ctx.lineTo(X(3e4 / 0.8), Y(3e4)); ctx.lineTo(X(3e4), Y(3e4)); ctx.fill();
  ctx.strokeStyle = "#555"; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(3e4), Y(3e4)); ctx.lineTo(X(8e6), Y(8e6)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#2f8f5b"; ctx.textAlign = "left"; ctx.fillText("shaded: within ±20%", m.l + 8, m.t + 12);
  EX = { pts: [], rows, ix, k };
  const order = unseenOn ? [...rows].sort((a, b) => a[ix.unseen_model] - b[ix.unseen_model]) : rows;
  for (const r of order) {
    const px = X(r[ix.actual]), py = Y(r[ix[k]]);
    ctx.fillStyle = unseenOn ? (r[ix.unseen_model] ? "rgba(181,69,43,0.85)" : "rgba(150,150,150,0.35)") : hexA(COL[s], 0.45);
    ctx.beginPath(); ctx.arc(px, py, 2.6, 0, 7); ctx.fill();
    EX.pts.push([px, py, r]);
  }
  exStats(design, s, k);
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}
function hover(ev) {
  const rect = ev.target.getBoundingClientRect(), x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  let best = null, bd = 64;
  for (const p of EX.pts) { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) { bd = d; best = p; } }
  const tip = $("tip");
  if (!best) { tip.hidden = true; return; }
  const r = best[2], ix = EX.ix, a = r[ix.actual], p = r[ix[EX.k]];
  tip.innerHTML = `<b>${esc(r[ix.brand_model])}</b> · ${r[ix.age]} yrs · ${Math.round(r[ix.km]).toLocaleString()} km · ${r[ix.fuel]} · ${r[ix.transmission]}<br>
    Actual ${L(a)} · predicted ${L(p)} (${pct((p / a - 1) * 100, 0, true)})${ix.unseen_model !== undefined && r[ix.unseen_model] ? "<br><i>model absent from training sources</i>" : ""}`;
  tip.hidden = false;
  const left = Math.min(best[0] + 12, rect.width - tip.offsetWidth - 4);
  tip.style.left = Math.max(0, left) + "px"; tip.style.top = Math.max(0, best[1] - 50) + "px";
}
function exStats(design, s, k) {
  const rs = R.metrics.random_split.by_source[s][k];
  const o = design === "loso" ? R.metrics.loso[s].overall[k] : rs;
  const lookup = design === "loso" ? R.metrics.loso[s].overall.lookup : R.metrics.random_split.by_source[s].lookup;
  const rel = (o.mae / lookup.mae - 1) * 100;
  let h = `<div class="stat"><div class="l">MAE (all ${o.n.toLocaleString()} test rows)</div><div class="v">${L(o.mae)}</div>
    ${design === "loso" ? `<div class="d">vs ${L(rs.mae)} on the random split: <span class="up">×${(o.mae / rs.mae).toFixed(2)}</span></div>` : ""}</div>
    <div class="stat"><div class="l">Median absolute % error · within ±20%</div><div class="v">${pct(o.medape, 1)} · ${pct(o.within20)}</div></div>
    <div class="stat"><div class="l">Bias (median signed % error)</div><div class="v ${Math.abs(o.bias_pct) > 10 ? "up" : ""}">${pct(o.bias_pct, 1, true)}</div>
      <div class="d">${o.bias_pct > 10 ? "systematically over-prices this source" : o.bias_pct < -10 ? "systematically under-prices this source" : "no large systematic bias"}</div></div>`;
  if (k !== "lookup") h += `<div class="stat"><div class="l">Compared with the model-median lookup</div><div class="v ${rel < 0 ? "down" : "up"}">${pct(rel, 1, true)} MAE</div></div>`;
  if (design === "loso") {
    const L2 = R.metrics.loso[s];
    h += `<div class="stat"><div class="l">Listings with a model absent from training</div><div class="v">${pct(L2.unseen_model_share, 1)}</div>
      ${L2.unseen ? `<div class="d">median % error: ${pct(L2.unseen[k].medape, 1)} unseen vs ${pct(L2.seen[k].medape, 1)} seen</div>` : `<div class="d">too few to compare</div>`}</div>`;
  }
  $("ex-stats").innerHTML = h;
}

function ablation() {
  const ab = R.metrics.random_split.ablation_specs, rs = R.metrics.random_split;
  $("ablation-text").innerHTML = `Adding engine, power, mileage, seats and owner count (median-imputed with missing flags, fitted on training data) lowers the random-split forest MAE from
    <strong>${L(rs.overall.forest.mae)}</strong> to <strong>${L(ab.overall.mae)}</strong> (median % error ${pct(rs.overall.forest.medape, 1)} → ${pct(ab.overall.medape, 1)}).
    The biggest gain is on source C (${L(rs.by_source.C.forest.mae)} → ${L(ab.by_source.C.mae)}), where engine and power separate luxury variants.
    So the specs carry real information, but each is missing for at least one whole source. A model relying on them would face an entirely imputed column when moved to that source.
    This is a second, structural form of dataset shift, which is why the main comparison uses only the shared features.`;
}

// ---------------------------------------------------------------- errors
function bandTable() {
  const B = R.diagnostics.by_band, bands = ["<3 L", "3–6 L", "6–10 L", "10–20 L", ">20 L"];
  let h = `<thead><tr><th>Held-out source</th>${bands.map((b) => `<th>${b}</th>`).join("")}</tr></thead><tbody>`;
  SRC.forEach((s) => {
    h += `<tr><td><span class="sw" style="background:${COL[s]}"></span>${s}</td>` + bands.map((b) => {
      const c = B[s][b];
      if (!c) return "<td>—</td>";
      const v = c.forest.medape, a = Math.min(v / 60, 1);
      return `<td style="background:rgba(181,69,43,${(a * 0.55).toFixed(2)})" title="n = ${c.n}">${v.toFixed(0)}% <span class="small">(${pct(c.forest.bias, 0, true)})</span><br><span class="small">n=${c.n}</span></td>`;
    }).join("") + "</tr>";
  });
  $("band-table").innerHTML = h + "</tbody>";
}
function worstTable() {
  const W = R.diagnostics.worst;
  let h = `<thead><tr><th>Source</th><th class="txt">Car</th><th>Actual</th><th>Lookup</th><th>Linear</th><th>Forest</th><th class="txt">Name in training sources?</th></tr></thead><tbody>`;
  W.forEach((w) => {
    h += `<tr><td>${w.source}</td><td class="txt">${esc(w.brand_model)} · ${w.age} yrs · ${Math.round(w.km).toLocaleString()} km · ${w.fuel} · ${w.transmission}</td>
      <td>${L(w.actual)}</td><td>${L(w.lookup)}</td><td>${L(w.linear)}</td><td><strong>${L(w.forest)}</strong></td><td class="txt">${w.unseen_model ? "name not seen" : "yes"}</td></tr>`;
  });
  $("worst-table").innerHTML = h + `</tbody><caption class="small" style="caption-side:bottom;text-align:left;padding-top:6px">Largest forest % errors on each held-out source (4 per source).</caption>`;
}

// ---------------------------------------------------------------- recalibration
const KS = [0, 5, 10, 25, 50, 100, 200];
function initRecal() {
  $("rc-source").addEventListener("change", drawRecal);
  $("rc-k").addEventListener("input", drawRecal);
  drawRecal();
}
function drawRecal() {
  const s = $("rc-source").value, ki = +$("rc-k").value, k = KS[ki];
  $("rc-kout").textContent = k;
  const D = R.diagnostics.recalibration[s], show = ["lookup", "linear", "forest"];
  const W = 620, H = 300, m = { l: 52, r: 16, t: 16, b: 40 };
  const vals = show.flatMap((n) => D[n].flatMap((r) => [r.mae, r.lo ?? r.mae, r.hi ?? r.mae]));
  const lo = Math.min(...vals) / 1e5 * 0.97, hi = Math.max(...vals) / 1e5 * 1.03;
  const ticks = niceTicks(lo, hi, 5);
  const X = (i) => m.l + (W - m.l - m.r) * i / (KS.length - 1);
  const Y = (v) => m.t + (H - m.t - m.b) * (1 - (v / 1e5 - lo) / (hi - lo));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "MAE against number of labelled listings used for recalibration" });
  ticks.forEach((t) => { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(t * 1e5), y2: Y(t * 1e5), stroke: "#ebe8e0" }, svg); text(svg, m.l - 6, Y(t * 1e5) + 4, t.toFixed(1), { "text-anchor": "end" }); });
  KS.forEach((kk, i) => text(svg, X(i), H - m.b + 16, kk, { "text-anchor": "middle" }));
  text(svg, (m.l + W) / 2, H - 6, "k labelled listings from the held-out source", { "text-anchor": "middle" });
  text(svg, m.l + 4, m.t + 2, "MAE (₹ L)", { "font-size": 11 });
  svgEl("line", { x1: X(ki), x2: X(ki), y1: m.t, y2: H - m.b, stroke: "#b5452b", "stroke-dasharray": "3 3" }, svg);
  show.forEach((n) => {
    const pts = D[n];
    const band = pts.filter((p) => p.lo != null);
    if (band.length) {
      const top = band.map((p) => `${X(KS.indexOf(p.k))},${Y(p.hi)}`), bot = band.map((p) => `${X(KS.indexOf(p.k))},${Y(p.lo)}`).reverse();
      svgEl("polygon", { points: [...top, ...bot].join(" "), fill: COL[n], opacity: 0.12 }, svg);
    }
    svgEl("polyline", { points: pts.map((p, i) => `${X(i)},${Y(p.mae)}`).join(" "), fill: "none", stroke: COL[n], "stroke-width": 2.2 }, svg);
    pts.forEach((p, i) => svgEl("circle", { cx: X(i), cy: Y(p.mae), r: i === ki ? 5 : 3, fill: COL[n] }, svg));
  });
  $("recal-chart").replaceChildren(svg);
  $("recal-stats").innerHTML = `<p class="small" style="margin:0">Held-out ${s}, k = ${k}. Shaded: 5–95% range over the 200 draws.</p>` + show.map((n) => {
    const r0 = D[n][0], r = D[n][ki], ch = (r.mae / r0.mae - 1) * 100;
    return `<div class="stat"><div class="l"><span class="sw" style="background:${COL[n]}"></span>${NAME[n]}</div><div class="v">${L(r.mae)}</div>
      <div class="d">${k === 0 ? "no correction" : `<span class="${ch < 0 ? "down" : "up"}">${pct(ch, 1, true)}</span> vs uncorrected ${L(r0.mae)}`}</div></div>`;
  }).join("");
}

// ---------------------------------------------------------------- predictor
function initPredictor() {
  const O = MOD.options;
  const brands = Object.keys(O).sort((a, b) => O[b].reduce((s, x) => s + x.n, 0) - O[a].reduce((s, x) => s + x.n, 0));
  $("p-brand").innerHTML = brands.map((b) => `<option>${esc(b)}</option>`).join("");
  $("p-brand").value = "maruti";
  $("compact-mae").textContent = L(MOD.forest.compact_mae_random_split);
  $("full-mae").textContent = L(MOD.forest.full_mae_random_split);
  const fillModels = () => {
    $("p-model").innerHTML = O[$("p-brand").value].map((m) => `<option value="${esc(m.m)}">${esc(m.m)} (${m.n})</option>`).join("");
    setDefaults();
  };
  const setDefaults = () => {
    const m = O[$("p-brand").value].find((x) => x.m === $("p-model").value);
    $("p-fuel").value = m.fuel; $("p-tr").value = m.tr; $("p-age").value = Math.round(m.age); $("p-km").value = Math.round(m.km / 1000) * 1000;
    predict();
  };
  $("p-brand").addEventListener("change", fillModels);
  $("p-model").addEventListener("change", setDefaults);
  ["p-age", "p-km", "p-fuel", "p-tr"].forEach((id) => $(id).addEventListener("input", predict));
  fillModels();
  $("p-model").value = "swift"; setDefaults();
}
function currentRow(ageOverride) {
  const b = $("p-brand").value, m = $("p-model").value;
  const age = Math.max(0, Math.min(30, +$("p-age").value || 0)), km = Math.max(0, +$("p-km").value || 0);
  return { brand: b, brand_model: b + " " + m, fuel: $("p-fuel").value, transmission: $("p-tr").value, age: ageOverride ?? age, km };
}
function predictAll(row) {
  return { mean: MOD.lookup.mean, lookup: CarModels.predictLookup(MOD.lookup, row),
           linear: CarModels.predictLinear(MOD.linear, row), forest: CarModels.predictForest(MOD.forest, row) };
}
function predict() {
  const row = currentRow(), p = predictAll(row), lo = R.metrics.loso;
  const err = Object.fromEntries(MODELS.map((k) => [k, SRC.reduce((s, x) => s + lo[x].overall[k].medape, 0) / 4]));
  const minV = Math.log10(5e4), maxV = Math.log10(1e7);
  const pos = (v) => ((Math.log10(Math.min(Math.max(v, 5e4), 1e7)) - minV) / (maxV - minV) * 100).toFixed(2) + "%";
  const n = (MOD.options[row.brand].find((x) => x.m === $("p-model").value) || {}).n || 0;
  $("p-out").innerHTML = MODELS.map((k) => {
    const v = p[k], e = err[k] / 100;
    return `<div class="prow"><span><span class="sw" style="background:${COL[k]}"></span>${NAME[k]}</span>
      <div class="pbar"><i style="left:${pos(v * (1 - e))};right:${100 - parseFloat(pos(v * (1 + e)))}%;background:${COL[k]}"></i><u style="left:${pos(v)};background:${COL[k]}"></u></div>
      <span class="pval">${L(v)}</span></div>`;
  }).join("") +
    `<p class="small" style="margin:4px 0 0">Bars: log scale from ₹0.5 L to ₹100 L. The shaded range is ± the model's average median % error on held-out sources
    (${MODELS.map((k) => `${SHORT[k]} ±${err[k].toFixed(0)}%`).join(", ")}), so about half of listings from a new source fall outside it.
    ${n < 5 ? `<br><strong style="color:#b5452b">This model has only ${n} listing${n === 1 ? "" : "s"}, so the ML models treat it as a "rare model". Treat their predictions with caution.</strong>` : ""}</p>`;
  drawDep(row);
}
function drawDep(row) {
  const ages = Array.from({ length: 21 }, (_, i) => i);
  const series = ["lookup", "linear", "forest"].map((k) => [k, ages.map((a) => predictAll({ ...row, age: a })[k])]);
  const all = series.flatMap((s) => s[1]);
  const W = 960, H = 260, m = { l: 56, r: 170, t: 14, b: 36 };
  const hi = Math.max(...all) / 1e5 * 1.08;
  const ticks = niceTicks(0, hi, 5);
  const X = (a) => m.l + (W - m.l - m.r) * a / 20, Y = (v) => m.t + (H - m.t - m.b) * (1 - v / 1e5 / hi);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Predicted price against age" });
  ticks.forEach((t) => { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(t * 1e5), y2: Y(t * 1e5), stroke: "#ebe8e0" }, svg); text(svg, m.l - 6, Y(t * 1e5) + 4, t + " L", { "text-anchor": "end" }); });
  [0, 5, 10, 15, 20].forEach((a) => text(svg, X(a), H - m.b + 16, a, { "text-anchor": "middle" }));
  text(svg, (m.l + W - m.r) / 2, H - 4, "Age (years)", { "text-anchor": "middle" });
  svgEl("line", { x1: X(row.age), x2: X(row.age), y1: m.t, y2: H - m.b, stroke: "#b5452b", "stroke-dasharray": "3 3" }, svg);
  const labs = series.map(([k, v]) => ({ k, y: Y(v[20]) + 4 })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < labs.length; i++) labs[i].y = Math.max(labs[i].y, labs[i - 1].y + 15);
  series.forEach(([k, v]) => {
    svgEl("polyline", { points: v.map((y, i) => `${X(i)},${Y(y)}`).join(" "), fill: "none", stroke: COL[k], "stroke-width": 2.4 }, svg);
    text(svg, W - m.r + 8, labs.find((l) => l.k === k).y, NAME[k], { fill: COL[k], "font-weight": 600 });
  });
  $("dep-chart").replaceChildren(svg);
}

// highlight current section in the nav
const links = [...document.querySelectorAll(".toc a")];
const obs = new IntersectionObserver((es) => es.forEach((e) => {
  if (e.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
}), { rootMargin: "-40% 0px -55% 0px" });
document.querySelectorAll("main section").forEach((s) => obs.observe(s));
