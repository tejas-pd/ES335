// In-browser inference for the exported models (mirrors src/models.py).
// Works in the browser (window.CarModels) and in Node (module.exports) for testing.
(function (root) {
  const CAT = ["brand", "brand_model", "fuel", "transmission"];

  function prepare(m) {
    // Build category -> column index maps once.
    let offset = 0;
    m._enc = m.encoder.map((spec) => {
      const map = new Map(spec.frequent.map((c, i) => [c, offset + i]));
      const infreq = spec.has_infrequent ? offset + spec.frequent.length : -1;
      offset += spec.frequent.length + (spec.has_infrequent ? 1 : 0);
      return { col: spec.col, map, infreq };
    });
    m._nCat = offset;
    return m;
  }

  function onehot(m, row, x) {
    for (const e of m._enc) {
      const idx = e.map.get(row[e.col]);
      if (idx !== undefined) x[idx] = 1;
      else if (e.infreq >= 0) x[e.infreq] = 1; // rare or unseen category
    }
  }

  function predictLinear(m, row) {
    if (!m._enc) prepare(m);
    const x = new Float64Array(m._nCat + 3);
    onehot(m, row, x);
    const raw = [row.age, row.age * row.age, Math.log1p(row.km)];
    for (let j = 0; j < 3; j++) x[m._nCat + j] = (raw[j] - m.num_mean[j]) / m.num_scale[j];
    let s = m.intercept;
    for (let j = 0; j < x.length; j++) if (x[j] !== 0) s += m.coef[j] * x[j];
    return Math.exp(s);
  }

  function predictForest(m, row) {
    if (!m._enc) prepare(m);
    const x = new Float64Array(m._nCat + 2);
    onehot(m, row, x);
    x[m._nCat] = row.age; x[m._nCat + 1] = row.km;
    let s = 0;
    for (const t of m.trees) {
      let i = 0;
      while (t.f[i] >= 0) i = x[t.f[i]] <= t.x[i] ? i + 1 : t.r[i];
      s += t.x[i];
    }
    return Math.exp(s / m.trees.length);
  }

  function predictLookup(l, row) {
    return l.model[row.brand_model] ?? l.brand[row.brand] ?? l.global;
  }

  const api = { predictLinear, predictForest, predictLookup, CAT };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CarModels = api;
})(this);
