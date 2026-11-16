"""Step 4: export precomputed results and small models for the static website.

The website has no server, so:
  * the linear model is exported as its coefficients,
  * a compact random forest (fewer, shallower trees) is exported as node arrays,
  * held-out predictions are subsampled for the scatter plot.
The "website" models are trained on all four sources with the hyperparameters
chosen on the random-split training data. The compact forest's accuracy is
checked against the full forest on the random-split test set and reported.

Usage: python src/export_web.py data/clean.csv docs/data
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from models import CAT, NUM, SEED, LookupBaseline, make_forest, make_linear  # noqa: E402

COMPACT = {"n_estimators": 20}


def encoder_spec(ct):
    enc = ct.named_transformers_["cat"]
    spec = []
    for col, cats, infreq in zip(CAT, enc.categories_, enc.infrequent_categories_):
        infreq = set() if infreq is None else set(infreq)
        frequent = [c for c in cats if c not in infreq]
        spec.append({"col": col, "frequent": frequent, "has_infrequent": bool(infreq)})
    return spec


def export_tree(t):
    """Compact form: f = feature (-1 for a leaf), x = threshold or leaf value (log price),
    r = right child. sklearn grows trees depth-first, so the left child is always i+1."""
    tr = t.tree_
    internal = tr.children_left >= 0
    assert np.all(tr.children_left[internal] == np.where(internal)[0] + 1)
    x = np.where(internal, tr.threshold, tr.value[:, 0, 0])
    return {"f": np.where(internal, tr.feature, -1).tolist(),
            "x": [round(float(v), 4) for v in x],
            "r": np.where(internal, tr.children_right, 0).tolist()}


def main(path, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    X, y = df[CAT + NUM], df.price.to_numpy(float)
    m = json.loads(Path("results/metrics.json").read_text())
    lin_p = m["random_split"]["tuning"]["linear"]["best"]
    rf_p = m["random_split"]["tuning"]["forest"]["best"]

    # Check the compact forest against the full one on the same random split.
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=df.source)
    small = make_forest(**rf_p, **COMPACT).fit(X.iloc[tr], y[tr])
    small_mae = float(np.mean(np.abs(small.predict(X.iloc[te]) - y[te])))
    full_mae = m["random_split"]["overall"]["forest"]["mae"]
    print(f"compact forest MAE {small_mae:.0f} vs full {full_mae:.0f}")

    # Deployment models on all data
    lin = make_linear(**lin_p).fit(X, y)
    rf = make_forest(**rf_p, **COMPACT).fit(X, y)
    look = LookupBaseline().fit(X, y)

    lin_pipe = lin.model
    ct_l = lin_pipe.named_steps["pre"]
    scaler = ct_l.named_transformers_["num"].named_steps["standardscaler"]
    linear = {"encoder": encoder_spec(ct_l), "coef": [round(float(c), 6) for c in lin_pipe.named_steps["reg"].coef_],
              "intercept": float(lin_pipe.named_steps["reg"].intercept_),
              "num_mean": scaler.mean_.tolist(), "num_scale": scaler.scale_.tolist(),
              "num_names": ["age", "age^2", "log(1+km)"], "params": lin_p}
    ct_f = rf.model.named_steps["pre"]
    forest = {"encoder": encoder_spec(ct_f), "trees": [export_tree(t) for t in rf.model.named_steps["reg"].estimators_],
              "params": {**rf_p, **COMPACT}, "compact_mae_random_split": small_mae,
              "full_mae_random_split": full_mae}

    # Options for the form: models per brand with their typical fuel/transmission
    opts = {}
    for (b, bm), g in df.groupby(["brand", "brand_model"]):
        opts.setdefault(b, []).append({"m": bm[len(b) + 1:], "n": int(len(g)),
                                       "fuel": g.fuel.mode()[0], "tr": g.transmission.mode()[0],
                                       "age": float(g.age.median()), "km": float(g.km.median())})
    for b in opts:
        opts[b].sort(key=lambda d: -d["n"])
    lookup = {"model": {k: round(v) for k, v in look.model_.items()},
              "brand": {k: round(v) for k, v in look.brand_.items()}, "global": look.global_,
              "mean": float(y.mean())}

    # Subsample held-out predictions for the scatter (all rows are used for metrics)
    pr = pd.read_csv("results/loso_predictions.csv")
    rnd = pd.read_csv("results/random_split_predictions.csv")
    meta = df[["brand_model", "age", "km", "fuel", "transmission"]]
    pts = {}
    for name, frame in [("loso", pr), ("random", rnd)]:
        frame = frame.merge(meta, left_on="row", right_index=True)
        samp = pd.concat([g.sample(min(len(g), 700), random_state=SEED)
                          for _, g in frame.groupby("source")])
        cols = ["source", "brand_model", "age", "km", "fuel", "transmission", "actual",
                "mean", "lookup", "linear", "forest"] + (["unseen_model"] if name == "loso" else [])
        recs = samp[cols].copy()
        for c in ["actual", "mean", "lookup", "linear", "forest", "km"]:
            recs[c] = recs[c].round(-2)
        pts[name] = recs.values.tolist()
        pts[name + "_cols"] = cols

    results = {"metrics": m, "eda": json.loads(Path("results/eda.json").read_text()),
               "diagnostics": json.loads(Path("results/diagnostics.json").read_text()),
               "cleaning": json.loads(Path("results/cleaning_log.json").read_text())}
    for t in ["linear", "forest"]:  # keep the page payload small
        for s in results["metrics"]["loso"].values():
            s["tuning"][t].pop("grid", None)

    def clean(o):  # NaN is not valid JSON; browsers reject it
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    def dump(name, obj):
        (out / name).write_text(json.dumps(clean(obj), separators=(",", ":"), allow_nan=False))
        print(name, f"{(out / name).stat().st_size / 1e6:.2f} MB")

    dump("results.json", results)
    dump("points.json", pts)
    dump("models.json", {"linear": linear, "forest": forest, "lookup": lookup, "options": opts})

    # Reference predictions to unit-test the JavaScript implementation
    ref = X.sample(25, random_state=1)
    test = {"rows": ref.to_dict(orient="records"), "linear": lin.predict(ref).tolist(),
            "forest": rf.predict(ref).tolist(), "lookup": look.predict(ref).tolist()}
    Path("results/js_reference.json").write_text(json.dumps(test))


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("data/clean.csv", "docs/data")))
