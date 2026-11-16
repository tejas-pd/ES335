"""Step 3: post-hoc error analysis on the held-out predictions.

Nothing here is used to choose or tune a model. These analyses look at the
leave-one-source-out (LOSO) test predictions to explain *why* errors grow.

  * error by price band and by age
  * the largest errors
  * few-shot recalibration: if k labelled listings from the new source were
    available, how much would one multiplicative correction factor help?
  * age-offset check: refit each LOSO model and shift the held-out source's
    recorded age by -4..+4 years to see which offset removes the median bias.

Usage: python src/diagnostics.py data/clean.csv
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from models import CAT, NUM, SEED, make_forest, make_linear  # noqa: E402

MODELS = ["mean", "lookup", "linear", "forest"]
COLORS = {"A": "#2a6fdb", "B": "#e07b39", "C": "#3a9e6e", "D": "#9b5cc4"}


def main(path):
    df = pd.read_csv(path)
    m = json.loads(Path("results/metrics.json").read_text())
    pr = pd.read_csv("results/loso_predictions.csv")
    pr = pr.merge(df[["brand", "brand_model", "age", "km", "fuel", "transmission"]],
                  left_on="row", right_index=True)
    out = {}

    # ---- error by price band (absolute % error, median) ----
    bands = [0, 3e5, 6e5, 1e6, 2e6, 1e9]
    names = ["<3 L", "3–6 L", "6–10 L", "10–20 L", ">20 L"]
    pr["band"] = pd.cut(pr.actual, bands, labels=names, right=False)
    out["by_band"] = {}
    for s, g in pr.groupby("source"):
        out["by_band"][s] = {}
        for b, h in g.groupby("band", observed=False):
            if len(h) == 0:
                continue
            out["by_band"][s][b] = {"n": int(len(h)), **{
                k: {"medape": float(np.median(np.abs(h[k] - h.actual) / h.actual) * 100),
                    "bias": float(np.median(h[k] / h.actual - 1) * 100)} for k in MODELS}}

    # ---- largest errors (forest), with context ----
    pr["ape_forest"] = np.abs(pr.forest - pr.actual) / pr.actual * 100
    worst = pr.sort_values("ape_forest", ascending=False).groupby("source").head(4)
    out["worst"] = worst[["source", "brand_model", "age", "km", "fuel", "transmission", "actual",
                          "lookup", "linear", "forest", "unseen_model"]].round(0).to_dict(orient="records")

    # ---- few-shot recalibration ----
    rng = np.random.default_rng(SEED)
    ks = [0, 5, 10, 25, 50, 100, 200]
    out["recalibration"] = {}
    for s, g in pr.groupby("source"):
        y = g.actual.to_numpy()
        out["recalibration"][s] = {}
        for k_model in MODELS:
            p = g[k_model].to_numpy()
            res = []
            for k in ks:
                if k == 0:
                    res.append({"k": 0, "mae": float(np.mean(np.abs(p - y))), "lo": None, "hi": None,
                                "medape": float(np.median(np.abs(p - y) / y) * 100)})
                    continue
                maes, apes = [], []
                for _ in range(200):
                    idx = rng.choice(len(y), size=k, replace=False)
                    rest = np.ones(len(y), bool); rest[idx] = False
                    factor = np.median(y[idx] / p[idx])  # robust multiplicative correction
                    q = p[rest] * factor
                    maes.append(np.mean(np.abs(q - y[rest])))
                    apes.append(np.median(np.abs(q - y[rest]) / y[rest]) * 100)
                res.append({"k": k, "mae": float(np.mean(maes)), "lo": float(np.percentile(maes, 5)),
                            "hi": float(np.percentile(maes, 95)), "medape": float(np.mean(apes))})
            out["recalibration"][s][k_model] = res

    # ---- age-offset check (refit LOSO linear + forest with their tuned params) ----
    X = df[CAT + NUM + ["source"]]
    y_all = df.price.to_numpy(float)
    offsets = list(range(-4, 5))
    out["age_offset"] = {}
    for s in "ABCD":
        tr = (df.source != s).to_numpy()
        lin = make_linear(**m["loso"][s]["tuning"]["linear"]["best"]).fit(X[tr], y_all[tr])
        rf = make_forest(**m["loso"][s]["tuning"]["forest"]["best"]).fit(X[tr], y_all[tr])
        te, yte = X[~tr].copy(), y_all[~tr]
        out["age_offset"][s] = {"linear": [], "forest": []}
        for o in offsets:
            t2 = te.assign(age=np.clip(te.age + o, 0, None))
            for name, mod in [("linear", lin), ("forest", rf)]:
                p = mod.predict(t2)
                out["age_offset"][s][name].append({"offset": o,
                                                   "bias": float(np.median(p / yte - 1) * 100),
                                                   "mae": float(np.mean(np.abs(p - yte)))})
        best = min(out["age_offset"][s]["linear"], key=lambda r: abs(r["bias"]))
        print(s, "linear offset removing bias:", best)

    Path("results/diagnostics.json").write_text(json.dumps(out, indent=2))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for s in "ABCD":
        r = out["recalibration"][s]["forest"]
        ax.plot([x["k"] for x in r], [x["mae"] / 1e5 for x in r], marker="o", color=COLORS[s],
                label=f"held-out {s}")
    ax.set_xscale("symlog", linthresh=5); ax.set_xlabel("Labelled listings from the new source (k)")
    ax.set_ylabel("Forest MAE (₹ lakh)"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("figures/recalibration.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for s in "ABCD":
        r = out["age_offset"][s]["linear"]
        ax.plot([x["offset"] for x in r], [x["bias"] for x in r], marker="o", color=COLORS[s],
                label=f"held-out {s}")
    ax.axhline(0, color="#888", lw=1)
    ax.set_xlabel("Years added to recorded age of held-out source")
    ax.set_ylabel("Median signed error (%)"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("figures/age_offset.png", dpi=150); plt.close(fig)
    print(json.dumps({s: {k: [round(r["mae"]) for r in v] for k, v in d.items() if k in ("lookup", "forest", "linear")}
                      for s, d in out["recalibration"].items()}, indent=0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/clean.csv")
