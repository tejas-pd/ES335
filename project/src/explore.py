"""Data-understanding summaries per reconstructed source (descriptive only, no fitting).

Usage: python src/explore.py data/clean.csv
Writes results/eda.json and figures/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"A": "#2a6fdb", "B": "#e07b39", "C": "#3a9e6e", "D": "#9b5cc4"}
FIG = Path("figures")
FIG.mkdir(exist_ok=True)


def main(path):
    df = pd.read_csv(path)
    eda = {"n": int(len(df)), "sources": {}}
    cols = ["engine", "owner", "location", "mileage", "power", "seats", "type"]
    for s, g in df.groupby("source"):
        eda["sources"][s] = {
            "n": int(len(g)),
            "price_median": float(g.price.median()), "price_mean": float(g.price.mean()),
            "price_p05": float(g.price.quantile(.05)), "price_p95": float(g.price.quantile(.95)),
            "age_median": float(g.age.median()), "km_median": float(g.km.median()),
            "automatic_pct": float((g.transmission == "automatic").mean() * 100),
            "diesel_pct": float((g.fuel == "diesel").mean() * 100),
            "n_brands": int(g.brand.nunique()), "n_models": int(g.brand_model.nunique()),
            "luxury_pct": float(g.brand.isin(["mercedes-benz", "bmw", "audi", "jaguar", "volvo",
                                              "land rover", "lexus", "porsche", "mini"]).mean() * 100),
            "recorded_pct": {c: float(g[c].notna().mean() * 100) for c in cols},
            "top_brands": g.brand.value_counts().head(5).to_dict(),
        }
    # Overlap of model vocabularies between sources
    sets = {s: set(g.brand_model) for s, g in df.groupby("source")}
    eda["model_overlap"] = {s: {t: float(np.mean([m in sets[t] for m in df[df.source == s].brand_model]) * 100)
                                for t in sets if t != s} for s in sets}

    # Price distribution per source (log scale)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(np.log10(4e4), np.log10(8e6), 60)
    for s, g in df.groupby("source"):
        ax.hist(g.price, bins=bins, histtype="step", density=True, lw=1.8, color=COLORS[s],
                label=f"Source {s} (n={len(g):,})")
    ax.set_xscale("log"); ax.set_xlabel("Selling price (₹, log scale)"); ax.set_ylabel("Density")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "price_by_source.png", dpi=150); plt.close(fig)

    # Price vs age by source (median line)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for s, g in df.groupby("source"):
        med = g[g.age <= 18].groupby("age").price.median()
        ax.plot(med.index, med.values / 1e5, marker="o", ms=3, color=COLORS[s], label=f"Source {s}")
    ax.set_xlabel("Age (years, as recorded)"); ax.set_ylabel("Median price (₹ lakh)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "price_vs_age.png", dpi=150); plt.close(fig)

    # Same-car comparison: for the 10 commonest models, median price per source at age 5-8
    common = df.brand_model.value_counts().head(10).index
    sub = df[df.brand_model.isin(common) & df.age.between(5, 8)]
    tab = sub.pivot_table(index="brand_model", columns="source", values="price", aggfunc="median") / 1e5
    eda["same_model_median_lakh_age5to8"] = tab.round(2).where(tab.notna(), None).to_dict(orient="index")

    Path("results/eda.json").write_text(json.dumps(eda, indent=2))
    print(json.dumps({s: {k: v for k, v in d.items() if k not in ("recorded_pct", "top_brands")}
                      for s, d in eda["sources"].items()}, indent=1))
    print(json.dumps(eda["model_overlap"], indent=1))
    print(tab.round(2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/clean.csv")
