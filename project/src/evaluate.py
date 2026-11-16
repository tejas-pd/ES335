"""Step 2: evaluate baselines and models.

Two evaluation designs, both fixed before any model was compared:
  1. Random split (in-distribution): stratified 80/20 split by source; tuning by
     5-fold CV on the 80% only; one final score on the 20%.
  2. Leave-one-source-out (dataset shift): for each source s, train on the other
     three, tune with an inner leave-one-source-out CV over those three (so tuning
     also rewards transfer), refit, and score once on s.

Usage: python src/evaluate.py data/clean.csv
Writes results/*.json, results/*.csv and figures/*.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from models import (CAT, FOREST_GRID, LINEAR_GRID, NUM, SEED, SPEC, LookupBaseline,  # noqa: E402
                    MeanBaseline, make_forest, make_linear)

RES = Path("results")
RES.mkdir(exist_ok=True)
MODELS = ["mean", "lookup", "linear", "forest"]
LABEL = {"mean": "Mean baseline", "lookup": "Model-median lookup",
         "linear": "Linear (ridge, log price)", "forest": "Random forest (log price)"}
N_BOOT = 1000


def metrics(y, p):
    e = p - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "medape": float(np.median(np.abs(e) / y) * 100),
            "bias_pct": float(np.median(p / y - 1) * 100),
            "within20": float(np.mean(np.abs(e) / y <= 0.2) * 100), "n": int(len(y))}


def tune(factory, grid, X, y, splits):
    """Return the grid point with the lowest mean validation MAE, and all scores."""
    scores = []
    for params in grid:
        maes = []
        for tr, va in splits:
            m = factory(**params).fit(X.iloc[tr], y[tr])
            maes.append(np.mean(np.abs(m.predict(X.iloc[va]) - y[va])))
        scores.append({"params": params, "cv_mae": float(np.mean(maes)),
                       "fold_mae": [float(v) for v in maes]})
    best = min(scores, key=lambda s: s["cv_mae"])
    return best["params"], scores


def fit_all(X, y, splits):
    lin_p, lin_s = tune(make_linear, LINEAR_GRID, X, y, splits)
    rf_p, rf_s = tune(make_forest, FOREST_GRID, X, y, splits)
    fitted = {"mean": MeanBaseline().fit(X, y), "lookup": LookupBaseline().fit(X, y),
              "linear": make_linear(**lin_p).fit(X, y), "forest": make_forest(**rf_p).fit(X, y)}
    return fitted, {"linear": {"best": lin_p, "grid": lin_s}, "forest": {"best": rf_p, "grid": rf_s}}


def bootstrap(y, preds, rng):
    """95% percentile intervals for each model's MAE and for paired MAE differences."""
    n = len(y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    ae = {k: np.abs(preds[k] - y) for k in preds}
    boot = {k: ae[k][idx].mean(axis=1) for k in ae}
    ci = {k: [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))] for k in boot}
    diffs = {}
    for a, b in [("forest", "linear"), ("linear", "lookup"), ("forest", "lookup"), ("lookup", "mean")]:
        d = boot[a] - boot[b]
        diffs[f"{a}_minus_{b}"] = {"est": float(ae[a].mean() - ae[b].mean()),
                                   "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return ci, diffs


def main(path):
    t0 = time.time()
    df = pd.read_csv(path)
    X = df[CAT + NUM + SPEC + ["source"]]
    y = df.price.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    out = {"label": LABEL}

    # ---------------- 1. random split -----------------
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED,
                                      stratify=df.source)
    Xtr, ytr, Xte, yte = X.iloc[tr_idx], y[tr_idx], X.iloc[te_idx], y[te_idx]
    splits = list(KFold(5, shuffle=True, random_state=SEED).split(Xtr))
    fitted, tuning = fit_all(Xtr, ytr, splits)
    preds = {k: fitted[k].predict(Xte) for k in MODELS}
    ci, diffs = bootstrap(yte, preds, rng)
    rand = {"overall": {k: metrics(yte, preds[k]) for k in MODELS}, "ci": ci, "diffs": diffs,
            "tuning": tuning, "by_source": {}}
    for s in "ABCD":
        mask = (Xte.source == s).to_numpy()
        rand["by_source"][s] = {k: metrics(yte[mask], preds[k][mask]) for k in MODELS}

    # Ablation: does adding the source-specific spec columns help in-distribution?
    rf_spec = make_forest(**tuning["forest"]["best"], with_specs=True).fit(Xtr, ytr)
    p_spec = rf_spec.predict(Xte)
    rand["ablation_specs"] = {"overall": metrics(yte, p_spec),
                              "by_source": {s: metrics(yte[(Xte.source == s).to_numpy()],
                                                       p_spec[(Xte.source == s).to_numpy()])
                                            for s in "ABCD"}}
    out["random_split"] = rand
    pd.DataFrame({"row": te_idx, "source": Xte.source.values, "actual": yte,
                  **{k: preds[k] for k in MODELS}}).to_csv(RES / "random_split_predictions.csv",
                                                           index=False)
    print(f"random split done {time.time()-t0:.0f}s", flush=True)

    # ---------------- 2. leave one source out -----------------
    loso, rows = {}, []
    for s in "ABCD":
        train = (X.source != s).to_numpy()
        Xtr, ytr, Xte, yte = X[train], y[train], X[~train], y[~train]
        inner = list(GroupKFold(n_splits=3).split(Xtr, groups=Xtr.source))
        fitted, tuning = fit_all(Xtr, ytr, inner)
        preds = {k: fitted[k].predict(Xte) for k in MODELS}
        ci, diffs = bootstrap(yte, preds, rng)
        seen_models = set(Xtr.brand_model)
        unseen = ~Xte.brand_model.isin(seen_models).to_numpy()
        loso[s] = {"overall": {k: metrics(yte, preds[k]) for k in MODELS}, "ci": ci, "diffs": diffs,
                   "tuning": tuning, "unseen_model_share": float(unseen.mean() * 100),
                   "seen": {k: metrics(yte[~unseen], preds[k][~unseen]) for k in MODELS},
                   "unseen": ({k: metrics(yte[unseen], preds[k][unseen]) for k in MODELS}
                              if unseen.sum() >= 20 else None)}
        rows.append(pd.DataFrame({"row": np.where(~train)[0], "source": s, "actual": yte,
                                  "unseen_model": unseen, **{k: preds[k] for k in MODELS}}))
        print(f"held-out {s} done {time.time()-t0:.0f}s  "
              + " ".join(f"{k}={loso[s]['overall'][k]['mae']:.0f}" for k in MODELS), flush=True)
    out["loso"] = loso
    pd.concat(rows).to_csv(RES / "loso_predictions.csv", index=False)

    (RES / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/clean.csv")
