"""Step 1: clean the combined dataset and reconstruct source identity.

The Kaggle "Top 4 Used Car Datasets Combined" file has no column saying which
of the four original datasets a row came from. Each original dataset recorded a
different set of columns, so the pattern of missing columns identifies the
source. All rules here are fixed, row-level rules (no statistics are learned
from the data), so applying them before splitting cannot leak test information.

Usage: python src/prepare.py data/raw.csv data/clean.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def assign_source(df: pd.DataFrame) -> pd.Series:
    eng, own = df.engine.notna(), df.owner.notna()
    loc, mil = df.location.notna(), df.mileage.notna()
    typ, seat = df.type.notna(), df.seats.notna()
    rules = [
        ~own & mil,                 # A: specs (engine/mileage/power/seats), never owner
        ~eng & own & (typ | seat),  # B: owner, location, seats, body type; never engine
        own & mil & loc,            # C: everything except body type
        own & ~loc & ~mil,          # D: core columns (+engine, owner) only
    ]
    return pd.Series(np.select(rules, list("ABCD"), "?"), index=df.index)


def main(raw_path: str, out_path: str) -> None:
    df = pd.read_csv(raw_path)
    log = {"raw_rows": int(len(df))}

    # Rows with no brand/model cannot be described at prediction time.
    bad = df.brand.isna() | df.model.isna() | df.age.isna()
    log["dropped_missing_identity"] = int(bad.sum())
    df = df[~bad].copy()

    df["source"] = assign_source(df)
    log["unassigned_source"] = int((df.source == "?").sum())
    df = df[df.source != "?"]
    log["source_counts_before_dedup"] = df.source.value_counts().sort_index().to_dict()

    # Exact duplicate listings (all within one source) would sit on both sides of a
    # random split and flatter the in-distribution score.
    dup = df.duplicated()
    log["dropped_exact_duplicates"] = int(dup.sum())
    df = df[~dup]

    # Physically implausible odometer values (>500,000 km; max was 6.5 million).
    bad_km = df.km > 500_000
    log["dropped_km_over_500k"] = int(bad_km.sum())
    df = df[~bad_km]

    # Zero is used as a missing-value code for these specs.
    for col in ["mileage", "seats", "engine", "power"]:
        n0 = int((df[col] == 0).sum())
        if n0:
            log[f"zero_to_missing_{col}"] = n0
        df.loc[df[col] == 0, col] = np.nan

    df["brand_model"] = df.brand + " " + df.model
    df["price_lakh"] = df.price / 1e5
    log["final_rows"] = int(len(df))
    log["source_counts"] = df.source.value_counts().sort_index().to_dict()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_csv(out_path, index=False)
    Path("results").mkdir(exist_ok=True)
    Path("results/cleaning_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("data/raw.csv", "data/clean.csv")))
