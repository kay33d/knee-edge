"""Recompute a Stage 2 run from its saved predictions: macro score, pair counts, bootstrap intervals.

Usage: analyse_run.py <run_dir>. Reads non-gold labels and metadata only; prints no study-level data.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "D:/claude files/knee-edge"
run = sys.argv[1]
sys.path.insert(0, f"{ROOT}/training")
import train_stage2 as ts  # noqa: E402

pred = pd.read_csv(f"{run}/val_predictions.csv", dtype={"StudyInstanceUID": str})
labels = pd.read_csv(f"{ROOT}/data_meta/pilkwang/report_labels_v2.csv", dtype={"StudyInstanceUID": str})
series = pd.read_csv(f"{ROOT}/data_meta/train_series.csv", dtype=str)
meta = pd.read_csv(f"{run}/cache_meta.csv", dtype={"series": str, "scanner": str})

merged = series.merge(meta, left_on="SeriesInstanceUID", right_on="series")
scanner_of = merged.groupby("StudyInstanceUID")["scanner"].first()

df = pred.merge(labels, on="StudyInstanceUID", suffixes=("_p", ""))
assert len(df) == len(pred)
names = ts.LABELS
p = df[[f"{n}_p" for n in names]].to_numpy(np.float64)
y = df[names].to_numpy(np.float64)
y[df[[f"{n}__conf" for n in names]].to_numpy(np.float64) <= ts.SILENT_CONF] = np.nan
strata = df["StudyInstanceUID"].map(scanner_of).fillna("unknown").to_numpy()

macro, each = ts.within_scanner_concordance(y, p, strata)
print(f"studies {len(df)}, scanners {len(np.unique(strata))}, recomputed macro {macro:.4f}")

rng = np.random.default_rng(0)
ids = {s: np.flatnonzero(strata == s) for s in np.unique(strata)}
draws = []
for _ in range(500):
    # Resample within each scanner so every draw has the same strata.
    pick = np.concatenate([rng.choice(v, len(v)) for v in ids.values()])
    draws.append(ts.within_scanner_concordance(y[pick], p[pick], strata[pick]))
macro_draws = np.array([d[0] for d in draws])
each_draws = np.array([d[1] for d in draws])
print(f"macro 95% interval {np.percentile(macro_draws, 2.5):.4f} to {np.percentile(macro_draws, 97.5):.4f}")

rows = []
for c, name in enumerate(names):
    pairs, stud = 0, 0
    for idx in ids.values():
        yc = y[idx, c]
        yc = yc[~np.isnan(yc)]
        stud += len(yc)
        pairs += int((yc[:, None] > yc[None, :]).sum())
    kept = y[:, c][~np.isnan(y[:, c])]
    rows.append({"finding": name, "c": round(each[c], 4),
                 "lo": round(np.nanpercentile(each_draws[:, c], 2.5), 3),
                 "hi": round(np.nanpercentile(each_draws[:, c], 97.5), 3),
                 "scored": stud, "share>0.5": round(float((kept > 0.5).mean()), 3),
                 "distinct": len(np.unique(np.round(kept, 3))), "pairs": pairs})
print(pd.DataFrame(rows).to_string(index=False))

sizes = pd.Series(strata).value_counts()
print("largest scanners:", sizes.head(5).to_dict())
print("scanners with under 5 studies:", int((sizes < 5).sum()))
print("prediction spread (std) per finding:", dict(zip(names, np.round(p.std(0), 3))))
print("mean prediction:", dict(zip(names, np.round(p.mean(0), 3))))
