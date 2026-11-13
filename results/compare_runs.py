"""Compare finished Stage 2 runs with the ResNet-50 baseline, paired by study.

Usage: compare_runs.py <baseline_dir> <run_dir>... Prints no study-level data.
A run directory needs val_predictions.csv, pilot_report.json and cache_meta.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "D:/claude files/knee-edge"
sys.path.insert(0, f"{ROOT}/training")
import train_stage2 as ts  # noqa: E402

names = ts.LABELS
labels = pd.read_csv(f"{ROOT}/data_meta/pilkwang/report_labels_v2.csv", dtype={"StudyInstanceUID": str})
series = pd.read_csv(f"{ROOT}/data_meta/train_series.csv", dtype=str)


def load(run):
    """A run folder, or "folder#w36" for its predictions with 36 windows."""
    folder, _, windows = run.partition("#")
    pred = pd.read_csv(f"{folder}/val_predictions{'_' + windows if windows else ''}.csv", dtype={"StudyInstanceUID": str})
    return pred.sort_values("StudyInstanceUID").reset_index(drop=True)


base_dir, runs = sys.argv[1], sys.argv[2:]
base = load(base_dir)
meta = pd.read_csv(f"{base_dir}/cache_meta.csv", dtype={"series": str, "scanner": str})
scanner_of = series.merge(meta, left_on="SeriesInstanceUID", right_on="series").groupby("StudyInstanceUID")["scanner"].first()
df = base[["StudyInstanceUID"]].merge(labels, on="StudyInstanceUID")
assert len(df) == len(base)
y = df[names].to_numpy(np.float64)
y[df[[f"{n}__conf" for n in names]].to_numpy(np.float64) <= ts.SILENT_CONF] = np.nan
strata = df["StudyInstanceUID"].map(scanner_of).fillna("unknown").to_numpy()
ids = [np.flatnonzero(strata == s) for s in np.unique(strata)]
no_syn = [i for i, n in enumerate(names) if n != "Synovitis"]
rng = np.random.default_rng(0)
picks = [np.concatenate([rng.choice(v, len(v)) for v in ids]) for _ in range(500)]


def score(p, idx=None):
    if idx is None:
        return ts.within_scanner_concordance(y, p, strata)
    return ts.within_scanner_concordance(y[idx], p[idx], strata[idx])


base_p = base[names].to_numpy(np.float64)
base_draws = [score(base_p, i)[1] for i in picks]
rows, per = [], {}
for run in runs:
    other = load(run)
    assert other["StudyInstanceUID"].equals(base["StudyInstanceUID"]), f"{run}: different validation studies"
    p = other[names].to_numpy(np.float64)
    macro, each = score(p)
    draws = [score(p, i)[1] for i in picks]
    diff = np.nanmean(np.array(draws) - np.array(base_draws), axis=1)
    with open(f"{run.partition('#')[0]}/pilot_report.json") as handle:
        report = json.load(handle)
    epochs = report.get("epochs") or [{"within_scanner_c": float("nan"), "train_s": 0, "val_s": 0}]  # A scored-only run has none.
    curve = [e["within_scanner_c"] for e in epochs]
    label = report["config"]["backbone"] + ("#" + run.partition("#")[2] if "#" in run else "")
    rows.append({"model": label, "status": report["status"], "macro": round(macro, 4),
                 "no_synovitis": round(float(np.nanmean(np.array(each)[no_syn])), 4),
                 "diff": round(macro - score(base_p)[0], 4), "diff_lo": round(np.percentile(diff, 2.5), 4),
                 "diff_hi": round(np.percentile(diff, 97.5), 4), "best_epoch": int(np.argmax(curve)),
                 "last3": [round(v, 4) for v in curve[-3:]], "train_h": round(sum(e["train_s"] + e["val_s"] for e in epochs) / 3600, 2),
                 "s_per_epoch": epochs[-1]["train_s"], "gpu_gb": report.get("gpu_peak_gb"),
                 "reused_cache": report["preprocess"].get("reused_cache"), "failed_series": report["preprocess"]["failed_series"]})
    per[label] = np.round(each, 3)
base_macro, base_each = score(base_p)
no_syn_base = float(np.nanmean(np.array(base_each)[no_syn]))
print(f"baseline macro {base_macro:.4f}, without Synovitis {no_syn_base:.4f}")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame({os.path.basename(os.path.normpath(base_dir)) + " (baseline)": np.round(base_each, 3), **per}, index=names).to_string())
