"""Phase A of the adaptive sampling study: is uniform sampling the problem, and how much can choosing windows recover?

Reads the Phase A result folders (results/phasea_*_fold0/<model>/) and the Stage 3 Atto run, and writes aggregates only
to results/phasea_summary.json. Study-level files (predictions, attention dumps) stay local and are never published.

Every comparison is paired by study against ConvNeXt-Atto scored with 72 uniform windows, with 500 bootstrap draws
within scanners (the same procedure as analyse_stage4.py). Scores: macro within-scanner concordance (primary) and AUC
against the LLM labels thresholded at 0.5. Gold results are AUC against radiologist labels, read from the run reports.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "D:/claude files/knee-edge"
R = f"{ROOT}/results"
sys.path.insert(0, f"{ROOT}/training")
import train_stage2 as ts  # noqa: E402

N_DRAWS = 500
NAMES = ts.LABELS
REFERENCE = f"{R}/stage3_atto_fold0/convnext_atto@eval/val_predictions.csv"  # Atto, 72 uniform windows.
MENISCI = ["Medial Meniscus", "Lateral Meniscus"]


def arms():
    """{label: (budget in windows, prediction file)} for every Phase A arm that exists."""
    out = {}
    stage3 = f"{R}/stage3_atto_fold0/convnext_atto@eval"
    for w, suffix in ((72, ""), (36, "_w36"), (18, "_w18")):
        out[f"uniform, trained on 48 [{w}]"] = (w, f"{stage3}/val_predictions{suffix}.csv")
    for mode in ("random", "centre"):
        folder = f"{R}/phasea_sampling_fold0/convnext_atto@{mode}"
        for w, suffix in ((72, ""), (36, "_w36"), (18, "_w18")):
            out[f"{mode}, trained on 48 [{w}]"] = (w, f"{folder}/val_predictions{suffix}.csv")
    for w in (18, 36, 72):
        out[f"uniform, trained at budget [{w}]"] = (w, f"{R}/phasea_train_fold0/convnext_atto@w{w}/val_predictions.csv")
    oracle = f"{R}/phasea_oracle_fold0/convnext_atto@oracle"
    out["dense, all windows [144]"] = (144, f"{oracle}/val_predictions_oracle_dense.csv")
    for w in (18, 36, 72):
        out[f"oracle, top by attention per slot [{w}]"] = (w, f"{oracle}/val_predictions_oracle_slot{w}.csv")
        out[f"oracle, top by attention per study [{w}]"] = (w, f"{oracle}/val_predictions_oracle_global{w}.csv")
    return {k: v for k, v in out.items() if os.path.exists(v[1])}


def auc_thresholded(y, p):
    each = []
    for c in range(y.shape[1]):
        keep = ~np.isnan(y[:, c])
        b = y[keep, c] > 0.5
        each.append(roc_auc_score(b, p[keep, c]) if 0 < b.sum() < len(b) else np.nan)
    return float(np.nanmean(each)), each


def interval(values):
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def load(path):
    return pd.read_csv(path, dtype={"StudyInstanceUID": str}).sort_values("StudyInstanceUID").reset_index(drop=True)


def missing_fs_groups(studies):
    """Validation studies by fluid-sensitive coverage: all three planes, no sagittal FS, no coronal FS."""
    series = pd.read_csv(f"{ROOT}/data_meta/train_series.csv", dtype=str)
    fs = series[series["Fluid_Sensitive"] == "1"]
    has = {plane: set(fs.loc[fs["Anatomical_Plane"] == plane, "StudyInstanceUID"]) for plane in ts.PLANES}
    s = pd.Series(studies)
    return {"all FS planes": s.isin(has["Sagittal"]) & s.isin(has["Coronal"]) & s.isin(has["Axial"]),
            "no sagittal FS": ~s.isin(has["Sagittal"]), "no coronal FS": ~s.isin(has["Coronal"])}


def attention_profiles(path, bins=10):
    """Where attention goes, from the oracle's dense pass: mass by relative position, per finding and slot plane.

    Consistency is the mean correlation between each study's position profile and the average profile; values near 1
    mean the same positions matter in every study, so a fixed non-uniform schedule could do as well as a learned one.
    """
    d = np.load(path, allow_pickle=True)
    w = d["weights"].astype(np.float32)  # [studies, slots*k, 12]
    pos = d["pos"].astype(np.float32)  # [studies, slots, k]
    mask = d["mask"]  # [studies, slots]
    n, s, k = pos.shape
    w = w.reshape(n, s, k, -1)
    edges = np.minimum((pos * bins).astype(int), bins - 1)
    out = {}
    for c, name in enumerate(NAMES):
        per_plane, consistency = {}, []
        for slot in range(s):
            present = mask[:, slot]
            if present.sum() < 10:
                continue
            prof = np.zeros((int(present.sum()), bins))
            for b in range(bins):
                prof[:, b] = (w[present, slot, :, c] * (edges[present, slot] == b)).sum(-1)
            mass = prof.sum(1, keepdims=True)
            share = prof / np.clip(mass, 1e-9, None)
            mean = share.mean(0)
            r = [np.corrcoef(row, mean)[0, 1] for row in share if row.std() > 0]
            consistency.append(float(np.nanmean(r)))
            label = f"{ts.PLANES[slot % 3]} {'FS' if slot < 3 else 'non-FS'}"
            per_plane[label] = {"mass": round(float(mass.mean()), 3), "profile": [round(float(v), 3) for v in mean]}
        top18 = np.sort(w[..., c].reshape(n, -1), 1)[:, ::-1][:, :18].sum(1)
        out[name] = {"planes": per_plane, "profile_consistency": round(float(np.nanmean(consistency)), 3),
                     "mass_in_top18_windows": round(float(top18.mean()), 3)}
    return out


def gold_results():
    rows = {}
    for path in glob.glob(f"{R}/phasea_*_fold0/*/pilot_report.json"):
        rep = json.load(open(path))
        if "gold" in rep:
            rows[path.replace("\\", "/").split("/")[-2]] = rep["gold"]
    return rows


def main():
    labels = pd.read_csv(f"{ROOT}/data_meta/pilkwang/report_labels_v2.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(f"{ROOT}/data_meta/train_series.csv", dtype=str)
    base = load(REFERENCE)
    meta = pd.read_csv(os.path.join(os.path.dirname(REFERENCE), "cache_meta.csv"), dtype={"series": str, "scanner": str})
    scanner_of = series.merge(meta, left_on="SeriesInstanceUID", right_on="series").groupby("StudyInstanceUID")["scanner"].first()
    df = base[["StudyInstanceUID"]].merge(labels, on="StudyInstanceUID")
    assert len(df) == len(base)
    y = df[NAMES].to_numpy(np.float64)
    y[df[[f"{n}__conf" for n in NAMES]].to_numpy(np.float64) <= ts.SILENT_CONF] = np.nan
    strata = df["StudyInstanceUID"].map(scanner_of).fillna("unknown").to_numpy()
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(strata == v) for v in np.unique(strata)]
    picks = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(N_DRAWS)]
    subgroups = missing_fs_groups(base["StudyInstanceUID"])

    def conc(p, idx=None):
        return ts.within_scanner_concordance(y, p, strata) if idx is None else ts.within_scanner_concordance(y[idx], p[idx], strata[idx])

    base_p = base[NAMES].to_numpy(np.float64)
    base_c, base_each = conc(base_p)
    base_draws = np.array([conc(base_p, i)[1] for i in picks])
    summary = {"reference": "ConvNeXt-Atto, 72 uniform windows (Stage 3 scoring of the Stage 2 model)", "studies": len(base),
               "subgroup_sizes": {k: int(v.sum()) for k, v in subgroups.items()}, "arms": {}}
    for label, (budget, path) in sorted(arms().items(), key=lambda kv: (kv[1][0], kv[0])):
        other = load(path)
        assert other["StudyInstanceUID"].equals(base["StudyInstanceUID"]), f"{path}: different validation studies"
        p = other[NAMES].to_numpy(np.float64)
        c, each = conc(p)
        draws = np.array([conc(p, i)[1] for i in picks])
        diff = np.nanmean(draws - base_draws, axis=1)
        idx_men = [NAMES.index(n) for n in MENISCI]
        men = draws[:, idx_men].mean(1) - base_draws[:, idx_men].mean(1)
        sub = {}
        for name, keep in subgroups.items():
            keep = keep.to_numpy()
            if keep.sum() >= 20:
                a, _ = auc_thresholded(y[keep], p[keep])
                b, _ = auc_thresholded(y[keep], base_p[keep])
                sub[name] = {"auc": round(a, 4), "auc_diff": round(a - b, 4)}
        summary["arms"][label] = {
            "windows": budget, "c": round(c, 4), "c_diff": round(c - base_c, 4), "c_diff_ci": interval(diff),
            "auc": round(auc_thresholded(y, p)[0], 4),
            "menisci_c_diff": round(float(np.mean([each[i] - base_each[i] for i in idx_men])), 4), "menisci_c_diff_ci": interval(men),
            "per_finding_c_diff": {n: round(float(a - b), 4) for n, a, b in zip(NAMES, each, base_each)}, "subgroups": sub}
    dump = f"{R}/phasea_oracle_fold0/convnext_atto@oracle/attention_dense.npz"
    if os.path.exists(dump):
        summary["attention"] = attention_profiles(dump)
    summary["gold"] = gold_results()
    with open(f"{R}/phasea_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"reference {base_c:.4f}; subgroups {summary['subgroup_sizes']}")
    for label, v in summary["arms"].items():
        print(f"{label:48s} C {v['c']:.4f} {v['c_diff']:+.4f} {v['c_diff_ci']}  AUC {v['auc']:.4f}  menisci {v['menisci_c_diff']:+.4f}")
    for model, g in summary["gold"].items():
        print(f"gold {model}: {g['macro_auc']} {g['macro_auc_95']}")
    if "attention" in summary:
        for name, a in summary["attention"].items():
            print(f"attention {name:18s} consistency {a['profile_consistency']:.2f}  top-18 mass {a['mass_in_top18_windows']:.2f}")


if __name__ == "__main__":
    main()
