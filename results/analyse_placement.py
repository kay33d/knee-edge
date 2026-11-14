"""Placement follow-up: evenly spaced (ends included), midpoint and centre-weighted windows on all 5 folds.

Reads results/place_fold{0..4}/<model>/ and results/w18c_fold0/, and writes aggregates only to results/placement_summary.json.

Each fold is compared with its own ConvNeXt-Atto scored at 72 evenly spaced windows, paired by study, with 500 bootstrap
draws within scanners. The pooled difference is the mean of the 5 fold differences; its interval comes from averaging one
bootstrap draw per fold, so it reflects study sampling within folds, not retraining noise.
"""
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
MENISCI = [NAMES.index(n) for n in ("Medial Meniscus", "Lateral Meniscus")]
MODES = ("uniform", "midpoint", "centre")


def files(folder):
    """{(placement, windows): path} for one scoring run."""
    out = {}
    for mode in MODES:
        for w in (72, 36, 18):
            path = f"{folder}/val_predictions_{mode}_w{w}.csv"
            if mode == "uniform" and not os.path.exists(path):
                path = f"{folder}/val_predictions.csv" if w == 72 else f"{folder}/val_predictions_w{w}.csv"
            if os.path.exists(path):
                out[(mode, w)] = path
    return out


def auc_thresholded(y, p):
    each = []
    for c in range(y.shape[1]):
        keep = ~np.isnan(y[:, c])
        b = y[keep, c] > 0.5
        each.append(roc_auc_score(b, p[keep, c]) if 0 < b.sum() < len(b) else np.nan)
    return float(np.nanmean(each))


class Fold:
    def __init__(self, reference):
        labels = pd.read_csv(f"{ROOT}/data_meta/pilkwang/report_labels_v2.csv", dtype={"StudyInstanceUID": str})
        series = pd.read_csv(f"{ROOT}/data_meta/train_series.csv", dtype=str)
        meta = pd.read_csv(os.path.join(os.path.dirname(reference), "cache_meta.csv"), dtype={"series": str, "scanner": str})
        scanner_of = series.merge(meta, left_on="SeriesInstanceUID", right_on="series").groupby("StudyInstanceUID")["scanner"].first()
        base = self.read(reference)
        self.ids = base["StudyInstanceUID"]
        df = self.ids.to_frame().merge(labels, on="StudyInstanceUID")
        assert len(df) == len(self.ids)
        self.y = df[NAMES].to_numpy(np.float64)
        self.y[df[[f"{n}__conf" for n in NAMES]].to_numpy(np.float64) <= ts.SILENT_CONF] = np.nan
        self.strata = df["StudyInstanceUID"].map(scanner_of).fillna("unknown").to_numpy()
        rng = np.random.default_rng(0)
        groups = [np.flatnonzero(self.strata == v) for v in np.unique(self.strata)]
        self.picks = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(N_DRAWS)]
        self.base = self.score(reference)

    @staticmethod
    def read(path):
        return pd.read_csv(path, dtype={"StudyInstanceUID": str}).sort_values("StudyInstanceUID").reset_index(drop=True)

    def score(self, path):
        frame = self.read(path)
        assert frame["StudyInstanceUID"].equals(self.ids), f"{path}: different validation studies"
        p = frame[NAMES].to_numpy(np.float64)
        c, each = ts.within_scanner_concordance(self.y, p, self.strata)
        draws = np.array([ts.within_scanner_concordance(self.y[i], p[i], self.strata[i])[1] for i in self.picks])
        return {"c": c, "each": np.array(each), "draws": draws, "auc": auc_thresholded(self.y, p)}

    def compare(self, path):
        a, b = self.score(path), self.base
        return {"c": a["c"], "auc": a["auc"], "c_diff": a["c"] - b["c"], "auc_diff": a["auc"] - b["auc"],
                "draws": np.nanmean(a["draws"] - b["draws"], axis=1),
                "men": a["draws"][:, MENISCI].mean(1) - b["draws"][:, MENISCI].mean(1),
                "men_diff": float(np.mean(a["each"][MENISCI] - b["each"][MENISCI])), "each_diff": a["each"] - b["each"]}


def ci(values):
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def main():
    summary = {"reference": "each fold's ConvNeXt-Atto at 72 evenly spaced windows", "five_folds": {}, "fold0_models": {}}
    per_arm = {}
    for fold in range(5):
        folder = f"{R}/place_fold{fold}/convnext_atto@place"
        found = files(folder)
        if ("uniform", 72) not in found:
            continue
        f = Fold(found[("uniform", 72)])
        for arm, path in found.items():
            per_arm.setdefault(arm, {})[fold] = f.compare(path)
        if fold == 0:  # The models trained at each budget, and the model trained on centre-weighted windows, against the same reference.
            runs = {f"trained at {w}": f"{R}/place_fold0/convnext_atto@w{w}_place" for w in (18, 36, 72)}
            runs["trained at 18, centre-weighted"] = f"{R}/w18c_fold0/convnext_atto@w18c"
            for name, run in runs.items():
                if not os.path.isdir(run):
                    continue
                if name.endswith("centre-weighted"):  # Scored only at its own budget: the main file is centre, 18 windows.
                    arms = {("centre", 18): f"{run}/val_predictions.csv",
                            **{(m, 18): f"{run}/val_predictions_{m}_w18.csv" for m in ("uniform", "midpoint")}}
                else:
                    arms = files(run)
                for (mode, w), path in arms.items():
                    if os.path.exists(path):
                        v = f.compare(path)
                        summary["fold0_models"].setdefault(name, {})[f"{mode} [{w}]"] = {
                            "c": round(v["c"], 4), "c_diff": round(v["c_diff"], 4), "c_diff_ci": ci(v["draws"]), "auc": round(v["auc"], 4),
                            "menisci_c_diff": round(v["men_diff"], 4)}
    for (mode, w), folds in sorted(per_arm.items(), key=lambda kv: (-kv[0][1], MODES.index(kv[0][0]))):
        n = sorted(folds)
        pooled = np.mean([folds[k]["draws"] for k in n], axis=0)
        pooled_men = np.mean([folds[k]["men"] for k in n], axis=0)
        summary["five_folds"][f"{mode} [{w}]"] = {
            "placement": mode, "windows": w, "folds": n,
            "c_diff_by_fold": [round(folds[k]["c_diff"], 4) for k in n], "auc_diff_by_fold": [round(folds[k]["auc_diff"], 4) for k in n],
            "c_by_fold": [round(folds[k]["c"], 4) for k in n],
            "c_diff_mean": round(float(np.mean([folds[k]["c_diff"] for k in n])), 4), "c_diff_ci": ci(pooled),
            "auc_diff_mean": round(float(np.mean([folds[k]["auc_diff"] for k in n])), 4),
            "auc_mean": round(float(np.mean([folds[k]["auc"] for k in n])), 4),
            "menisci_c_diff_mean": round(float(np.mean([folds[k]["men_diff"] for k in n])), 4), "menisci_c_diff_ci": ci(pooled_men),
            "per_finding_c_diff_mean": {name: round(float(np.mean([folds[k]["each_diff"][i] for k in n])), 4) for i, name in enumerate(NAMES)}}
    # Paired placement contrasts at each budget, pooled over folds: is centre-weighted better than midpoint?
    summary["contrasts"] = {}
    for w in (72, 36, 18):
        for a, b in (("centre", "midpoint"), ("midpoint", "uniform"), ("centre", "uniform")):
            if (a, w) in per_arm and (b, w) in per_arm:
                n = sorted(set(per_arm[(a, w)]) & set(per_arm[(b, w)]))
                d = np.mean([per_arm[(a, w)][k]["draws"] - per_arm[(b, w)][k]["draws"] for k in n], axis=0)
                summary["contrasts"][f"{a} minus {b} [{w}]"] = {
                    "folds": n, "c_diff_mean": round(float(np.mean([per_arm[(a, w)][k]["c_diff"] - per_arm[(b, w)][k]["c_diff"] for k in n])), 4),
                    "c_diff_ci": ci(d), "by_fold": [round(per_arm[(a, w)][k]["c_diff"] - per_arm[(b, w)][k]["c_diff"], 4) for k in n]}
    with open(f"{R}/placement_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    for k, v in summary["five_folds"].items():
        print(f"{k:16s} folds {v['folds']}  C diff {v['c_diff_mean']:+.4f} {v['c_diff_ci']}  by fold {v['c_diff_by_fold']}  menisci {v['menisci_c_diff_mean']:+.4f}")
    for k, v in summary["contrasts"].items():
        print(f"{k:28s} {v['c_diff_mean']:+.4f} {v['c_diff_ci']}  by fold {v['by_fold']}")
    for name, arms in summary["fold0_models"].items():
        for arm, v in arms.items():
            print(f"fold 0, {name:32s} {arm:14s} C {v['c']:.4f} {v['c_diff']:+.4f} {v['c_diff_ci']}  AUC {v['auc']:.4f}")


if __name__ == "__main__":
    main()
