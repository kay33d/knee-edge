"""Stage 4: how much each precision changes discrimination and calibration, paired by study against true FP32.

Reads results/stage4_*_fold0/<model>/val_predictions_<precision>.csv. The reference for each model is its FP32 file.
Prints and saves aggregates only, never study-level data.

Discrimination: macro within-scanner concordance, 500 paired bootstrap draws within scanners.
Calibration, against the soft labels with silent labels left out (the labels are LLM readings of reports, not truth):
  ECE with 10 equal-width bins, ECE with 10 equal-count bins, and Brier score, each macro over findings.
"After recalibration" fits slope and intercept on the logits per finding, cross-fitted in 5 parts of the validation
studies, so no study is calibrated by a fit that saw it. The fit is not repeated inside each bootstrap draw.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "D:/claude files/knee-edge"
sys.path.insert(0, f"{ROOT}/training")
import train_stage2 as ts  # noqa: E402

N_DRAWS = 500
# Folder of a QAT result: (the folder of its starting model, label prefix). "plain" is FP32 with the fine-tuned weights.
QAT_RUNS = {"convnext_atto@qat": ("convnext_atto@eval", ""), "convnext_tiny@qat": ("convnext_tiny@eval", ""),
            "convnext_tiny@tf_qat": ("convnext_tiny@tf_eval", ""),
            "convnext_atto@qat_eval": ("convnext_atto@eval", "qat_"), "convnext_tiny@qat_eval": ("convnext_tiny@eval", "qat_"),
            "convnext_tiny@tf_qat_eval": ("convnext_tiny@tf_eval", "qat_")}


def ece_finding(y, p, bins=10, equal_count=False):
    keep = ~np.isnan(y)
    y, p = y[keep], p[keep]
    if not len(y):
        return float("nan")
    if equal_count:
        order = np.argsort(p, kind="stable")
        groups = [g for g in np.array_split(order, bins) if len(g)]
    else:
        edges = np.minimum((p * bins).astype(int), bins - 1)
        groups = [np.flatnonzero(edges == b) for b in np.unique(edges)]
    return float(sum(len(g) * abs(p[g].mean() - y[g].mean()) for g in groups) / len(y))


def calibration(y, p):
    """Macro (ECE equal width, ECE equal count, Brier)."""
    n = y.shape[1]
    width = np.nanmean([ece_finding(y[:, c], p[:, c]) for c in range(n)])
    count = np.nanmean([ece_finding(y[:, c], p[:, c], equal_count=True) for c in range(n)])
    brier = np.nanmean(np.nanmean((p - y) ** 2, axis=0))
    return float(width), float(count), float(brier)


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def soft_bce(y, z):
    return float(np.mean(np.logaddexp(0, z) - y * z))


def fit_platt(z, y, steps=50):
    """Slope and intercept on logits by damped Newton steps. Soft targets keep the problem convex."""
    X = np.stack([z, np.ones_like(z)], 1)
    theta = np.array([1.0, 0.0])
    loss = soft_bce(y, X @ theta)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ theta)))
        gradient = X.T @ (p - y) / len(y)
        hessian = (X.T * (p * (1 - p))) @ X / len(y) + 1e-8 * np.eye(2)
        step, rate = np.linalg.solve(hessian, gradient), 1.0
        while rate > 1e-4 and soft_bce(y, X @ (theta - rate * step)) > loss:
            rate /= 2
        if rate <= 1e-4:
            break
        theta -= rate * step
        new = soft_bce(y, X @ theta)
        if loss - new < 1e-10:
            loss = new
            break
        loss = new
    return theta


def recalibrated(y, p, parts=5, seed=0):
    """Cross-fitted slope-and-intercept recalibration per finding. Silent labels keep their prediction and are not scored."""
    out = p.copy()
    z = to_logit(p)
    for c in range(y.shape[1]):
        known = np.flatnonzero(~np.isnan(y[:, c]))
        if len(known) < parts * 2:
            continue
        shuffled = np.random.default_rng(seed + c).permutation(known)
        for held in np.array_split(shuffled, parts):
            fit = np.setdiff1d(known, held)
            a, b = fit_platt(z[fit, c], y[fit, c])
            out[held, c] = 1 / (1 + np.exp(-(a * z[held, c] + b)))
    return out


def interval(values):
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def find_runs(root):
    """{model: {precision label: file}} from every Stage 4 result folder."""
    found = {}
    for path in sorted(glob.glob(f"{root}/stage4_*_fold0/*/val_predictions_*.csv")):
        path = path.replace("\\", "/")
        folder, model, name = path.rsplit("/", 3)[1:]
        precision = name[len("val_predictions_"):-len(".csv")]
        if precision.startswith("w"):
            continue
        if model in QAT_RUNS:  # QAT results are compared with the FP32 model they started from.
            model, prefix = QAT_RUNS[model]
            precision = prefix + precision
        found.setdefault(model, {}).setdefault(precision, []).append((folder, path))
    runs = {}
    for model, precisions in found.items():
        runs[model] = {}
        for precision, sources in precisions.items():
            for folder, path in sources:
                label = precision if len(sources) == 1 else f"{precision} [{folder.replace('stage4_', '').replace('_fold0', '')}]"
                runs[model][label] = path
    return runs


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/stage4_summary.json"
    names = ts.LABELS
    labels = pd.read_csv(f"{ROOT}/data_meta/pilkwang/report_labels_v2.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(f"{ROOT}/data_meta/train_series.csv", dtype=str)
    runs = find_runs(f"{ROOT}/results")
    summary = {}
    for model, files in runs.items():
        if "fp32" not in files:
            print(f"{model}: no FP32 reference, skipped")
            continue
        load = lambda path: pd.read_csv(path, dtype={"StudyInstanceUID": str}).sort_values("StudyInstanceUID").reset_index(drop=True)
        base = load(files["fp32"])
        meta = pd.read_csv(os.path.join(os.path.dirname(files["fp32"]), "cache_meta.csv"), dtype={"series": str, "scanner": str})
        scanner_of = series.merge(meta, left_on="SeriesInstanceUID", right_on="series").groupby("StudyInstanceUID")["scanner"].first()
        df = base[["StudyInstanceUID"]].merge(labels, on="StudyInstanceUID")
        assert len(df) == len(base)
        y = df[names].to_numpy(np.float64)
        y[df[[f"{n}__conf" for n in names]].to_numpy(np.float64) <= ts.SILENT_CONF] = np.nan
        strata = df["StudyInstanceUID"].map(scanner_of).fillna("unknown").to_numpy()
        ids = [np.flatnonzero(strata == s) for s in np.unique(strata)]
        rng = np.random.default_rng(0)
        picks = [np.concatenate([rng.choice(v, len(v)) for v in ids]) for _ in range(N_DRAWS)]

        def score(p, idx=None):
            return ts.within_scanner_concordance(y, p, strata) if idx is None else ts.within_scanner_concordance(y[idx], p[idx], strata[idx])

        base_p = base[names].to_numpy(np.float64)
        base_c, base_each = score(base_p)
        base_draws = np.array([score(base_p, i)[1] for i in picks])
        base_cal = np.array([calibration(y[i], base_p[i]) for i in picks])
        base_re = recalibrated(y, base_p)
        base_re_cal = np.array([calibration(y[i], base_re[i]) for i in picks])
        summary[model] = {}
        for label, path in sorted(files.items(), key=lambda kv: (kv[0] != "fp32", kv[0])):
            other = load(path)
            assert other["StudyInstanceUID"].equals(base["StudyInstanceUID"]), f"{path}: different validation studies"
            p = other[names].to_numpy(np.float64)
            macro, each = score(p)
            draws = np.array([score(p, i)[1] for i in picks])
            diff = np.nanmean(draws - base_draws, axis=1)
            cal, cal_draws = calibration(y, p), np.array([calibration(y[i], p[i]) for i in picks])
            again = recalibrated(y, p)
            re_cal, re_draws = calibration(y, again), np.array([calibration(y[i], again[i]) for i in picks])
            known = ~np.isnan(y)
            summary[model][label] = {
                "c": round(macro, 4), "c_diff": round(macro - base_c, 4), "c_diff_ci": interval(diff),
                "per_finding_c_diff": {n: round(float(a - b), 4) for n, a, b in zip(names, each, base_each)},
                "ece": round(cal[0], 4), "ece_diff_ci": interval(cal_draws[:, 0] - base_cal[:, 0]),
                "ece_equal_count": round(cal[1], 4), "ece_equal_count_diff_ci": interval(cal_draws[:, 1] - base_cal[:, 1]),
                "brier": round(cal[2], 5), "brier_diff_ci": [round(float(np.percentile(cal_draws[:, 2] - base_cal[:, 2], q)), 5) for q in (2.5, 97.5)],
                "ece_after_recal": round(re_cal[0], 4), "ece_after_recal_diff_ci": interval(re_draws[:, 0] - base_re_cal[:, 0]),
                "brier_after_recal": round(re_cal[2], 5),
                "brier_after_recal_diff_ci": [round(float(np.percentile(re_draws[:, 2] - base_re_cal[:, 2], q)), 5) for q in (2.5, 97.5)],
                "mean_abs_prob_change": round(float(np.abs(p - base_p)[known].mean()), 4),
                "mean_prob_shift": round(float((p - base_p)[known].mean()), 4),
            }
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for model, rows in summary.items():
        table = pd.DataFrame({label: {"C": v["c"], "dC": v["c_diff"], "dC 95%": str(v["c_diff_ci"]), "ECE": v["ece"], "dECE 95%": str(v["ece_diff_ci"]),
                                      "Brier": v["brier"], "dBrier 95%": str(v["brier_diff_ci"]), "ECE recal": v["ece_after_recal"],
                                      "Brier recal": v["brier_after_recal"], "|dp|": v["mean_abs_prob_change"], "shift": v["mean_prob_shift"]}
                              for label, v in rows.items()}).T
        print(f"\n{model}\n{table.to_string()}")


if __name__ == "__main__":
    main()
