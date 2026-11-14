"""Rebuild docs/project_report.md from the result files. Run it after any new result.

The tables and headline figures are read or computed here, so they cannot drift from the results.
Only aggregates are written; no study-level data or labels. Paired intervals are cached in results/report_cache.json.
"""
import json
import os
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd

ROOT = "D:/claude files/knee-edge"
R = f"{ROOT}/results"
sys.path.insert(0, f"{ROOT}/training")
import train_stage2 as ts  # noqa: E402

NAMES = ts.LABELS
DRAWS = 500
CACHE_PATH = f"{R}/report_cache.json"
META = f"{R}/stage3_atto_fold0/convnext_atto@eval/cache_meta.csv"
NOT_YET = "Not available yet."


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def signed(value, digits=3):
    return f"{value:+.{digits}f}"


def span(ci, digits=3):
    return f"{ci[0]:+.{digits}f} to {ci[1]:+.{digits}f}"


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


class Scorer:
    """Paired comparison on the fold 0 validation studies, as in compare_runs.py."""

    def __init__(self, reference):
        labels = pd.read_csv(f"{ROOT}/data_meta/pilkwang/report_labels_v2.csv", dtype={"StudyInstanceUID": str})
        series = pd.read_csv(f"{ROOT}/data_meta/train_series.csv", dtype=str)
        meta = pd.read_csv(META, dtype={"series": str, "scanner": str})
        scanner_of = series.merge(meta, left_on="SeriesInstanceUID", right_on="series").groupby("StudyInstanceUID")["scanner"].first()
        self.ids = self.read(reference)["StudyInstanceUID"]
        df = self.ids.to_frame().merge(labels, on="StudyInstanceUID")
        assert len(df) == len(self.ids)
        self.y = df[NAMES].to_numpy(np.float64)
        self.y[df[[f"{n}__conf" for n in NAMES]].to_numpy(np.float64) <= ts.SILENT_CONF] = np.nan
        self.strata = df["StudyInstanceUID"].map(scanner_of).fillna("unknown").to_numpy()
        rng = np.random.default_rng(0)
        groups = [np.flatnonzero(self.strata == s) for s in np.unique(self.strata)]
        self.picks = [np.concatenate([rng.choice(v, len(v)) for v in groups]) for _ in range(DRAWS)]
        self.cache = read_json(CACHE_PATH) or {}

    @staticmethod
    def read(path):
        return pd.read_csv(path, dtype={"StudyInstanceUID": str}).sort_values("StudyInstanceUID").reset_index(drop=True)

    def result(self, path):
        key = f"{os.path.relpath(path, R)}|{int(os.path.getmtime(path))}"
        if key not in self.cache:
            frame = self.read(path)
            assert frame["StudyInstanceUID"].equals(self.ids), f"{path}: different validation studies"
            p = frame[NAMES].to_numpy(np.float64)
            macro, each = ts.within_scanner_concordance(self.y, p, self.strata)
            draws = [ts.within_scanner_concordance(self.y[i], p[i], self.strata[i])[1] for i in self.picks]
            self.cache[key] = {"macro": macro, "each": each, "draws": np.round(draws, 5).tolist()}
            with open(CACHE_PATH, "w") as handle:
                json.dump(self.cache, handle)
        return self.cache[key]

    def compare(self, path, base):
        a, b = self.result(path), self.result(base)
        diff = np.nanmean(np.array(a["draws"], float) - np.array(b["draws"], float), axis=1)
        return {"c": a["macro"], "diff": a["macro"] - b["macro"], "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                "each": a["each"], "each_diff": [x - y for x, y in zip(a["each"], b["each"])]}


def stage1_section():
    rows = read_json(f"{R}/stage1_summary.json") or []
    body = [[r["kind"], r["model"], r["params_m"], r["fp32_mb"], r.get("gmacs_per_study") or "-", r["s_4core"], r["s_6core"],
             r["peak_mb_4core"]] for r in sorted(rows, key=lambda v: v["s_4core"])]
    return table(["Kind", "Encoder", "Parameters (M)", "FP32 size (MB)", "GMACs per study", "s per study, 4 cores", "s per study, 6 cores",
                  "Peak RAM, 4 cores (MB)"], body)


STAGE2 = [("MobileNetV4-Conv-Small", "small_fold0/mobilenetv4_conv_small", "Mobile CNN"),
          ("EfficientNetV2-B0", "small_fold0/tf_efficientnetv2_b0", "Mobile CNN"),
          ("ResNet-50 (torchvision V2 weights)", "session3_fold0/resnet50.tv2_in1k", "Classic CNN"),
          ("ConvNeXt-Atto", "small_fold0/convnext_atto", "Modern CNN"),
          ("ConvNeXt-Tiny", "session2_fold0/convnext_tiny", "Modern CNN"),
          ("CoAtNet-0", "session3_fold0/coatnet_0_rw_224", "Hybrid"),
          ("DINOv2-Small, shared settings", "session2_fold0/vit_small_patch14_dinov2", "ViT"),
          ("DINOv2-Small, tuned", "dinov2_tuning_fold0/vit_small_patch14_dinov2@llrd", "ViT"),
          ("R3D-18, 192 px", "r3d18_fold0", "3D CNN")]


def stage2_section(score):
    base = f"{R}/resnet50_fold0/val_predictions.csv"
    rows, scores = [], {}
    for label, folder, kind in STAGE2:
        path = f"{R}/{folder}/val_predictions.csv"
        if not os.path.exists(path):
            continue
        v = score.compare(path, base)
        scores[label] = (v["c"], kind)
        rows.append([kind, label, f"{v['c']:.4f}", signed(v["diff"]), span(v["ci"])])
    rows.sort(key=lambda row: -float(row[2]))
    return table(["Kind", "Encoder", "Concordance", "Against ResNet-50 baseline", "95% interval"], rows), scores


ARMS = [("Mean pooling head", "mean"), ("Transformer head", "tf"), ("192 px input", "r192"), ("5 slices per window", "s5"), ("Slices 3.5 mm apart", "gap35")]
ARM_DIRS = {"convnext_atto": "stage3_atto_fold0", "convnext_tiny": "stage3_tiny_{}_fold0"}


def stage3_paths(model, arm):
    if model == "convnext_atto":
        return f"{R}/stage3_atto_fold0/convnext_atto@{arm}/val_predictions.csv"
    return f"{R}/stage3_tiny_{'b' if arm in ('s5', 'gap35') else 'a'}_fold0/convnext_tiny@{arm}/val_predictions.csv"


def stage3_section(score):
    bases = {m: f"{R}/stage3_atto_fold0/{m}@eval/val_predictions.csv" for m in ("convnext_atto", "convnext_tiny")}
    rows = []
    for label, arm in ARMS:
        row = [label]
        for model, base in bases.items():
            v = score.compare(stage3_paths(model, arm), base)
            row += [f"{v['c']:.4f}", f"{signed(v['diff'])} ({span(v['ci'])})"]
        rows.append(row)
    for windows in (36, 18):
        row = [f"{windows} windows (scored, not retrained)"]
        for model, base in bases.items():
            v = score.compare(base.replace("val_predictions.csv", f"val_predictions_w{windows}.csv"), base)
            row += [f"{v['c']:.4f}", f"{signed(v['diff'])} ({span(v['ci'])})"]
        rows.append(row)
    base_row = ["Stage 2 setting"] + [x for b in bases.values() for x in (f"{score.result(b)['macro']:.4f}", "reference")]
    main = table(["Change", "Atto", "Atto change (95% interval)", "Tiny", "Tiny change (95% interval)"], [base_row] + rows)
    per = []
    drops = {}
    for i, name in enumerate(NAMES):
        values = [score.compare(b.replace("val_predictions.csv", "val_predictions_w18.csv"), b)["each_diff"][i] for b in bases.values()]
        drops[name] = float(np.mean(values))
        per.append([name, signed(values[0]), signed(values[1]), signed(drops[name])])
    per.sort(key=lambda row: float(row[3]))
    w18 = float(np.mean([score.compare(b.replace("val_predictions.csv", "val_predictions_w18.csv"), b)["diff"] for b in bases.values()]))
    return main, table(["Finding", "Atto", "Tiny", "Mean of both"], per), drops, w18


PRECISION_NAMES = {"fp32": "FP32", "fp16": "FP16", "int8_minmax": "INT8, MinMax ranges", "int8_percentile": "INT8, percentile ranges",
                   "int8_minmax_core": "INT8, MinMax, Conv and MatMul only", "int8_percentile_core": "INT8, percentile, Conv and MatMul only",
                   "qat_plain": "QAT weights in FP32", "qat_int8_minmax": "QAT, then INT8 MinMax", "qat_int8_percentile": "QAT, then INT8 percentile",
                   "qat_int8_minmax_core": "QAT, then INT8 MinMax, Conv and MatMul only",
                   "qat_int8_percentile_core": "QAT, then INT8 percentile, Conv and MatMul only"}
MODEL_NAMES = {"convnext_atto@eval": "ConvNeXt-Atto", "convnext_tiny@eval": "ConvNeXt-Tiny", "convnext_tiny@tf_eval": "ConvNeXt-Tiny, transformer head"}
ORDER = list(PRECISION_NAMES)


def refresh_stage4():
    """Rerun the Stage 4 analysis when any prediction file is newer than its summary."""
    import glob

    summary = f"{R}/stage4_summary.json"
    newest = max([os.path.getmtime(p) for p in glob.glob(f"{R}/stage4_*_fold0/*/val_predictions_*.csv")] or [0])
    if not os.path.exists(summary) or os.path.getmtime(summary) < newest:
        subprocess.run([sys.executable, f"{R}/analyse_stage4.py", summary], check=True, stdout=subprocess.DEVNULL)
    return read_json(summary) or {}


def stage4_section(summary):
    rows, per_rows, sizes = [], [], []
    for model, runs in summary.items():
        for key in sorted(runs, key=lambda k: ORDER.index(k.split(" [")[0]) if k.split(" [")[0] in ORDER else 99):
            v = runs[key]
            name = PRECISION_NAMES.get(key, key)
            rows.append([MODEL_NAMES.get(model, model), name, f"{v['c']:.4f}",
                         "reference" if key == "fp32" else f"{signed(v['c_diff'])} ({span(v['c_diff_ci'])})",
                         f"{v['ece']:.4f}", f"{v['brier']:.4f}", f"{v['ece_after_recal']:.4f}", f"{v['brier_after_recal']:.4f}",
                         signed(v["mean_prob_shift"])])
    for model, folder in [("ConvNeXt-Atto", "convnext_atto@eval"), ("ConvNeXt-Tiny", "convnext_tiny@eval"), ("ConvNeXt-Tiny, transformer head", "convnext_tiny@tf_eval")]:
        report = read_json(f"{R}/stage4_minmax_fold0/{folder}/pilot_report.json")
        info = (report or {}).get("precisions", {}).get("int8_minmax")
        if info:
            sizes.append([model, info["fp32_mb"], info["int8_mb"]])
    return table(["Model", "Precision", "Concordance", "Change from FP32 (95% interval)", "ECE", "Brier", "ECE recalibrated", "Brier recalibrated",
                  "Mean probability shift"], rows), table(["Model", "FP32 encoder (MB)", "INT8 encoder (MB)"], sizes)


def seed_section(score):
    folder = f"{R}/stage4_seed1_fold0"
    paths = {("attention", 0): f"{R}/stage3_atto_fold0/convnext_tiny@eval/val_predictions.csv",
             ("transformer", 0): f"{R}/stage3_tiny_a_fold0/convnext_tiny@tf/val_predictions.csv",
             ("attention", 1): f"{folder}/convnext_tiny@seed1/val_predictions.csv",
             ("transformer", 1): f"{folder}/convnext_tiny@tf_seed1/val_predictions.csv"}
    if not all(os.path.exists(p) for p in paths.values()):
        return None, None
    rows, diffs = [], []
    for seed in (0, 1):
        v = score.compare(paths["transformer", seed], paths["attention", seed])
        diffs.append(v["diff"])
        rows.append([seed, f"{score.result(paths['attention', seed])['macro']:.4f}", f"{v['c']:.4f}", f"{signed(v['diff'])} ({span(v['ci'])})"])
    return table(["Training seed", "Attention head", "Transformer head", "Transformer minus attention (95% interval)"], rows), diffs


def timing_section():
    """One row per model at full input (72 windows, or 6 clips for 3D), both limits side by side. Returns (table, {label: seconds at 4 cores})."""
    found = {}
    for profile in ("pi5-class", "orin-class-cpu"):
        for r in read_json(f"{R}/stage3_timing_{profile}.json") or []:
            if "error" in r or r["windows"] not in (72, "6 clips of 32"):
                continue
            found.setdefault(r["label"], {})[profile] = r
    if not found:
        return None, {}
    rows, four = [], {}
    for label, by in found.items():
        a, b = by.get("pi5-class"), by.get("orin-class-cpu")
        first = a or b
        cell = lambda r: f"{r['total_s']:.2f} ({r['total_range_s'][0]:.2f}-{r['total_range_s'][1]:.2f})" if r else "-"
        peak = max(r["encoder_peak_rss_mb"] for r in (a, b) if r)
        rows.append([label, first["backbone"], first.get("precision", "fp32").upper(), cell(a), cell(b), f"{peak:.0f}", f"{first['encoder_mb']:.1f}"])
        if a:
            four[label] = a["total_s"]
    header = ["Run", "Encoder", "Precision", "4 cores: s per study (range)", "6 cores: s per study (range)", "Peak memory MB", "Encoder file MB"]
    return table(header, rows), four


FOLD_RUNS = {"ConvNeXt-Atto": {0: ("small_fold0/convnext_atto", "stage3_atto_fold0/convnext_atto@eval")},
             "ConvNeXt-Tiny, transformer head": {0: ("stage3_tiny_a_fold0/convnext_tiny@tf", None)}}
for _f in range(1, 5):
    FOLD_RUNS["ConvNeXt-Atto"][_f] = (f"stage5_fold{_f}/convnext_atto", None)
    FOLD_RUNS["ConvNeXt-Tiny, transformer head"][_f] = (f"stage5_fold{_f}/convnext_tiny@tf", None)


def fold_scores():
    """{model: {fold: {c, auc, c36, c18}}} from each run's last epoch (the EMA weights it saves) and its window scoring."""
    out = {}
    for model, folds in FOLD_RUNS.items():
        for fold, (train, windows) in folds.items():
            r, w = read_json(f"{R}/{train}/pilot_report.json"), read_json(f"{R}/{windows or train}/pilot_report.json")
            if not r or not w or "epochs" not in r or "36" not in w.get("eval_windows", {}):
                continue
            last = r["epochs"][-1]
            out.setdefault(model, {})[fold] = {"c": last["within_scanner_c"], "auc": last["auc_thresholded"],
                                               "c36": w["eval_windows"]["36"]["within_scanner_c"], "c18": w["eval_windows"]["18"]["within_scanner_c"]}
    return out


def stage5_section(scores):
    if not scores or any(len(v) < 5 for v in scores.values()):
        return None, None
    rows = []
    for model, folds in scores.items():
        for key, label in (("c", "Concordance, 72 windows"), ("auc", "AUC, 72 windows"), ("c36", "Concordance, 36 windows"), ("c18", "Concordance, 18 windows")):
            v = np.array([folds[f][key] for f in range(5)])
            rows.append([model, label] + [f"{x:.4f}" for x in v] + [f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"])
    a, b = scores["ConvNeXt-Atto"], scores["ConvNeXt-Tiny, transformer head"]
    gap = np.array([b[f]["c"] - a[f]["c"] for f in range(5)])
    loss18 = {m: np.array([s[f]["c18"] - s[f]["c"] for f in range(5)]) for m, s in scores.items()}
    return table(["Model", "Score"] + [f"Fold {f}" for f in range(5)] + ["Mean ± SD"], rows), {"gap": gap, "loss18": loss18}


def gold_section():
    """Gold AUC of each saved model, paired with the LLM labels' own AUC on the same gold studies."""
    import glob
    from sklearn.metrics import roc_auc_score

    found = {}
    for path in sorted(glob.glob(f"{R}/phasea_*_fold0/*/pilot_report.json")):
        rep = read_json(path)
        preds = os.path.join(os.path.dirname(path), "gold_predictions.csv")
        if rep and "gold" in rep and os.path.exists(preds):
            found[os.path.basename(os.path.dirname(path))] = (rep["gold"], preds)
    if not found:
        return None, None
    gold = pd.read_csv(f"{ROOT}/data_meta/train.csv", dtype={"StudyInstanceUID": str})
    gold = gold[gold[NAMES].notna().any(axis=1)].set_index("StudyInstanceUID")
    llm = pd.read_csv(f"{ROOT}/data_meta/pilkwang/report_labels_v2.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    common = sorted(set(gold.index) & set(llm.index))
    y, q = gold.loc[common, NAMES].to_numpy(float), llm.loc[common, NAMES].to_numpy(float)

    def macro(p, idx):
        each = []
        for c in range(len(NAMES)):
            t = y[idx, c]
            keep = ~np.isnan(t)
            each.append(roc_auc_score(t[keep] > 0.5, p[idx, c][keep]) if 0 < (t[keep] > 0.5).sum() < keep.sum() else np.nan)
        return float(np.nanmean(each))

    rng = np.random.default_rng(0)
    picks = [rng.choice(len(common), len(common)) for _ in range(DRAWS)]
    full = np.arange(len(common))
    names = {"convnext_atto@oracle": "ConvNeXt-Atto", "convnext_tiny@gold": "ConvNeXt-Tiny", "convnext_tiny@tf_gold": "ConvNeXt-Tiny, transformer head"}
    rows, worst, paired = [], {}, {}
    rows.append(["LLM report labels (training labels)", len(common), f"{macro(q, full):.4f}", "-", "reference"])
    for key, (rep, preds) in found.items():
        p = pd.read_csv(preds, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[common, NAMES].to_numpy(float)
        d = np.array([macro(p, i) - macro(q, i) for i in picks])
        ci = rep["macro_auc_95"]
        rows.append([names.get(key, key), rep["studies"], f"{rep['macro_auc']:.4f}", f"{ci[0]:.3f} to {ci[1]:.3f}",
                     f"{signed(macro(p, full) - macro(q, full))} ({span([np.percentile(d, 2.5), np.percentile(d, 97.5)])})"])
        worst[names.get(key, key)] = sorted(rep["per_finding_auc"].items(), key=lambda kv: kv[1])[:2]
        paired[names.get(key, key)] = (macro(p, full) - macro(q, full), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), rep["macro_auc"])
    header = ["Scored", "Gold studies", "Macro AUC", "95% interval", f"Against LLM labels on the {len(common)} shared studies (95% interval)"]
    return table(header, rows), (worst, paired, len(common), macro(q, full))


SAMPLING_ORDER = ["uniform, trained on 48", "random, trained on 48", "centre, trained on 48", "uniform, trained at budget",
                  "oracle, top by attention per slot", "oracle, top by attention per study", "dense, all windows"]


def sampling_section():
    summary = read_json(f"{R}/phasea_summary.json")
    if not summary or not summary.get("arms"):
        return None, None
    rows = []
    arms = summary["arms"]
    for label in sorted(arms, key=lambda k: (-arms[k]["windows"], SAMPLING_ORDER.index(k.split(" [")[0]) if k.split(" [")[0] in SAMPLING_ORDER else 99)):
        v = arms[label]
        shown = label.split(" [")[0].replace("uniform", "evenly spaced").replace("centre,", "centre-weighted,")
        rows.append([shown, v["windows"], f"{v['c']:.4f}", "reference" if v["c_diff_ci"] == [0.0, 0.0] else f"{signed(v['c_diff'])} ({span(v['c_diff_ci'])})",
                     f"{v['auc']:.4f}", signed(v["menisci_c_diff"])])
    main = table(["Window placement", "Windows", "Concordance", "Against 72 evenly spaced (95% interval)", "AUC", "Menisci change"], rows)
    att = summary.get("attention")
    att_table = table(["Finding", "Profile consistency", "Attention mass in the top 18 of 144 windows"],
                      [[n, f"{v['profile_consistency']:.2f}", f"{v['mass_in_top18_windows']:.2f}"] for n, v in att.items()]) if att else None
    groups = summary.get("subgroup_sizes", {})
    sub_rows = [[label.split(" [")[0].replace("uniform", "evenly spaced").replace("centre,", "centre-weighted,"), arms[label]["windows"]]
                + [f"{arms[label]['subgroups'][g]['auc']:.4f} ({signed(arms[label]['subgroups'][g]['auc_diff'])})" for g in groups]
                for label in ("uniform, trained on 48 [72]", "uniform, trained on 48 [18]", "uniform, trained at budget [18]", "centre, trained on 48 [18]",
                              "oracle, top by attention per slot [18]") if label in arms]
    sub_table = table(["Window placement", "Windows"] + [f"{g} ({n} studies)" for g, n in groups.items()], sub_rows)
    ranges = None
    if att:
        cons = [v["profile_consistency"] for v in att.values()]
        mass = [v["mass_in_top18_windows"] for v in att.values()]
        ranges = (min(cons), max(cons), min(mass), max(mass))
    return main, {"arms": arms, "attention": att_table, "subgroups": sub_table, "ranges": ranges}


def main():
    score = Scorer(f"{R}/stage3_atto_fold0/convnext_atto@eval/val_predictions.csv")
    s1 = stage1_section()
    s2, s2_scores = stage2_section(score)
    s3, s3_findings, drops, w18 = stage3_section(score)
    summary = refresh_stage4()
    s4, sizes = stage4_section(summary) if summary else (NOT_YET, NOT_YET)
    seeds, seed_diffs = seed_section(score)
    timing, t4 = timing_section()
    s5, s5_stats = stage5_section(fold_scores())
    gold, gold_info = gold_section()
    sampling, extra = sampling_section()
    arms = (extra or {}).get("arms")
    oracle_ok = arms and "oracle, top by attention per slot [18]" in arms
    if oracle_ok:
        o18, c18, u72 = arms["oracle, top by attention per slot [18]"], arms["centre, trained on 48 [18]"], arms["uniform, trained on 48 [72]"]
        g18 = arms["oracle, top by attention per study [18]"]
        b18 = arms.get("uniform, trained at budget [18]")
        a4_gap = u72["auc"] - max(o18["auc"], g18["auc"])
        beats = o18["c_diff"] - c18["c_diff"]
        verdict = (f"**Result.** The oracle did not pass either test. Its best 18-window AUC was {max(o18['auc'], g18['auc']):.4f}, "
                   f"{a4_gap:.3f} below 72 evenly spaced windows, outside the 0.02 line set in advance. "
                   f"Choosing within each slot scored no better than centre-weighted placement in concordance ({signed(o18['c_diff'])} against {signed(c18['c_diff'])}, a difference of {signed(beats)}), "
                   f"and choosing across the whole study did worse ({signed(g18['c_diff'])}). The rule added after the centre-weighted result needed a gain above 0.01, so Phase B, "
                   "the learned selector, does not go ahead. A selector trained to copy the model's attention could at best match the oracle, and the oracle only matches a fixed schedule that costs nothing. "
                   "One reason shows in the findings: the oracle ranks windows by attention averaged over all 12 findings, which kept windows useful for most findings but dropped ones that effusion needs "
                   f"(effusion {signed(o18['per_finding_c_diff']['Effusion'])} under the oracle against {signed(c18['per_finding_c_diff']['Effusion'])} centre-weighted).\n\n"
                   + (f"Retraining at the budget helps but less than moving the windows: 18 evenly spaced windows trained at 18 lost {signed(b18['c_diff'])}, against "
                      f"{signed(arms['uniform, trained on 48 [18]']['c_diff'])} for the model trained on 48, and {signed(c18['c_diff'])} for the model trained on 48 with centre-weighted windows. "
                      "Training at the budget with centre-weighted windows has not been tried.\n" if b18 else ""))


    best_cnn = max((v for v in s2_scores.values() if v[1] not in ("Hybrid", "ViT")), key=lambda v: v[0])[0]
    best_other = max((v for v in s2_scores.values() if v[1] in ("Hybrid", "ViT")), key=lambda v: v[0])[0]
    worst_first = sorted(drops, key=drops.get)
    tiny = summary.get("convnext_tiny@eval", {})
    atto = summary.get("convnext_atto@eval", {})
    tf = summary.get("convnext_tiny@tf_eval", {})
    have_qat = any(k.startswith("qat_int8") for runs in summary.values() for k in runs)

    def int8_line(name, runs):
        parts = [f"{PRECISION_NAMES[k]} {signed(runs[k]['c_diff'])}" for k in ORDER if k in runs and k.startswith("int8")]
        return f"  - {name}: " + ("; ".join(parts) if parts else "no INT8 result yet") + "."

    def qat_compare(key):
        rows = [(n, r) for n, r in (("Atto", atto), ("Tiny", tiny), ("Tiny with transformer head", tf)) if f"qat_{key}" in r and key in r]
        return ", ".join(f"{n} {signed(r['qat_' + key]['c_diff'])} against {signed(r[key]['c_diff'])}" for n, r in rows)

    qat_line = (f"- **Quantization-aware training.** It did not reduce the INT8 loss. With Conv and MatMul conversion and percentile ranges, "
                f"QAT weights scored {qat_compare('int8_percentile_core')} without QAT, all against FP32. With ONNX Runtime's default conversion "

                f"and MinMax ranges, QAT helped Atto slightly but made both Tiny models worse ({qat_compare('int8_minmax')}). Calibration after the core conversion was similar. "
                "This QAT is a simple version (3 epochs, learned ranges discarded), so a fuller version could still do better.\n"
                if have_qat else "- **Quantization-aware training.** Running; not in this version.\n")
    def sec(label):
        return f"{t4[label]:.2f} s" if label in t4 else "not timed"

    timing_line = (f"- **CPU cost.** On 4 cores, FP32, 72 windows: ConvNeXt-Atto {sec('atto')}, ConvNeXt-Tiny {sec('tiny')}, CoAtNet-0 {sec('coatnet0_256')}, "
                   f"DINOv2-Small {sec('dinov2_tuned')}, R3D-18 {sec('r3d18_192')}. Every model is far inside the 80 s and 1.5 GB targets. "
                   f"INT8 gained little speed on this CPU: Tiny took {sec('tiny_int8_core_pct')} with Conv and MatMul conversion, about a fifth faster, and Atto became slower ({sec('atto_int8_core_pct')}). "
                   "The laptop's CPU lacks VNNI instructions, so 8-bit products are not much cheaper than 32-bit ones here. INT8 still cuts encoder file size by about 4x and peak memory by 20-40%.\n"
                   if t4 else "")
    head_line = (f"- **Transformer head.** On ConvNeXt-Tiny it scored {signed(seed_diffs[0])} and {signed(seed_diffs[1])} against the attention head in two training seeds, on the edge of the 0.01 tie margin.\n"
                 if seed_diffs else "")
    fold_line = ""
    if s5_stats:
        g, l18 = s5_stats["gap"], s5_stats["loss18"]["ConvNeXt-Atto"]
        fold_line = (f"- **Five folds.** ConvNeXt-Tiny with the transformer head beat ConvNeXt-Atto on all 5 folds, by {g.mean():+.3f} on average "
                     f"(range {g.min():+.3f} to {g.max():+.3f}), at about 10 times the CPU time. The same model varies by up to "
                     f"{max(np.ptp([v['c'] for v in s.values()]) for s in fold_scores().values()):.3f} between folds, so single-fold gaps under about 0.01 should not be read as real. "
                     f"Cutting Atto to 18 evenly spaced windows cost {l18.min():+.3f} to {l18.max():+.3f} on every fold.\n")
    gold_line = ""
    if gold:
        worst, paired, n_common, llm_auc = gold_info
        above = all(v[1] > 0 for v in paired.values())
        gold_line = (f"- **Radiologist labels.** On the 58 gold studies, which no training or selection used, macro AUC against the official labels was "
                     + "; ".join(f"{k} {v[3]:.3f}" for k, v in paired.items())
                     + f". The LLM report labels the models were trained on score {llm_auc:.3f}. Paired on the {n_common} studies that have both, the models are ahead of their training labels by "
                     + "; ".join(f"{signed(v[0])} ({span(v[1:3])}) for {k}" for k, v in paired.items())
                     + (". So the models do better than the labels they learned from, though the intervals come close to zero. " if above else ". Not every interval excludes zero. ")
                     + "The weakest findings across the models are " + (lambda w: ", ".join(w[:-1]) + " and " + w[-1])(sorted({n for v in worst.values() for n, _ in v})) + " (section 10).\n")
    place_line = ""
    if arms and "centre, trained on 48 [18]" in arms:
        u18, c18, c36 = arms["uniform, trained on 48 [18]"], arms["centre, trained on 48 [18]"], arms["centre, trained on 48 [36]"]
        place_line = (f"- **Where the windows go.** With 18 windows, placing them towards the middle of each series cut Atto's loss from {signed(u18['c_diff'])} to "
                      f"{signed(c18['c_diff'])}, and the menisci loss from {signed(u18['menisci_c_diff'])} to {signed(c18['menisci_c_diff'])}, with no retraining. "
                      f"With 36 centre-weighted windows Atto matched 72 evenly spaced ones ({signed(c36['c_diff'])}), at half the CPU time. "
                      + (f"Picking the 18 windows the model attends to most, from a pass over all 144, did no better ({signed(arms['oracle, top by attention per slot [18]']['c_diff'])}), "
                         "so a learned window selector is not worth building. " if "oracle, top by attention per slot [18]" in arms else "")
                      + "Fold 0 only (section 11).\n")
    text = f"""# CPU-deployable knee MRI classification: project report

Last rebuilt {date.today().isoformat()} by `results/build_report.py`. Tables and headline figures come from the result files. Fold 0, one training seed unless stated. Research work, not a clinical tool.

## 1. Summary

This project asks which knee MRI classifiers can run on an ordinary CPU, and what accuracy is lost by making them cheaper. It uses the RSNA Knee Abnormality Detection data, where each study is scored on {len(NAMES)} findings. Models are trained on Kaggle GPUs and timed on a Ryzen 5 5500U laptop in Docker, limited to 4 or 6 cores and 3 GB.

Where things stand:

- **Encoder choice.** The best hybrid or vision-transformer encoder scored {best_other:.4f}, against {best_cnn:.4f} for the best convolutional encoder, so they did not beat the CNNs by more than the 0.01 tie margin (H1, fold 0). CoAtNet-0 takes about as long as ConvNeXt-Tiny, while ConvNeXt-Atto is within 0.01 of both at about a tenth of the time.
- **Fewer windows.** Going from 72 to 18 windows costs {abs(w18):.3f} on average for the two ConvNeXt models. The losses are largest for {worst_first[0]} and {worst_first[1]}.
- **Precision.** FP16 changes nothing measurable. The cost of INT8 depends mostly on which layers are converted. ONNX Runtime's default also rounds around LayerNormalization, which costs ConvNeXt-Tiny 0.02-0.03; converting only Conv and MatMul layers removes nearly all of it. Change in concordance from FP32:
{int8_line('ConvNeXt-Atto', atto)}
{int8_line('ConvNeXt-Tiny', tiny)}
{int8_line('ConvNeXt-Tiny, transformer head', tf)}
{timing_line}{head_line}{qat_line}{fold_line}{gold_line}{place_line}
## 2. Questions

| ID | Hypothesis | Status |
|---|---|---|
| H1 | At equal CPU latency, hybrid and ViT encoders do not beat modern CNNs by more than the noise margin | Supported in fold 0: at the same latency as ConvNeXt-Tiny, CoAtNet-0 ties it, and DINOv2-Small is slower and scores lower (sections 5 and 8) |
| H2 | Cutting windows hurts focal findings more than global ones | Partly supported: see section 6 |
| H3 | Post-training INT8 loses more discrimination and calibration than quantization-aware training, and the gap depends on model family | {'Not supported here: converting only Conv and MatMul keeps post-training INT8 within the tie margin for all three models, and QAT did not reduce the loss. The conversion mattered more than the model family (section 7)' if have_qat else 'In progress: post-training INT8 measured, QAT running'} |
| H4 | Label quality moves AUC more than encoder choice | Not tested yet: needs a second label set |
| H5 | At a fixed window budget, choosing windows beats evenly spaced windows | {'Supported for a fixed centre-weighted schedule, in fold 0. Not supported for choosing by the model' + chr(39) + 's attention, which did no better than the fixed schedule, so no learned selector is built (section 11)' if oracle_ok else 'Fixed centre-weighted placement already does, in fold 0; whether a learned choice adds more waits for the oracle run (section 11)' if place_line else 'Running (section 11)'} |

## 3. Method

- **Input.** Each study is up to 6 series slots (3 planes, fat-suppressed or not). Each slot gives 12 windows of 3 adjacent slices at 256 px, so 72 windows per study. A per-finding attention head combines them.
- **Labels.** `pilkwang/rsna-knee-llm-labels`, soft labels read from radiology reports by an LLM. Labels where the report was silent are left out of scoring. The 58 gold studies are excluded from training and selection.
- **Primary score.** Within-scanner concordance: for each finding, the share of same-scanner study pairs ranked correctly, averaged over findings.
- **Comparisons.** Differences are paired by study on the same 870 validation studies, with 500 bootstrap draws within scanners. Differences under 0.01 count as ties.
- **Calibration.** ECE (10 bins) and Brier score against the soft labels, so they measure agreement with report-derived labels rather than ground truth.
- **INT8 conversion.** ONNX Runtime static quantization in QDQ form, with 8-bit activations and per-channel 7-bit weights, because the laptop's CPU has no VNNI instructions and 8-bit products could saturate. Ranges come from 32 training-split studies. Only the encoder is converted; the head costs 1-7 ms and stays FP32. Two conversions are compared: ONNX Runtime's default, which also places rounding around LayerNormalization even though that layer still computes in floating point, and one limited to Conv and MatMul layers.
- **Quantization-aware training.** Models are fine-tuned for 3 epochs under simulated INT8 noise. The simulation is removed and the weights go through the same conversion, so the comparison isolates the weights. Ranges learned in the simulation are not carried over, so this is a simpler form of QAT.

## 4. Stage 1: untrained encoders on CPU

Encoder only, 72 windows per study (6 clips for 3D), no study head or DICOM reading.

{s1}

No candidate missed the 80 s or 1.5 GB targets.

## 5. Stage 2: encoder comparison

ResNet-50 baseline (`resnet50_fold0`): the reference for the change column. Models whose interval excludes zero differ from it beyond bootstrap noise, but separately trained models also differ by about 0.01, so one fold and one seed only narrows the list.

{s2}

ConvNeXt-Atto and ConvNeXt-Tiny went forward to Stage 3.

## 6. Stage 3: head and scale

One change at a time from the Stage 2 setting. Paired against that setting.

{s3}

Change at 18 windows per finding, worst first:

{s3_findings}

Slices 3.5 mm apart changed only about 5% of series, since the median series spacing is already 3.5 mm, so that row says little.

## 7. Stage 4: precision and calibration

All rows use the same weights unless marked QAT. Differences are against that model's true FP32 score. Mixed-precision GPU scoring, used in Stages 2 and 3, gave the same scores as true FP32 to 4 decimals.

{s4}

Encoder file sizes:

{sizes}

Recalibrated columns fit a slope and intercept per finding, cross-fitted in 5 parts of the validation fold. "Mean probability shift" is the average change in predicted probability from FP32.

### Second seed

{seeds or 'Running: ConvNeXt-Tiny with both heads, training seed 1, same folds.'}

## 8. CPU cost of trained models

Encoder over all windows plus the study head, after the model is loaded. Each figure is the median of 5 separate timing processes, in Docker with the stated core limit and 3 GB of memory. Time grows in proportion to the number of windows, so 36 windows take half as long. Labels ending `_tf`, `_mean` and `_r192` are the Stage 3 variants; INT8 labels ending `_core_pct` convert only Conv and MatMul layers with percentile ranges.

{timing}

The 6-core limit was not faster than the 4-core one: ConvNeXt-Tiny took about the same time, and ConvNeXt-Atto was slower. A likely cause is that WSL's 6 virtual processors do not map to 6 separate physical cores, so the extra threads compete with each other. The two limits therefore do not separate as intended, and the 4-core figures are the ones to quote.

{'' if timing else 'Pending. A first attempt was discarded because other applications loaded the laptop: the same encoder varied by up to 2x. It will be repeated on an idle machine with 5 separate timing processes per encoder. That run also covers CoAtNet-0 at 256 px and R3D-18 at 192 px, and INT8 encoders.'}

## 9. Five folds

The two finalists, trained from scratch on each of the 5 folds with the Stage 2 setting. Each fold scores a different 870 studies, so scores are not paired across folds. Fold 0 is the run used in the earlier sections.

{s5 or 'Running: folds 1-4 for ConvNeXt-Atto and ConvNeXt-Tiny with the transformer head.'}

## 10. Radiologist gold labels

Saved fold 0 models scored on the 58 gold studies against the official radiologist labels. The 95% interval resamples studies (2,000 draws). The last column pairs each model with the LLM report labels, used as scores, on the same studies (500 draws). Both are AUC against the official labels.

{gold or 'Not scored yet.'}

## 11. Window placement (Phase A, fold 0)

ConvNeXt-Atto trained on 48 windows, then scored with windows placed in different ways. "Evenly spaced" is the Stage 2 placement. "Centre-weighted" packs windows towards the middle of each series. "Random" draws positions at random with a fixed seed. "Trained at budget" retrains with the same number of windows it is scored with. "Oracle" scores all 144 windows, then keeps the ones the model's attention ranks highest. It is an upper bound for any selector that imitates the model, not a usable method, because it costs the full dense pass. Menisci change is the mean concordance change for the two menisci.

{sampling or 'Running.'}

The adaptive-selection plan in `study_design.md` set a go/no-go rule before these runs: build a learned selector only if the oracle at 18 windows comes within 0.02 AUC of 72 evenly spaced windows. Centre-weighted placement at 18 windows already came close to that line without any selector, so a dated note added a second condition before the oracle result: the oracle must also beat centre-weighted placement by more than 0.01 in concordance.

{verdict if oracle_ok else 'The oracle run is not in yet.'}
{('### Where the attention goes' + chr(10) + chr(10) + 'From the dense pass (144 windows). Profile consistency is the mean correlation between each study' + chr(39) + 's attention-by-position profile and the average profile: 1 means every study attends to the same positions. Mass in the top 18 is how much of the attention the 18 highest windows hold.' + chr(10) + chr(10) + extra['attention'] + chr(10) + chr(10) + 'Attention is only partly consistent across studies (' + '{:.2f}-{:.2f}'.format(*extra['ranges'][:2]) + ') and spread out: the top 18 windows hold ' + '{:.2f}-{:.2f}'.format(*extra['ranges'][2:]) + ' of the attention, depending on the finding. For about half the findings, the top 18 windows hold less than 60% of the attention, which fits a fixed central schedule doing as well as picking windows per study.' + chr(10)) if extra and extra.get('attention') else ''}
{('### Studies missing a fat-suppressed plane' + chr(10) + chr(10) + 'AUC within each group, with the change from 72 evenly spaced windows. The small groups have wide uncertainty and no intervals are given.' + chr(10) + chr(10) + extra['subgroups'] + chr(10)) if extra else ''}
## 12. Limits

- Most comparisons use one fold and one seed. Folds differ by up to about 0.02 for the same model, so close models cannot be separated without the five-fold runs.
- Timing describes this laptop under Docker limits, not a Raspberry Pi or Jetson, because Docker can limit cores and memory but not device speed.
- INT8 speed depends on the CPU. This one has no VNNI instructions, so INT8 gives little or no speed-up; CPUs with VNNI or ARM dot-product instructions usually gain more.
- Labels come from reports through an LLM, so concordance and calibration measure agreement with those labels. Only 58 studies have radiologist labels, which makes gold intervals wide.
- The data has study IDs but no patient IDs, so two studies of the same patient could fall in different folds.
- The 2.5D and 3D comparison mixes in resolution and pretraining differences.
- The QAT tested is simple: 3 epochs of fine-tuning, with the ranges it learns discarded before conversion. A fuller version that keeps them might close the small remaining gap.
- Study-level data, labels and trained weights are not published here. The MIRA agreement limits redistribution, and whether trained weights count as derivatives still needs asking RSNA.

## 13. Next steps

1. Check centre-weighted placement at 18 and 36 windows on all 5 folds, scoring the saved Stage 5 models.
2. Try training at the budget with centre-weighted windows, the one combination not yet run.
3. Time centre-weighted Atto at 18 and 36 windows on the laptop, to state the accuracy for each CPU cost.
4. Test fat-suppressed series only against all 6 slots (Phase C), and a second label set for H4.
5. The learned window selector (Phase B) is dropped: the attention oracle did no better than a fixed schedule.
"""
    os.makedirs(f"{ROOT}/docs", exist_ok=True)
    with open(f"{ROOT}/docs/project_report.md", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote docs/project_report.md, {len(text)} characters")


if __name__ == "__main__":
    main()
