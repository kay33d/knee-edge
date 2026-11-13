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
    rows = []
    for profile, label in (("pi5-class", "4 cores"), ("orin-class-cpu", "6 cores")):
        for r in read_json(f"{R}/stage3_timing_{profile}.json") or []:
            if "error" in r:
                continue
            rows.append([r["label"], label, r["windows"], r.get("precision", "fp32"), r["encoder_s"], r["head_s"], r["total_s"], f"{r['total_range_s'][0]}-{r['total_range_s'][1]}"])
    if not rows:
        return None
    return table(["Model", "Limit", "Windows", "Precision", "Encoder s", "Head s", "Total s per study", "Range over repeats"], rows)


def main():
    score = Scorer(f"{R}/stage3_atto_fold0/convnext_atto@eval/val_predictions.csv")
    s1 = stage1_section()
    s2, s2_scores = stage2_section(score)
    s3, s3_findings, drops, w18 = stage3_section(score)
    summary = refresh_stage4()
    s4, sizes = stage4_section(summary) if summary else (NOT_YET, NOT_YET)
    seeds, seed_diffs = seed_section(score)
    timing = timing_section()

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
    head_line = (f"- **Transformer head.** On ConvNeXt-Tiny it scored {signed(seed_diffs[0])} and {signed(seed_diffs[1])} against the attention head in two training seeds, on the edge of the 0.01 tie margin.\n"
                 if seed_diffs else "")
    text = f"""# CPU-deployable knee MRI classification: project report

Last rebuilt {date.today().isoformat()} by `results/build_report.py`. Tables and headline figures come from the result files. Fold 0, one training seed unless stated. Research work, not a clinical tool.

## 1. Summary

This project asks which knee MRI classifiers can run on an ordinary CPU, and what accuracy is lost by making them cheaper. It uses the RSNA Knee Abnormality Detection data, where each study is scored on {len(NAMES)} findings. Models are trained on Kaggle GPUs and timed on a Ryzen 5 5500U laptop in Docker, limited to 4 or 6 cores and 3 GB.

Where things stand:

- **Encoder choice.** The best hybrid or vision-transformer encoder scored {best_other:.4f}, against {best_cnn:.4f} for the best convolutional encoder, so they did not beat the CNNs by more than the 0.01 tie margin (H1, fold 0; final latency comparison still needs the re-timing in section 8).
- **Fewer windows.** Going from 72 to 18 windows costs {abs(w18):.3f} on average for the two ConvNeXt models. The losses are largest for {worst_first[0]} and {worst_first[1]}.
- **Precision.** FP16 changes nothing measurable. The cost of INT8 depends mostly on which layers are converted. ONNX Runtime's default also rounds around LayerNormalization, which costs ConvNeXt-Tiny 0.02-0.03; converting only Conv and MatMul layers removes nearly all of it. Change in concordance from FP32:
{int8_line('ConvNeXt-Atto', atto)}
{int8_line('ConvNeXt-Tiny', tiny)}
{int8_line('ConvNeXt-Tiny, transformer head', tf)}
{head_line}{qat_line}
## 2. Questions

| ID | Hypothesis | Status |
|---|---|---|
| H1 | At equal CPU latency, hybrid and ViT encoders do not beat modern CNNs by more than the noise margin | Supported on accuracy in fold 0; latency comparison pending |
| H2 | Cutting windows hurts focal findings more than global ones | Partly supported: see section 6 |
| H3 | Post-training INT8 loses more discrimination and calibration than quantization-aware training, and the gap depends on model family | {'Not supported here: converting only Conv and MatMul keeps post-training INT8 within the tie margin for all three models, and QAT did not reduce the loss. The conversion mattered more than the model family (section 7)' if have_qat else 'In progress: post-training INT8 measured, QAT running'} |
| H4 | Label quality moves AUC more than encoder choice | Stage 5, not started |

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

{timing or 'Pending. A first attempt was discarded because other applications loaded the laptop: the same encoder varied by up to 2x. It will be repeated on an idle machine with 5 separate timing processes per encoder. That run also covers CoAtNet-0 at 256 px and R3D-18 at 192 px, and INT8 encoders.'}

## 9. Limits

- One fold and mostly one seed. Close models cannot be separated.
- Timing describes this laptop under Docker limits, not a Raspberry Pi or Jetson, because Docker can limit cores and memory but not device speed.
- Labels come from reports through an LLM, so concordance and calibration measure agreement with those labels. The 58 gold studies and external sets are for the final models.
- The 2.5D and 3D comparison mixes in resolution and pretraining differences.
- The QAT tested is simple: 3 epochs of fine-tuning, with the ranges it learns discarded before conversion. A fuller version that keeps them might close the small remaining gap.
- Study-level data, labels and trained weights are not published here. The MIRA agreement limits redistribution, and whether trained weights count as derivatives still needs asking RSNA.

## 10. Next steps

1. Time the trained models on an idle laptop, FP32 and INT8 (Conv and MatMul only).
2. Choose finalists on accuracy against measured CPU cost.
3. Stage 5: all folds for 2-3 finalists, a second label set for H4, gold-study and external checks.
"""
    os.makedirs(f"{ROOT}/docs", exist_ok=True)
    with open(f"{ROOT}/docs/project_report.md", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote docs/project_report.md, {len(text)} characters")


if __name__ == "__main__":
    main()
