"""Checks train_stage2.py on synthetic DICOMs, on CPU, before any GPU time is spent."""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_stage2 as ts  # noqa: E402

IOP = {"Sagittal": [0, 1, 0, 0, 0, -1], "Coronal": [1, 0, 0, 0, 0, -1], "Axial": [1, 0, 0, 0, 1, 0]}
NORMAL_AXIS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def write_slice(path, pixels, iop, position, instance, scanner="Acme", spacing=0.5):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.ImageOrientationPatient, ds.ImagePositionPatient = iop, position
    ds.PixelSpacing, ds.InstanceNumber = [spacing, spacing], instance
    ds.Manufacturer, ds.ManufacturerModelName = scanner, "X1"
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(path)


def write_series(folder, plane, n=6, scanner="Acme", seed=0, reverse_files=False):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        pixels = rng.integers(0, 200, (64, 64)) + (i + 1) * 300
        position = [0.0, 0.0, 0.0]
        position[NORMAL_AXIS[plane]] = float(i * 3)
        name = n - i if reverse_files else i
        write_slice(f"{folder}/{name}.dcm", pixels, IOP[plane], position, i + 1, scanner)


def test_metric():
    y = np.array([[0.1], [0.9], [0.2], [0.8]], np.float32)
    strata = np.array(["a", "a", "b", "b"])
    assert ts.within_scanner_concordance(y, y.copy(), strata)[0] == 1.0
    assert ts.within_scanner_concordance(y, -y, strata)[0] == 0.0
    assert ts.within_scanner_concordance(y, np.zeros_like(y), strata)[0] == 0.5
    # Wrong order only across scanners must not count: each scanner is ranked correctly on its own.
    across = np.array([[0.7], [0.8], [0.1], [0.2]], np.float32)
    assert ts.within_scanner_concordance(y, across, strata)[0] == 1.0
    missing = np.array([[0.1], [0.9], [np.nan], [np.nan]], np.float32)
    assert ts.within_scanner_concordance(missing, missing.copy(), strata)[0] == 1.0


def test_side_rules():
    assert ts.resolve_side(["", "R", ""], [90.0, 90.0, 90.0]) == ("R", "tag"), "the tag must win over coordinates"
    assert ts.resolve_side(["", ""], [-80.0, -70.0]) == ("R", "coordinates")
    assert ts.resolve_side(["", ""], [80.0, float("nan")]) == ("L", "coordinates")
    assert ts.resolve_side(["", ""], [5.0, -3.0]) == ("", "unresolved"), "near the midline must not be guessed"
    assert ts.resolve_side([], [float("nan")]) == ("", "unresolved")

    vol = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    assert np.array_equal(ts.to_common_side(vol, "Coronal", "L"), vol)
    assert np.array_equal(ts.to_common_side(vol, "Coronal", ""), vol)
    assert np.array_equal(ts.to_common_side(vol, "Axial", "R"), vol[:, :, ::-1])
    assert np.array_equal(ts.to_common_side(vol, "Sagittal", "R"), vol[::-1])


def test_folds_keep_reports_together():
    studies = [f"s{i}" for i in range(100)]
    reports = [f"r{i % 30}" for i in range(100)]
    folds = ts.make_folds(studies, reports, 5, 0)
    by_report = {}
    for study, report in zip(studies, reports):
        by_report.setdefault(report, set()).add(folds[study])
    assert all(len(v) == 1 for v in by_report.values())
    assert set(folds.values()) == set(range(5))
    assert folds == ts.make_folds(studies, reports, 5, 0)


def test_slice_order_and_orientation(tmp):
    a, b = f"{tmp}/order_a", f"{tmp}/order_b"
    write_series(a, "Sagittal", reverse_files=False)
    write_series(b, "Sagittal", reverse_files=True)
    va, ia = ts.load_series(a)
    vb, _ = ts.load_series(b)
    assert va.shape == (6, ts.CACHE_SIZE, ts.CACHE_SIZE) and ia["scanner"] == "Acme|X1"
    assert np.array_equal(va, vb), "file order changed the volume"
    means = va[:, 100:188, 100:188].mean((1, 2))
    assert np.all(np.diff(means) != 0)

    # The same anatomy with flipped row direction must come out identical.
    pixels = np.zeros((64, 64), np.uint16)
    pixels[:20, :] = 1000
    pixels[0:5, 0:5] = 2000
    normal_dir, flipped_dir = f"{tmp}/normal", f"{tmp}/flipped"
    os.makedirs(normal_dir), os.makedirs(flipped_dir)
    for i in range(4):
        position = [float(i * 3), 0.0, 0.0]
        write_slice(f"{normal_dir}/{i}.dcm", pixels, IOP["Sagittal"], position, i + 1)
        write_slice(f"{flipped_dir}/{i}.dcm", pixels[:, ::-1], [0, -1, 0, 0, 0, -1], position, i + 1)
    vn, _ = ts.load_series(normal_dir)
    vf, _ = ts.load_series(flipped_dir)
    assert np.array_equal(vn, vf), "orientation was not canonicalised"


def test_end_to_end(tmp):
    comp, out = f"{tmp}/comp", f"{tmp}/out"
    studies, series_rows, train_rows, label_rows = [], [], [], []
    rng = np.random.default_rng(0)
    planes = [("Sagittal", 1), ("Coronal", 0), ("Axial", 1)]
    for i in range(14):
        study = f"1.{i}"
        is_gold = i < 2
        studies.append(study)
        for j, (plane, fat) in enumerate(planes):
            if i == 5 and plane == "Axial":
                continue  # A missing plane must be masked, not crash.
            series = f"{study}.{j}"
            write_series(f"{comp}/train_series/{study}/{series}", plane, scanner="A" if i % 2 else "B", seed=i * 10 + j)
            series_rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series, "Fluid_Sensitive": 1,
                                "Fat_Suppression": fat, "Anatomical_Plane": plane})
        official = {name: (1 if is_gold else np.nan) for name in ts.LABELS}
        train_rows.append({"StudyInstanceUID": study, "Report": f"report {i % 9}", **official})
        soft = {name: float(rng.random()) for name in ts.LABELS}
        soft.update({f"{name}__conf": 0.95 for name in ts.LABELS})
        soft["Fracture"], soft["Fracture__conf"] = 0.28, 0.05  # A silent report.
        if i == 6:
            soft["Effusion"] = np.nan
        label_rows.append({"StudyInstanceUID": study, **soft})
    pd.DataFrame(train_rows).to_csv(f"{comp}/train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(f"{comp}/train_series.csv", index=False)
    labels_csv = f"{tmp}/labels.csv"
    pd.DataFrame(label_rows).to_csv(labels_csv, index=False)

    cfg = {**ts.DEFAULTS, "backbone": "resnet18", "pretrained": False, "limit_train": 8, "limit_val": 3, "epochs": 2,
           "batch": 2, "accum": 2, "res": 64, "k_train": 2, "k_eval": 2, "dim": 32, "workers": 0, "preprocess_workers": 2,
           "install_decoders": False, "comp_dir": comp, "labels_csv": labels_csv, "out_dir": out, "cache_dir": f"{tmp}/cache"}
    report = ts.run(cfg)

    assert report["status"] == "finished" and report["gold_excluded"] == 2
    assert sum(report["sides"][k] for k in ("from_tag", "from_coordinates", "unresolved")) == 11
    assert report["silent_label_share"]["Fracture"] == 1.0 and report["silent_label_share"]["ACL"] == 0.0
    assert np.isnan(report["epochs"][-1]["per_finding_c"]["Fracture"]), "silent labels were scored"
    assert report["studies"]["pool"] == 12
    folds = pd.read_csv(f"{out}/folds.csv", dtype=str)
    assert not {"1.0", "1.1"} & set(folds["StudyInstanceUID"]), "gold reached the fold table"
    predictions = pd.read_csv(f"{out}/val_predictions.csv", dtype={"StudyInstanceUID": str})
    assert not {"1.0", "1.1"} & set(predictions["StudyInstanceUID"]), "gold was scored"
    assert predictions[ts.LABELS].notna().all().all() and report["epochs"][-1]["loss"] > 0
    cached = {name[:-4] for name in os.listdir(f"{tmp}/cache")}
    assert not any(name.startswith(("1.0.", "1.1.")) for name in cached), "gold was preprocessed"
    with open(f"{out}/pilot_report.json") as handle:
        assert json.load(handle)["cost_estimate"]["train_s_per_study"] > 0

    # A second model in the session reuses the cache, sees the same data, and a failed model loses nothing else.
    many = f"{tmp}/many"
    os.makedirs(many)
    results = ts.train_many({**cfg, "backbones": ["no_such_model", "resnet18"], "out_dir": many})
    assert results["resnet18"] == "finished" and results["no_such_model"].startswith("failed"), results
    with open(f"{many}/resnet18/pilot_report.json") as handle:
        second = json.load(handle)
    assert second["preprocess"]["reused_cache"] and not report["preprocess"]["reused_cache"]
    assert second["sides"] == report["sides"] and second["studies"] == report["studies"]
    assert second["preprocess"]["failed_series"] == report["preprocess"]["failed_series"]
    assert second["cost_estimate"]["strata_in_val"] == report["cost_estimate"]["strata_in_val"]

    # Stage 3: fewer windows are scored, slice spacing is recorded, and every arm trains.
    assert set(report["eval_windows"]) == {"36", "18"} and os.path.exists(f"{out}/val_predictions_w18.csv")
    assert report["slice_mm_quantiles"][0.5] > 0, report["slice_mm_quantiles"]
    arms = ["resnet18@mean", "resnet18@tf", "resnet18@s5", "resnet18@gap35"]
    stage3 = ts.train_many({**cfg, "epochs": 1, "backbones": arms, "out_dir": f"{tmp}/stage3"})
    assert all(v == "finished" for v in stage3.values()), stage3
    with open(f"{tmp}/stage3/resnet18@s5/pilot_report.json") as handle:
        assert json.load(handle)["preflight"]["slices"] == 5
    # Scoring the saved model reproduces its own predictions without training.
    again = ts.run({**cfg, "eval_from": f"{out}/model_ema.pt", "out_dir": f"{tmp}/evaluated"})
    assert again["status"] == "evaluated" and "epochs" not in again
    before = pd.read_csv(f"{out}/val_predictions.csv")[ts.LABELS].to_numpy()
    after = pd.read_csv(f"{tmp}/evaluated/val_predictions.csv")[ts.LABELS].to_numpy()
    assert np.allclose(before, after, atol=1e-4), "a saved model scored differently"
    # Stage 4: each precision scores the same model; FP32 reproduces it and INT8 stays close.
    precisions = ["fp32", "fp16", "int8_minmax", "int8_percentile", "int8_minmax_core", "int8_percentile_core"]
    quant = ts.run({**cfg, "eval_from": f"{out}/model_ema.pt", "out_dir": f"{tmp}/stage4", "precisions": precisions,
                    "calib_studies": 2})
    assert quant["status"] == "evaluated" and set(quant["precisions"]) == set(precisions), quant.get("precisions")
    for precision in precisions:
        p = pd.read_csv(f"{tmp}/stage4/val_predictions_{precision}.csv")[ts.LABELS].to_numpy()
        assert np.abs(p - before).max() < (1e-4 if precision == "fp32" else 0.05), (precision, np.abs(p - before).max())
    int8 = quant["precisions"]["int8_minmax"]
    assert int8["int8_mb"] < int8["fp32_mb"] / 2 and int8["feature_rel_err"] < 0.1, int8
    try:
        ts.run({**cfg, "backbone": "resnet18@tf", "eval_from": f"{out}/model_ema.pt", "out_dir": f"{tmp}/wrong"})
        raise RuntimeError("weights were loaded into a model with a different head")
    except AssertionError as error:
        assert "head" in str(error)

    # A new training seed keeps the validation studies, and changes the model.
    other_seed = ts.run({**cfg, "epochs": 1, "seed": 1, "out_dir": f"{tmp}/seed1"})
    assert other_seed["status"] == "finished"
    assert pd.read_csv(f"{tmp}/seed1/folds.csv", dtype=str).equals(pd.read_csv(f"{out}/folds.csv", dtype=str)), "a seed moved the folds"
    seed1 = pd.read_csv(f"{tmp}/seed1/val_predictions.csv", dtype=str)["StudyInstanceUID"]
    assert seed1.equals(pd.read_csv(f"{out}/val_predictions.csv", dtype=str)["StudyInstanceUID"]), "a seed moved the validation studies"

    # QAT on a real ConvNeXt: fine-tunes under simulation, saves plain weights, and those convert to INT8 like any others.
    atto = {**cfg, "backbone": "convnext_atto", "epochs": 1, "out_dir": f"{tmp}/atto"}
    assert ts.run(atto)["status"] == "finished"
    real_find = ts.find_input
    try:
        ts.find_input = lambda name: f"{tmp}/atto/model_ema.pt"
        qat = ts.run({**cfg, "backbone": "convnext_atto@qat", "qat_observe": 2, "out_dir": f"{tmp}/qat"})
    finally:
        ts.find_input = real_find
    assert qat["status"] == "finished" and len(qat["epochs"]) == 3 and "qat_plain_fp32" in qat
    import torch

    start, tuned = (torch.load(f"{tmp}/{d}/model_ema.pt", weights_only=False)["model"] for d in ("atto", "qat"))
    assert set(tuned) == set(start), "the saved QAT weights do not fit a plain model"
    assert any(not torch.equal(tuned[k], start[k]) for k in tuned if k.startswith("enc.")), "QAT did not change the encoder"
    simulated = pd.read_csv(f"{tmp}/qat/val_predictions.csv")[ts.LABELS].to_numpy()
    unrounded = pd.read_csv(f"{tmp}/qat/val_predictions_qat_plain.csv")[ts.LABELS].to_numpy()
    assert np.abs(simulated - unrounded).max() > 1e-6, "the simulation changed nothing"
    converted = ts.run({**cfg, "backbone": "convnext_atto", "eval_from": f"{tmp}/qat/model_ema.pt", "out_dir": f"{tmp}/qat_int8",
                        "precisions": ["int8_minmax"], "calib_studies": 2, "int8_save": True, "int8_score": False})
    assert converted["precisions"]["int8_minmax"]["scored"] is False and os.path.exists(f"{tmp}/qat_int8/encoder_int8_minmax.onnx")
    assert not os.path.exists(f"{tmp}/qat_int8/val_predictions_int8_minmax.csv")

    # R3D-18: pauses cleanly at the session limit, then resumes and finishes all epochs.
    assert ts.MODEL_OVERRIDES["r3d_18"]["res"] == 192 and ts.MODEL_OVERRIDES["r3d_18"]["k_train"] == 32
    real = ts.MODEL_OVERRIDES["r3d_18"]
    small = {"res": 64, "batch": 2, "accum": 2, "k_train": 4, "k_eval": 4}
    try:
        ts.MODEL_OVERRIDES["r3d_18"] = {**small, "max_hours": 1e-9}
        paused = ts.run({**cfg, "backbone": "r3d_18", "out_dir": f"{tmp}/r3d"})
        assert paused["status"] == "paused" and len(paused["epochs"]) == 1
        assert not os.path.exists(f"{tmp}/r3d/val_predictions.csv")
        ts.MODEL_OVERRIDES["r3d_18"] = small
        done = ts.run({**cfg, "backbone": "r3d_18", "out_dir": f"{tmp}/r3d2", "resume_from": f"{tmp}/r3d/checkpoint.pt"})
    finally:
        ts.MODEL_OVERRIDES["r3d_18"] = real
    assert done["status"] == "finished" and [e["epoch"] for e in done["epochs"]] == [0, 1]
    # Compared field by field: the silent finding's score is NaN, which never equals itself.
    for key in ("epoch", "loss", "within_scanner_c"):
        assert done["epochs"][0][key] == paused["epochs"][0][key], "resuming lost the earlier history"
    assert not np.isnan(done["epochs"][-1]["within_scanner_c"])
    assert pd.read_csv(f"{tmp}/r3d2/val_predictions.csv")[ts.LABELS].notna().all().all()


def test_stage2_models_build(tmp):
    """Every Stage 2 encoder must build, step and evaluate at its real input size, with its own normalisation."""
    expected_res = {"vit_small_patch14_dinov2": 252}
    for backbone in ["mobilenetv4_conv_small", "tf_efficientnetv2_b0", "resnet50", "convnext_atto", "convnext_tiny",
                     "coatnet_0_rw_224", "vit_small_patch14_dinov2", "resnet34"]:
        cfg = {**ts.DEFAULTS, "backbone": backbone, "pretrained": False, "batch": 1, "k_train": 1, "k_eval": 1,
               "install_decoders": False, "preflight_only": True, "out_dir": f"{tmp}/pre", "cache_dir": f"{tmp}/pre_cache"}
        result = ts.run(cfg)["preflight"]
        assert result["res"] == expected_res.get(backbone, 256), backbone
        if backbone.startswith("coatnet"):
            assert result["mean"] == [0.5, 0.5, 0.5], "CoAtNet weights expect 0.5 normalisation"
        print("built", backbone, result["params_m"], "M")


def test_variant_settings(tmp):
    """A "name@variant" run uses the same encoder with its own learning rates and pooling."""
    cfg = {**ts.DEFAULTS, "backbone": "vit_small_patch14_dinov2@llrd_avg", "pretrained": False, "batch": 1, "k_train": 1,
           "k_eval": 1, "install_decoders": False, "preflight_only": True, "out_dir": f"{tmp}/variant", "cache_dir": f"{tmp}/variant_cache"}
    result = ts.run(cfg)
    assert result["preflight"]["res"] == 252 and result["config"]["layer_decay"] == 0.75
    assert result["config"]["global_pool"] == "avg" and result["config"]["lr"] == 5e-5


def test_stage3_rules():
    """Window slices, spacing steps and variant settings for the Stage 3 arms."""
    assert ts.window_slices(5, 10, 3, 1).tolist() == [4, 5, 6], "3 adjacent slices must match Stage 2"
    assert ts.window_slices(1, 10, 5, 2).tolist() == [0, 0, 1, 3, 5], "edges repeat the first slice"
    assert ts.window_slices(9, 10, 3, 3).tolist() == [6, 9, 9]
    assert ts.slice_step(1.0, 3.5) == 4 and ts.slice_step(4.0, 3.5) == 1 and ts.slice_step(3.3, 0) == 1
    assert ts.slice_step(float("nan"), 3.5) == 1, "unknown spacing falls back to adjacent slices"
    base = {**ts.DEFAULTS}
    tiny = ts.settings_for({**base, "backbone": "convnext_tiny@mean"})
    assert tiny["head"] == "mean" and tiny["batch"] == 2 and tiny["accum"] == 4, "a variant lost the encoder's settings"
    assert ts.settings_for({**base, "backbone": "convnext_atto@eval"})["eval_from"] == "convnext_atto_model_ema.pt"
    assert ts.settings_for({**base, "backbone": "convnext_atto@r192"})["res"] == 192
    dino = ts.settings_for({**base, "backbone": "vit_small_patch14_dinov2@llrd"})
    assert dino["res"] == 252 and dino["layer_decay"] == 0.75
    try:
        ts.settings_for({**base, "backbone": "convnext_atto@typo"})
        raise AssertionError("an unknown variant must fail")
    except AssertionError as error:
        assert "unknown variant" in str(error)


def test_probe_survives_a_bad_model(tmp):
    cfg = {**ts.DEFAULTS, "probe_backbones": ["resnet18", "no_such_model"], "pretrained": False, "batch": 1, "k_train": 1,
           "k_eval": 1, "res": 64, "out_dir": f"{tmp}/probe", "cache_dir": f"{tmp}/probe_cache"}
    results = ts.probe(cfg)
    assert [r["fits"] for r in results] == [True, False]
    assert os.path.exists(f"{tmp}/probe/gpu_probe.json")


def main():
    test_metric()
    test_side_rules()
    test_folds_keep_reports_together()
    with tempfile.TemporaryDirectory() as tmp:
        test_slice_order_and_orientation(tmp)
        test_end_to_end(tmp)
        test_stage2_models_build(tmp)
        test_probe_survives_a_bad_model(tmp)
        test_variant_settings(tmp)
    test_stage3_rules()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
