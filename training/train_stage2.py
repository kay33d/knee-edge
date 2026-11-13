"""Train one Stage 2 encoder on one fold and report what the full run would cost.

Preprocessing and model structure are adapted from goodpjw2008/rsna-knee-2-5d-convnext-reader
(Apache-2.0). Changes: a smaller pixel cache, per-finding attention with no transformer,
masked loss for missing labels, within-scanner scoring, and no use of the gold studies.

The gold studies are never loaded for training, validation or model selection.
Override DEFAULTS with a JSON file named by the KNEE_CONFIG environment variable.
"""
import copy
import gc
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import types
import zlib
from multiprocessing import Pool

import numpy as np

DEFAULTS = {
    "backbone": "resnet50",
    "pretrained": True,
    "limit_train": 0,  # 0 uses every study
    "limit_val": 0,
    "epochs": 14,
    "batch": 4,
    "accum": 2,
    "res": 256,
    "k_train": 8,
    "k_eval": 12,  # 12 windows x 6 slots = 72 per study, as benchmarked in Stage 1
    "lr": 1e-4,
    "lr_head": 5e-4,
    "wd": 0.02,
    "ema": 0.998,
    "dim": 384,
    "workers": 4,
    "preprocess_workers": 4,
    "fold": 0,
    "n_folds": 5,
    "seed": 0,  # Training randomness only. The folds use fold_seed, so a new seed keeps the same validation studies.
    "fold_seed": 0,
    "planned_epochs": 14,
    "install_decoders": True,
    "comp_dir": None,
    "labels_csv": None,
    "out_dir": "/kaggle/working",
    "cache_dir": "/tmp/knee_cache",  # /kaggle/working is capped at 19.5 GB; the full cache is about 56 GB
    "preflight_only": False,
    "probe_backbones": [],  # Non-empty: measure these on the GPU with synthetic data and train nothing.
    "backbones": [],  # Non-empty: train these in turn in one session, sharing the preprocessing.
    "max_hours": 0,  # Above 0: keep a checkpoint each epoch and stop cleanly before this much session time is used.
    "resume_from": None,  # Path of a checkpoint from a run that stopped at max_hours.
    "layer_decay": 0,  # Above 0: each earlier encoder block gets this times the learning rate of the block after it.
    "global_pool": None,  # For example "avg" to pool patch tokens instead of using a ViT's class token.
    "head": "attn",  # Study-level head: "attn" (per-finding attention), "mean" (masked mean), "attn_tf" (2-layer transformer, then attention).
    "slices": 3,  # Slices per 2.5D window, used as input channels.
    "slice_gap_mm": 0,  # 0: adjacent slices. Above 0: slices this many mm apart (rounded to whole slices, at least 1).
    "eval_windows": [6, 3],  # Extra windows per slot to score the final model at: 6 and 3 give 36 and 18 per study.
    "eval_from": None,  # File name of a saved model_ema.pt under /kaggle/input: score it and train nothing.
    # Stage 4, with eval_from: score the encoder at these precisions instead of the usual scoring. "fp32" turns off
    # mixed precision; "fp16" is a half-precision copy on this device; "int8_minmax" and "int8_percentile" are ONNX Runtime
    # static quantization on CPU. The head stays FP32 in every case: it costs milliseconds.
    "precisions": [],
    "calib_studies": 32,  # Training-split studies whose windows set the INT8 ranges; never validation studies.
    "int8_save": False,  # Keep each INT8 encoder in out_dir, for timing on the laptop.
    "int8_score": True,  # False: build the INT8 encoder and skip scoring it (CPU scoring takes hours).
    "init_from": None,  # Weights to start training from: a local path or a file name under /kaggle/input.
    "qat": False,  # Fine-tune the encoder under simulated INT8, then save plain weights for the usual INT8 conversion.
    "qat_observe": 100,  # Forward passes that only measure activation ranges before quantization noise starts.
}

# DINOv2's patch size is 14, so 252 px is the nearest valid size to 256.
# ConvNeXt-Tiny and CoAtNet-0 run out of memory on a T4 at batch 4. Batch 2 with 4 accumulation steps
# keeps 8 studies per update and the same number of updates per epoch.
MODEL_OVERRIDES = {"vit_small_patch14_dinov2": {"res": 252},
                   "convnext_tiny": {"batch": 2, "accum": 4},
                   "coatnet_0_rw_224": {"batch": 2, "accum": 4},
                   # One clip per series: 32 slices at 192 px, about 0.74 mm/pixel. 256 px would need about 15 h.
                   "r3d_18": {"res": 192, "batch": 2, "accum": 4, "k_train": 32, "k_eval": 32, "max_hours": 11},
                   # Tuning check added after DINOv2 underfit at the shared settings. Not part of the planned grid.
                   "vit_small_patch14_dinov2@llrd": {"res": 252, "lr": 5e-5, "layer_decay": 0.75},
                   # Second seed: same folds, new training randomness. Both heads, so their difference can be checked.
                   "convnext_tiny@seed1": {"seed": 1},
                   "convnext_tiny@tf_seed1": {"head": "attn_tf", "seed": 1},
                   # Stage 4 quantization-aware fine-tuning, and scoring its result.
                   "convnext_tiny@tf_qat": {"head": "attn_tf", "qat": True, "init_from": "convnext_tiny_tf_model_ema.pt", "epochs": 3,
                                            "planned_epochs": 3, "lr": 1e-5, "lr_head": 5e-5},
                   "convnext_atto@qat_eval": {"eval_from": "convnext_atto_qat_model_ema.pt"},
                   "convnext_tiny@qat_eval": {"eval_from": "convnext_tiny_qat_model_ema.pt"},
                   "convnext_tiny@tf_qat_eval": {"head": "attn_tf", "eval_from": "convnext_tiny_tf_qat_model_ema.pt"},
                   # Stage 4: the best Stage 3 arm, scored from its saved weights.
                   "convnext_tiny@tf_eval": {"head": "attn_tf", "eval_from": "convnext_tiny_tf_model_ema.pt"},
                   "vit_small_patch14_dinov2@llrd_avg": {"res": 252, "lr": 5e-5, "layer_decay": 0.75, "global_pool": "avg"}}
# Stage 3 arms, applied to any encoder as "name@variant", on top of the encoder's own settings.
VARIANTS = {"mean": {"head": "mean"}, "tf": {"head": "attn_tf"}, "r192": {"res": 192}, "s5": {"slices": 5},
            # About the slice spacing of a typical 2D knee sequence, so thin series are sampled like thick ones.
            "gap35": {"slice_gap_mm": 3.5},
            # The saved Stage 2 model, scored at fewer windows without training.
            "eval": {"eval_from": "{arch}_model_ema.pt"},
            "qat": {"qat": True, "init_from": "{arch}_model_ema.pt", "epochs": 3, "planned_epochs": 3, "lr": 1e-5, "lr_head": 5e-5}}
SESSION_START = time.time()

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = ["Sagittal", "Coronal", "Axial"]
N_SLOTS = 6
FOV_MM = 153.6
CACHE_SIZE = 288  # 153.6 mm at 0.533 mm/pixel; the model sees a 0.92 crop at 256 px, about 0.55 mm/pixel
MM_PER_PX = FOV_MM / CACHE_SIZE
MAX_SLICES = 64
FULL_SERIES = 24371  # rows in train_series.csv
SILENT_CONF = 0.05
SIDE_MIN_MM = 20.0

# plane -> (slice-order sign, (col axis, sign), (row axis, sign)) in LPS coordinates
CANON = {
    0: (+1, (1, +1), (2, -1)),
    1: (+1, (0, +1), (2, -1)),
    2: (-1, (0, +1), (1, +1)),
}


def _read_dicom(path):
    import pydicom

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    slope, inter = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if slope is not None and inter is not None:
        arr = arr * float(slope) + float(inter)
    iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
    ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
    spacing = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
    instance = float(ds.get("InstanceNumber") or 0)
    scanner = f"{ds.get('Manufacturer', '') or ''}|{ds.get('ManufacturerModelName', '') or ''}"
    tag = str(ds.get("Laterality", "") or ds.get("ImageLaterality", "") or "").strip().upper()
    centre_x = None
    if iop and ipp and spacing:
        # The first direction cosines run along a row, so they pair with the column spacing.
        centre_x = ipp[0] + iop[0] * spacing[1] * int(ds.Columns) / 2 + iop[3] * spacing[0] * int(ds.Rows) / 2
    frames = list(arr) if arr.ndim == 3 else [arr]
    return [(f, iop, ipp, spacing, instance + i * 1e-3, scanner, tag, centre_x) for i, f in enumerate(frames)]


def _canon_slice(img, iop, spacing, plane):
    """Reorient one slice to the canonical in-plane orientation."""
    spacing = spacing or [MM_PER_PX * img.shape[0] / CACHE_SIZE] * 2
    if iop is None:
        return img, spacing
    r, c = np.array(iop[:3]), np.array(iop[3:])
    _, (cax, csgn), (rax, rsgn) = CANON[plane]
    if abs(r[cax]) < abs(r[rax]):
        img, r, c, spacing = img.T, c, r, spacing[::-1]
    if r[cax] * csgn < 0:
        img = img[:, ::-1]
    if c[rax] * rsgn < 0:
        img = img[::-1]
    return img, spacing


def _resample(img, spacing):
    """Centre crop to the field of view, resample to the cache spacing, zero-pad."""
    import cv2

    out = np.zeros((CACHE_SIZE, CACHE_SIZE), np.float32)
    h, w = img.shape
    ch = min(h, int(round(FOV_MM / spacing[0])))
    cw = min(w, int(round(FOV_MM / spacing[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    nh = max(1, min(CACHE_SIZE, int(round(ch * spacing[0] / MM_PER_PX))))
    nw = max(1, min(CACHE_SIZE, int(round(cw * spacing[1] / MM_PER_PX))))
    interp = cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (nw, nh), interpolation=interp)
    oy, ox = (CACHE_SIZE - nh) // 2, (CACHE_SIZE - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = resized
    return out


def load_series(series_dir, plane_name=None):
    """Return a canonical uint8 volume [n, H, W] and an info dict. Unreadable slices are skipped."""
    slices, bad = [], 0
    for name in os.listdir(series_dir):
        try:
            slices.extend(_read_dicom(os.path.join(series_dir, name)))
        except Exception:
            bad += 1
    info = {"n_raw": len(slices), "n_bad": bad, "scanner": "", "tag": "", "centre_x": float("nan"), "slice_mm": float("nan")}
    if not slices:
        return np.zeros((1, CACHE_SIZE, CACHE_SIZE), np.uint8), {**info, "n": 0}
    info["scanner"] = slices[0][5]
    info["tag"] = next((s[6] for s in slices if s[6] in ("L", "R")), "")
    centres = [s[7] for s in slices if s[7] is not None]
    info["centre_x"] = float(np.mean(centres)) if centres else float("nan")

    iop = next((s[1] for s in slices if s[1] is not None), None)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:])
        plane = int(np.abs(normal).argmax())
    else:
        normal, plane = None, PLANES.index(plane_name) if plane_name in PLANES else 0

    if normal is not None and all(s[2] is not None for s in slices):
        sign = CANON[plane][0] * np.sign(normal[plane])
        slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sign)
    else:
        slices.sort(key=lambda s: s[4])
    if len(slices) > MAX_SLICES:
        keep = np.linspace(0, len(slices) - 1, MAX_SLICES).round().astype(int)
        slices = [slices[i] for i in keep]
    info["slice_mm"] = float("nan")  # Distance between cached slices, measured along the slice normal.
    if normal is not None and len(slices) > 1 and all(s[2] is not None for s in slices):
        depth = np.array([float(np.dot(s[2], normal)) for s in slices])
        info["slice_mm"] = float(np.median(np.abs(np.diff(depth))))

    vol = np.stack([_resample(*_canon_slice(s[0], s[1], s[3], plane)) for s in slices])
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    info["n"] = len(vol)
    return (vol * 255).round().astype(np.uint8), info


def _cache_job(args):
    series_dir, series, plane, out_dir = args
    out = f"{out_dir}/{series}.npy"
    try:
        vol, info = load_series(series_dir, plane)
        np.save(out + ".tmp.npy", vol)
        os.replace(out + ".tmp.npy", out)
        info["bytes"] = os.path.getsize(out)
        info["error"] = ""
    except Exception as error:
        info = {"n": 0, "n_raw": 0, "n_bad": 0, "scanner": "", "tag": "", "centre_x": float("nan"), "bytes": 0,
                "slice_mm": float("nan"), "error": f"{type(error).__name__}: {error}"[:200]}
    return {"series": series, **info}


def resolve_side(tags, centres):
    """Return ("R" | "L" | "", source) for a study: the DICOM tag first, then patient coordinates.

    +x points to the patient's left. On non-gold studies with a tag, coordinates agreed with it
    98.6% of the time at 20 mm or more from the midline. Closer studies stay unresolved, not guessed.
    """
    known = [tag for tag in tags if tag in ("L", "R")]
    if known:
        return max(set(known), key=known.count), "tag"
    values = [c for c in centres if c == c]
    if values and abs(float(np.median(values))) >= SIDE_MIN_MM:
        return ("R" if float(np.median(values)) < 0 else "L"), "coordinates"
    return "", "unresolved"


def to_common_side(vol, plane, side):
    """Mirror right knees onto left. Medial-lateral is in plane for coronal and axial, and the slice axis for sagittal."""
    if side != "R":
        return vol
    return vol[::-1] if plane == "Sagittal" else vol[:, :, ::-1]


def make_folds(studies, reports, n_folds, seed):
    """Assign folds so identical report text never spans two folds."""
    groups = {}
    for study, report in zip(studies, reports):
        groups.setdefault(report, []).append(study)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda key: -len(groups[key]))  # stable, so ties keep their shuffled order
    sizes, folds = [0] * n_folds, {}
    for key in keys:
        fold = int(np.argmin(sizes))
        sizes[fold] += len(groups[key])
        for study in groups[key]:
            folds[study] = fold
    return folds


def within_scanner_concordance(y, p, strata):
    """Macro pairwise concordance, counting only pairs from the same scanner model.

    Pairs tied on the target are dropped, prediction ties count as one half, and missing
    targets are ignored. Returns (macro, per-finding list).
    """
    each = []
    for c in range(y.shape[1]):
        score, pairs = 0.0, 0
        for stratum in np.unique(strata):
            keep = (strata == stratum) & ~np.isnan(y[:, c])
            yc, pc = y[keep, c], p[keep, c]
            higher = yc[:, None] > yc[None, :]
            diff = pc[:, None] - pc[None, :]
            score += float((diff[higher] > 0).sum()) + 0.5 * float((diff[higher] == 0).sum())
            pairs += int(higher.sum())
        each.append(score / pairs if pairs else float("nan"))
    return float(np.nanmean(each)), each


def macro_auc_thresholded(y, p):
    """Secondary measure: AUC against soft labels thresholded at 0.5."""
    from sklearn.metrics import roc_auc_score

    each = []
    for c in range(y.shape[1]):
        keep = ~np.isnan(y[:, c])
        binary = y[keep, c] > 0.5
        each.append(float(roc_auc_score(binary, p[keep, c])) if 0 < binary.sum() < len(binary) else float("nan"))
    return float(np.nanmean(each)), each


def window_centres(n, k, train, lo=0.04, hi=0.96):
    """k slice indices (centres of 3-slice windows) spread over the series."""
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    if train:
        edges = np.linspace(a, b, k + 1)
        c = edges[:-1] + np.random.rand(k) * (edges[1:] - edges[:-1])
    else:
        c = np.linspace(a, b, k)
    return np.clip(np.round(c).astype(int), 1, max(1, n - 2))


def window_slices(centre, n, slices, step):
    """Slice indices of one window: `slices` slices `step` apart around `centre`, repeating the edge slice."""
    half = slices // 2
    return np.clip(centre + step * np.arange(-half, half + 1), 0, n - 1)


def slice_step(slice_mm, gap_mm):
    """Slices between window members: 1 for adjacent, otherwise the nearest whole number for gap_mm (at least 1)."""
    if not gap_mm or not slice_mm == slice_mm or slice_mm <= 0:
        return 1
    return max(1, int(round(gap_mm / slice_mm)))


def build_study_table(series_rows, n_slices):
    """Return {study: [[series of slot 0], ...]}, largest series first. Fat-suppressed slots come first."""
    table = {}
    for study, series, fat, plane in series_rows:
        if n_slices.get(series, 0) < 3:
            continue
        slot = PLANES.index(plane) + 3 * (1 - int(fat))
        table.setdefault(study, [[] for _ in range(N_SLOTS)])[slot].append(series)
    for slots in table.values():
        for candidates in slots:
            candidates.sort(key=lambda s: -n_slices[s])
    return table


def find_input(name):
    """Look a few levels down first: a recursive search walks every DICOM file in the competition data."""
    for depth in range(1, 5):
        found = glob.glob("/kaggle/input/" + "*/" * depth + name)
        if found:
            return found[0]
    raise FileNotFoundError(name)


def settings_for(cfg):
    """The encoder's own settings, then the variant's, then any entry for the exact "name@variant"."""
    base_name, _, variant = cfg["backbone"].partition("@")
    assert not variant or variant in VARIANTS or cfg["backbone"] in MODEL_OVERRIDES, f"unknown variant {variant}"
    extra = {k: (v.format(arch=base_name) if isinstance(v, str) else v) for k, v in VARIANTS.get(variant, {}).items()}
    return {**cfg, **MODEL_OVERRIDES.get(base_name, {}), **extra, **(MODEL_OVERRIDES.get(cfg["backbone"], {}) if variant else {})}


def run(cfg):
    cfg = settings_for(cfg)
    if cfg["install_decoders"]:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylibjpeg", "pylibjpeg-libjpeg",
                        "pylibjpeg-openjpeg"], check=True)
        if any(p.startswith("int8") for p in cfg["precisions"]):  # The versions in the CPU timing image.
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime==1.31.1", "onnx==1.23.2"], check=True)

    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    # Each encoder gets the normalisation its weights were trained with.
    arch = cfg["backbone"].split("@")[0]  # "name@variant" is the same encoder with the variant's overrides.
    is_3d = arch == "r3d_18"
    if is_3d:  # The normalisation published with torchvision's Kinetics-400 weights.
        enc_cfg = types.SimpleNamespace(tag="kinetics400_v1", mean=(0.43216, 0.394666, 0.37645),
                                        std=(0.22803, 0.22145, 0.216989), fixed_input_size=False)
    else:
        enc_cfg = timm.get_pretrained_cfg(arch)
    enc_kwargs = {"img_size": cfg["res"]} if enc_cfg.fixed_input_size else {}
    chans = 3 if is_3d else cfg["slices"]
    if chans != 3:
        enc_kwargs["in_chans"] = chans  # timm adapts the pretrained first layer to the new channel count.
    # With other than 3 channels, each channel gets the average of the weights' per-channel values.
    norm_mean = list(enc_cfg.mean) if chans == 3 else [float(np.mean(enc_cfg.mean))] * chans
    norm_std = list(enc_cfg.std) if chans == 3 else [float(np.mean(enc_cfg.std))] * chans
    if cfg["global_pool"]:
        # Keep the final norm before pooling, so the pretrained "norm" weights still load (timm would otherwise swap in fc_norm).
        enc_kwargs.update(global_pool=cfg["global_pool"], fc_norm=False)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    report = {"config": cfg, "device": device, "status": "started"}
    report_path = os.path.join(cfg["out_dir"], "pilot_report.json")

    def save_report():
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=2)

    def say(message):
        print(message, flush=True)

    if device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
    save_report()

    def make_views(x, train, group):
        """Crop, resize and normalise; training adds geometry and intensity jitter shared within a series."""
        fov, res = 0.92, cfg["res"]
        n = x.shape[0]
        ns = n // group
        dev = x.device
        x = x.float().div_(255.0)
        if train:
            angle = (torch.rand(ns, device=dev) - 0.5) * 2 * math.radians(12)
            scale = fov * (1 + (torch.rand(ns, device=dev) - 0.5) * 0.24)
            tx = (torch.rand(ns, device=dev) - 0.5) * 0.12
            ty = (torch.rand(ns, device=dev) - 0.5) * 0.12
            gamma = torch.exp((torch.rand(ns, device=dev) - 0.5) * 0.5)
            gain = 1 + (torch.rand(ns, device=dev) - 0.5) * 0.2
        else:
            angle = torch.zeros(ns, device=dev)
            scale = torch.full((ns,), fov, device=dev)
            tx = ty = torch.zeros(ns, device=dev)
        cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
        theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
        theta = theta.repeat_interleave(group, 0)
        grid = F.affine_grid(theta, (n, x.shape[1], res, res), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if train:  # No flips: left and right knees must stay distinct.
            x = x.clamp_min(1e-4).pow(gamma.repeat_interleave(group).view(n, 1, 1, 1))
            x = x * gain.repeat_interleave(group).view(n, 1, 1, 1)
        mean = torch.tensor(norm_mean, device=dev).view(1, -1, 1, 1)
        std = torch.tensor(norm_std, device=dev).view(1, -1, 1, 1)
        return (x - mean) / std

    class KneeNet(nn.Module):
        """2D encoder on 3-slice windows, tagged by slot and position, with one attention pool per finding."""

        def __init__(self):
            super().__init__()
            if is_3d:
                import torchvision

                weights = torchvision.models.video.R3D_18_Weights.KINETICS400_V1 if cfg["pretrained"] else None
                self.enc = torchvision.models.video.r3d_18(weights=weights)
                out_dim = self.enc.fc.in_features
                self.enc.fc = nn.Identity()
            else:
                self.enc = timm.create_model(arch, pretrained=cfg["pretrained"], num_classes=0, **enc_kwargs)
                # MobileNetV4's final convolution makes its output wider than num_features.
                out_dim = getattr(self.enc, "head_hidden_size", None) or self.enc.num_features
            dim = cfg["dim"]
            self.proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(out_dim, dim), nn.LayerNorm(dim))
            self.slot_emb = nn.Embedding(N_SLOTS, dim)
            self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
            self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, len(LABELS)))
            # Mixes windows before pooling, so a window's score can depend on the others.
            self.mix = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 8, 2 * dim, 0.1, batch_first=True, norm_first=True),
                                             2, enable_nested_tensor=False) if cfg["head"] == "attn_tf" else None
            self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
            self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))

        def forward(self, x, mask, pos, train):
            b, s, k = x.shape[:3]
            keep = mask.view(-1)  # Run the encoder only on series that exist.
            flat = x.view(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
            with torch.amp.autocast(device, enabled=False):
                views = make_views(flat, train, k)
            if is_3d:  # The slices of each series become one clip [series, 3, slices, H, W], so one token per slot.
                clips = views.view(int(keep.sum()), k, *views.shape[1:]).transpose(1, 2)
                encoded = self.enc(clips)
                pos = pos.mean(-1, keepdim=True)
                k = 1
            else:
                encoded = self.enc(views)
            feat = encoded.new_zeros(b * s, k, encoded.shape[-1])
            feat[keep] = encoded.view(-1, k, encoded.shape[-1])
            return self.head(feat.view(b, s, k, -1), mask, pos)

        def head(self, feat, mask, pos):
            """Study-level part: encoder features [b, slots, k, d] to 12 logits. Exported on its own for CPU timing."""
            b, s, k = feat.shape[:3]
            slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
            h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
            h = h.view(b, s * k, -1)
            pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
            pad = pad & ~pad.all(1, keepdim=True)  # A study with no usable series attends to everything.
            if self.mix is not None:
                h = self.mix(h, src_key_padding_mask=pad)
            if cfg["head"] == "mean":  # The same pooled vector for every finding.
                keep_w = (~pad).to(h.dtype).unsqueeze(-1)
                pooled = ((h * keep_w).sum(1) / keep_w.sum(1).clamp_min(1)).unsqueeze(1)
                return (pooled * self.cls_w).sum(-1) + self.cls_b
            weights = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
            pooled = torch.einsum("btc,btd->bcd", weights.to(h.dtype), h)
            return (pooled * self.cls_w).sum(-1) + self.cls_b

    class WeightFakeQuant(nn.Module):
        """Per-channel symmetric 7-bit weights, as ONNX Runtime's reduce_range gives. Gradients pass straight through."""

        def forward(self, w):
            step = w.detach().abs().amax(tuple(range(1, w.ndim)), keepdim=True).clamp_min(1e-12) / 63
            return w + (torch.round(w / step) * step - w).detach()

    class ActivationFakeQuant(nn.Module):
        """8-bit range tracked during training, as calibration would measure it. Measures only until `observe` passes.

        The rounding and clipping are one detached correction added to the input, so the gradient is the input's own and
        autograd keeps nothing extra. Building them from differentiable steps tripled GPU memory (Atto 4.6 to 13 GB).
        """

        def __init__(self, observe):
            super().__init__()
            self.observe = observe
            self.register_buffer("lo", torch.tensor(0.0))
            self.register_buffer("hi", torch.tensor(0.0))
            self.register_buffer("seen", torch.tensor(0, dtype=torch.long))

        def forward(self, x):
            with torch.no_grad():
                value = x.float()
                if self.training:
                    first = self.seen == 0
                    self.lo.copy_(value.min() if first else 0.99 * self.lo + 0.01 * value.min())
                    self.hi.copy_(value.max() if first else 0.99 * self.hi + 0.01 * value.max())
                    self.seen += 1
                if self.seen < self.observe:
                    return x
                lo, hi = self.lo.clamp_max(0.0), self.hi.clamp_min(0.0)  # A quantized range always contains zero.
                step = (hi - lo).clamp_min(1e-8) / 255
                clipped = torch.minimum(torch.maximum(value, lo), hi)
                correction = (torch.round((clipped - lo) / step) * step + lo - value).to(x.dtype)
            return x + correction

    def fake_quant_output(module, _, output):
        return module.act_fq(output)

    def prepare_qat(enc):
        """Simulate INT8 where the converter will put it: weights of linear layers, and the outputs of these layers."""
        from torch.nn.utils import parametrize

        for module in list(enc.modules()):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parametrize.register_parametrization(module, "weight", WeightFakeQuant())
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.GELU, nn.LayerNorm)):
                module.add_module("act_fq", ActivationFakeQuant(cfg["qat_observe"]))
                module.register_forward_hook(fake_quant_output)

    def strip_qat(enc):
        """Back to a plain encoder holding the fine-tuned weights, unrounded, so its file loads into an ordinary model."""
        from torch.nn.utils import parametrize

        for module in list(enc.modules()):
            if parametrize.is_parametrized(module, "weight"):
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
            if hasattr(module, "act_fq"):
                del module.act_fq
                module._forward_hooks.clear()

    if cfg.get("build_only"):  # For CPU timing and conversion: the untrained model and its settings, nothing else.
        return KneeNet(), cfg

    def preflight():
        """One real-sized training and validation step on synthetic data, on a throwaway model.

        Running out of GPU memory here costs seconds; after preprocessing it would cost over an hour.
        """
        net = KneeNet()
        if cfg["qat"]:  # With the simulation fully on, so its memory cost is part of this check.
            prepare_qat(net.enc)
            for module in net.enc.modules():
                if isinstance(module, ActivationFakeQuant):
                    module.observe = 0
        net = net.to(device)
        shadow = copy.deepcopy(net).eval()
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"])
        grad_scaler = torch.amp.GradScaler(device, enabled=use_amp)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def inputs(k):
            shape = (cfg["batch"], N_SLOTS, k, chans, CACHE_SIZE, CACHE_SIZE)
            return (torch.randint(0, 256, shape, dtype=torch.uint8, device=device),
                    torch.ones((cfg["batch"], N_SLOTS), dtype=torch.bool, device=device),
                    torch.rand((cfg["batch"], N_SLOTS, k), device=device))

        def settle():
            if device == "cuda":
                torch.cuda.synchronize()

        net.train()
        times = []
        for _ in range(4):  # The first step includes set-up, so it is not timed.
            x, mask, pos = inputs(cfg["k_train"])
            start = time.time()
            with torch.amp.autocast(device, enabled=use_amp):
                logits = net(x, mask, pos, True)
            loss = F.binary_cross_entropy_with_logits(logits.float(), torch.zeros_like(logits).float())
            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt)
            grad_scaler.update()
            opt.zero_grad(set_to_none=True)
            settle()
            times.append(time.time() - start)
        x, mask, pos = inputs(cfg["k_eval"])
        start = time.time()
        with torch.no_grad(), torch.amp.autocast(device, enabled=use_amp):
            shadow(x, mask, pos, False)
        settle()
        result = {"backbone": cfg["backbone"], "res": cfg["res"], "pretrained_tag": enc_cfg.tag,
                  "mean": norm_mean, "std": norm_std, "head": cfg["head"], "slices": chans,
                  "params_m": round(sum(p.numel() for p in net.enc.parameters()) / 1e6, 2),
                  "train_step_s": round(float(np.median(times[1:])), 3), "val_step_s": round(time.time() - start, 3)}
        if device == "cuda":
            result["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            result["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        del net, shadow, opt, grad_scaler
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        # The real run starts from the same random state as if this step had never happened.
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        return result

    if not cfg["eval_from"]:  # Scoring only needs no training-memory check.
        report["preflight"] = preflight()
        say(f"preflight: {report['preflight']}")
        save_report()
    if cfg["preflight_only"]:
        return report
    import pandas as pd  # Imported late, so building a model for CPU timing needs no pandas.

    comp = cfg["comp_dir"] or os.path.dirname(find_input("train_series.csv"))
    labels_csv = cfg["labels_csv"] or find_input("report_labels_v2.csv")
    say(f"competition data: {comp}; labels: {labels_csv}")

    train_df = pd.read_csv(f"{comp}/train.csv", dtype={"StudyInstanceUID": str})
    series_df = pd.read_csv(f"{comp}/train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    labels_df = pd.read_csv(labels_csv, dtype={"StudyInstanceUID": str})

    # Gold is identified from the official labels and kept out of everything below.
    gold = set(train_df.loc[train_df[LABELS].notna().any(axis=1), "StudyInstanceUID"])
    labels_df = labels_df[~labels_df["StudyInstanceUID"].isin(gold)]
    assert not gold & set(labels_df["StudyInstanceUID"]), "gold study in the training pool"
    report["gold_excluded"] = len(gold)
    conf_cols = [f"{name}__conf" for name in LABELS]
    # A report that never mentions a finding gets a placeholder score with this confidence.
    silent = labels_df[conf_cols].to_numpy(np.float32) <= SILENT_CONF
    report["silent_label_share"] = {name: round(float(share), 3) for name, share in zip(LABELS, silent.mean(0))}

    pool = labels_df.merge(train_df[["StudyInstanceUID", "Report"]], on="StudyInstanceUID", how="inner")
    pool["Report"] = pool["Report"].fillna("")
    folds = make_folds(pool["StudyInstanceUID"].tolist(), pool["Report"].tolist(), cfg["n_folds"], cfg["fold_seed"])
    pool["fold"] = pool["StudyInstanceUID"].map(folds)
    pool[["StudyInstanceUID", "fold"]].to_csv(os.path.join(cfg["out_dir"], "folds.csv"), index=False)

    train_all = pool[pool["fold"] != cfg["fold"]]
    val_all = pool[pool["fold"] == cfg["fold"]]
    train_part = train_all.sample(cfg["limit_train"], random_state=cfg["fold_seed"]) if cfg["limit_train"] else train_all
    val_part = val_all.sample(cfg["limit_val"], random_state=cfg["fold_seed"]) if cfg["limit_val"] else val_all
    report["studies"] = {"pool": len(pool), "full_train": len(train_all), "full_val": len(val_all),
                         "used_train": len(train_part), "used_val": len(val_part)}
    say(f"studies: {report['studies']}")
    save_report()

    # Preprocess only the series this run needs.
    used = set(train_part["StudyInstanceUID"]) | set(val_part["StudyInstanceUID"])
    used_series = series_df[series_df["StudyInstanceUID"].isin(used)]
    jobs = [(f"{comp}/train_series/{a}/{b}", b, p, cfg["cache_dir"]) for a, b, p in
            zip(used_series.StudyInstanceUID, used_series.SeriesInstanceUID, used_series.Anatomical_Plane)]
    start = time.time()
    shared_meta = os.path.join(cfg["cache_dir"], "cache_meta.csv")
    reused_cache = os.path.exists(shared_meta)  # An earlier model in this session already preprocessed the data.
    if reused_cache:
        meta = pd.read_csv(shared_meta, dtype={"series": str})
        text_cols = ["scanner", "tag", "error"]
        meta[text_cols] = meta[text_cols].astype(object).where(meta[text_cols].notna(), "")
    else:
        with Pool(cfg["preprocess_workers"]) as workers:
            rows = workers.map(_cache_job, jobs, chunksize=4)
        meta = pd.DataFrame(rows)
        meta.to_csv(shared_meta, index=False)
    seconds = time.time() - start
    if "slice_mm" not in meta:
        meta["slice_mm"] = np.nan
    meta.to_csv(os.path.join(cfg["out_dir"], "cache_meta.csv"), index=False)
    spacing = meta["slice_mm"].dropna()
    report["slice_mm_quantiles"] = {q: round(float(spacing.quantile(q)), 2) for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if len(spacing) else {}

    sample = meta[meta["n"] > 0]["series"].head(50).tolist()
    raw = packed = 0
    for series in sample:
        data = np.load(f"{cfg['cache_dir']}/{series}.npy").tobytes()
        raw += len(data)
        packed += len(zlib.compress(data, 6))
    ratio = packed / raw if raw else float("nan")
    per_series = meta["bytes"].sum() / max(len(meta), 1)
    report["preprocess"] = {
        "reused_cache": reused_cache,
        "series": len(meta),
        "failed_series": int((meta["n"] == 0).sum()),
        "unreadable_slices": int(meta["n_bad"].sum()),
        "first_errors": meta.loc[meta["error"].ne(""), "error"].head(5).tolist(),
        "seconds": round(seconds, 1),
        "mean_slices": round(float(meta["n"].mean()), 1),
        "full_hours_estimate": round(seconds / max(len(meta), 1) * FULL_SERIES / 3600, 2),
        "full_raw_gb_estimate": round(per_series * FULL_SERIES / 2**30, 1),
        "zlib_ratio": round(ratio, 3),
        "full_zlib_gb_estimate": round(per_series * ratio * FULL_SERIES / 2**30, 1),
    }
    say(f"preprocess: {report['preprocess']}")
    report["preprocess"]["cache_disk_free_gb"] = round(shutil.disk_usage(cfg["cache_dir"]).free / 2**30, 1)
    save_report()
    # A full disk shows up as failed series, which would otherwise silently shrink the training data.
    assert report["preprocess"]["failed_series"] <= 0.01 * len(meta), f"too many failed series: {report['preprocess']['first_errors']}"

    n_slices = dict(zip(meta["series"], meta["n"]))
    step_of = {series: slice_step(mm, cfg["slice_gap_mm"]) for series, mm in zip(meta["series"], meta["slice_mm"])}
    table = build_study_table(zip(used_series.StudyInstanceUID, used_series.SeriesInstanceUID,
                                  used_series.Fat_Suppression, used_series.Anatomical_Plane), n_slices)
    by_study = used_series[["StudyInstanceUID", "SeriesInstanceUID"]].merge(
        meta, left_on="SeriesInstanceUID", right_on="series", how="inner").groupby("StudyInstanceUID")
    scanner_of = {study: (group["scanner"].iloc[0] or "unknown") for study, group in by_study}
    sides = {study: resolve_side(group["tag"].fillna("").tolist(), group["centre_x"].tolist()) for study, group in by_study}
    side_of = {study: value[0] for study, value in sides.items()}
    report["sides"] = {"right": sum(v[0] == "R" for v in sides.values()), "left": sum(v[0] == "L" for v in sides.values()),
                       "from_tag": sum(v[1] == "tag" for v in sides.values()),
                       "from_coordinates": sum(v[1] == "coordinates" for v in sides.values()),
                       "unresolved": sum(v[1] == "unresolved" for v in sides.values())}
    say(f"sides: {report['sides']}")
    save_report()

    class StudyDataset(torch.utils.data.Dataset):
        """One item is one study: uint8 windows [slots, k, 3, H, W], slot mask, positions, target."""

        def __init__(self, frame, k, train):
            self.studies = frame["StudyInstanceUID"].tolist()
            self.targets = frame[LABELS].to_numpy(np.float32)
            # The label authors' weighting, so silent labels count for little in training.
            self.weights = 0.25 + 0.75 * frame[conf_cols].to_numpy(np.float32)
            self.k, self.train = k, train

        def __len__(self):
            return len(self.studies)

        def __getitem__(self, i):
            slots = table.get(self.studies[i], [[] for _ in range(N_SLOTS)])
            x = np.zeros((N_SLOTS, self.k, chans, CACHE_SIZE, CACHE_SIZE), np.uint8)
            mask = np.zeros(N_SLOTS, bool)
            pos = np.zeros((N_SLOTS, self.k), np.float32)
            for s, candidates in enumerate(slots):
                if not candidates:
                    continue
                series = candidates[np.random.randint(len(candidates))] if self.train else candidates[0]
                vol = np.load(f"{cfg['cache_dir']}/{series}.npy", mmap_mode="r")
                vol = to_common_side(vol, PLANES[s % 3], side_of.get(self.studies[i], ""))
                n = len(vol)
                centres = window_centres(n, self.k, self.train)
                # 2.5D: neighbouring slices as channels. 3D: one slice repeated, because depth is the clip axis.
                step = step_of.get(series, 1)
                x[s] = np.stack([vol[[j, j, j] if is_3d else window_slices(j, n, chans, step)] for j in centres])
                mask[s], pos[s] = True, centres / max(n - 1, 1)
            return (torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos),
                    torch.from_numpy(self.targets[i]), torch.from_numpy(self.weights[i]))

    loader_args = {"num_workers": cfg["workers"], "pin_memory": use_amp, "persistent_workers": cfg["workers"] > 0}
    train_loader = torch.utils.data.DataLoader(StudyDataset(train_part, cfg["k_train"], True), batch_size=cfg["batch"],
                                               shuffle=True, drop_last=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(StudyDataset(val_part, cfg["k_eval"], False), batch_size=cfg["batch"],
                                             shuffle=False, **loader_args)

    model = KneeNet()
    if cfg["init_from"]:
        source = cfg["init_from"] if os.path.exists(cfg["init_from"]) else find_input(cfg["init_from"])
        state = torch.load(source, map_location="cpu", weights_only=False)
        for key in ("head", "res", "slices", "slice_gap_mm", "global_pool", "dim"):
            assert state["config"].get(key, DEFAULTS[key]) == cfg[key], f"{key}: saved {state['config'].get(key)} vs {cfg[key]}"
        model.load_state_dict(state["model"])
        say(f"started from {source}")
    if cfg["qat"]:
        prepare_qat(model.enc)
    model = model.to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    if cfg["layer_decay"]:
        from timm.optim import param_groups_layer_decay

        # timm also exempts norms and biases from weight decay here, as layer-decay recipes usually do.
        enc_groups = param_groups_layer_decay(model.enc, weight_decay=cfg["wd"], layer_decay=cfg["layer_decay"])
        for group in enc_groups:
            group["lr"] = cfg["lr"] * group.pop("lr_scale", 1.0)
    else:
        enc_groups = [{"params": list(model.enc.parameters()), "lr": cfg["lr"]}]
    optimizer = torch.optim.AdamW(enc_groups + [{"params": head_params, "lr": cfg["lr_head"]}], weight_decay=cfg["wd"])
    updates_per_epoch = max(1, len(train_loader) // cfg["accum"])
    total_updates = cfg["epochs"] * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda u: min(1.0, (u + 1) / updates_per_epoch) * 0.5 * (1 + math.cos(math.pi * min(1.0, u / total_updates))))
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    @torch.no_grad()
    def predict(net, loader=None, amp=use_amp):
        net.eval()
        out = []
        for x, mask, pos, _, _ in loader or val_loader:
            x, mask, pos = x.to(device, non_blocking=True), mask.to(device), pos.to(device)
            with torch.amp.autocast(device, enabled=amp):
                out.append(torch.sigmoid(net(x, mask, pos, False).float()).cpu())
        return torch.cat(out).numpy()

    strata = val_part["StudyInstanceUID"].map(scanner_of).fillna("unknown").to_numpy()
    val_y = val_part[LABELS].to_numpy(np.float32)
    # Silence is not evidence, so silent labels are left out of scoring.
    val_y[val_part[conf_cols].to_numpy(np.float32) <= SILENT_CONF] = np.nan
    def score_windows(net):
        """Score the final model with fewer windows, as a deployment would. Same model, no retraining."""
        report["eval_windows"] = {}
        for k in cfg["eval_windows"]:
            loader = torch.utils.data.DataLoader(StudyDataset(val_part, k, False), batch_size=cfg["batch"], shuffle=False,
                                                 num_workers=cfg["workers"], pin_memory=use_amp)
            p = predict(net, loader)
            macro_k, each_k = within_scanner_concordance(val_y, p, strata)
            report["eval_windows"][str(k * N_SLOTS)] = {
                "within_scanner_c": round(macro_k, 4), "auc_thresholded": round(macro_auc_thresholded(val_y, p)[0], 4),
                "per_finding_c": {name: round(value, 4) for name, value in zip(LABELS, each_k)}}
            pd.DataFrame(p, columns=LABELS, index=val_part["StudyInstanceUID"]).to_csv(
                os.path.join(cfg["out_dir"], f"val_predictions_w{k * N_SLOTS}.csv"))
            say(f"windows {k * N_SLOTS}: {report['eval_windows'][str(k * N_SLOTS)]['within_scanner_c']}")
            save_report()

    def window_views(x, mask):
        """The encoder's input for a batch of studies, exactly as KneeNet.forward builds it."""
        k = x.shape[2]
        flat = x.view(-1, k, *x.shape[3:])[mask.view(-1)].flatten(0, 1)
        return make_views(flat.to(device), False, k)

    def int8_encoder(net, method):
        import onnxruntime as ort
        from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                              quantize_static)
        from onnxruntime.quantization.calibrate import create_calibrator, save_tensors_data
        from onnxruntime.quantization.shape_inference import quant_pre_process

        work = os.path.join(cfg["cache_dir"], "onnx")
        os.makedirs(work, exist_ok=True)
        fp32_path, pre_path, int8_path = (os.path.join(work, f"{arch}_{name}.onnx") for name in ("fp32", "pre", method))
        enc = copy.deepcopy(net.enc).float().cpu().eval()
        torch.onnx.export(enc, torch.rand(1, chans, cfg["res"], cfg["res"]), fp32_path, opset_version=17, dynamo=False,
                          input_names=["x"], output_names=["feat"], dynamic_axes={"x": {0: "n"}, "feat": {0: "n"}})
        quant_pre_process(fp32_path, pre_path)
        calib_part = train_part.head(cfg["calib_studies"])
        calib_loader = torch.utils.data.DataLoader(StudyDataset(calib_part, cfg["k_eval"], False), batch_size=1,
                                                   shuffle=False, num_workers=cfg["workers"])

        def batches(size=12):
            """Calibration windows in batches of one size across studies, which the percentile calibrator needs."""
            pending = []
            for x, mask, _, _, _ in calib_loader:
                pending += list(window_views(x, mask).cpu().numpy())
                while len(pending) >= size:
                    yield np.stack(pending[:size])
                    pending = pending[size:]
            if pending:  # Fewer windows than one batch in all: use what there is.
                yield np.stack(pending)

        class Reader(CalibrationDataReader):
            def __init__(self, items):
                self.items = iter(items)

            def get_next(self):
                batch = next(self.items, None)
                return None if batch is None else {"x": batch}

        start = time.time()
        # "_core" converts only Conv and MatMul. By default ONNX Runtime also rounds around LayerNormalization, which still
        # computes in floating point, so that rounding costs accuracy and buys no speed.
        core = method.endswith("_core")
        base_method = method[:-len("_core")] if core else method
        calibrate_method = CalibrationMethod.MinMax if base_method == "int8_minmax" else CalibrationMethod.Percentile
        # ONNX Runtime's calibrators keep every intermediate tensor of a collection call in memory (8 batches of Tiny
        # ran a 30 GB session out of memory), so ranges are collected one batch at a time; MinMax merges ranges and
        # Percentile merges histograms across calls.
        # The ranges do not depend on which layers are then converted, so the two conversions share one measurement.
        cache_path = os.path.join(work, f"{cfg['backbone']}_{base_method}_ranges.json")
        n_batches = 0
        if not os.path.exists(cache_path):
            calibrator = create_calibrator(pre_path, None, augmented_model_path=os.path.join(work, "augmented.onnx"),
                                           calibrate_method=calibrate_method,
                                           extra_options={"percentile": 99.999} if base_method == "int8_percentile" else {})
            for batch in batches():
                calibrator.collect_data(Reader([batch]))
                n_batches += 1
            save_tensors_data(calibrator.compute_data(), cache_path)
        # U8 activations, S8 weights, per channel. Weights use 7 bits (reduce_range) because this laptop's CPU has no
        # VNNI: on such CPUs 8-bit products can saturate, which would make results depend on the CPU they run on.
        quantize_static(pre_path, int8_path, None, quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=True,
                        op_types_to_quantize=["Conv", "MatMul"] if core else None,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=calibrate_method,
                        calibration_cache_path=cache_path)
        if cfg["int8_save"]:
            shutil.copy(int8_path, os.path.join(cfg["out_dir"], f"encoder_{method}.onnx"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = os.cpu_count() or 4
        sessions = {name: ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
                    for name, path in (("fp32", fp32_path), ("int8", int8_path))}

        class OrtEncoder(nn.Module):
            def __init__(self, session):
                super().__init__()
                self.session = session

            def forward(self, views):
                return torch.from_numpy(self.session.run(None, {"x": views.float().cpu().numpy()})[0]).to(views.device)

        # How far INT8 features move from the FP32 export, on calibration windows (no validation data).
        x, mask, _, _, _ = next(iter(calib_loader))
        probe = window_views(x, mask)[:12].cpu().numpy()
        ref, got = (sessions[name].run(None, {"x": probe})[0] for name in ("fp32", "int8"))
        info = {"quantize_s": round(time.time() - start, 1), "calib_studies": len(calib_part), "calib_batches": n_batches,
                "fp32_mb": round(os.path.getsize(fp32_path) / 2**20, 2), "int8_mb": round(os.path.getsize(int8_path) / 2**20, 2),
                "feature_rel_err": round(float(np.linalg.norm(got - ref) / np.linalg.norm(ref)), 5)}
        return OrtEncoder(sessions["int8"]), info

    def score_precisions(net):
        """Stage 4: the same model with its encoder at each precision, at the evaluation window count."""
        assert not is_3d, "Stage 4 covers the 2.5D models"
        report["precisions"] = {}
        original = net.enc
        for precision in cfg["precisions"]:
            start, info = time.time(), {}
            if precision == "fp32":
                encoder = original
            elif precision == "fp16":
                class Half(nn.Module):
                    def __init__(self, enc):
                        super().__init__()
                        self.enc = copy.deepcopy(enc).half()

                    def forward(self, views):
                        return self.enc(views.half()).float()

                encoder = Half(original)
            elif precision in ("int8_minmax", "int8_percentile", "int8_minmax_core", "int8_percentile_core"):
                encoder, info = int8_encoder(net, precision)
                if not cfg["int8_score"]:
                    report["precisions"][precision] = {**info, "scored": False}
                    say(f"{precision}: built, not scored: {info}")
                    save_report()
                    continue
            else:
                raise ValueError(f"unknown precision {precision}")
            net.enc = encoder
            try:
                p = predict(net, amp=False)  # Mixed precision off, so each row is the precision it names.
            finally:
                net.enc = original
            macro_p, each_p = within_scanner_concordance(val_y, p, strata)
            report["precisions"][precision] = {
                **info, "device": "cpu" if precision.startswith("int8") else device, "score_s": round(time.time() - start, 1),
                "within_scanner_c": round(macro_p, 4), "auc_thresholded": round(macro_auc_thresholded(val_y, p)[0], 4),
                "per_finding_c": {name: round(value, 4) for name, value in zip(LABELS, each_p)}}
            pd.DataFrame(p, columns=LABELS, index=val_part["StudyInstanceUID"]).to_csv(
                os.path.join(cfg["out_dir"], f"val_predictions_{precision}.csv"))
            say(f"{precision}: {report['precisions'][precision]}")
            save_report()

    if cfg["eval_from"]:
        source = cfg["eval_from"] if os.path.exists(cfg["eval_from"]) else find_input(cfg["eval_from"])
        state = torch.load(source, map_location=device, weights_only=False)
        saved = state["config"]
        for key in ("head", "res", "slices", "slice_gap_mm", "global_pool", "dim"):  # The weights must fit this model.
            assert saved.get(key, DEFAULTS[key]) == cfg[key], f"{key}: saved {saved.get(key)} vs {cfg[key]}"
        ema.load_state_dict(state["model"])
        if cfg["precisions"]:
            score_precisions(ema)
            report["status"] = "evaluated"
            save_report()
            return report
        preds = predict(ema)
        macro, each = within_scanner_concordance(val_y, preds, strata)
        report["evaluated"] = {"within_scanner_c": round(macro, 4),
                               "per_finding_c": {name: round(value, 4) for name, value in zip(LABELS, each)}}
        say(f"evaluated {cfg['eval_from']}: {report['evaluated']['within_scanner_c']}")
        pd.DataFrame(preds, columns=LABELS, index=val_part["StudyInstanceUID"]).to_csv(os.path.join(cfg["out_dir"], "val_predictions.csv"))
        score_windows(ema)
        report["status"] = "evaluated"
        save_report()
        return report

    report["epochs"], updates, first_epoch = [], 0, 0
    if cfg["resume_from"]:
        state = torch.load(cfg["resume_from"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        report["epochs"], updates, first_epoch = state["epochs"], state["updates"], len(state["epochs"])
        # New random draws, so the resumed epochs do not repeat the first session's augmentation.
        np.random.seed(cfg["seed"] + first_epoch)
        torch.manual_seed(cfg["seed"] + first_epoch)
    for epoch in range(first_epoch, cfg["epochs"]):
        model.train()
        start, total, steps = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for x, mask, pos, y, weight in train_loader:
            x, mask, pos, y = x.to(device, non_blocking=True), mask.to(device), pos.to(device), y.to(device)
            known = ~torch.isnan(y)  # Missing labels are masked, not guessed.
            with torch.amp.autocast(device, enabled=use_amp):
                logits = model(x, mask, pos, True)
            loss = F.binary_cross_entropy_with_logits(logits.float(), torch.nan_to_num(y), reduction="none")
            loss = (loss * known * weight.to(device)).sum() / known.sum().clamp_min(1)
            scaler.scale(loss / cfg["accum"]).backward()
            steps += 1
            total += loss.item()
            if steps % cfg["accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                updates += 1
                decay = min(cfg["ema"], (1 + updates) / (10 + updates))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
        train_seconds = time.time() - start

        start = time.time()
        preds = predict(ema)
        val_seconds = time.time() - start
        macro, each = within_scanner_concordance(val_y, preds, strata)
        auc, _ = macro_auc_thresholded(val_y, preds)
        row = {"epoch": epoch, "loss": round(total / max(steps, 1), 4), "train_s": round(train_seconds, 1),
               "val_s": round(val_seconds, 1), "within_scanner_c": round(macro, 4), "auc_thresholded": round(auc, 4),
               "per_finding_c": {name: round(value, 4) for name, value in zip(LABELS, each)}}
        report["epochs"].append(row)
        say(json.dumps({k: v for k, v in row.items() if k != "per_finding_c"}))
        save_report()
        if cfg["max_hours"]:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epochs": report["epochs"], "updates": updates}, os.path.join(cfg["out_dir"], "checkpoint.pt"))
            spent = (time.time() - SESSION_START) / 3600
            next_epoch = 1.2 * (train_seconds + val_seconds) / 3600
            if epoch + 1 < cfg["epochs"] and spent + next_epoch > cfg["max_hours"]:
                report["status"] = "paused"
                save_report()
                say(f"paused after epoch {epoch} at {spent:.2f} h; resume from checkpoint.pt")
                return report

    # The first epoch includes start-up costs, so cost estimates prefer the later ones.
    later = report["epochs"][1:] or report["epochs"]
    train_per_study = float(np.median([r["train_s"] for r in later])) / max(len(train_part), 1)
    val_per_study = float(np.median([r["val_s"] for r in later])) / max(len(val_part), 1)
    epoch_hours = (train_per_study * len(train_all) + val_per_study * len(val_all)) / 3600
    report["cost_estimate"] = {
        "train_s_per_study": round(train_per_study, 3),
        "val_s_per_study": round(val_per_study, 3),
        "full_epoch_hours": round(epoch_hours, 3),
        "planned_epochs": cfg["planned_epochs"],
        "full_run_hours": round(epoch_hours * cfg["planned_epochs"], 2),
        "strata_in_val": int(len(np.unique(strata))),
    }
    if device == "cuda":
        report["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    pd.DataFrame(preds, columns=LABELS, index=val_part["StudyInstanceUID"]).to_csv(os.path.join(cfg["out_dir"], "val_predictions.csv"))
    if cfg["qat"]:  # val_predictions.csv above is the simulation; the saved weights are plain, for the real conversion.
        strip_qat(ema.enc)
        plain = predict(ema)
        macro_plain, each_plain = within_scanner_concordance(val_y, plain, strata)
        report["qat_plain_fp32"] = {"within_scanner_c": round(macro_plain, 4),
                                    "per_finding_c": {name: round(value, 4) for name, value in zip(LABELS, each_plain)}}
        say(f"QAT weights without the simulation: {macro_plain:.4f}")
        pd.DataFrame(plain, columns=LABELS, index=val_part["StudyInstanceUID"]).to_csv(
            os.path.join(cfg["out_dir"], "val_predictions_qat_plain.csv"))
    torch.save({"model": ema.state_dict(), "config": cfg}, os.path.join(cfg["out_dir"], "model_ema.pt"))
    if not cfg["qat"]:
        score_windows(ema)
    report["status"] = "finished"
    save_report()
    say(json.dumps({k: report.get(k) for k in ["cost_estimate", "gpu_peak_gb", "gpu", "status"]}))
    return report


def probe(cfg):
    """Measure GPU memory and step time for each encoder, keeping going if one fails."""
    import torch

    os.makedirs(cfg["out_dir"], exist_ok=True)
    results = []
    for backbone in cfg["probe_backbones"]:
        one = {**cfg, "backbone": backbone, "preflight_only": True, "install_decoders": False}
        try:
            row = {"fits": True, **run(one)["preflight"]}
        except Exception as error:  # An out-of-memory or a load failure must not stop the other models.
            row = {"backbone": backbone, "fits": False, "error": f"{type(error).__name__}: {str(error)[:300]}"}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results.append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(cfg["out_dir"], "gpu_probe.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    return results


def train_many(cfg):
    """Train several encoders in one session. One failing must not lose the others."""
    import torch

    results = {}
    for backbone in cfg["backbones"]:
        if "paused" in results.values():  # Session time is nearly used up; a new model would be cut off mid-epoch.
            results[backbone] = "not started: session time"
            continue
        one = {**cfg, "backbone": backbone, "out_dir": os.path.join(cfg["out_dir"], backbone)}
        try:
            results[backbone] = run(one)["status"]
        except Exception as error:
            results[backbone] = f"failed: {type(error).__name__}: {str(error)[:300]}"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({"backbone": backbone, "result": results[backbone]}), flush=True)
    with open(os.path.join(cfg["out_dir"], "session_summary.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    return results


def main():
    cfg = dict(DEFAULTS)
    path = os.environ.get("KNEE_CONFIG")
    if path:
        with open(path) as handle:
            cfg.update(json.load(handle))
    if cfg["probe_backbones"]:
        probe(cfg)
    elif cfg["backbones"]:
        train_many(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
