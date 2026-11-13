"""Feasibility screen: export untrained candidates to ONNX and time them on this CPU.

CPU latency depends on the architecture, not the trained weight values, so this can rule
models out before any GPU time is spent. Each timing runs bench_onnx.py in its own process,
so peak RAM excludes PyTorch. Results cover the encoder only, without DICOM preprocessing
or the study-level head.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import timm
import torch
import torchvision
from fvcore.nn import FlopCountAnalysis

# (kind, name, passes per study, input shape). 2.5D models see 72 three-slice windows.
# The 3D model sees each of the 6 slots once, as 32 slices.
CANDIDATES = [
    ("cnn-mobile", "mobilenetv4_conv_small", 72, (1, 3, 256, 256)),
    ("cnn-mobile", "tf_efficientnetv2_b0", 72, (1, 3, 256, 256)),
    ("cnn-classic", "resnet18", 72, (1, 3, 256, 256)),
    ("cnn-classic", "resnet34", 72, (1, 3, 256, 256)),
    ("cnn-classic", "resnet50", 72, (1, 3, 256, 256)),
    ("cnn-modern", "convnext_atto", 72, (1, 3, 256, 256)),
    ("cnn-modern", "convnext_pico", 72, (1, 3, 256, 256)),
    ("cnn-modern", "convnext_tiny", 72, (1, 3, 256, 256)),
    ("hybrid", "fastvit_t8", 72, (1, 3, 256, 256)),
    ("hybrid", "mobilevitv2_100", 72, (1, 3, 256, 256)),
    ("hybrid", "tiny_vit_5m_224", 72, (1, 3, 224, 224)),
    ("hybrid", "coatnet_0_rw_224", 72, (1, 3, 224, 224)),
    ("vit", "deit_tiny_patch16_224", 72, (1, 3, 224, 224)),
    ("vit", "vit_small_patch14_dinov2", 72, (1, 3, 252, 252)),
    ("3d-cnn", "r3d_18", 6, (1, 3, 32, 256, 256)),
]


def build(name, shape):
    if name == "r3d_18":
        model = torchvision.models.video.r3d_18(weights=None)
        model.fc = torch.nn.Identity()
        return model.eval()
    kwargs = {"pretrained": False, "num_classes": 0}
    if name == "vit_small_patch14_dinov2":
        kwargs["img_size"] = shape[-1]
    return timm.create_model(name, **kwargs).eval()


def fill_zero_init(model):
    """Replace all-zero parameters, which the exporter would merge and a trained model would not have."""
    filled = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.any():
                parameter.normal_(0.0, 0.01)
                filled += 1
    return filled


def count_gmacs(model, x):
    try:
        counter = FlopCountAnalysis(model, x).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        return counter.total() / 1e9
    except Exception:  # Some models cannot be traced by fvcore; the timing still stands.
        return None


def export(model, x, path):
    """Prefer the TorchScript exporter and fall back to the newer one, recording which worked."""
    with torch.no_grad():
        try:
            torch.onnx.export(model, x, path, opset_version=17, dynamo=False, input_names=["x"], output_names=["y"])
            return "torchscript"
        except Exception:
            torch.onnx.export(model, x, path, opset_version=18, dynamo=True, external_data=False)
            return "dynamo"


def bench(path, passes, threads, studies):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_onnx.py")
    command = [sys.executable, script, "--model", path, "--windows", str(passes),
               "--studies", str(studies), "--warmup", "2", "--threads", str(threads)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        # A negative code means a signal, usually the out-of-memory killer under the cap.
        raise RuntimeError(f"bench exit {done.returncode}: {done.stderr[-200:]}")
    return json.loads(done.stdout)


def screen(kind, name, passes, shape, threads, studies, out_dir):
    model = build(name, shape)
    zero_params_filled = fill_zero_init(model)
    x = torch.rand(*shape)
    gmacs = count_gmacs(model, x)
    path = os.path.join(out_dir, f"{name}.onnx")
    exporter = export(model, x, path)
    params = sum(p.numel() for p in model.parameters())
    del model
    result = bench(path, passes, threads, studies)
    os.remove(path)
    return {
        "kind": kind,
        "model": name,
        "params_m": round(params / 1e6, 2),
        "fp32_mb": round(result["model_mb"], 1),
        "gmacs_per_study": None if gmacs is None else round(gmacs * passes, 1),
        "passes_per_study": passes,
        "input": list(shape),
        "exporter": exporter,
        "zero_params_filled": zero_params_filled,
        "median_s_per_study": result["median_s_per_study"],
        "first_s": result["first_s"],
        "last_s": result["last_s"],
        "peak_rss_mb": result["peak_rss_mb"],
        "threads": threads,
        "enforced_limits": result["enforced_limits"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--studies", type=int, default=3)
    parser.add_argument("--only", nargs="*", help="Limit the screen to these model names")
    args = parser.parse_args()

    torch.set_num_threads(2)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for kind, name, passes, shape in CANDIDATES:
            if args.only and name not in args.only:
                continue
            try:
                row = screen(kind, name, passes, shape, args.threads, args.studies, tmp)
            except Exception as error:  # A failed export is itself a finding, so keep going.
                row = {"kind": kind, "model": name, "error": f"{type(error).__name__}: {error}"[:300]}
            rows.append(row)
            print(json.dumps(row), flush=True)
            with open(args.out, "w") as handle:
                json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
