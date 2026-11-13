"""Time trained models on CPU: the encoder over every window of a study, plus the study-level head.

Each checkpoint is rebuilt from its saved settings, loaded, and exported as two ONNX files: the
encoder (one window per call, as in Stage 1) and the head (all windows' features at once). The
encoder is timed in its own process by bench_onnx.py, so peak RAM excludes PyTorch. Both exports
are checked against PyTorch on the same input. DICOM reading and resampling are still excluded.

time_trained.py --threads 4 --out result.json label=path/to/model_ema.pt[@72,36,18][~encoder_int8.onnx] ...

With ~file the encoder is that ready-made INT8 ONNX file instead of an FP32 export of the checkpoint; the head is unchanged.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import screen_encoders as se  # noqa: E402
import train_stage2 as ts  # noqa: E402


class Head(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, feat, mask, pos):
        return self.net.head(feat, mask, pos)


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def time_one(label, path, windows, threads, studies, tmp, repeats, encoder_cache, int8_path=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved = state["config"]
    net, cfg = ts.run({**ts.DEFAULTS, **saved, "build_only": True, "pretrained": False, "install_decoders": False,
                       "out_dir": os.path.join(tmp, "report"), "cache_dir": os.path.join(tmp, "cache")})
    net.load_state_dict(state["model"])
    net.eval()
    is_3d = cfg["backbone"].split("@")[0] == "r3d_18"
    res = cfg["res"]
    shape = (1, 3, cfg["k_eval"], res, res) if is_3d else (1, cfg["slices"], res, res)
    x = torch.rand(*shape)
    enc_path = int8_path or os.path.join(tmp, "enc.onnx")
    exporter = "ready-made INT8 file" if int8_path else se.export(net.enc, x, enc_path)
    with torch.no_grad():
        enc_ref = net.enc(x).numpy()
    enc_onnx = session(enc_path, threads).run(None, {session(enc_path, threads).get_inputs()[0].name: x.numpy()})[0]
    enc_err = float(np.abs(enc_ref - enc_onnx).max())
    dim = enc_ref.shape[-1]

    # Windows run one call each, so a study costs windows x the per-call cost. Each distinct encoder is timed
    # once, in several separate processes, so background load shows up as spread rather than as a difference.
    key = (cfg["backbone"].split("@")[0], res, shape[1], int8_path)
    if key not in encoder_cache:
        passes = ts.N_SLOTS if is_3d else 72
        runs = [se.bench(enc_path, passes, threads, studies) for _ in range(repeats)]
        per_call = sorted(r["median_s_per_study"] / passes for r in runs)
        encoder_cache[key] = {"per_call_s": statistics.median(per_call), "per_call_min_s": per_call[0],
                              "per_call_max_s": per_call[-1], "peak_rss_mb": max(r["peak_rss_mb"] for r in runs),
                              "model_mb": runs[0]["model_mb"], "enforced_limits": runs[0]["enforced_limits"]}
    enc = encoder_cache[key]

    rows = []
    for total in ([ts.N_SLOTS] if is_3d else windows):
        k = 1 if is_3d else total // ts.N_SLOTS
        calls = ts.N_SLOTS if is_3d else total
        feat = torch.randn(1, ts.N_SLOTS, k, dim)
        mask = torch.ones(1, ts.N_SLOTS, dtype=torch.bool)
        pos = torch.linspace(0, 1, k).repeat(1, ts.N_SLOTS, 1)
        head_path = os.path.join(tmp, f"head{k}.onnx")
        with torch.no_grad():
            head_ref = net.head(feat, mask, pos).numpy()
            # A wrapper starts in training mode, and export restores that mode on the whole model afterwards.
            torch.onnx.export(Head(net).eval(), (feat, mask, pos), head_path, opset_version=17, dynamo=False,
                              input_names=["feat", "mask", "pos"], output_names=["logits"])
        assert not net.training
        head_sess = session(head_path, threads)
        feeds = {"feat": feat.numpy(), "mask": mask.numpy(), "pos": pos.numpy().astype(np.float32)}
        head_out = head_sess.run(None, feeds)[0]
        for _ in range(5):
            head_sess.run(None, feeds)
        times = []
        for _ in range(50):
            start = time.perf_counter()
            head_sess.run(None, feeds)
            times.append(time.perf_counter() - start)
        head_s = statistics.median(times)
        rows.append({
            "label": label, "backbone": cfg["backbone"], "head": cfg["head"], "res": res, "slices": cfg["slices"],
            "windows": total if not is_3d else f"{ts.N_SLOTS} clips of {cfg['k_eval']}", "threads": threads,
            "encoder_s": round(enc["per_call_s"] * calls, 4), "head_s": round(head_s, 4),
            "total_s": round(enc["per_call_s"] * calls + head_s, 4),
            "total_range_s": [round(enc["per_call_min_s"] * calls + head_s, 4), round(enc["per_call_max_s"] * calls + head_s, 4)],
            "repeats": repeats, "encoder_peak_rss_mb": enc["peak_rss_mb"], "encoder_mb": enc["model_mb"],
            "head_mb": round(os.path.getsize(head_path) / 2**20, 2),
            "encoder_max_abs_err": enc_err, "head_max_abs_err": float(np.abs(head_ref - head_out).max()),
            "exporter": exporter, "precision": "int8" if int8_path else "fp32",
            "enforced_limits": enc["enforced_limits"],
        })
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "enforced_limits"}), flush=True)
        os.remove(head_path)
    if not int8_path:
        os.remove(enc_path)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="label=path[@72,36,18]")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--studies", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5, help="Separate timing processes per distinct encoder")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    # The fused eval-mode transformer kernel has no ONNX form; the plain path computes the same thing.
    torch.backends.mha.set_fastpath_enabled(False)
    rows, encoder_cache = [], {}
    for item in args.models:
        label, spec = item.split("=", 1)
        spec, _, int8_path = spec.partition("~")
        path, _, counts = spec.rpartition("@")
        if not counts.replace(",", "").isdigit():  # Folder names contain "@" too; only a trailing count list is one.
            path, counts = spec, ""
        windows = [int(v) for v in counts.split(",")] if counts else [72]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                rows += time_one(label, path, windows, args.threads, args.studies, tmp, args.repeats, encoder_cache, int8_path or None)
            except Exception as error:  # One model failing must not lose the others.
                rows.append({"label": label, "error": f"{type(error).__name__}: {str(error)[:300]}"})
                print(json.dumps(rows[-1]), flush=True)
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
