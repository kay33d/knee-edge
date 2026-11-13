"""Time an ONNX encoder over the windows of one simulated study and record peak RAM.

This covers the encoder only. DICOM preprocessing and the study-level head are not
included yet, so these numbers are not end-to-end study latency.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
import psutil
from onnx import TensorProto, helper, numpy_helper


def build_smoke_model(path, size):
    """Write a tiny Conv + global pool model so the pipeline can be tested without weights."""
    weights = numpy_helper.from_array(np.ones((8, 3, 3, 3), dtype=np.float32), "w")
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "w"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("GlobalAveragePool", ["c"], ["y"]),
        ],
        "smoke",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def peak_rss_mb():
    """Return true peak RSS on Linux and Windows; other platforms get the current RSS."""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / 2**20
    # VmHWM resets on exec. ru_maxrss does not, so a child would report its parent's peak.
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return info.rss / 2**20


def enforced_limits():
    """Read the limits actually applied to this process, so each result records them."""
    limits = {"cpus_usable": len(psutil.Process().cpu_affinity())}
    for key, path in [("cgroup_cpu_max", "/sys/fs/cgroup/cpu.max"), ("cgroup_memory_max", "/sys/fs/cgroup/memory.max")]:
        try:
            with open(path) as handle:
                limits[key] = handle.read().strip()
        except OSError:
            limits[key] = None
    return limits


def run(args):
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(args.model, options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    # Models are exported with a fixed shape, so 2D and 3D inputs need no special handling.
    # A dynamic batch axis (the INT8 files have one) is timed as a single window.
    shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
    # A fixed seed keeps the input identical across models and runs.
    window = np.random.default_rng(0).random(shape, dtype=np.float32)

    for _ in range(args.warmup):
        session.run(None, {input_name: window})

    study_seconds = []
    for _ in range(args.studies):
        start = time.perf_counter()
        for _ in range(args.windows):
            session.run(None, {input_name: window})
        study_seconds.append(time.perf_counter() - start)

    ordered = sorted(study_seconds)
    return {
        "model": os.path.basename(args.model) if args.model else None,
        "model_mb": round(os.path.getsize(args.model) / 2**20, 3),
        "input_shape": shape,
        "windows_per_study": args.windows,
        "studies": args.studies,
        "threads": args.threads,
        "median_s_per_study": round(statistics.median(ordered), 4),
        "p95_s_per_study": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        # Comparing first and last studies shows thermal slowdown over a sustained run.
        "first_s": round(study_seconds[0], 4),
        "last_s": round(study_seconds[-1], 4),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "onnxruntime": ort.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "in_container": os.path.exists("/.dockerenv"),
        "enforced_limits": enforced_limits(),
        "scope": "encoder only, single simulated window, no DICOM preprocessing",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Path to an ONNX encoder with one image input")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny generated model")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--windows", type=int, default=72)
    parser.add_argument("--studies", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--out", help="Write the JSON result to this path")
    args = parser.parse_args()

    if bool(args.model) == args.smoke:
        parser.error("pass exactly one of --model or --smoke")

    if args.smoke:
        with tempfile.TemporaryDirectory() as tmp:
            args.model = os.path.join(tmp, "smoke.onnx")
            build_smoke_model(args.model, args.size)
            result = run(args)
    else:
        result = run(args)

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
