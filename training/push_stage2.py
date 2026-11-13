"""Build and push one private Kaggle kernel.

push_stage2.py [--cpu] [--dataset <owner/slug>]... [--set key=json]... <backbone> | --group <name> <backbones...> | --probe <backbones...>

A Kaggle script kernel is a single file, so the model choice is written into a copy of train_stage2.py.
--cpu asks for a CPU-only session, which uses no GPU quota (Stage 4 INT8 scoring runs on CPU anyway).
"""
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
USER = os.environ["KAGGLE_USERNAME"]  # Set KAGGLE_USERNAME and KAGGLE_KEY for the account to push from.


def main():
    args, datasets, settings, cpu = sys.argv[1:], [], [], False
    while args[0] in ("--dataset", "--set", "--cpu"):
        if args[0] == "--cpu":
            cpu, args = True, args[1:]
            continue
        (datasets if args[0] == "--dataset" else settings).append(args[1])
        args = args[2:]
    probe = args[0] == "--probe"
    group = args[1] if args[0] == "--group" else None
    names = args[1:] if probe else args[2:] if group else args[:1]
    with open(f"{HERE}/train_stage2.py", encoding="utf-8") as handle:
        source = handle.read()
    for setting in settings:  # Changes one entry of DEFAULTS, for example max_hours=11.3.
        key, value = setting.split("=", 1)
        pattern = re.compile(rf'^    "{key}": [^\n]*?,(?=[ \t]*(#|$))', re.M)
        assert len(pattern.findall(source)) == 1, key
        source = pattern.sub(lambda _: f'    "{key}": {json.loads(value)!r},', source, count=1)  # Python literals: True, not true.
    if probe:
        slug, old, new = "knee-edge-gpu-probe", '"probe_backbones": [],', f'"probe_backbones": {json.dumps(names)},'
    elif group:
        slug, old, new = f"knee-edge-stage2-{group}-fold0", '"backbones": [],', f'"backbones": {json.dumps(names)},'
    else:
        slug = "knee-edge-stage2-" + names[0].replace("_", "-") + "-fold0"
        old, new = '"backbone": "resnet50",', f'"backbone": "{names[0]}",'
    assert source.count(old) == 1, old
    folder = f"{HERE}/kernels/{slug}"
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    with open(f"{folder}/train_stage2.py", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(source.replace(old, new))
    metadata = {
        "id": f"{USER}/{slug}", "title": slug.replace("-", " "), "code_file": "train_stage2.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": not cpu, "enable_tpu": False, "enable_internet": True,
        **({} if cpu else {"machine_shape": "NvidiaTeslaT4"}), "dataset_sources": datasets + ([] if probe else ["pilkwang/rsna-knee-llm-labels"]),
        "competition_sources": [] if probe else ["rsna-knee-abnormality-detection"], "kernel_sources": [],
    }
    with open(f"{folder}/kernel-metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    subprocess.run(["kaggle", "kernels", "push", "-p", folder], check=True)
    print(f"{USER}/{slug}")


if __name__ == "__main__":
    main()
