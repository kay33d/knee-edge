# knee-edge

A study of knee MRI classifiers that can run on a CPU at the edge. It compares encoders, slice-window settings, attention heads and INT8 conversion on the RSNA knee MRI data (12 findings per study), with training on Kaggle and timing in Docker under fixed CPU and memory limits. Research prototype, no clinical validation.

**Results:** [docs/project_report.md](docs/project_report.md), rebuilt from the result files by `results/build_report.py`. The plan and hypotheses are in [study_design.md](study_design.md).

## What is not here

No images, labels, study-level predictions or trained weights. The RSNA MIRA agreement bans redistributing the competition data, and trained weights are held back until it is clear whether they count as derivatives. To reproduce, accept the competition rules on Kaggle and use the data there.

## Layout

| Folder | Contents |
|---|---|
| `training/` | `train_stage2.py` (training and scoring, run as a Kaggle kernel), `push_stage2.py` (builds and pushes kernels), `test_train_stage2.py` (end-to-end test on toy data) |
| `results/` | Analysis scripts and aggregate summaries only |
| `benchmarks/` | ONNX Runtime timing of encoders and trained models |
| `setup/` | WSL and Docker setup, timing run scripts |

`push_stage2.py` reads `KAGGLE_USERNAME` and `KAGGLE_KEY` from the environment.

## Benchmark environment

The image holds code and dependencies only. Competition data and weights stay outside it, because the RSNA MIRA agreement bans redistribution.

### Setup (once)

WSL and `.wslconfig` are already in place. After restarting Windows, run this in a normal PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File "D:\claude files\knee-edge\setup\finish_docker_install.ps1"
```

This puts Ubuntu and all Docker data on `D:\wsl\ubuntu`.

### Build and test

Open Ubuntu as root with `wsl -d Ubuntu-24.04 -u root`, then:

```bash
cd "/mnt/d/claude files/knee-edge"
docker build -t knee-edge:0.1 .
docker run --rm --cpuset-cpus=0-3 --memory=3g --memory-swap=3g knee-edge:0.1 --smoke --threads 4
```

### Benchmark a model

Limits and their sources are in [hardware_profiles.md](hardware_profiles.md). Run each model under both profiles.

```bash
# pi5-class: 4 cores
docker run --rm --cpuset-cpus=0-3 --memory=3g --memory-swap=3g \
  -v "$PWD/models:/models:ro" -v "$PWD/results:/results" \
  knee-edge:0.1 --model /models/encoder.onnx --threads 4 --out /results/encoder_pi5-class.json

# orin-class-cpu: 6 cores
docker run --rm --cpuset-cpus=0-5 --memory=3g --memory-swap=3g \
  -v "$PWD/models:/models:ro" -v "$PWD/results:/results" \
  knee-edge:0.1 --model /models/encoder.onnx --threads 6 --out /results/encoder_orin-class-cpu.json
```

These profiles limit cores and memory only. Results describe this laptop under those limits, not a Pi or Jetson.

### Run conditions to keep fixed

- Laptop plugged in, same Windows power mode, other apps closed.
- The same `.wslconfig`, CPU and memory limits for every run.
- Report the difference between one native run and one container run.
