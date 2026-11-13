#!/usr/bin/env bash
# Re-measures peak RAM after the VmHWM fix. Timing results stay in the stage1_<profile>.json files.
set -euo pipefail
cd "/mnt/d/claude files/knee-edge"

docker build -q -f Dockerfile.export -t knee-edge-export:0.2 .

echo "--- pi5-class"
docker run --rm --cpuset-cpus=0-3 --memory=3g --memory-swap=3g -v "$PWD/results:/results" \
  knee-edge-export:0.2 --threads 4 --studies 1 --out /results/stage1_memory_pi5-class.json
echo "--- orin-class-cpu"
docker run --rm --cpuset-cpus=0-5 --memory=3g --memory-swap=3g -v "$PWD/results:/results" \
  knee-edge-export:0.2 --threads 6 --studies 1 --out /results/stage1_memory_orin-class-cpu.json
