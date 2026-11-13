#!/usr/bin/env bash
# Stage 1: time every candidate under both hardware profiles. Usage: run_stage1.sh [model names...]
set -euo pipefail
cd "/mnt/d/claude files/knee-edge"

docker build -q -f Dockerfile.export -t knee-edge-export:0.1 .

only=()
suffix=""
if [ "$#" -gt 0 ]; then
  only=(--only "$@")
  suffix="_test"
fi

echo "--- pi5-class"
docker run --rm --cpuset-cpus=0-3 --memory=3g --memory-swap=3g -v "$PWD/results:/results" \
  knee-edge-export:0.1 --threads 4 --out "/results/stage1_pi5-class$suffix.json" "${only[@]}"
echo "--- orin-class-cpu"
docker run --rm --cpuset-cpus=0-5 --memory=3g --memory-swap=3g -v "$PWD/results:/results" \
  knee-edge-export:0.1 --threads 6 --out "/results/stage1_orin-class-cpu$suffix.json" "${only[@]}"
