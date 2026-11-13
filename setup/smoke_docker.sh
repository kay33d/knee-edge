#!/usr/bin/env bash
# Confirms the image runs and that both hardware profiles are actually enforced.
set -euo pipefail
cd "/mnt/d/claude files/knee-edge"

docker images knee-edge:0.1 --format '{{.Repository}}:{{.Tag}} {{.Size}}'
echo "--- pi5-class"
docker run --rm --cpuset-cpus=0-3 --memory=3g --memory-swap=3g -v "$PWD/results:/results" \
  knee-edge:0.1 --smoke --threads 4 --studies 5 --warmup 2 --out /results/docker_smoke_pi5-class.json
echo "--- orin-class-cpu"
docker run --rm --cpuset-cpus=0-5 --memory=3g --memory-swap=3g -v "$PWD/results:/results" \
  knee-edge:0.1 --smoke --threads 6 --studies 5 --warmup 2 --out /results/docker_smoke_orin-class-cpu.json
echo "--- WSL"
nproc
free -h | head -3
df -h / | tail -1
grep -m1 'model name' /proc/cpuinfo
