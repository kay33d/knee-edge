#!/usr/bin/env bash
# Runs as root inside Ubuntu. Uses Ubuntu's own docker.io package rather than Docker Desktop to save RAM.
set -euo pipefail

grep -q '^systemd=true' /etc/wsl.conf 2>/dev/null || printf '[boot]\nsystemd=true\n' >> /etc/wsl.conf

apt-get update -y
apt-get install -y docker.io
systemctl enable docker || true
