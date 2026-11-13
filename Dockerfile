# CPU-only inference benchmark image. Contains no competition data or weights:
# mount models/ and results/ at run time (the MIRA agreement bans redistribution).
FROM python:3.12.7-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmarks/ benchmarks/

ENTRYPOINT ["python", "benchmarks/bench_onnx.py"]
