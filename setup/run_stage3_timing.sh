#!/usr/bin/env bash
# Stage 3 CPU timing of trained models (encoder over all windows + study head), under both Stage 1 profiles.
set -euo pipefail
cd "/mnt/d/claude files/knee-edge"
docker build -q -f Dockerfile.export -t knee-edge-export:0.1 .
R=/work/results
S3A="$R/stage3_atto_fold0"; S3T="$R/stage3_tiny_a_fold0"
E="$R/stage4_export_fold0"
C="$R/stage4_core_a_fold0"
MODELS=(
  "atto=$R/small_fold0/convnext_atto/model_ema.pt@72,36,18"
  "atto_tf=$S3A/convnext_atto@tf/model_ema.pt@72,36"
  "atto_r192=$S3A/convnext_atto@r192/model_ema.pt@72,36"
  "atto_mean=$S3A/convnext_atto@mean/model_ema.pt@72"
  "tiny=$R/session2_fold0/convnext_tiny/model_ema.pt@72,36,18"
  "tiny_tf=$S3T/convnext_tiny@tf/model_ema.pt@72,36"
  "tiny_r192=$S3T/convnext_tiny@r192/model_ema.pt@72,36"
  "tiny_mean=$S3T/convnext_tiny@mean/model_ema.pt@72"
  "mobilenetv4=$R/small_fold0/mobilenetv4_conv_small/model_ema.pt"
  "efficientnetv2_b0=$R/small_fold0/tf_efficientnetv2_b0/model_ema.pt"
  "resnet50_tv2=$R/session3_fold0/resnet50.tv2_in1k/model_ema.pt"
  "coatnet0_256=$R/session3_fold0/coatnet_0_rw_224/model_ema.pt"
  "dinov2_tuned=$R/dinov2_tuning_fold0/vit_small_patch14_dinov2@llrd/model_ema.pt"
  "r3d18_192=$R/r3d18_fold0/model_ema.pt"
  "atto_int8_core_pct=$R/small_fold0/convnext_atto/model_ema.pt@72,36,18~$C/convnext_atto@eval/encoder_int8_percentile_core.onnx"
  "tiny_int8_core_pct=$R/session2_fold0/convnext_tiny/model_ema.pt@72,36,18~$C/convnext_tiny@eval/encoder_int8_percentile_core.onnx"
  "tiny_tf_int8_core_pct=$S3T/convnext_tiny@tf/model_ema.pt@72,36~$R/stage4_core_b_fold0/convnext_tiny@tf_eval/encoder_int8_percentile_core.onnx"
  "atto_int8_minmax=$R/small_fold0/convnext_atto/model_ema.pt@72,36,18~$E/convnext_atto@eval/encoder_int8_minmax.onnx"
  "atto_int8_pct=$R/small_fold0/convnext_atto/model_ema.pt@72,36,18~$E/convnext_atto@eval/encoder_int8_percentile.onnx"
  "tiny_int8_minmax=$R/session2_fold0/convnext_tiny/model_ema.pt@72,36,18~$E/convnext_tiny@eval/encoder_int8_minmax.onnx"
  "tiny_int8_pct=$R/session2_fold0/convnext_tiny/model_ema.pt@72,36,18~$E/convnext_tiny@eval/encoder_int8_percentile.onnx"
  "tiny_tf_int8_pct=$S3T/convnext_tiny@tf/model_ema.pt@72,36~$E/convnext_tiny@tf_eval/encoder_int8_percentile.onnx"
)
for profile in "pi5-class 0-3 4" "orin-class-cpu 0-5 6"; do
  set -- $profile
  echo "--- $1"
  docker run --rm --cpuset-cpus=$2 --memory=3g --memory-swap=3g -v "$PWD:/work" --entrypoint python \
    knee-edge-export:0.1 /work/benchmarks/time_trained.py --threads $3 --repeats 5 --out "$R/stage3_timing_$1.json" "${MODELS[@]}"
done
