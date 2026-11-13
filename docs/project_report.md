# CPU-deployable knee MRI classification: project report

Last rebuilt 2026-11-13 by `results/build_report.py`. Tables and headline figures come from the result files. Fold 0, one training seed unless stated. Research work, not a clinical tool.

## 1. Summary

This project asks which knee MRI classifiers can run on an ordinary CPU, and what accuracy is lost by making them cheaper. It uses the RSNA Knee Abnormality Detection data, where each study is scored on 12 findings. Models are trained on Kaggle GPUs and timed on a Ryzen 5 5500U laptop in Docker, limited to 4 or 6 cores and 3 GB.

Where things stand:

- **Encoder choice.** The best hybrid or vision-transformer encoder scored 0.8382, against 0.8417 for the best convolutional encoder, so they did not beat the CNNs by more than the 0.01 tie margin (H1, fold 0; final latency comparison still needs the re-timing in section 8).
- **Fewer windows.** Going from 72 to 18 windows costs 0.056 on average for the two ConvNeXt models. The losses are largest for Medial Meniscus and Lateral Meniscus.
- **Precision.** FP16 changes nothing measurable. The cost of INT8 depends mostly on which layers are converted. ONNX Runtime's default also rounds around LayerNormalization, which costs ConvNeXt-Tiny 0.02-0.03; converting only Conv and MatMul layers removes nearly all of it. Change in concordance from FP32:
  - ConvNeXt-Atto: INT8, MinMax ranges -0.007; INT8, percentile ranges -0.001; INT8, MinMax, Conv and MatMul only -0.000; INT8, percentile, Conv and MatMul only -0.000.
  - ConvNeXt-Tiny: INT8, MinMax ranges -0.034; INT8, percentile ranges -0.021; INT8, MinMax, Conv and MatMul only -0.002; INT8, percentile, Conv and MatMul only -0.001.
  - ConvNeXt-Tiny, transformer head: INT8, MinMax ranges -0.033; INT8, percentile ranges -0.018; INT8, MinMax, Conv and MatMul only -0.004; INT8, percentile, Conv and MatMul only -0.004.
- **Transformer head.** On ConvNeXt-Tiny it scored +0.009 and +0.010 against the attention head in two training seeds, on the edge of the 0.01 tie margin.
- **Quantization-aware training.** It did not reduce the INT8 loss. With Conv and MatMul conversion and percentile ranges, QAT weights scored Atto -0.001 against -0.000, Tiny -0.005 against -0.001, Tiny with transformer head -0.007 against -0.004 without QAT, all against FP32. With ONNX Runtime's default conversion and MinMax ranges, QAT helped Atto slightly but made both Tiny models worse (Atto -0.003 against -0.007, Tiny -0.045 against -0.034, Tiny with transformer head -0.050 against -0.033). Calibration after the core conversion was similar. This QAT is a simple version (3 epochs, learned ranges discarded), so a fuller version could still do better.

## 2. Questions

| ID | Hypothesis | Status |
|---|---|---|
| H1 | At equal CPU latency, hybrid and ViT encoders do not beat modern CNNs by more than the noise margin | Supported on accuracy in fold 0; latency comparison pending |
| H2 | Cutting windows hurts focal findings more than global ones | Partly supported: see section 6 |
| H3 | Post-training INT8 loses more discrimination and calibration than quantization-aware training, and the gap depends on model family | Not supported here: converting only Conv and MatMul keeps post-training INT8 within the tie margin for all three models, and QAT did not reduce the loss. The conversion mattered more than the model family (section 7) |
| H4 | Label quality moves AUC more than encoder choice | Stage 5, not started |

## 3. Method

- **Input.** Each study is up to 6 series slots (3 planes, fat-suppressed or not). Each slot gives 12 windows of 3 adjacent slices at 256 px, so 72 windows per study. A per-finding attention head combines them.
- **Labels.** `pilkwang/rsna-knee-llm-labels`, soft labels read from radiology reports by an LLM. Labels where the report was silent are left out of scoring. The 58 gold studies are excluded from training and selection.
- **Primary score.** Within-scanner concordance: for each finding, the share of same-scanner study pairs ranked correctly, averaged over findings.
- **Comparisons.** Differences are paired by study on the same 870 validation studies, with 500 bootstrap draws within scanners. Differences under 0.01 count as ties.
- **Calibration.** ECE (10 bins) and Brier score against the soft labels, so they measure agreement with report-derived labels rather than ground truth.
- **INT8 conversion.** ONNX Runtime static quantization in QDQ form, with 8-bit activations and per-channel 7-bit weights, because the laptop's CPU has no VNNI instructions and 8-bit products could saturate. Ranges come from 32 training-split studies. Only the encoder is converted; the head costs 1-7 ms and stays FP32. Two conversions are compared: ONNX Runtime's default, which also places rounding around LayerNormalization even though that layer still computes in floating point, and one limited to Conv and MatMul layers.
- **Quantization-aware training.** Models are fine-tuned for 3 epochs under simulated INT8 noise. The simulation is removed and the weights go through the same conversion, so the comparison isolates the weights. Ranges learned in the simulation are not carried over, so this is a simpler form of QAT.

## 4. Stage 1: untrained encoders on CPU

Encoder only, 72 windows per study (6 clips for 3D), no study head or DICOM reading.

| Kind | Encoder | Parameters (M) | FP32 size (MB) | GMACs per study | s per study, 4 cores | s per study, 6 cores | Peak RAM, 4 cores (MB) |
|---|---|---|---|---|---|---|---|
| cnn-mobile | mobilenetv4_conv_small | 2.49 | 9.5 | 17.7 | 0.455 | 0.3948 | 112.3 |
| cnn-modern | convnext_atto | 3.37 | 12.9 | 51.9 | 1.6678 | 1.4212 | 141.5 |
| cnn-mobile | tf_efficientnetv2_b0 | 5.86 | 22.4 | 68.3 | 2.0244 | 1.5744 | 147.9 |
| hybrid | fastvit_t8 | 3.26 | 12.5 | 50.7 | 2.0874 | 1.6671 | 140.2 |
| vit | deit_tiny_patch16_224 | 5.52 | 21.2 | 77.7 | 2.5194 | 2.3668 | 123.6 |
| cnn-classic | resnet18 | 11.18 | 42.6 | 171.0 | 2.7929 | 2.2864 | 223.3 |
| hybrid | tiny_vit_5m_224 | 5.07 | 19.8 | 84.5 | 3.1442 | 2.6908 | 142.7 |
| cnn-modern | convnext_pico | 8.53 | 32.6 | 129.2 | 3.3865 | 3.1313 | 205.6 |
| hybrid | mobilevitv2_100 | 4.39 | 16.8 | 132.7 | 3.8592 | 3.3128 | 153.3 |
| cnn-classic | resnet34 | 21.28 | 81.2 | 345.2 | 5.935 | 4.405 | 299.1 |
| cnn-classic | resnet50 | 23.51 | 89.6 | 386.5 | 7.2114 | 5.2173 | 317.3 |
| hybrid | coatnet_0_rw_224 | 26.67 | 113.4 | 304.8 | 8.0176 | 6.8273 | 371.9 |
| cnn-modern | convnext_tiny | 27.82 | 106.2 | 420.3 | 10.4999 | 8.1975 | 245.4 |
| vit | vit_small_patch14_dinov2 | 21.65 | 82.6 | 503.3 | 12.5276 | 10.8401 | 248.9 |
| 3d-cnn | r3d_18 | 33.17 | 126.5 | 2554.1 | 46.4164 | 37.9657 | 1030.8 |

No candidate missed the 80 s or 1.5 GB targets.

## 5. Stage 2: encoder comparison

ResNet-50 baseline (`resnet50_fold0`): the reference for the change column. Models whose interval excludes zero differ from it beyond bootstrap noise, but separately trained models also differ by about 0.01, so one fold and one seed only narrows the list.

| Kind | Encoder | Concordance | Against ResNet-50 baseline | 95% interval |
|---|---|---|---|---|
| Modern CNN | ConvNeXt-Tiny | 0.8417 | +0.036 | +0.024 to +0.048 |
| Classic CNN | ResNet-50 (torchvision V2 weights) | 0.8391 | +0.033 | +0.022 to +0.046 |
| Hybrid | CoAtNet-0 | 0.8382 | +0.032 | +0.021 to +0.043 |
| Modern CNN | ConvNeXt-Atto | 0.8373 | +0.032 | +0.019 to +0.043 |
| ViT | DINOv2-Small, tuned | 0.8336 | +0.028 | +0.015 to +0.039 |
| Mobile CNN | EfficientNetV2-B0 | 0.8228 | +0.017 | +0.006 to +0.028 |
| Mobile CNN | MobileNetV4-Conv-Small | 0.8208 | +0.015 | +0.005 to +0.024 |
| 3D CNN | R3D-18, 192 px | 0.8095 | +0.004 | -0.012 to +0.019 |
| ViT | DINOv2-Small, shared settings | 0.7164 | -0.089 | -0.107 to -0.071 |

ConvNeXt-Atto and ConvNeXt-Tiny went forward to Stage 3.

## 6. Stage 3: head and scale

One change at a time from the Stage 2 setting. Paired against that setting.

| Change | Atto | Atto change (95% interval) | Tiny | Tiny change (95% interval) |
|---|---|---|---|---|
| Stage 2 setting | 0.8373 | reference | 0.8417 | reference |
| Mean pooling head | 0.8315 | -0.006 (-0.015 to +0.003) | 0.8394 | -0.002 (-0.012 to +0.008) |
| Transformer head | 0.8375 | +0.000 (-0.010 to +0.009) | 0.8505 | +0.009 (-0.001 to +0.019) |
| 192 px input | 0.8344 | -0.003 (-0.010 to +0.006) | 0.8380 | -0.004 (-0.013 to +0.007) |
| 5 slices per window | 0.8292 | -0.008 (-0.017 to +0.003) | 0.8386 | -0.003 (-0.014 to +0.007) |
| Slices 3.5 mm apart | 0.8382 | +0.001 (-0.005 to +0.007) | 0.8440 | +0.002 (-0.007 to +0.012) |
| 36 windows (scored, not retrained) | 0.8309 | -0.006 (-0.012 to +0.001) | 0.8286 | -0.013 (-0.020 to -0.005) |
| 18 windows (scored, not retrained) | 0.7821 | -0.055 (-0.068 to -0.043) | 0.7850 | -0.057 (-0.069 to -0.045) |

Change at 18 windows per finding, worst first:

| Finding | Atto | Tiny | Mean of both |
|---|---|---|---|
| Medial Meniscus | -0.123 | -0.137 | -0.130 |
| Lateral Meniscus | -0.081 | -0.125 | -0.103 |
| Lateral OA | -0.053 | -0.073 | -0.063 |
| Contusion | -0.061 | -0.063 | -0.062 |
| Effusion | -0.048 | -0.073 | -0.060 |
| Baker's | -0.046 | -0.066 | -0.056 |
| MCL | -0.077 | -0.029 | -0.053 |
| Synovitis | -0.051 | -0.044 | -0.047 |
| ACL | -0.042 | -0.031 | -0.037 |
| PF OA | -0.024 | -0.026 | -0.025 |
| Medial OA | -0.026 | -0.013 | -0.020 |
| Fracture | -0.031 | -0.001 | -0.016 |

Slices 3.5 mm apart changed only about 5% of series, since the median series spacing is already 3.5 mm, so that row says little.

## 7. Stage 4: precision and calibration

All rows use the same weights unless marked QAT. Differences are against that model's true FP32 score. Mixed-precision GPU scoring, used in Stages 2 and 3, gave the same scores as true FP32 to 4 decimals.

| Model | Precision | Concordance | Change from FP32 (95% interval) | ECE | Brier | ECE recalibrated | Brier recalibrated | Mean probability shift |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Atto | FP32 | 0.8373 | reference | 0.0431 | 0.0567 | 0.0236 | 0.0548 | +0.000 |
| ConvNeXt-Atto | FP16 | 0.8373 | +0.000 (-0.000 to +0.000) | 0.0431 | 0.0567 | 0.0241 | 0.0548 | -0.000 |
| ConvNeXt-Atto | INT8, MinMax ranges | 0.8304 | -0.007 (-0.013 to -0.000) | 0.0500 | 0.0609 | 0.0214 | 0.0569 | -0.011 |
| ConvNeXt-Atto | INT8, percentile ranges | 0.8359 | -0.001 (-0.006 to +0.005) | 0.0575 | 0.0600 | 0.0238 | 0.0561 | -0.029 |
| ConvNeXt-Atto | INT8, MinMax, Conv and MatMul only | 0.8372 | -0.000 (-0.005 to +0.005) | 0.0432 | 0.0584 | 0.0223 | 0.0558 | -0.004 |
| ConvNeXt-Atto | INT8, percentile, Conv and MatMul only | 0.8372 | -0.000 (-0.005 to +0.005) | 0.0449 | 0.0577 | 0.0226 | 0.0553 | -0.004 |
| ConvNeXt-Atto | QAT weights in FP32 | 0.8366 | -0.001 (-0.004 to +0.002) | 0.0449 | 0.0570 | 0.0238 | 0.0549 | +0.005 |
| ConvNeXt-Atto | QAT, then INT8 MinMax | 0.8340 | -0.003 (-0.010 to +0.003) | 0.0454 | 0.0597 | 0.0206 | 0.0565 | +0.001 |
| ConvNeXt-Atto | QAT, then INT8 percentile | 0.8352 | -0.002 (-0.007 to +0.004) | 0.0580 | 0.0604 | 0.0233 | 0.0562 | -0.029 |
| ConvNeXt-Atto | QAT, then INT8 MinMax, Conv and MatMul only | 0.8359 | -0.001 (-0.006 to +0.003) | 0.0444 | 0.0580 | 0.0239 | 0.0557 | +0.005 |
| ConvNeXt-Atto | QAT, then INT8 percentile, Conv and MatMul only | 0.8365 | -0.001 (-0.005 to +0.004) | 0.0447 | 0.0574 | 0.0216 | 0.0551 | +0.004 |
| ConvNeXt-Tiny | FP32 | 0.8417 | reference | 0.0423 | 0.0561 | 0.0227 | 0.0536 | +0.000 |
| ConvNeXt-Tiny | FP16 | 0.8418 | +0.000 (-0.000 to +0.000) | 0.0423 | 0.0562 | 0.0225 | 0.0536 | +0.000 |
| ConvNeXt-Tiny | INT8, MinMax ranges | 0.8080 | -0.034 (-0.044 to -0.021) | 0.1774 | 0.1231 | 0.0389 | 0.0660 | -0.063 |
| ConvNeXt-Tiny | INT8, percentile ranges | 0.8206 | -0.021 (-0.030 to -0.013) | 0.1590 | 0.0958 | 0.0254 | 0.0601 | +0.051 |
| ConvNeXt-Tiny | INT8, MinMax, Conv and MatMul only | 0.8398 | -0.002 (-0.007 to +0.004) | 0.0511 | 0.0581 | 0.0232 | 0.0547 | -0.014 |
| ConvNeXt-Tiny | INT8, percentile, Conv and MatMul only | 0.8408 | -0.001 (-0.005 to +0.004) | 0.0493 | 0.0576 | 0.0244 | 0.0543 | -0.010 |
| ConvNeXt-Tiny | QAT weights in FP32 | 0.8396 | -0.002 (-0.006 to +0.003) | 0.0449 | 0.0574 | 0.0207 | 0.0542 | +0.003 |
| ConvNeXt-Tiny | QAT, then INT8 MinMax | 0.7964 | -0.045 (-0.058 to -0.033) | 0.1921 | 0.1291 | 0.0386 | 0.0681 | -0.064 |
| ConvNeXt-Tiny | QAT, then INT8 percentile | 0.8177 | -0.024 (-0.032 to -0.014) | 0.1825 | 0.1091 | 0.0315 | 0.0610 | +0.094 |
| ConvNeXt-Tiny | QAT, then INT8 MinMax, Conv and MatMul only | 0.8392 | -0.003 (-0.008 to +0.003) | 0.0452 | 0.0570 | 0.0223 | 0.0543 | -0.003 |
| ConvNeXt-Tiny | QAT, then INT8 percentile, Conv and MatMul only | 0.8363 | -0.005 (-0.009 to -0.001) | 0.0455 | 0.0572 | 0.0224 | 0.0545 | -0.000 |
| ConvNeXt-Tiny, transformer head | FP32 | 0.8505 | reference | 0.0407 | 0.0543 | 0.0254 | 0.0523 | +0.000 |
| ConvNeXt-Tiny, transformer head | FP16 | 0.8505 | +0.000 (-0.000 to +0.000) | 0.0405 | 0.0543 | 0.0247 | 0.0523 | +0.000 |
| ConvNeXt-Tiny, transformer head | INT8, MinMax ranges | 0.8180 | -0.033 (-0.043 to -0.020) | 0.1369 | 0.0961 | 0.0307 | 0.0663 | -0.078 |
| ConvNeXt-Tiny, transformer head | INT8, percentile ranges | 0.8329 | -0.018 (-0.027 to -0.009) | 0.1089 | 0.0802 | 0.0260 | 0.0598 | -0.065 |
| ConvNeXt-Tiny, transformer head | INT8, MinMax, Conv and MatMul only | 0.8467 | -0.004 (-0.010 to +0.002) | 0.0459 | 0.0562 | 0.0226 | 0.0531 | -0.012 |
| ConvNeXt-Tiny, transformer head | INT8, percentile, Conv and MatMul only | 0.8462 | -0.004 (-0.011 to +0.001) | 0.0457 | 0.0556 | 0.0241 | 0.0527 | -0.010 |
| ConvNeXt-Tiny, transformer head | QAT weights in FP32 | 0.8467 | -0.004 (-0.008 to +0.001) | 0.0414 | 0.0548 | 0.0228 | 0.0528 | +0.009 |
| ConvNeXt-Tiny, transformer head | QAT, then INT8 MinMax | 0.8004 | -0.050 (-0.064 to -0.037) | 0.1493 | 0.1050 | 0.0342 | 0.0696 | -0.082 |
| ConvNeXt-Tiny, transformer head | QAT, then INT8 percentile | 0.8254 | -0.025 (-0.035 to -0.015) | 0.1058 | 0.0799 | 0.0264 | 0.0608 | -0.044 |
| ConvNeXt-Tiny, transformer head | QAT, then INT8 MinMax, Conv and MatMul only | 0.8447 | -0.006 (-0.012 to -0.000) | 0.0429 | 0.0554 | 0.0225 | 0.0529 | -0.001 |
| ConvNeXt-Tiny, transformer head | QAT, then INT8 percentile, Conv and MatMul only | 0.8435 | -0.007 (-0.013 to -0.002) | 0.0413 | 0.0555 | 0.0210 | 0.0530 | +0.002 |

Encoder file sizes:

| Model | FP32 encoder (MB) | INT8 encoder (MB) |
|---|---|---|
| ConvNeXt-Atto | 12.92 | 3.63 |
| ConvNeXt-Tiny | 106.2 | 27.21 |
| ConvNeXt-Tiny, transformer head | 106.2 | 27.21 |

Recalibrated columns fit a slope and intercept per finding, cross-fitted in 5 parts of the validation fold. "Mean probability shift" is the average change in predicted probability from FP32.

### Second seed

| Training seed | Attention head | Transformer head | Transformer minus attention (95% interval) |
|---|---|---|---|
| 0 | 0.8417 | 0.8505 | +0.009 (-0.001 to +0.019) |
| 1 | 0.8419 | 0.8521 | +0.010 (+0.000 to +0.020) |

## 8. CPU cost of trained models

Pending. A first attempt was discarded because other applications loaded the laptop: the same encoder varied by up to 2x. It will be repeated on an idle machine with 5 separate timing processes per encoder. That run also covers CoAtNet-0 at 256 px and R3D-18 at 192 px, and INT8 encoders.

## 9. Limits

- One fold and mostly one seed. Close models cannot be separated.
- Timing describes this laptop under Docker limits, not a Raspberry Pi or Jetson, because Docker can limit cores and memory but not device speed.
- Labels come from reports through an LLM, so concordance and calibration measure agreement with those labels. The 58 gold studies and external sets are for the final models.
- The 2.5D and 3D comparison mixes in resolution and pretraining differences.
- The QAT tested is simple: 3 epochs of fine-tuning, with the ranges it learns discarded before conversion. A fuller version that keeps them might close the small remaining gap.
- Study-level data, labels and trained weights are not published here. The MIRA agreement limits redistribution, and whether trained weights count as derivatives still needs asking RSNA.

## 10. Next steps

1. Time the trained models on an idle laptop, FP32 and INT8 (Conv and MatMul only).
2. Choose finalists on accuracy against measured CPU cost.
3. Stage 5: all folds for 2-3 finalists, a second label set for H4, gold-study and external checks.
