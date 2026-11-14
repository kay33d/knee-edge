# CPU-deployable knee MRI classification: project report

Last rebuilt 2026-11-14 by `results/build_report.py`. Tables and headline figures come from the result files. Fold 0, one training seed unless stated. Research work, not a clinical tool.

## 1. Summary

This project asks which knee MRI classifiers can run on an ordinary CPU, and what accuracy is lost by making them cheaper. It uses the RSNA Knee Abnormality Detection data, where each study is scored on 12 findings. Models are trained on Kaggle GPUs and timed on a Ryzen 5 5500U laptop in Docker, limited to 4 or 6 cores and 3 GB.

Where things stand:

- **Encoder choice.** The best hybrid or vision-transformer encoder scored 0.8382, against 0.8417 for the best convolutional encoder, so they did not beat the CNNs by more than the 0.01 tie margin (H1, fold 0). CoAtNet-0 takes about as long as ConvNeXt-Tiny, while ConvNeXt-Atto is within 0.01 of both at about a tenth of the time.
- **Fewer windows.** Going from 72 to 18 windows costs 0.056 on average for the two ConvNeXt models. The losses are largest for Medial Meniscus and Lateral Meniscus.
- **Precision.** FP16 changes nothing measurable. The cost of INT8 depends mostly on which layers are converted. ONNX Runtime's default also rounds around LayerNormalization, which costs ConvNeXt-Tiny 0.02-0.03; converting only Conv and MatMul layers removes nearly all of it. Change in concordance from FP32:
  - ConvNeXt-Atto: INT8, MinMax ranges -0.007; INT8, percentile ranges -0.001; INT8, MinMax, Conv and MatMul only -0.000; INT8, percentile, Conv and MatMul only -0.000.
  - ConvNeXt-Tiny: INT8, MinMax ranges -0.034; INT8, percentile ranges -0.021; INT8, MinMax, Conv and MatMul only -0.002; INT8, percentile, Conv and MatMul only -0.001.
  - ConvNeXt-Tiny, transformer head: INT8, MinMax ranges -0.033; INT8, percentile ranges -0.018; INT8, MinMax, Conv and MatMul only -0.004; INT8, percentile, Conv and MatMul only -0.004.
- **CPU cost.** On 4 cores, FP32, 72 windows: ConvNeXt-Atto 0.96 s, ConvNeXt-Tiny 9.88 s, CoAtNet-0 9.40 s, DINOv2-Small 11.56 s, R3D-18 25.21 s. Every model is far inside the 80 s and 1.5 GB targets. INT8 gained little speed on this CPU: Tiny took 7.79 s with Conv and MatMul conversion, about a fifth faster, and Atto became slower (1.65 s). The laptop's CPU lacks VNNI instructions, so 8-bit products are not much cheaper than 32-bit ones here. INT8 still cuts encoder file size by about 4x and peak memory by 20-40%.
- **Transformer head.** On ConvNeXt-Tiny it scored +0.009 and +0.010 against the attention head in two training seeds, on the edge of the 0.01 tie margin.
- **Quantization-aware training.** It did not reduce the INT8 loss. With Conv and MatMul conversion and percentile ranges, QAT weights scored Atto -0.001 against -0.000, Tiny -0.005 against -0.001, Tiny with transformer head -0.007 against -0.004 without QAT, all against FP32. With ONNX Runtime's default conversion and MinMax ranges, QAT helped Atto slightly but made both Tiny models worse (Atto -0.003 against -0.007, Tiny -0.045 against -0.034, Tiny with transformer head -0.050 against -0.033). Calibration after the core conversion was similar. This QAT is a simple version (3 epochs, learned ranges discarded), so a fuller version could still do better.
- **Five folds.** ConvNeXt-Tiny with the transformer head beat ConvNeXt-Atto on all 5 folds, by +0.010 on average (range +0.005 to +0.017), at about 10 times the CPU time. The same model varies by up to 0.024 between folds, so single-fold gaps under about 0.01 should not be read as real. Cutting Atto to 18 evenly spaced windows cost -0.055 to -0.047 on every fold.
- **Radiologist labels.** On the 58 gold studies, which no training or selection used, macro AUC against the official labels was ConvNeXt-Tiny 0.895; ConvNeXt-Tiny, transformer head 0.906; ConvNeXt-Atto 0.900. The LLM report labels the models were trained on score 0.870. Paired on the 57 studies that have both, the models are ahead of their training labels by +0.028 (+0.001 to +0.056) for ConvNeXt-Tiny; +0.038 (+0.008 to +0.067) for ConvNeXt-Tiny, transformer head; +0.033 (+0.001 to +0.060) for ConvNeXt-Atto. So the models do better than the labels they learned from, though the intervals come close to zero. The weakest findings across the models are Lateral OA, PF OA and Synovitis (section 10).
- **Where the windows go.** With 18 windows, placing them towards the middle of each series cut Atto's loss from -0.055 to -0.015, and the menisci loss from -0.102 to -0.019, with no retraining. With 36 centre-weighted windows Atto matched 72 evenly spaced ones (+0.001), at half the CPU time. Picking the 18 windows the model attends to most, from a pass over all 144, did no better (-0.015), so a learned window selector is not worth building. Fold 0 only (section 11).

## 2. Questions

| ID | Hypothesis | Status |
|---|---|---|
| H1 | At equal CPU latency, hybrid and ViT encoders do not beat modern CNNs by more than the noise margin | Supported in fold 0: at the same latency as ConvNeXt-Tiny, CoAtNet-0 ties it, and DINOv2-Small is slower and scores lower (sections 5 and 8) |
| H2 | Cutting windows hurts focal findings more than global ones | Partly supported: see section 6 |
| H3 | Post-training INT8 loses more discrimination and calibration than quantization-aware training, and the gap depends on model family | Not supported here: converting only Conv and MatMul keeps post-training INT8 within the tie margin for all three models, and QAT did not reduce the loss. The conversion mattered more than the model family (section 7) |
| H4 | Label quality moves AUC more than encoder choice | Not tested yet: needs a second label set |
| H5 | At a fixed window budget, choosing windows beats evenly spaced windows | Supported for a fixed centre-weighted schedule, in fold 0. Not supported for choosing by the model's attention, which did no better than the fixed schedule, so no learned selector is built (section 11) |

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

Encoder over all windows plus the study head, after the model is loaded. Each figure is the median of 5 separate timing processes, in Docker with the stated core limit and 3 GB of memory. Time grows in proportion to the number of windows, so 36 windows take half as long. Labels ending `_tf`, `_mean` and `_r192` are the Stage 3 variants; INT8 labels ending `_core_pct` convert only Conv and MatMul layers with percentile ranges.

| Run | Encoder | Precision | 4 cores: s per study (range) | 6 cores: s per study (range) | Peak memory MB | Encoder file MB |
|---|---|---|---|---|---|---|
| atto | convnext_atto | FP32 | 0.96 (0.78-1.01) | 1.50 (1.44-1.57) | 141 | 12.9 |
| atto_tf | convnext_atto@tf | FP32 | 0.97 (0.79-1.01) | 1.50 (1.45-1.57) | 141 | 12.9 |
| atto_r192 | convnext_atto@r192 | FP32 | 0.96 (0.87-1.10) | 0.88 (0.85-0.90) | 133 | 12.9 |
| atto_mean | convnext_atto@mean | FP32 | 0.96 (0.78-1.01) | 1.50 (1.44-1.57) | 141 | 12.9 |
| tiny | convnext_tiny | FP32 | 9.88 (9.49-10.39) | 9.78 (9.67-10.10) | 291 | 106.2 |
| tiny_tf | convnext_tiny@tf | FP32 | 9.88 (9.49-10.39) | 9.78 (9.67-10.10) | 291 | 106.2 |
| tiny_r192 | convnext_tiny@r192 | FP32 | 5.41 (5.34-7.77) | 5.70 (5.55-5.73) | 272 | 106.2 |
| tiny_mean | convnext_tiny@mean | FP32 | 9.88 (9.49-10.39) | 9.78 (9.67-10.10) | 291 | 106.2 |
| mobilenetv4 | mobilenetv4_conv_small | FP32 | 0.38 (0.32-0.39) | 0.39 (0.38-0.46) | 113 | 9.5 |
| efficientnetv2_b0 | tf_efficientnetv2_b0 | FP32 | 1.55 (1.32-1.66) | 1.66 (1.63-1.75) | 151 | 22.4 |
| resnet50_tv2 | resnet50.tv2_in1k | FP32 | 6.23 (5.61-6.71) | 6.33 (6.20-6.47) | 328 | 89.6 |
| coatnet0_256 | coatnet_0_rw_224 | FP32 | 9.40 (9.26-10.10) | 9.74 (9.61-9.94) | 402 | 121.7 |
| dinov2_tuned | vit_small_patch14_dinov2@llrd | FP32 | 11.56 (11.30-11.64) | 11.73 (11.47-12.21) | 249 | 82.7 |
| r3d18_192 | r3d_18 | FP32 | 25.21 (24.51-26.58) | 25.70 (23.29-26.59) | 599 | 126.5 |
| atto_int8_core_pct | convnext_atto | INT8 | 1.65 (1.59-1.73) | 1.78 (1.66-2.53) | 114 | 3.6 |
| tiny_int8_core_pct | convnext_tiny | INT8 | 7.79 (7.29-8.21) | 7.84 (7.72-8.12) | 180 | 27.2 |
| tiny_tf_int8_core_pct | convnext_tiny@tf | INT8 | 7.99 (7.77-8.22) | 7.85 (7.72-8.08) | 180 | 27.2 |
| atto_int8_minmax | convnext_atto | INT8 | 1.35 (1.31-1.44) | 1.43 (1.38-1.46) | 112 | 3.6 |
| atto_int8_pct | convnext_atto | INT8 | 1.39 (1.37-1.42) | 1.43 (1.42-1.48) | 114 | 3.6 |
| tiny_int8_minmax | convnext_tiny | INT8 | 8.64 (8.26-8.86) | 8.40 (8.22-8.49) | 186 | 27.2 |
| tiny_int8_pct | convnext_tiny | INT8 | 9.26 (8.84-9.88) | 8.31 (8.23-8.52) | 186 | 27.2 |
| tiny_tf_int8_pct | convnext_tiny@tf | INT8 | 8.28 (8.16-8.51) | 8.26 (8.21-8.42) | 186 | 27.2 |

The 6-core limit was not faster than the 4-core one: ConvNeXt-Tiny took about the same time, and ConvNeXt-Atto was slower. A likely cause is that WSL's 6 virtual processors do not map to 6 separate physical cores, so the extra threads compete with each other. The two limits therefore do not separate as intended, and the 4-core figures are the ones to quote.



## 9. Five folds

The two finalists, trained from scratch on each of the 5 folds with the Stage 2 setting. Each fold scores a different 870 studies, so scores are not paired across folds. Fold 0 is the run used in the earlier sections.

| Model | Score | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± SD |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Atto | Concordance, 72 windows | 0.8373 | 0.8407 | 0.8393 | 0.8326 | 0.8564 | 0.8413 ± 0.0090 |
| ConvNeXt-Atto | AUC, 72 windows | 0.8897 | 0.8755 | 0.8865 | 0.8724 | 0.8946 | 0.8837 ± 0.0095 |
| ConvNeXt-Atto | Concordance, 36 windows | 0.8309 | 0.8270 | 0.8299 | 0.8193 | 0.8430 | 0.8300 ± 0.0086 |
| ConvNeXt-Atto | Concordance, 18 windows | 0.7821 | 0.7899 | 0.7922 | 0.7773 | 0.8098 | 0.7903 ± 0.0125 |
| ConvNeXt-Tiny, transformer head | Concordance, 72 windows | 0.8505 | 0.8576 | 0.8491 | 0.8393 | 0.8619 | 0.8517 ± 0.0087 |
| ConvNeXt-Tiny, transformer head | AUC, 72 windows | 0.8971 | 0.8925 | 0.9003 | 0.8876 | 0.9012 | 0.8957 ± 0.0057 |
| ConvNeXt-Tiny, transformer head | Concordance, 36 windows | 0.8348 | 0.8440 | 0.8417 | 0.8297 | 0.8519 | 0.8404 ± 0.0086 |
| ConvNeXt-Tiny, transformer head | Concordance, 18 windows | 0.7969 | 0.7977 | 0.7973 | 0.7805 | 0.8007 | 0.7946 ± 0.0080 |

## 10. Radiologist gold labels

Saved fold 0 models scored on the 58 gold studies against the official radiologist labels. The 95% interval resamples studies (2,000 draws). The last column pairs each model with the LLM report labels, used as scores, on the same studies (500 draws). Both are AUC against the official labels.

| Scored | Gold studies | Macro AUC | 95% interval | Against LLM labels on the 57 shared studies (95% interval) |
|---|---|---|---|---|
| LLM report labels (training labels) | 57 | 0.8700 | - | reference |
| ConvNeXt-Tiny | 58 | 0.8955 | 0.860 to 0.926 | +0.028 (+0.001 to +0.056) |
| ConvNeXt-Tiny, transformer head | 58 | 0.9060 | 0.874 to 0.933 | +0.038 (+0.008 to +0.067) |
| ConvNeXt-Atto | 58 | 0.8999 | 0.868 to 0.927 | +0.033 (+0.001 to +0.060) |

## 11. Window placement (Phase A, fold 0)

ConvNeXt-Atto trained on 48 windows, then scored with windows placed in different ways. "Evenly spaced" is the Stage 2 placement. "Centre-weighted" packs windows towards the middle of each series. "Random" draws positions at random with a fixed seed. "Trained at budget" retrains with the same number of windows it is scored with. "Oracle" scores all 144 windows, then keeps the ones the model's attention ranks highest. It is an upper bound for any selector that imitates the model, not a usable method, because it costs the full dense pass. Menisci change is the mean concordance change for the two menisci.

| Window placement | Windows | Concordance | Against 72 evenly spaced (95% interval) | AUC | Menisci change |
|---|---|---|---|---|---|
| dense, all windows | 144 | 0.8423 | +0.005 (+0.001 to +0.009) | 0.8908 | +0.009 |
| evenly spaced, trained on 48 | 72 | 0.8373 | reference | 0.8897 | +0.000 |
| random, trained on 48 | 72 | 0.8320 | -0.005 (-0.012 to +0.002) | 0.8809 | -0.007 |
| centre-weighted, trained on 48 | 72 | 0.8408 | +0.004 (-0.004 to +0.010) | 0.8898 | +0.005 |
| evenly spaced, trained at budget | 72 | 0.8310 | -0.006 (-0.014 to +0.002) | 0.8892 | -0.000 |
| oracle, top by attention per slot | 72 | 0.8405 | +0.003 (-0.003 to +0.008) | 0.8860 | +0.004 |
| oracle, top by attention per study | 72 | 0.8422 | +0.005 (-0.000 to +0.010) | 0.8870 | +0.009 |
| evenly spaced, trained on 48 | 36 | 0.8309 | -0.006 (-0.012 to +0.001) | 0.8787 | -0.011 |
| random, trained on 48 | 36 | 0.8185 | -0.019 (-0.029 to -0.011) | 0.8713 | -0.020 |
| centre-weighted, trained on 48 | 36 | 0.8388 | +0.001 (-0.005 to +0.008) | 0.8880 | -0.004 |
| evenly spaced, trained at budget | 36 | 0.8327 | -0.005 (-0.013 to +0.003) | 0.8838 | -0.006 |
| oracle, top by attention per slot | 36 | 0.8362 | -0.001 (-0.011 to +0.008) | 0.8765 | -0.005 |
| oracle, top by attention per study | 36 | 0.8324 | -0.005 (-0.013 to +0.003) | 0.8766 | -0.001 |
| evenly spaced, trained on 48 | 18 | 0.7821 | -0.055 (-0.068 to -0.043) | 0.8344 | -0.102 |
| random, trained on 48 | 18 | 0.8012 | -0.036 (-0.049 to -0.024) | 0.8446 | -0.046 |
| centre-weighted, trained on 48 | 18 | 0.8224 | -0.015 (-0.025 to -0.007) | 0.8705 | -0.019 |
| evenly spaced, trained at budget | 18 | 0.8094 | -0.028 (-0.040 to -0.015) | 0.8601 | -0.081 |
| oracle, top by attention per slot | 18 | 0.8224 | -0.015 (-0.027 to -0.004) | 0.8653 | -0.019 |
| oracle, top by attention per study | 18 | 0.8126 | -0.025 (-0.036 to -0.014) | 0.8555 | -0.024 |

The adaptive-selection plan in `study_design.md` set a go/no-go rule before these runs: build a learned selector only if the oracle at 18 windows comes within 0.02 AUC of 72 evenly spaced windows. Centre-weighted placement at 18 windows already came close to that line without any selector, so a dated note added a second condition before the oracle result: the oracle must also beat centre-weighted placement by more than 0.01 in concordance.

**Result.** The oracle did not pass either test. Its best 18-window AUC was 0.8653, 0.024 below 72 evenly spaced windows, outside the 0.02 line set in advance. Choosing within each slot scored no better than centre-weighted placement in concordance (-0.015 against -0.015, a difference of +0.000), and choosing across the whole study did worse (-0.025). The rule added after the centre-weighted result needed a gain above 0.01, so Phase B, the learned selector, does not go ahead. A selector trained to copy the model's attention could at best match the oracle, and the oracle only matches a fixed schedule that costs nothing. One reason shows in the findings: the oracle ranks windows by attention averaged over all 12 findings, which kept windows useful for most findings but dropped ones that effusion needs (effusion -0.071 under the oracle against -0.018 centre-weighted).

Retraining at the budget helps but less than moving the windows: 18 evenly spaced windows trained at 18 lost -0.028, against -0.055 for the model trained on 48, and -0.015 for the model trained on 48 with centre-weighted windows. Training at the budget with centre-weighted windows has not been tried.

### Where the attention goes

From the dense pass (144 windows). Profile consistency is the mean correlation between each study's attention-by-position profile and the average profile: 1 means every study attends to the same positions. Mass in the top 18 is how much of the attention the 18 highest windows hold.

| Finding | Profile consistency | Attention mass in the top 18 of 144 windows |
|---|---|---|
| ACL | 0.64 | 0.72 |
| MCL | 0.66 | 0.67 |
| Medial Meniscus | 0.66 | 0.75 |
| Lateral Meniscus | 0.64 | 0.60 |
| Medial OA | 0.69 | 0.53 |
| Lateral OA | 0.71 | 0.45 |
| PF OA | 0.71 | 0.46 |
| Effusion | 0.74 | 0.59 |
| Synovitis | 0.68 | 0.40 |
| Baker's | 0.58 | 0.64 |
| Contusion | 0.55 | 0.64 |
| Fracture | 0.48 | 0.47 |

Attention is only partly consistent across studies (0.48-0.74) and spread out: the top 18 windows hold 0.40-0.75 of the attention, depending on the finding. For about half the findings, the top 18 windows hold less than 60% of the attention, which fits a fixed central schedule doing as well as picking windows per study.

### Studies missing a fat-suppressed plane

AUC within each group, with the change from 72 evenly spaced windows. The small groups have wide uncertainty and no intervals are given.

| Window placement | Windows | all FS planes (789 studies) | no sagittal FS (49 studies) | no coronal FS (32 studies) |
|---|---|---|---|---|
| evenly spaced, trained on 48 | 72 | 0.8933 (+0.000) | 0.8568 (+0.000) | 0.8423 (+0.000) |
| evenly spaced, trained on 48 | 18 | 0.8409 (-0.052) | 0.7854 (-0.071) | 0.7666 (-0.076) |
| evenly spaced, trained at budget | 18 | 0.8664 (-0.027) | 0.8309 (-0.026) | 0.7920 (-0.050) |
| centre-weighted, trained on 48 | 18 | 0.8738 (-0.019) | 0.8418 (-0.015) | 0.8159 (-0.026) |
| oracle, top by attention per slot | 18 | 0.8704 (-0.023) | 0.8319 (-0.025) | 0.8115 (-0.031) |

## 12. Limits

- Most comparisons use one fold and one seed. Folds differ by up to about 0.02 for the same model, so close models cannot be separated without the five-fold runs.
- Timing describes this laptop under Docker limits, not a Raspberry Pi or Jetson, because Docker can limit cores and memory but not device speed.
- INT8 speed depends on the CPU. This one has no VNNI instructions, so INT8 gives little or no speed-up; CPUs with VNNI or ARM dot-product instructions usually gain more.
- Labels come from reports through an LLM, so concordance and calibration measure agreement with those labels. Only 58 studies have radiologist labels, which makes gold intervals wide.
- The data has study IDs but no patient IDs, so two studies of the same patient could fall in different folds.
- The 2.5D and 3D comparison mixes in resolution and pretraining differences.
- The QAT tested is simple: 3 epochs of fine-tuning, with the ranges it learns discarded before conversion. A fuller version that keeps them might close the small remaining gap.
- Study-level data, labels and trained weights are not published here. The MIRA agreement limits redistribution, and whether trained weights count as derivatives still needs asking RSNA.

## 13. Next steps

1. Check centre-weighted placement at 18 and 36 windows on all 5 folds, scoring the saved Stage 5 models.
2. Try training at the budget with centre-weighted windows, the one combination not yet run.
3. Time centre-weighted Atto at 18 and 36 windows on the laptop, to state the accuracy for each CPU cost.
4. Test fat-suppressed series only against all 6 slots (Phase C), and a second label set for H4.
5. The learned window selector (Phase B) is dropped: the attention oracle did no better than a fixed schedule.
