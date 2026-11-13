# Study design (draft)

Status: draft, before any data is seen. Freeze it before Stage 2. Research prototype, no clinical validation.

## Problem

Given one knee MRI study (a variable set of DICOM series and slices), output 12 finding probabilities: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.

Maximise macro discrimination, subject to running on a CPU within the limits in [hardware_profiles.md](hardware_profiles.md): 80 s per study and 1.5 GB of memory, under the 4-core and 6-core profiles.

**Research question:** study cost is roughly cost per pass x passes per study. For multi-label, weakly labelled MRI, which reduction loses the least per finding: model family, resolution, window count, or numeric precision?

## Hypotheses

| ID | Hypothesis | Judged by |
|---|---|---|
| H1 | At equal CPU latency, hybrid and ViT encoders do not beat modern CNNs by more than the noise margin | Primary metric, paired by study |
| H2 | Cutting windows hurts focal findings (MCL, Lateral Meniscus, Fracture) more than global ones (Effusion, Baker's) | Per-finding change |
| H3 | INT8 post-training quantization loses more discrimination and shifts calibration more than quantization-aware training, and the gap depends on model family | Change in AUC, ECE and Brier score |
| H4 | Label quality moves AUC more than encoder choice among feasible models | Gold and private score only, because a label-based metric cannot judge its own labels |

## Factors

**A. Encoder kind**

| Kind | Candidates |
|---|---|
| Mobile CNN | MobileNetV4-Conv-Small, EfficientNetV2-B0 |
| Classic CNN | ResNet-18, ResNet-34, ResNet-50 |
| Modern CNN | ConvNeXt-Atto, ConvNeXt-Pico, ConvNeXt-Tiny |
| Hybrid | FastViT-T8, MobileViTv2, TinyViT-5M, CoAtNet-0 |
| ViT | DeiT-Tiny, DINOv2-Small |
| 3D CNN | R3D-18, whole series instead of windows |

Excluded: Mamba models, because they rely on custom kernels with weak runtime support (arXiv 2602.00910).

**B. Study-level head:** mean pooling, per-finding attention, per-finding attention with a 2-layer transformer. Mean pooling is the baseline, since it often matched attention in brain MRI work.

**C. Scale:** windows per study 72, 36, 18; input 256 px, 192 px.

**D. Precision:** FP32, FP16, INT8 post-training, INT8 quantization-aware.

A full grid would be far too expensive, so factors are tested in stages.

## Stages

| Stage | Work | Where | Output |
|---|---|---|---|
| 0 | Freeze labels, folds and preprocessing | Kaggle | One fixed data definition |
| 1 | Export untrained candidates, measure MACs, size and CPU latency | This laptop | Feasible list; no GPU used |
| 2 | Train up to 8 encoders, at most 2 per kind, same head, fold 0 | Kaggle | H1 |
| 3 | Vary head and scale on the top 2-3 encoders | Kaggle | H2 |
| 4 | Apply precision options to those models, calibrate after converting | Kaggle, then laptop | H3 |
| 5 | Retrain 2-3 finalists on all folds, including a second label set for H4 | Kaggle | Final results |

A model that misses 80 s at 72 windows stays in Stage 2 only if 18 windows could bring it within the limit.

Stage 1 result: no candidate missed the 80 s or 1.5 GB targets, so the Stage 2 list is chosen for variety and past results, not feasibility. Models are compared on accuracy against measured latency.

## Fixed choices

- **Preprocessing:** 153.6 mm field of view cached at 288 px (0.533 mm/pixel; the model sees a 0.92 crop at 256 px, about 0.55 mm/pixel), canonical orientation from `ImageOrientationPatient`, slices sorted along the normal, one intensity window per series (1st-99.5th percentile), series picked from metadata.
- **Input:** 6 slots (3 planes x fat-suppressed or not), 3 adjacent slices per window, missing slots masked.
- **Laterality:** no flip augmentation. Right knees are mirrored onto left: flip in plane for coronal and axial, reverse slice order for sagittal. Side comes from the DICOM tag, otherwise from patient coordinates at 20 mm or more from the midline; closer studies stay unresolved. On non-gold studies with a tag (49.6%), coordinates agreed with it 98.6% of the time. Vendors with no tag, such as GE and Toshiba, could not be checked this way.
- **Folds:** 5 folds grouped by report text, recording scanner vendor and model for each study.
- **Stage 2 models:** MobileNetV4-Conv-Small, EfficientNetV2-B0, ResNet-50, ConvNeXt-Atto, ConvNeXt-Tiny, CoAtNet-0, DINOv2-Small (fine-tuned, not frozen), R3D-18.
- **R3D-18:** 32 slices per series at 192 px (about 0.74 mm/pixel), one clip per slot, each slice repeated across the 3 input channels, Kinetics-400 weights, same attention head over slots. The user chose 192 px over 256 px (about 15 h of training) and over replacing it with ResNet-34. So any 2.5D vs 3D difference is confounded by resolution and pretraining, and is reported that way. Stage 1 timed it at 256 px, so it must be re-timed at 192 px.
- **Stage 3 additions:** 3 or 5 slices per window; adjacent slices or a fixed thickness in mm. Temporal Shift Module is optional, if it exports to ONNX.
- **Stage 3 runs (started 2026-11-06):** ConvNeXt-Atto and ConvNeXt-Tiny, chosen by the user from the Stage 2 tie. One change at a time from the Stage 2 setting (attention head, 72 windows, 256 px, 3 adjacent slices): mean head; attention after a 2-layer transformer (8 heads); 192 px; 5 slices; slices 3.5 mm apart (rounded to whole slices, at least 1; about the spacing of a typical 2D knee sequence, so thin series are sampled like thick ones). Window count (36 and 18) is tested by scoring each final model with fewer windows, not by retraining, as a deployment would cut them; the Stage 2 models are scored the same way. Temporal Shift Module is not run. With 5 slices the first layer is adapted by timm and every channel gets the mean of the weights' normalisation values. Each session stops cleanly before 11.3 h.
- **Stage 4 runs (started 2026-11-08):** ConvNeXt-Atto and ConvNeXt-Tiny at the Stage 2 setting, plus ConvNeXt-Tiny with the transformer head (best Stage 3 arm), scored from saved weights at 72 windows. Only the encoder changes precision; the head stays FP32 (it costs 1-7 ms). Stage 2 and 3 scores were computed under GPU mixed precision, so Stage 4 rescored in true FP32 as the reference; the scores were identical to 4 decimals (Atto 0.8373, Tiny 0.8417, Tiny with transformer head 0.8505). FP16 is a half-precision copy of the encoder on the T4. INT8 is ONNX Runtime 1.31.1 static quantization in QDQ form: U8 activations, S8 per-channel weights at 7 bits (reduce_range, because the Ryzen 5500U has no VNNI and 8-bit products could saturate, which would make scores depend on the CPU). Ranges come from 32 training-split studies (160 batches of 12 windows, 1,920 in all, because some studies lack a slot), never validation, with MinMax or 99.999th-percentile calibration, collected one batch at a time. INT8 is scored on Kaggle CPU-only sessions. Each encoder's INT8 file is also built by a separate export run, for timing on the laptop.
- **Quantization-aware training (started 2026-11-09):** the same three models, fine-tuned from their saved weights for 3 epochs (encoder learning rate 1e-5, head 5e-5). The encoder trains under simulated INT8: per-channel symmetric 7-bit weights on Conv2d and Linear layers, and 8-bit ranges on the outputs of Conv2d, Linear, GELU and LayerNorm, with the head left in FP32. The simulation is then removed and the plain weights go through the same ONNX Runtime conversion as post-training INT8, so the two differ only in the weights. This is not full QAT: the ranges the simulation learned are discarded and measured again at conversion.
- **Second seed (started 2026-11-09):** ConvNeXt-Tiny with the attention head and with the transformer head, training seed 1. `seed` now covers training randomness only and `fold_seed` (0) sets the folds, so the validation studies are unchanged.
- **Labels:** `pilkwang/rsna-knee-llm-labels` (`report_labels_v2.csv`) for Stages 2-4, excluding the 58 gold studies. Training weight is 0.25 + 0.75 x confidence, as its authors use. Labels with confidence 0.05 mean the report was silent, so they are left out of scoring. `drizlo/rsna-knee-mri-llm-soft-labels` is the second set for H4, subject to the data-handling question. Do not republish any study-level labels: the MIRA agreement limits redistribution and derivatives, whatever licence a dataset page shows.
- **Encoder settings:** each encoder uses the normalisation of its pretrained weights (0.5/0.5 for CoAtNet-0, ImageNet values for the others). DINOv2 uses 252 px because its patch size is 14; Stage 1 timed CoAtNet-0 at 224 px, so it must be re-timed at 256 px.
- **Pretraining:** ImageNet for all 2D encoders, Kinetics-400 for R3D-18. RadImageNet ResNet-50 as one extra arm, with weights licensed CC BY-NC-SA.

## Evaluation

| Use | Measure | Limit on use |
|---|---|---|
| Primary, Stages 2-4 | Macro within-scanner concordance on out-of-fold studies | Cannot judge label changes |
| Generalisation | Same measure with one vendor held out | Final models |
| Gold check | AUC on the 58 gold studies | Finalists only, never for selection |
| Final score | Private score from late submissions | At most one per finalist, plus H4 |
| External | fastMRI+ coronal; MRNet for ACL and meniscus | Partial labels; MRNet bypasses DICOM steps |
| Calibration | ECE and Brier score on the final converted model | Out-of-fold soft labels; gold has too few studies per finding |
| Deployment | Median and p95 study latency, peak RAM, sustained speed | Whole pipeline in Docker for finalists |

**Synovitis:** 84% of reports are silent on it, leaving 127 scored studies and 274 comparable pairs in fold 0 (95% interval 0.656-0.848). Report the macro score with and without it.

**Noise margin:** treat differences under 0.01 as ties, since separately trained models differed by about 0.007-0.010 on the public board in one team's estimate. Report paired bootstrap intervals, resampling studies. Rerun finalists with a second seed.

## Training budget

Measured, ResNet-50 on a T4 with all data: 4.6 h in total (about 0.16 h start-up, 1.2 h preprocessing, 3.25 h for 14 epochs), peak GPU memory 11.0 GB at batch 4, about 57 GB of cache in `/tmp`. The score stopped improving by about epoch 10-12.

A synthetic GPU step predicted ResNet-50 closely (10.9 GB and 0.84 s per step, against 11.0 GB and 0.86 s in the real run). Using it for the others, training time is a lower bound, because data loading may limit the smaller models:

| Model | Batch x accumulation | Peak GPU | s per step | Estimated training h |
|---|---|---|---|---|
| MobileNetV4-Conv-Small | 4 x 2 | 2.2 GB | 0.17 | 0.7 |
| ConvNeXt-Atto | 4 x 2 | 4.6 GB | 0.35 | 1.4 |
| EfficientNetV2-B0 | 4 x 2 | 7.5 GB | 0.44 | 1.7 |
| DINOv2-Small (252 px) | 4 x 2 | 11.7 GB | 0.91 | 3.4 |
| ConvNeXt-Tiny | 2 x 4 (batch 4 ran out of memory) | 7.9 GB | 0.53 | 4.1 |
| CoAtNet-0 | 2 x 4 (batch 4 ran out of memory) | 10.8 GB | 0.77 | 5.8 |
| R3D-18 (192 px) | 2 x 4 | 6.5 GB | 1.44 | 10.6, so two sessions at the 11 h stop |

Batch 2 with 4 accumulation steps keeps 8 studies per update and the same updates per epoch. Models can share a session and its preprocessing. Stages 3-5 depend on Stage 2 results.

## Stage 4 results

Current tables are in `docs/project_report.md`, rebuilt by `results/build_report.py`. So far (fold 0, one seed): FP16 changes nothing. Post-training INT8 with ONNX Runtime's default conversion costs ConvNeXt-Tiny 0.018-0.034 and worsens calibration; converting only Conv and MatMul layers cuts the loss to 0.001-0.004 for every model, because the default also rounds around LayerNormalization, which still computes in floating point. Quantization-aware training (finished 2026-11-13) did not reduce the loss: after the Conv and MatMul conversion, QAT weights lost 0.001-0.007 against 0.000-0.004 without it, and with the default conversion they lost more on both Tiny models. H3 is not supported in this setup; the conversion mattered more than the model family.

## Stage 3 and 4 timing (2026-11-13)

Clean run, 5 separate timing processes per model, laptop idle. On 4 cores, FP32, 72 windows: ConvNeXt-Atto 0.96 s, ConvNeXt-Tiny 9.88 s, CoAtNet-0 9.40 s, DINOv2-Small 11.56 s, R3D-18 25.2 s; peak memory under 600 MB. INT8 is about a fifth faster for Tiny and slower for Atto, because this CPU lacks VNNI. The 6-core limit was not faster than the 4-core one, so 4-core figures are quoted. Tables in `docs/project_report.md`.

## Known limits

- One fold and one seed cannot separate close models, so Stage 2 only narrows the list.
- Results describe this Ryzen 5 5500U under stated limits, not a Pi or Jetson.
- Check the MIRA agreement before publishing weights. Its derivative-works clause does not say whether trained weights count.

## Stage 3 result (fold 0, one seed, 2026-11-07)

Paired difference from the Stage 2 model of the same encoder (Atto 0.8373, Tiny 0.8417), 95% interval:

| Change | ConvNeXt-Atto | ConvNeXt-Tiny |
|---|---|---|
| Mean head | -0.006 (-0.015 to 0.003) | -0.002 (-0.012 to 0.008) |
| Transformer, then attention | +0.000 (-0.010 to 0.009) | +0.009 (-0.001 to 0.019) |
| 192 px | -0.003 (-0.010 to 0.006) | -0.004 (-0.013 to 0.007) |
| 5 slices | -0.008 (-0.017 to 0.003) | -0.003 (-0.014 to 0.007) |
| 3.5 mm gaps | +0.001 (-0.005 to 0.007) | +0.002 (-0.007 to 0.012) |
| 36 windows, scored | -0.007 (-0.012 to 0.001) | -0.013 (-0.020 to -0.005) |
| 18 windows, scored | -0.055 (-0.068 to -0.043) | -0.057 (-0.069 to -0.045) |

- Measured slice spacing: median 3.5 mm (5th-95th percentile 2.4-5.0), so the 3.5 mm arm changed only 5% of series and is close to a repeat run.
- H2, partly supported: at 18 windows the menisci lose most (-0.08 to -0.14), MCL loses -0.08 / -0.03, but Fracture loses less (-0.03 / -0.00) than Effusion and Baker's (-0.05 to -0.07). Windows were cut at scoring time; models trained with fewer windows may lose less.
- Scoring a saved Stage 2 model reproduced its score exactly, which checks the scoring path.
