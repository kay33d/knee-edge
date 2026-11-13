# Hardware restrictions and their sources

Every limit below comes from a published source, except where marked as a machine limit. Docker can enforce core count and memory. It cannot reproduce a device's speed, cache, GPU or power use.

## Enforced in Docker

| Profile | Cores | Source | What it does not reproduce |
|---|---|---|---|
| `pi5-class` | 4 | Raspberry Pi 5, quad-core Cortex-A76 (MedMambaLite) | ARM speed, cache and memory bandwidth |
| `orin-class-cpu` | 6 | Jetson Orin Nano, 6-core Cortex-A78AE (MedMambaLite) | The GPU, which produced the paper's Jetson results |

Both devices have more memory than this laptop can give a container (Orin Nano 8 GB, Pi 5 16 GB in MedMambaLite). Machine limit, not from a paper: the container gets 3 GB with no swap, which is tighter than either device.

## Pass/fail targets

| Target | Value | Source | Caveat |
|---|---|---|---|
| Study latency | 80 s | Browser-based 3D medical AI on consumer laptops (SPIE 2023) | Segmentation of a 256x256x256 volume, read from a search excerpt only |
| Memory | 1.5 GB | Same study | Average use, while this benchmark records peak use |

## Reported only, never measured here

| Quantity | Published value | Source |
|---|---|---|
| Effective compute, Pi 5 | 2.9-9.8 GFLOPS across three models | MedMambaLite |
| Effective compute, Orin Nano | 11.7-77.4 GFLOPS across three models | MedMambaLite |
| Power, Orin Nano | 2.7-3.3 W | MedMambaLite |
| Power, Orin Nano | about 5.4-6.0 W at the wall | Monkeypox study |

Device estimates derived from these rates must be labelled as estimates. The rates come from 2D MedMNIST models and vary by model, so they may not carry over to a CNN on 256 px windows.

## Stage 1 finding

All 15 untrained FP32 encoders met both targets on this laptop under both profiles (slowest 46 s, highest 1,031 MB), so these limits did not rule out any model. This measures encoders only. The laptop ran at about 24-61 G/s on 4 cores, 2.5-21 times MedMambaLite's published Pi 5 rates, so passing here says nothing about passing on a Pi. Data: `results/stage1_summary.json`.

## Removed

The 20 MB model size cap: no paper reviewed sets one.

## Limitation to state in the paper

No reviewed study validated cgroup-limited containers as a stand-in for real edge devices. Results describe this Ryzen 5 5500U under stated limits, not a Pi or Jetson.

## Sources

- MedMambaLite: https://arxiv.org/abs/2508.05049
- Monkeypox study: https://arxiv.org/abs/2507.17123
- SPIE 2023: https://doi.org/10.1117/12.2653626
- Containers at the edge: https://arxiv.org/html/2505.02082
