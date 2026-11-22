# CA-DCF-Net: A Dual-Stream Dynamic Framework for Resolving Small and Subtle Thoracic Abnormalities

Official PyTorch implementation, configuration files, and trained checkpoints for **CA-DCF-Net**, a dual-stream YOLOv11s-based framework integrating Context-Aware Dynamic Compositional Filters (CA-DCF) for thoracic abnormality detection in chest radiographs (CXR).

## Overview
- **Stream I (Structure-Preserving Baseline, M1):** Unmodified YOLOv11s trained under standard augmentation.
- **Stream II (Texture-Aware Expert, M2):** YOLOv11s with Neck `C3k2` blocks (nodes 13, 16, 19, and 22) replaced by **CA-DCF** modules ($K = 8$, Gabor initialization, module-specific learnable dictionaries) trained under heavy augmentation.
- **Dual-Stream Ensemble (M3):** Late fusion of Stream I and Stream II predictions via class-aware global NMS ($\tau_{\text{iou}} = 0.60$, operating confidence threshold $s_{\text{conf}} = 0.13$).

## Benchmark Summary
| Evaluation Cohort | Model | Overall mAP@0.5 | Class-wise Nodule/Mass AP@0.5 |
| :--- | :--- | :---: | :---: |
| **VinDr-CXR** (Held-out Test, $N = 3,000$) | YOLOv11s Baseline (M1) | `0.323 ± 0.003` | `0.224 ± 0.004` |
| **VinDr-CXR** (Held-out Test, $N = 3,000$) | CA-DCF-Net (M3) | `0.341 ± 0.003` | `0.274 ± 0.005` |
| **ChestX-Det** (Zero-Shot, $N = 553$, 9 classes) | YOLOv11s Baseline (M1) | `0.221` (95% CI: `0.206–0.253`) | `0.176` (95% CI: `0.137–0.247`) |
| **ChestX-Det** (Zero-Shot, $N = 553$, 9 classes) | CA-DCF-Net (M3) | `0.268` (95% CI: `0.251–0.298`) | `0.225` (95% CI: `0.183–0.298`) |

*Note: VinDr-CXR variability reports mean ± SD across three independent training seeds (`{0, 42, 123}`); external ChestX-Det 95% CIs are image-level percentile bootstrap intervals ($B = 1,000$, RNG seed `0`) on the primary reference checkpoint (`seed = 0`).*

## Environment & Key Hyperparameters
- **Framework:** PyTorch + `ultralytics==8.3.162`
- **Input Resolution:** `1024 × 1024 × 3`
- **Optimizer:** AdamW (`lr0 = 5e-4`, `lrf = 0.01`, `weight_decay = 5e-4`, `batch_size = 12`)
- **Seeds:** Primary `seed = 0`; multi-run stability `{0, 42, 123}` (`deterministic = false`)

## Datasets
Raw radiographs are not redistributed in this repository. Please obtain the public datasets from their official sources:
1. **VinDr-CXR:** Nguyen et al., *Scientific Data* (2022), https://doi.org/10.1038/s41597-022-01498-w
2. **ChestX-Det:** Liu et al., *arXiv:2006.10550* (2020)

## Clinical Disclaimer
CA-DCF-Net is an experimental research framework evaluated strictly on retrospective public benchmarks and is not validated for clinical deployment or automated screening.

## Authors
Melika Ghasemi, Hamid Reza Ghaffari*, Hamed SabbaghGol
