# NepalCropNet

Maize disease detection using Vision Transformers and few-shot learning.

**Status:** NB1 + NB2 + NB3 complete. NB4 (cross-dataset domain gap on PlantDoc) next.

## Results — PlantVillage Maize (4 classes, 3,852 images, 70/15/15 split)

### Supervised baselines

| Model | Test Accuracy | Test Macro F1 | Labels used |
|---|---|---|---|
| ResNet50 (ImageNet V2) | 0.9792 | 0.9698 | 2,696 |
| **Swin-T (ImageNet V1)** | **0.9827** | **0.9752** | 2,696 |

Per-class F1 — Swin-T's gain over ResNet50 is concentrated on the hardest class:

| Class | ResNet50 | Swin-T | Δ |
|---|---|---|---|
| Gray Leaf Spot | 0.919 | 0.934 | **+0.015** |
| Common Rust | 1.000 | 1.000 | 0 |
| Northern Leaf Blight | 0.960 | 0.966 | +0.006 |
| Healthy | 1.000 | 1.000 | 0 |

### Few-shot with frozen Swin-T features (1000 episodes per setting)

| Method | Accuracy | 95% CI | Labels used |
|---|---|---|---|
| 4-way 1-shot ProtoNet (Cosine + L2-norm) | **88.94%** | ±0.37% | **4** |
| 4-way 5-shot ProtoNet (Euclidean) | **95.17%** | ±0.19% | **20** |

**Key finding:** 5-shot ProtoNet hits 95.17% with just 20 labeled images — within 3.1 points of fully-supervised Swin-T (which uses 2,696 training images). **A 99.3% reduction in labels for a 3-point accuracy hit.**

## Repo structure

- `notebooks/` — Kaggle notebooks (NB1–NB4)
- `results/` — metrics JSONs and figures for each NB
- `paper/` — draft sections (Week 7)

## Pipeline

| NB | What | Status | Headline |
|---|---|---|---|
| NB1 | ResNet50 baseline | ✅ | 0.9698 macro F1 |
| NB2 | Swin-T fine-tuning | ✅ | 0.9752 macro F1 (+0.005 over ResNet50) |
| NB3 | Few-shot Prototypical Networks | ✅ | 88.94% (1-shot), 95.17% (5-shot) |
| NB4 | Cross-dataset domain gap (PlantVillage → PlantDoc) | ⏳ | TBD |

## Reproducing

1. Get a Kaggle account, attach `abdallahalidev/plantvillage-dataset` to a notebook with GPU T4.
2. Upload notebooks from `notebooks/` and run all cells.
3. NB1 trains in ~12 min. NB2 (two-stage) in ~7 min. NB3 (uses NB2 checkpoint) in ~2 min.
