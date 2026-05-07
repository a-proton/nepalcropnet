# NepalCropNet

Maize disease detection for Nepal using Vision Transformers and few-shot learning.

**Status:** NB1 + NB2 complete. NB3 (Prototypical Networks) in progress.

## Results so far

PlantVillage Maize, 4 classes (Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy), 3,852 images, 70/15/15 stratified split.

| Model | Test Accuracy | Test Macro F1 | Notes |
|---|---|---|---|
| ResNet50 (ImageNet V2) | 0.9792 | 0.9698 | Full fine-tuning, 15 epochs |
| **Swin-T (ImageNet V1)** | **0.9827** | **0.9752** | Two-stage: head warmup → unfreeze last block |

Swin-T edges out ResNet50 by +0.005 macro F1, **with the entire gain concentrated on Gray Leaf Spot — the hardest class to discriminate from Northern Leaf Blight.**

### Per-class F1

| Class | ResNet50 | Swin-T | Δ |
|---|---|---|---|
| Gray Leaf Spot | 0.919 | 0.934 | **+0.015** |
| Common Rust | 1.000 | 1.000 | 0 |
| Northern Leaf Blight | 0.960 | 0.966 | +0.006 |
| Healthy | 1.000 | 1.000 | 0 |

### Confusion (errors only)
- ResNet50: 12 total errors (9 Gray→Northern, 3 Northern→Gray)
- Swin-T: 10 total errors (6 Gray→Northern, 4 Northern→Gray)

## Repo structure

- `notebooks/` — Kaggle notebooks (NB1–NB4)
- `results/` — metrics JSONs and figures for each NB
- `paper/` — draft sections (coming in Week 7)

## Pipeline

| NB | What | Status | Headline |
|---|---|---|---|
| NB1 | ResNet50 baseline | ✅ | 0.9698 test macro F1 |
| NB2 | Swin-T fine-tuning | ✅ | 0.9752 test macro F1 (+0.005 over ResNet50) |
| NB3 | Few-shot (Prototypical Networks) | ⏳ | TBD |
| NB4 | Domain gap on Nepali field images | ⏳ | TBD (blocked on field collection) |

## Reproducing the baselines

1. Get a Kaggle account, attach the `abdallahalidev/plantvillage-dataset` dataset to a notebook with GPU T4 enabled.
2. Upload the corresponding notebook from `notebooks/` and run all cells.
3. Outputs (checkpoints, JSONs, figures) appear in `/kaggle/working/`.

NB1 trains in ~12 min. NB2 (two-stage) trains in ~7 min total.
