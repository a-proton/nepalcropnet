cat > README.md << 'EOF'

# NepalCropNet

Maize disease detection for Nepal using Vision Transformers and few-shot learning.

**Status:** NB1 (ResNet50 baseline) complete.

## Results so far

| Model                  | Test Macro F1 | Accuracy |
| ---------------------- | ------------- | -------- |
| ResNet50 (ImageNet V2) | 0.9698        | 0.9792   |

Per-class F1: Gray Leaf Spot 0.919, Common Rust 1.000, Northern Leaf Blight 0.960, Healthy 1.000.

## Repo structure

- `notebooks/` — Kaggle notebooks (NB1–NB4)
- `results/` — metrics JSONs and figures for each NB
- `paper/` — draft sections (coming in Week 7)

## Reproducing NB1

1. Get a Kaggle account, attach the `abdallahalidev/plantvillage-dataset` dataset.
2. Upload `notebooks/nb1_resnet50_baseline.ipynb` and run all cells (T4 GPU, ~12 min).
3. Output goes to `/kaggle/working/`.

## Pipeline

| NB  | What                              | Status |
| --- | --------------------------------- | ------ |
| NB1 | ResNet50 baseline                 | ✅     |
| NB2 | Swin-T fine-tuning                | ⏳     |
| NB3 | Few-shot (Prototypical Networks)  | ⏳     |
| NB4 | Domain gap on Nepali field images | ⏳     |

EOF
