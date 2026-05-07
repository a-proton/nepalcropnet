
# NepalCropNet

**Maize disease classification with Vision Transformers and few-shot learning, evaluated under real-world domain shift.**

A complete ML pipeline benchmarking convolutional and transformer architectures for plant disease detection, exploring few-shot adaptation, and quantifying the gap between lab-quality training data and real-world field deployment.

---

## TL;DR

| Stage                                 | Method                      | Test Accuracy | Labels Used |
| ------------------------------------- | --------------------------- | ------------- | ----------- |
| Baseline                              | ResNet50 (ImageNet V2)      | 97.92%        | 2,696       |
| Architecture comparison               | **Swin-T (ImageNet V1)**    | **98.27%**    | 2,696       |
| In-domain label efficiency            | 4-way 5-shot ProtoNet       | 95.17%        | **20**      |
| **Cross-domain (PlantDoc) zero-shot** | **Swin-T**                  | **39.42% ⚠️** | 0           |
| **Cross-domain (PlantDoc) zero-shot** | **ResNet50**                | **23.81% ⚠️** | 0           |
| Cross-domain few-shot                 | 5-shot ProtoNet on PlantDoc | 51.02%        | 15          |

**Headline finding:** Lab-trained models generalize poorly to real-world field imagery, losing 59–74 percentage points on PlantDoc. Vision Transformers are noticeably more robust to domain shift than CNNs (-59 pts vs -74 pts). Naive prototype-based few-shot adaptation provides modest improvement but does not close severe domain gaps, motivating the need for target-domain feature adaptation.

---

## Problem Statement

Plant disease detection models trained exclusively on PlantVillage — a large dataset of plant leaves photographed under controlled lab conditions — are widely cited as performing well, often above 95% accuracy. However, real-world deployment in agricultural fields involves significant domain shift: different lighting, soil backgrounds, leaf orientations, image quality, and disease stages.

This project investigates four research questions:

1. **RQ1:** How large is the performance drop when a PlantVillage-trained model is applied to real-world field imagery?
2. **RQ2:** Does a Swin Transformer fine-tuned on PlantVillage outperform ResNet50 under domain shift?
3. **RQ3:** Can few-shot learning (Prototypical Networks) reduce the labeled-data requirement for adaptation?
4. **RQ4:** What is the failure mode of these models — do they make safe mistakes (e.g., predicting the wrong disease) or unsafe ones (e.g., predicting "healthy" on diseased plants)?

---

## Datasets

| Dataset          | Source                 | Setting            | Maize Classes                                              | Total Maize Images Used |
| ---------------- | ---------------------- | ------------------ | ---------------------------------------------------------- | ----------------------- |
| **PlantVillage** | Hughes & Salathé, 2015 | Lab (controlled)   | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy | 3,852 (70/15/15 split)  |
| **PlantDoc**     | Singh et al., 2020     | Field (real-world) | Gray Leaf Spot, Common Rust, Northern Leaf Blight          | 378 (no Healthy class)  |

Kaggle dataset slugs:

- PlantVillage: `abdallahalidev/plantvillage-dataset`
- PlantDoc: `nirmalsankalana/plantdoc-dataset`

---

## Methodology

The project is structured as four sequential notebooks, each producing a paper-ready result.

### NB1 — ResNet50 Baseline

- ResNet50 pretrained on ImageNet V2, head replaced with `Dropout(0.3) → Linear(2048, 4)`
- Full fine-tuning, 15 epochs, AdamW + cosine LR schedule, weighted sampler for class imbalance
- Trained on PlantVillage maize (4 classes, 70/15/15 split, seed=42)

### NB2 — Swin-T with Two-Stage Fine-Tuning

- Swin Transformer Tiny pretrained on ImageNet V1, head replaced with 4-class output
- **Stage 1 (5 epochs):** entire backbone frozen, only the new head trains (3,076 trainable params)
- **Stage 2 (10 epochs):** unfreeze the last Swin stage + final LayerNorm + head (~14M trainable params, lower LR=1e-4)
- Same data and split as NB1 for direct comparison

### NB3 — Prototypical Networks for Few-Shot (In-Domain)

- Uses NB2's fine-tuned Swin-T as a frozen feature extractor (768-d embeddings)
- 4-way K-shot episodic evaluation on PlantVillage test split
- 1,000 episodes per setting; K ∈ {1, 5}; 15 query images per class per episode
- Two distance metrics tested: Euclidean (raw features) and Cosine (L2-normalized features)

### NB4 — Cross-Dataset Domain Gap (PlantVillage → PlantDoc)

- **Zero-shot evaluation:** Run NB1 ResNet50 and NB2 Swin-T directly on PlantDoc with no adaptation
- **Few-shot adaptation:** Use Swin-T features, sample K PlantDoc images per class as support, evaluate on the rest
- 3-way K-shot (PlantDoc lacks Corn Healthy class), 1,000 episodes per setting
- Compare against in-domain (PlantVillage) results to quantify the domain gap

---

## Results

### NB1 — ResNet50 on PlantVillage Maize

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 0.9792 |
| Test Macro F1 | 0.9698 |

Per-class F1:

| Class                | F1     | Notes                |
| -------------------- | ------ | -------------------- |
| Gray Leaf Spot       | 0.9189 | Hardest class        |
| Common Rust          | 1.0000 | Visually distinctive |
| Northern Leaf Blight | 0.9603 |                      |
| Healthy              | 1.0000 | Visually distinctive |

12 of 578 test errors total. 9 Gray→Northern, 3 Northern→Gray (the visually similar pair).

### NB2 — Swin-T on PlantVillage Maize

| Metric        | Value      | vs. ResNet50 |
| ------------- | ---------- | ------------ |
| Test Accuracy | 0.9827     | +0.0035      |
| Test Macro F1 | **0.9752** | **+0.0054**  |

Per-class F1 (Swin-T vs ResNet50):

| Class                | ResNet50 | Swin-T    | Δ          |
| -------------------- | -------- | --------- | ---------- |
| Gray Leaf Spot       | 0.919    | **0.934** | **+0.015** |
| Common Rust          | 1.000    | 1.000     | 0          |
| Northern Leaf Blight | 0.960    | 0.966     | +0.006     |
| Healthy              | 1.000    | 1.000     | 0          |

Swin-T's entire macro-F1 gain over ResNet50 is concentrated on the hardest class (Gray Leaf Spot), suggesting attention-based architectures handle subtle visual discriminations better than CNNs even on clean lab data.

### NB3 — Few-Shot Prototypical Networks (In-Domain)

Embedding clustering (Swin-T features on PlantVillage test): inter/intra L2 ratio = 1.62 (well-clustered).

| Setting                            | Accuracy   | 95% CI     | Labels Used |
| ---------------------------------- | ---------- | ---------- | ----------- |
| 4-way 1-shot, Euclidean (raw)      | 87.91%     | ±0.40%     | 4           |
| **4-way 1-shot, Cosine (L2-norm)** | **88.94%** | **±0.37%** | **4**       |
| **4-way 5-shot, Euclidean (raw)**  | **95.17%** | **±0.19%** | **20**      |
| 4-way 5-shot, Cosine (L2-norm)     | 94.64%     | ±0.20%     | 20          |

**Key finding:** With just 20 labeled images (5 per class), 5-shot ProtoNet achieves 95.17% — within 3.1 percentage points of fully-supervised Swin-T (98.27%, 2,696 labels). A 99.3% reduction in labels for a 3-point accuracy hit.

Methodological note: Cosine distance with L2-normalized features wins at 1-shot (noisier prototypes benefit from normalization), Euclidean wins at 5-shot (stable prototypes don't need it). Consistent with SimpleShot (Wang et al. 2019).

### NB4 — Cross-Dataset Domain Gap

#### Zero-shot evaluation on PlantDoc

| Model    | PlantVillage Acc | PlantDoc Acc | Drop          |
| -------- | ---------------- | ------------ | ------------- |
| ResNet50 | 97.92%           | **23.81%**   | **−74.1 pts** |
| Swin-T   | 98.27%           | **39.42%**   | **−58.9 pts** |

Both models drop catastrophically. ResNet50's 23.81% is below random chance for 3 classes (33.3%) — the model is systematically biased toward wrong predictions.

#### Per-class recall on PlantDoc

| Class                | ResNet50    | Swin-T |
| -------------------- | ----------- | ------ |
| Gray Leaf Spot       | 50.7%       | 46.3%  |
| Common Rust          | **3.4% ⚠️** | 27.4%  |
| Northern Leaf Blight | 26.8%       | 44.3%  |

Swin-T's advantage over ResNet50 is largest on Common Rust (8x higher recall), suggesting attention better handles texture variations across domains.

#### Deployment-risk: false-positive "Healthy" predictions

| Model    | "Healthy" predictions on PlantDoc | All ground truth is diseased |
| -------- | --------------------------------- | ---------------------------- |
| ResNet50 | 24% of images                     | All are false positives      |
| Swin-T   | 21% of images                     | All are false positives      |

A deployed model trained on PlantVillage would tell ~22% of farmers their diseased crops are healthy.

#### Few-shot adaptation on PlantDoc

Embedding clustering (Swin-T features on PlantDoc): inter/intra L2 ratio = **1.03** (essentially no clustering — features were optimized for lab images).

| Setting                            | Accuracy   | 95% CI     | Labels Used |
| ---------------------------------- | ---------- | ---------- | ----------- |
| 3-way 1-shot, Euclidean            | 40.11%     | ±0.49%     | 3           |
| **3-way 1-shot, Cosine (L2-norm)** | **41.57%** | **±0.54%** | **3**       |
| **3-way 5-shot, Euclidean**        | **51.02%** | **±0.50%** | **15**      |
| 3-way 5-shot, Cosine               | 49.72%     | ±0.49%     | 15          |

5-shot ProtoNet improves over zero-shot by +12 points but plateaus around 51%. With an inter/intra ratio of just 1.03, the embedding space itself is the bottleneck — the lab-trained features simply don't cluster field images by disease class. This is consistent with Chen et al. 2019, who showed that severe domain shifts require target-domain feature adaptation, not just classifier modification.

---

## Key Findings

1. **On clean lab data, transformers and CNNs are nearly tied.** Swin-T edges out ResNet50 by only 0.5 percentage points on PlantVillage. Both are within 2.5 percentage points of perfect on this benchmark.

2. **In-domain few-shot is highly effective.** 5-shot ProtoNet recovers 99.3% of full-supervision performance with 0.7% of the labels — strong evidence that label efficiency is achievable when source and target distributions match.

3. **The lab-to-field domain gap is severe.** Both architectures lose 59–74 percentage points when tested on real-world PlantDoc imagery. This is consistent with prior literature warning against relying on PlantVillage-only benchmarks.

4. **Vision Transformers generalize meaningfully better than CNNs under shift.** Swin-T's 15-point smaller drop (compared to ResNet50) suggests attention-based architectures are less brittle to distribution change. This is a methodological recommendation for any practitioner deploying plant disease models in real settings.

5. **Naive few-shot prototypes are insufficient under severe shift.** When the embedding space itself doesn't cluster the target data (inter/intra ratio dropped from 1.62 to 1.03), prototype averaging cannot recover meaningful performance. Future work should explore feature adaptation methods (head fine-tuning, parameter-efficient adapters, or full backbone updates).

6. **The deployment-risk failure mode is unsafe-by-default.** Both models predicted "Healthy" for ~22% of diseased field images. A deployed system would silently miss roughly one in four disease cases.

---

## Repository Structure
nepalcropnet/
├── README.md                           # this file
├── .gitignore                          # excludes .pt checkpoints, dataset/, OS files
├── notebooks/
│   ├── nb1-resnet50-baseline.ipynb     # ResNet50 fine-tuning
│   ├── nb2-swin-t-baseline.ipynb       # Swin-T two-stage fine-tuning
│   ├── nb3-prototypical-networks.ipynb # Few-shot in-domain
│   └── nb4-plantdoc-domain-gap.ipynb   # Cross-dataset evaluation
├── results/
│   ├── nb1_resnet50_results.json
│   ├── resnet50_confusion_matrix.png
│   ├── resnet50_gradcam.png
│   ├── nb2_swin_t_results.json
│   ├── swin_t_confusion_matrix.png
│   ├── swin_t_training_curves.png
│   ├── nb3_protonet_results.json
│   ├── nb3_episode_distributions.png
│   ├── nb3_supervised_vs_fewshot.png
│   ├── nb4_domain_gap_results.json
│   ├── nb4_domain_gap.png
│   ├── nb4_per_class_recall.png
│   └── nb4_fewshot_recovery.png
└── paper/                              # project writeup (work in progress)

---

## Reproducing the Results

### Prerequisites

- Kaggle account with phone verification (enables free GPU)
- Both Kaggle datasets attached:
  - `abdallahalidev/plantvillage-dataset`
  - `nirmalsankalana/plantdoc-dataset` (only needed for NB4)
- Trained checkpoints from earlier notebooks (only needed for NB3 and NB4)

### Running each notebook

1. Create a new Kaggle notebook, attach the required datasets/checkpoints (see each notebook's Cell 1 for specifics)
2. Set Accelerator to **GPU T4 x2** and Internet to **On** in Session options
3. Upload the `.ipynb` from the `notebooks/` folder of this repo
4. Run all cells in order

| Notebook | Required inputs | Approx. runtime on T4 |
|---|---|---|
| NB1 | PlantVillage dataset | ~12 min |
| NB2 | PlantVillage dataset | ~7 min |
| NB3 | PlantVillage dataset + NB2 checkpoint (`swin_t_best.pt`) | ~2 min |
| NB4 | PlantDoc dataset + NB1 + NB2 checkpoints | ~3 min |

### Saving outputs

After each notebook completes successfully:

1. Click **Save Version → Quick Save** at the top right of the Kaggle notebook (this preserves outputs permanently)
2. Download the four artifact files (model checkpoint, confusion matrix, training curves, results JSON)
3. Either upload the `.pt` checkpoint as a new private Kaggle dataset (for use in subsequent notebooks) or store it locally

---

## Tech Stack

- **Framework:** PyTorch 2.10
- **Models:** torchvision (ResNet50, Swin-T)
- **Data:** scikit-learn (stratified splits, metrics)
- **Visualization:** matplotlib
- **Compute:** Kaggle T4 GPU (free tier)
- **Storage:** Kaggle Datasets, GitHub
- **Few-shot:** Custom Prototypical Networks implementation following Snell et al. 2017

---

## References

- **PlantVillage:** Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
- **PlantDoc:** Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection.* CoDS-COMAD.
- **ResNet:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- **Swin Transformer:** Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV.
- **Prototypical Networks:** Snell, J., Swersky, K., & Zemel, R. (2017). *Prototypical Networks for Few-shot Learning.* NeurIPS.
- **SimpleShot:** Wang, Y., Chao, W. L., Weinberger, K. Q., & van der Maaten, L. (2019). *SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning.* arXiv:1911.04623.
- **Cross-domain few-shot baseline:** Chen, W. Y., Liu, Y. C., Kira, Z., Wang, Y. C. F., & Huang, J. B. (2019). *A Closer Look at Few-shot Classification.* ICLR.

---

## Limitations & Future Work

**Limitations of this study:**

- PlantDoc is a small dataset (378 maize images across 3 classes). The 95% CIs on few-shot results are tight, but absolute conclusions about cross-domain performance would be more robust with a larger field benchmark.
- PlantDoc's lack of a Corn Healthy class restricts the cross-dataset evaluation to 3 classes. The 4-class story is incomplete on the field side.
- All experiments use a single random seed for splits and training; full reporting would include multiple seeds and report variability.
- The few-shot adaptation tested here uses *only* prototype averaging on frozen features. More aggressive baselines (head fine-tuning, LoRA adapters, full fine-tuning on the support set) would likely yield substantially better cross-domain results.

**Promising directions:**

- **Target-domain head fine-tuning:** train a fresh classifier head on K labeled PlantDoc images for ~30 epochs while keeping the backbone frozen. Likely recovers to 70–85% accuracy.
- **Parameter-efficient adaptation:** LoRA on Swin-T's last block, fine-tuned on K-shot support. Combines few-shot regimes with feature update.
- **Self-supervised pretraining on field imagery:** instead of relying on PlantVillage as the only source, pretrain on unlabeled field images (e.g., scraped agricultural photos) before fine-tuning.
- **Multi-source training:** train jointly on PlantVillage + a small subset of PlantDoc to harden the feature extractor against domain shift before any deployment.
- **Larger field-image benchmark:** the most direct improvement would be expanding the field test set with more images per class, ideally with multiple geographic regions and lighting conditions.

---

## Acknowledgments

Models trained on Kaggle's free T4 GPU tier. Datasets sourced from public Kaggle repositories (credit to original authors of PlantVillage and PlantDoc). Methodology informed by the cited references.

---

## License

This repository is currently private. Code and results are released under the MIT License once the repository is made public. Original dataset licenses (PlantVillage, PlantDoc) belong to their respective authors.
