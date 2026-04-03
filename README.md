# 🛰️ EarthSeg-Imbalance
### Novel Deep Learning for Class-Imbalanced Satellite Image Segmentation

> **AI/ML Hackathon — Problem Statement 5**
> Optimizing Deep Learning in Earth Observation with Imbalanced Data
> Dataset: LISS-IV Multispectral Imagery (Bhoonidhi, NRSC/ISRO)

---

## 📌 Problem

In remote sensing segmentation, rare land cover classes — water bodies, wetlands, sparse vegetation — are drastically underrepresented against dominant classes like urban area and bare soil. Standard cross-entropy training causes models to simply ignore minority classes, yielding near-zero IoU for exactly the classes that matter most in environmental monitoring.

This project tackles that head-on with a multi-layer strategy spanning **loss engineering**, **sampling**, **architecture**, and **evaluation discipline**.

---

## 🏗️ Architecture

We implement a **custom Attention U-Net with deep supervision**, trained under a composite loss regime with hard example mining.

```
Input (LISS-IV Multispectral Patches, 128×128×4)
        │
┌───────▼────────┐
│   Encoder      │  ResNet-34 / EfficientNet-B2 backbone
│  (pretrained)  │  Extracts multi-scale spatial features
└───────┬────────┘
        │  skip connections (with Attention Gates)
┌───────▼────────┐
│  Bottleneck    │  Deepest feature representation
└───────┬────────┘
        │
┌───────▼────────┐
│   Decoder      │  Progressive upsampling with skip merges
│  (4 stages)    │──► Auxiliary Head (Stage 2) ─► Deep Supervision Loss
│                │──► Auxiliary Head (Stage 3) ─► Deep Supervision Loss
└───────┬────────┘
        │
┌───────▼────────┐
│  Final Head    │──► Primary Segmentation Output
└────────────────┘
        │
  [N_classes probability maps]
```

### Key Architectural Components

| Component | Purpose |
|---|---|
| **Attention Gates** | Suppress non-target feature activations in skip connections; model learns to focus on rare class boundaries |
| **Deep Supervision** | Auxiliary segmentation heads at intermediate decoder stages; prevents gradient starvation in early layers |
| **Multi-scale Skip Connections** | Preserve fine spatial detail lost during downsampling — critical for small water bodies / wetland patches |
| **Pretrained Encoder** | ResNet-34 ImageNet weights adapted to 4-band LISS-IV via modified first conv layer |

---

## 🧮 Loss Function Engineering

This is the highest-impact lever. We use a **composite loss** that combines global overlap quality with per-pixel hard-example focus:

### Primary Loss: Tversky-Focal Composite

```python
L_total = λ₁ · L_tversky + λ₂ · L_focal + λ₃ · L_bce
```

**Tversky Loss** (generalisation of Dice):
```
TL(P, G) = 1 - |P ∩ G| / (|P ∩ G| + α|FP| + β|FN|)
```
- `α = 0.3`, `β = 0.7` — asymmetric penalty: **missing a rare class is penalised 2.3× harder than a false alarm**
- At `α = β = 0.5`, reduces to standard Dice loss

**Focal Loss** (per-pixel hard mining):
```
FL(p) = -α_f (1 - p)^γ log(p)
```
- `γ = 2` (tunable): suppresses confident background predictions, redirects gradient to ambiguous boundary pixels
- Complements Tversky's global overlap term with pixel-level difficulty weighting

**Why not Dice alone?** Dice loses sensitivity when a class has very few pixels (near-zero denominator instability). Focal + BCE anchor the optimization numerically.

### Deep Supervision Loss
Each auxiliary decoder head contributes at reduced weight:
```python
L_final = L_total(head_final) + 0.4 · L_total(head_stage3) + 0.2 · L_total(head_stage2)
```

---

## 🎯 Sampling Strategy

The loss alone is not enough if 99% of training batches are background pixels.

### Patch Oversampling
- Full LISS-IV scenes are too large; we crop **128×128 patches**
- Batch composition guarantee: **≥ 50% of patches per batch must contain at least one rare-class pixel**
- Rare class index computed from annotation statistics at dataset initialization

### Online Hard Example Mining (OHEM)
- Per batch, only the **top-40% highest-loss pixels** contribute to the backward pass
- Low-loss background pixels are masked out — they add noise, not signal
- Forces the model to focus gradient updates on the uncertain, ambiguous regions where classes actually blur

### Class-Frequency Weighted Sampling
```python
class_weights = 1 / (freq_per_class + ε)
class_weights = class_weights / class_weights.sum()
```
Rare classes receive proportionally higher sample weight during batch construction.

---

## 📊 Evaluation Metrics

We report the **full production metric suite** — not just accuracy.

| Metric | What It Catches |
|---|---|
| **Per-class IoU** | Primary competition metric; rare class IoU is the key signal |
| **Macro IoU** | Unweighted average across all classes — penalises ignoring any class |
| **F1 / Dice (per class)** | Overlap quality at pixel level |
| **Recall / Sensitivity** | Are we finding all rare-class pixels? |
| **Precision** | Are our detections real? |
| **AUPR** | Area under Precision-Recall curve — the right summary metric for imbalanced problems |

> ⚠️ We do **not** report overall pixel accuracy or AUC-ROC as primary metrics — they are misleading in high-imbalance regimes.

---




---

## 🔬 Ablation Study Plan

We run systematic ablations to isolate the contribution of each component:

| Experiment | Loss | Sampling | Architecture |
|---|---|---|---|
| Baseline | BCE | Uniform | Plain U-Net |
| + Dice Loss | Dice | Uniform | Plain U-Net |
| + Tversky | Tversky (0.3/0.7) | Uniform | Plain U-Net |
| + Focal | Tversky + Focal | Uniform | Plain U-Net |
| + OHEM | Tversky + Focal | OHEM | Plain U-Net |
| + Oversampling | Tversky + Focal | OHEM + Patch OS | Plain U-Net |
| + Attention Gates | Tversky + Focal | OHEM + Patch OS | Attention U-Net |
| **Full System** | **Tversky + Focal + DS** | **OHEM + Patch OS** | **Attention U-Net + DeepSupervision** |

---

## 📈 Innovations Summary

1. **Asymmetric Tversky loss** with domain-calibrated α/β — rare class misses penalised over false alarms, aligned with real-world Earth observation stakes
2. **OHEM integrated into composite loss** — gradient budget spent only on hard, uncertain pixels
3. **Attention-gated skip connections** — architecture-level suppression of majority-class feature dominance
4. **Deep supervision with auxiliary heads** — prevents gradient starvation; intermediate layers learn to detect small structures
5. **Guaranteed rare-class patch batching** — sampling strategy enforces minority exposure independent of loss function

---

## 🛠️ Tech Stack

- **Framework**: PyTorch
- **Geospatial I/O**: Rasterio, GDAL
- **Augmentation**: Albumentations (flips, rotations, spectral jitter, elastic transforms)
- **Experiment Tracking**: TensorBoard / W&B
- **Visualization**: QGIS, Matplotlib
- **Evaluation**: Scikit-learn + custom per-class IoU

---

## 👥 Team

ADHITHYALAKSHMAN N
DHANESH V C 
SHARVESHRAM N

---



---

*Built for AI/ML Hackathon — Problem Statement 5 | ISRO LISS-IV Earth Observation Track*
