# MotionCLIP-Based Skeleton Anomaly Detection

This repository contains the implementation and experiments for my MSc thesis on anomaly action detection using pretrained MotionCLIP embeddings on the NTU RGB+D dataset.

The project explores how pretrained motion-language representations can be adapted for anomaly detection in human skeleton sequences using:
- contrastive fine-tuning,
- Gaussian distribution modeling,
- hyperbolic embedding spaces,
- and anomaly-aware representation learning.

---

# Repository Structure

```text
├── MotionCLIP_experiment/
│   ├── finetuning/
│   ├── anomaly_detection/
│   ├── hyperbolic/
│   ├── visualization/
│   ├── evaluation/
│   ├── checkpoints/
│   └── scripts/
│
├── MotionCLIP_ready_datasetFinalAll_1-60/
├── MotionCLIP_ready_datasetFinalAll_61-120/
├── MotionCLIP_ready_datasetFinalAll_1-120/
│
├── shellScripts/
│
├── combinedDataset.py
├── ntu_60-120.txt
├── smpl_pkl_files_120.txt
└── .gitignore
```

---

# Project Overview

The goal of this thesis is to investigate whether MotionCLIP embeddings can be adapted into a compact and discriminative representation space for anomaly action detection.

Instead of relying purely on classification confidence, the project models the distribution of **normal actions** inside the embedding space and detects anomalies using distance-based scoring.

The project primarily evaluates:
- MotionCLIP pretrained on BABEL60,
- fine-tuned MotionCLIP embeddings,
- hyperbolic embedding extensions,
- and anomaly-aware representation learning inspired by AA-CLIP.

---

# Datasets

## NTU RGB+D 60 / 120

Experiments are conducted on:
- NTU RGB+D 60
- NTU RGB+D 120

The repository contains preprocessed MotionCLIP-compatible datasets:

```text
MotionCLIP_ready_datasetFinalAll_1-60/
MotionCLIP_ready_datasetFinalAll_61-120/
MotionCLIP_ready_datasetFinalAll_1-120/
```

Each sequence is represented using:
- 60 frames,
- 25 joints,
- rot6D pose representation,
- global translation enabled.

---

# MotionCLIP Fine-Tuning

The pretrained MotionCLIP encoder is fine-tuned on NTU skeleton data to improve anomaly separability.

Typical encoder configuration:

```python
Encoder_TRANSFORMER(
    modeltype='motionclip',
    njoints=25,
    nfeats=6,
    num_frames=60,
    pose_rep='rot6d',
    translation=True,
    glob=True,
    latent_dim=512
)
```

The extracted embedding used for anomaly detection is:

```python
output['mu']
```

which produces a 512-dimensional motion embedding.

---

# Fine-Tuning Strategies

## 1. Supervised Contrastive Learning

Normal classes are pulled closer together using supervised contrastive loss.

Goals:
- improve compactness of normal clusters,
- reduce overlap,
- improve anomaly separability.

Experiments include:
- class-aware batching,
- temperature tuning,
- partial encoder unfreezing,
- contrastive-only training,
- CE + contrastive combinations.

---

## 2. Positive-Only Attraction Loss

Instead of strongly pushing classes apart, this method:
- primarily pulls normal samples together,
- reduces fragmentation,
- creates smoother normal manifolds.

This often improves anomaly detection because anomaly detection depends more on compact normal regions than large inter-class margins.

---

## 3. Unsupervised Normal Collapse

An unsupervised variation where:
- labels are ignored,
- all normal samples are pulled together,
- the encoder learns a compact one-class representation.

---

# Gaussian-Based Anomaly Detection

After fine-tuning:
- embeddings are extracted,
- Gaussian distributions are fitted over normal samples.

Different variants explored:

## Single Gaussian
All normal embeddings modeled using:
- one global mean,
- one covariance matrix.

## Multimodal Gaussian
Each normal class receives:
- its own Gaussian,
- separate covariance structure.

Anomaly score:

```text
Mahalanobis Distance(x)
```

Higher distance:
→ more anomalous.

---

# Hyperbolic Embedding Extension

This repository also explores converting the MotionCLIP embedding space into a hyperbolic representation space.

The idea is based on:
- hierarchical hyperbolic learning,
- prototype-based OOD detection,
- balanced hyperbolic embeddings.

Reference:
- "Balanced Hyperbolic Embeddings Are Natural Out-of-Distribution Detectors" :contentReference[oaicite:0]{index=0}

---

# Hyperbolic Pipeline

## Stage 1 — Learn Hyperbolic Prototypes

A hierarchy is defined over normal actions.

Class prototypes are optimized inside a Poincaré ball using:
- distortion loss,
- norm balancing loss.

Goals:
- preserve hierarchy structure,
- avoid prototype collapse,
- improve semantic organization.

---

## Stage 2 — Align MotionCLIP Embeddings

MotionCLIP embeddings are:
1. projected into hyperbolic space using exponential mapping,
2. optimized to land near the correct prototype.

This creates:
- compact class-aware regions,
- stronger semantic structure,
- improved anomaly separability.

---

# AA-CLIP Inspired Extensions

Inspired by AA-CLIP, the project also investigates:
- anomaly-aware representation adaptation,
- embedding disentanglement,
- stronger separation between normal and abnormal semantics.

Reference:
- "AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP" :contentReference[oaicite:1]{index=1}

Key ideas adapted:
- lightweight residual adaptation,
- controlled fine-tuning,
- preserving pretrained generalization,
- improving anomaly discrimination without overfitting.

---

# Evaluation

Primary evaluation metric:
- AUROC

Additional analyses:
- PCA visualization
- t-SNE visualization
- UMAP visualization
- distance histograms
- embedding overlap analysis

Comparisons include:
- pretrained MotionCLIP,
- fine-tuned MotionCLIP,
- hyperbolic MotionCLIP,
- graph-based anomaly baselines,
- prompt-guided anomaly detection methods.

---

# Main Findings

## Fine-tuning improves anomaly detection
Even simple fine-tuning significantly improves separability over pretrained MotionCLIP embeddings.

## Random splits are easier
Random normal classes often form naturally separable clusters.

## Meaningful splits are harder
Semantically related actions heavily overlap in embedding space.

## Hyperbolic geometry is promising
Hyperbolic prototypes provide:
- better hierarchical organization,
- stronger uncertainty modeling,
- improved separation for nuanced actions.

---

# Running Experiments

## Fine-Tuning

```bash
python finetune.py
```

## Extract Embeddings

```bash
python extract_embeddings.py
```

## Run Anomaly Detection

```bash
python anomaly_detection.py
```

## Hyperbolic Training

```bash
python train_hyperbolic.py
```

---

# References

## MotionCLIP
Tevet et al. — Human Motion as a Foreign Language

## AA-CLIP
Ma et al. — Anomaly-Aware CLIP for Zero-Shot Anomaly Detection :contentReference[oaicite:2]{index=2}

## Balanced Hyperbolic Embeddings
Kasarla et al. — Balanced Hyperbolic Embeddings Are Natural OOD Detectors :contentReference[oaicite:3]{index=3}

## Markovitz et al.
Graph-based skeleton anomaly detection baseline.
