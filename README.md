# PRAM: Activating Latent Multimodal Knowledge through Partwise Reasoning

<p align="center">
  <a href="https://luo1yan.github.io/PRAM/"><img src="https://img.shields.io/badge/Project%20Page-PRAM-2565a3?style=flat-square" alt="Project Page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-3a46d6?style=flat-square" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square" alt="PyTorch 2.0+">
</p>

Official PyTorch implementation of **PRAM: Activating Latent Multimodal Knowledge through Partwise Reasoning**.

**Authors:** Haidong Chen<sup>1</sup>, Ying Guo<sup>1</sup>, Shangze Li<sup>2</sup>, Zhongkuan Mao<sup>3</sup>, Linrui Xu<sup>4</sup>, Canran Xiao<sup>5</sup>, Fei Shen<sup>6,&dagger;</sup>, Tat-Seng Chua<sup>6</sup>  
<sup>1</sup>North China University of Technology &middot; <sup>2</sup>Nanjing University of Science and Technology &middot; <sup>3</sup>Sichuan University &middot; <sup>4</sup>Central South University &middot; <sup>5</sup>Sun Yat-sen University &middot; <sup>6</sup>National University of Singapore, NExT++ Research Centre  
<sup>&dagger;</sup>Corresponding author: Fei Shen (`shenfei29@nus.edu.sg`)

<p align="center">
  <img src="docs/assets/figures/fig2.png" alt="Overview of PRAM" width="96%">
</p>

## Overview

**PRAM (Partwise Reasoning across Modalities)** activates latent cross-modal knowledge in frozen multimodal encoders without direct target-modality pairs or encoder fine-tuning. Given two separately provided modality–pivot collections $\mathcal{D}_{AP}$ and $\mathcal{D}_{PB}$, PRAM operates in three closed-form statistical stages:

1. **Multi-Granularity Covariance Regularization (`pram/mcr.py`)** — Retains both global and partwise local vectors from frozen encoders, applies edge-wise centering with modality scalar scaling, and stabilizes local covariance estimation via adaptive cross-part shrinkage.
2. **Pivot-Based Spectral Relational Reasoning (`pram/psrr.py`)** — Composes regularized $A$–$P$ and $P$–$B$ covariances through equal-edge pivot averaging, whitens both endpoints with relative ridge $\lambda = 0.5$, extracts paired projection directions and bounded correlation strengths ($\sigma_{gk} \le 1$) via SVD, and computes composite Gaussian log-likelihood ratio matching scores.
3. **Multi-Granularity Score Fusion (`pram/msf.py`)** — Aggregates global and local correlation scores alongside covariance-weighted pivot-map cosine scores, applies symmetric training-bank standardization (up to 2,000 fixed rows per endpoint), and incorporates cross-fitted discriminant pivot-association and class-semantic scores when pivot states or class prompts repeat.

Across ten datasets on frozen ImageBind and LanguageBind encoders, PRAM ranks first in **16 of 17** cross-modal classification settings and **19 of 20** cross-modal retrieval settings, and leads all **16/16** metrics across three real-world sensing scenarios (robot localization, cross-device vehicle localization, and wearable IMU action recognition).

## Installation

```bash
git clone https://github.com/luo1yan/PRAM.git
cd PRAM
pip install -e .
```

Or install the core dependencies directly:

```bash
pip install -r requirements.txt
```

## Quickstart

### 1. Run the End-to-End Example

```bash
python examples/quickstart.py --device cpu
```

This script generates a disjoint-collection linear–Gaussian multi-part benchmark (no direct $A$–$B$ pairs) and compares `Frozen Cosine`, `Global-only PRAM`, and `Full PRAM` on bidirectional retrieval and class-prompt classification:

```text
Method               A->B R@1   B->A R@1   Mean R@1  Prompt acc.  Balanced acc.
Frozen Cosine            1.4%       0.0%       0.7%         6.2%           6.2%
Global-only PRAM         1.4%       2.8%       2.1%         7.8%           7.8%
Full PRAM              100.0%     100.0%     100.0%       100.0%         100.0%
```

### 2. Minimal Python API

```python
import torch
from pram import (
    ModalityInput,
    PRAM,
    PRAMConfig,
    evaluate_bidirectional_retrieval,
    evaluate_classification,
)

# Frozen global embeddings [N, d0] and optional aligned local features [N, G, d_loc]
ap_a = ModalityInput(global_feat=a_global_train, local_feat=a_local_train)
ap_p = ModalityInput(global_feat=p_global_ap,    local_feat=p_local_ap)
pb_p = ModalityInput(global_feat=p_global_pb,    local_feat=p_local_pb)
pb_b = ModalityInput(global_feat=b_global_train, local_feat=b_local_train)

# Fit strictly on separate A-P and P-B collections (no A-B pairs required)
model = PRAM(PRAMConfig()).fit(ap_a, ap_p, pb_p, pb_b, is_class_prompt=False)

# Retrieval readout R (supports optional chunk_size for large galleries)
ret_scores = model.score_retrieval(test_a, test_b, chunk_size=256)
ret_metrics = evaluate_bidirectional_retrieval(ret_scores, ks=(1, 5, 10))

# Classification readout S (class-prompt matching or 10-neighbor label transfer)
cls_scores = model.score_classification(prompts_a, test_b, chunk_size=256)
cls_metrics = evaluate_classification(cls_scores.T, test_labels_b, protocol="class_prompt")
```

Single-vector modalities (such as text or global audio/IMU embeddings) simply pass `local_feat=None`; PRAM automatically broadcasts them across the aligned local grid while excluding constant broadcast self-blocks from shrinkage estimation and setting the structural weight $\omega = 1/G$ when exactly one modality has multiple local parts.

### 3. Configuration & Ablation Switches

All fixed estimator controls from the paper are exposed via `PRAMConfig`:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `relative_ridge` | `0.5` | Relative ridge coefficient $\lambda$ for all covariance routes |
| `max_directions` | `256` | Maximum retained projection directions per component |
| `max_bank_size` | `2000` | Maximum training reference-bank rows per endpoint (`seed=0`) |
| `max_gallery_size` | `10000` | Maximum labeled training gallery size for 10-NN transfer (`seed=0`) |
| `knn_k` | `10` | Neighbor count for cross-modal label transfer |
| `state_decimals` | `4` | Rounding precision for pivot and class-prompt state quantization |
| `lbfgs_max_iter` | `200` | Maximum L-BFGS steps for 2-fold cross-fitted posterior pooling (`seed=0`) |
| `scale_floor` | `1e-12` | Numerical floor before square roots and divisions |
| `corr_cap` | `1 - 1e-6` | Upper correlation clip preventing singular Gaussian coordinates |
| `use_global` / `use_local` | `True` / `True` | Toggle global and partwise local components |
| `use_shrinkage` / `use_quadratic` | `True` / `True` | Toggle MCR cross-part shrinkage and Gaussian quadratic terms |
| `use_covariance_score` / `use_standardization` | `True` / `True` | Toggle covariance-weighted pivot maps ($s_V$) and reference-bank standardization |
| `use_pivot_associations` / `use_class_semantics` | `True` / `True` | Toggle pivot-association ($s_T$) and class-semantic ($s_A$) scores |

## Repository Structure

```text
PRAM/
├── pram/
│   ├── __init__.py       # Public package exports
│   ├── config.py         # PRAMConfig controls and deterministic sampling
│   ├── mcr.py            # Multi-Granularity Covariance Regularization (MCR)
│   ├── psrr.py           # Pivot-Based Spectral Relational Reasoning (PSRR)
│   ├── msf.py            # Multi-Granularity Score Fusion (MSF) & reference banks
│   ├── metrics.py        # Tie-aware expected Recall@K and classification metrics
│   └── model.py          # High-level PRAM and ModalityInput APIs
├── examples/
│   └── quickstart.py     # Self-contained CPU/GPU synthetic benchmark example
├── tests/
│   └── test_pram_e2e.py  # End-to-end verification suite (14 tests)
├── docs/                 # Bilingual (EN / 中文) GitHub Pages project website
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Testing

Run the full test suite on CPU:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -v
```

## Citation

If you find PRAM useful in your research, please cite:

```bibtex
@article{chen2026pram,
  title={PRAM: Activating Latent Multimodal Knowledge through Partwise Reasoning},
  author={Chen, Haidong and Guo, Ying and Li, Shangze and Mao, Zhongkuan and Xu, Linrui and Xiao, Canran and Shen, Fei and Chua, Tat-Seng},
  journal={arXiv preprint},
  year={2026},
  url={https://luo1yan.github.io/PRAM/}
}
```

## License

This project is released under the [MIT License](LICENSE).
