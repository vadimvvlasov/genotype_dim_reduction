# Genotype Dimensionality Reduction

Pipeline for population structure analysis from SNP/genotype data using PCA, t-SNE, and UMAP.

## Overview

A dimensionality reduction workflow to detect population structure from genetic markers. Demonstrates three methods on synthetic SNP data (1000 samples × 10,000 SNPs) to identify two population clusters.

## Quick Start

```bash
source .venv/bin/activate
jupyter notebook snp_explorer.ipynb
```

## Methods

| Method | Purpose | Parameters |
|--------|---------|------------|
| PCA | Linear dimensionality reduction | 20 components, random_state=42 |
| t-SNE | Non-linear visualization | perplexity=30, init=pca |
| UMAP | Non-linear visualization | n_neighbors=15, min_dist=0.1 |

Standard workflow: SNP matrix → PCA (20 PC) → t-SNE/UMAP → visualization.

## Data

- 1000 samples (2 populations × 500)
- 10,000 SNP markers
- 3,000 divergent markers with allele frequency shift = 0.3
- Seed: 42

## Output

- `plots/scree_plot.png` — PCA variance explained
- `plots/dashboard.png` — PCA/t-SNE/UMAP comparison

## Requirements

- Python 3.13+
- numpy, pandas, scikit-learn, matplotlib, seaborn, umap-learn

Install: `pip install -e .`