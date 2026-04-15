# AGENTS.md

## Project Overview
Dimensionality reduction pipeline for SNP/genotype data (PCA, t-SNE, UMAP).

## Key Entry Points
- **`snp_explorer.ipynb`**: Main Jupyter notebook with all analysis code — this is where the actual work happens
- **`main.py`**: Stub entry point (prints hello, not the main analysis)
- **`export_plots.py`**: Exports plots generated in notebook to `plots/`

## Running the Project
```bash
# Activate virtual environment (already set up)
source .venv/bin/activate

# Open notebook
jupyter notebook snp_explorer.ipynb

# Or run as script (if converted to .py)
python main.py
```

## Dependencies
- Python 3.13+
- numpy, pandas, scikit-learn, matplotlib, seaborn, umap-learn, jupyter

## Testing
No formal test suite exists. Run notebook cells to verify.

## Notes
- `plots/` directory contains output figures (dashboard.png, scree_plot.png)
- Synthetic data is generated with seed=42 in the notebook
- Workflow: generate SNP matrix → PCA → t-SNE/UMAP → visualize