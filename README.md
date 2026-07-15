# RSF-PHATE: Survival Clustering for Customer Churn Prediction

A Python library implementing the RSF-PHATE method: a novel approach for **survival-aware clustering** that combines Random Survival Forests, cophenetic similarity, diffusion smoothing, PHATE embedding, and spectral clustering.

## Context

This repository contains the implementation and results from the **Master's Thesis: "Survival Clustering in Energy Commercial Contracts: An RSF-PHATE Approach"** (Diego Suárez Marañón, 2024).

**Goal**: Segment electricity customers into risk-based groups (Churn Risk, Loyal, Moderate Risk) to enable targeted retention strategies, validated on real commercial data and synthetic survival data.

**Key contributions**:
- Path-aware random survival forest proximities
- Heat-kernel diffusion smoothing with teleportation parameter optimization
- PHATE embedding for non-linear manifold learning
- Survival-clustered segmentation via spectral clustering (Yu-Shi algorithm)
- Robustness under informative censoring (p-boxes)
- Commercial interpretability: actionable cluster profiles with business recommendations

See the full thesis document: `ENTREGA_TFG/TFG_Diego_Suarez_Maranon.docx`

## Installation

### Option 1: Install from source (recommended for reproduction)

```bash
# Clone and navigate
git clone https://github.com/yourusername/rsfphate.git
cd rsfphate

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Option 2: Install just dependencies

```bash
pip install -r requirements.txt
```

**Python version**: 3.10+ (tested on 3.10, 3.11)

## Quick Start

### Minimal example (synthetic data)

```python
from rsfphate import RSFPhate, make_donut_survival

# Generate synthetic survival-clustering problem
X, time, event, truth = make_donut_survival(
    n_samples=400,
    censoring_fraction=0.10,
    random_state=42,
)

# Train RSF-PHATE with 2 clusters
model = RSFPhate(
    n_clusters=2,
    n_estimators=100,
    random_state=42,
    diffusion_time=7.0,  # Optimized for real data (see thesis Sec 4.3)
)

labels = model.fit_predict(X, time, event)

print(f"Cluster labels: {labels[:10]}")
print(f"PHATE embedding shape: {model.embedding_.shape}")
print(f"Proximity matrix shape: {model.smoothed_proximity_.shape}")
```

### Reproduce thesis results (real data)

See the notebooks in `notebooks/`:

1. **`eda_datos_reales.ipynb`** — Exploratory data analysis of real electricity contracts
2. **`eda_datos_sinteticos.ipynb`** — Exploratory analysis of synthetic validation data
3. **`analisis_churn_survival.ipynb`** — Main analysis: RSF-PHATE on real contracts → 3 clusters (Churn Risk, Loyal, Moderate Risk)
4. **`validacion_sintetica.ipynb`** — Validate method on synthetic data with known ground truth (ARI=0.48)
5. **`analisis_multiproducto.ipynb`** — Multi-product analysis: Does gas data improve prediction? (ARI=0.26)

Run all notebooks:
```bash
jupyter notebook notebooks/
```

Expected runtime: ~10-15 min total (on modern CPU)

## Key Results from Thesis

### Real Data (1,200 electricity contracts, subsampled)

**3 discovered clusters:**

| Cluster | n | % | Churn Rate | Median Tenure | Profile |
|---------|---|---|----------|---------|---------|
| 0 | 226 | 18.8% | 15.9% | 278 days | **"New at Risk"** — Recent customers in promo period, high abandonment risk |
| 1 | 577 | 48.0% | 2.4% | 3,541 days | **"Core Loyal"** — Established (10 year avg), very low churn |
| 2 | 365 | 30.4% | 8.8% | 1,139 days | **"Moderate Risk"** — Intermediate tenure, evaluating provider |

**Business impact:**
- Cluster 0 → Target with renewal offers before promo expires
- Cluster 1 → VIP retention programs + cross-sell (gas, digital)
- Cluster 2 → Monitor closely, improve service

### Synthetic Validation

- Dataset: 500K synthetic contracts with known ground truth (Weibull survival times, k=3 true clusters)
- RSF-PHATE recovery: **ARI = 0.48** (Adjusted Rand Index — moderate but robust validation)
- Confirms method works under realistic censoring and noise

### Multiproduct Analysis

- Dataset: Electricity + gas (1,200 + 312 cross-product links)
- Joint model: **ARI = 0.26** (clustering less separable with multiproduct features)
- Implication: Product diversification weakens pure-churn signals

## Public API

### `RSFPhate`

Main estimator. After fitting, available attributes:

- `labels_` — Cluster assignments
- `embedding_` — PHATE embedding (2D or n-D)
- `proximity_` — Cophenetic similarity matrix
- `smoothed_proximity_` — After heat-kernel diffusion
- `rf_model_` — Trained RandomSurvivalForest
- `phate_operator_` — PHATE transformer

Main methods:

- `fit(X, time, event)` — Fit to survival data
- `fit_predict(X, time, event)` — Fit and return labels
- `fit_transform(X, time, event)` — Fit and return embedding

### `to_survival_array(time, event)`

Convert time/event vectors to scikit-survival structured array.

### `make_donut_survival(...)`

Generate synthetic survival-clustering problem (concentric rings).

## Examples

Basic usage:
```bash
python examples/basic_usage.py
```

Full thesis reproduction:
```bash
jupyter notebook notebooks/
# Run notebooks in order: 1→2→3→4→5
```

## Design Notes

- **Dense proximity matrix**: Default (vs sparse) for simplicity with PHATE/clustering.
- **Clustering algorithm**: Yu-Shi spectral discretization on smoothed similarity.
- **Diffusion parameter**: `diffusion_time=7.0` optimized for real electricity data (tested 4.0, 7.0, 10.0; see thesis Sec. 4.3).
- **Subsample strategy**: Stratified subsampling to 1,200 contracts (O(N²) complexity limit).

## Repository Structure

- `src/rsfphate/` — Main library (model.py, forest.py, spectral.py, datasets.py)
- `notebooks/` — Reproduction scripts for thesis results (EDA, main analysis, validation, multiproducto)
- `examples/` — Minimal usage examples
- `tests/` — Unit tests
- `ENTREGA_TFG/` — Final delivery folder with thesis PDF, results, figures
- `requirements.txt` — Python dependencies

## Citation

If you use this method, cite the thesis:

```bibtex
@mastersthesis{suarez_maranon_2024,
  author={Suárez Marañón, Diego},
  title={Survival Clustering in Energy Commercial Contracts: An RSF-PHATE Approach},
  school={Universidad de Oviedo},
  year={2024}
}
```

Or cite via GitHub (CITATION.cff):
```bash
gh repo create rsfphate --citation
```

## License

See `LICENSE` file (open source).

## Contact

- **Author**: Diego Suárez Marañón (diegosuarezma@gmail.com)
- **Advisor**: Luciano Sánchez

For questions or issues, open a GitHub issue or contact the author.
