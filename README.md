# `rsfphate`

`rsfphate` is a public Python library that exposes the core contribution of the paper `Fusing Covariates and Censored Outcomes via Reliability-Weighted Diffusion`:

- path-aware random survival forest proximities,
- diffusion smoothing with teleportation,
- PHATE embedding,
- and survival clustering through spectral discretization.

The goal is to provide a clean, reusable library that another user can import into their own project with a minimal API.

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quick start

```python
from rsfphate import RSFPhate, make_donut_survival

X, time, event, truth = make_donut_survival(
    n_samples=400,
    censoring_fraction=0.10,
    random_state=42,
)

model = RSFPhate(
    n_clusters=2,
    n_estimators=100,
    random_state=42,
    diffusion_time=3.0,
)

labels = model.fit_predict(X, time, event)

print(labels[:10])
print(model.embedding_.shape)
print(model.smoothed_proximity_.shape)
```

## Public API

### `RSFPhate`

Main estimator. After fitting, the following learned attributes are available:

- `labels_`
- `embedding_`
- `proximity_`
- `smoothed_proximity_`
- `rf_model_`
- `phate_operator_`

Main methods:

- `fit(X, time, event)`
- `fit_predict(X, time, event)`
- `fit_transform(X, time, event)`

### `to_survival_array(time, event)`

Build a `scikit-survival` structured target array from separate time and event vectors.

### `make_donut_survival(...)`

Generate a small synthetic survival-clustering problem inspired by the paper's concentric-ring example.

## Example script

A runnable example lives in:

- `examples/basic_usage.py`
- `notebooks/quickstart.ipynb`

Run it with:

```bash
python examples/basic_usage.py
```

## Design notes

- The public package defaults to a dense proximity matrix because it is the simplest interface for downstream PHATE and clustering.
- Clustering defaults to Yu-Shi spectral discretization on the smoothed similarity matrix.


## Repository layout

- `src/rsfphate/`: library code
- `examples/`: minimal usage examples
- `notebooks/`: quickstart notebook for GitHub
- `tests/`: smoke tests
- `CHANGELOG.md`: release notes
- `CITATION.cff`: citation metadata for GitHub
