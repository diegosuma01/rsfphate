"""Small synthetic datasets for examples and smoke tests."""

from __future__ import annotations

import numpy as np


def make_donut_survival(
    n_samples: int = 1000,
    censoring_fraction: float = 0.10,
    random_state: int | None = 42,
):
    """Generate a concentric-ring survival clustering example.

    Parameters
    ----------
    n_samples:
        Total number of samples.
    censoring_fraction:
        Fraction of observations to censor.
    random_state:
        Optional random seed.

    Returns
    -------
    X:
        Array of shape `(n_samples, 2)`.
    time:
        Observed event or censoring times.
    event:
        Event indicator, with `True` meaning the event was observed.
    labels:
        Ground-truth ring labels.
    """

    rng = np.random.default_rng(random_state)

    X = np.zeros((n_samples, 2), dtype=float)
    observed_time = np.zeros(n_samples, dtype=float)
    censored = np.zeros(n_samples, dtype=bool)

    n_inner = int(0.25 * n_samples)
    n_outer = n_samples - n_inner

    radii_inner = np.sqrt(rng.random(n_inner))
    angles_inner = 2.0 * np.pi * rng.random(n_inner)
    X[:n_inner, 0] = radii_inner * np.cos(angles_inner)
    X[:n_inner, 1] = radii_inner * np.sin(angles_inner)
    observed_time[:n_inner] = 2.0 * radii_inner

    radii_outer = 2.0 + rng.random(n_outer)
    angles_outer = 2.0 * np.pi * rng.random(n_outer)
    X[n_inner:, 0] = radii_outer * np.cos(angles_outer)
    X[n_inner:, 1] = radii_outer * np.sin(angles_outer)
    observed_time[n_inner:] = 10.0 * radii_outer

    censored_indices = rng.choice(n_samples, int(censoring_fraction * n_samples), replace=False)
    censored[censored_indices] = True
    observed_time[censored_indices] *= rng.random(len(censored_indices))

    labels = np.concatenate(
        [np.zeros(n_inner, dtype=int), np.ones(n_outer, dtype=int)]
    )
    event = ~censored
    return X, observed_time, event, labels

