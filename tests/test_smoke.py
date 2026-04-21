import numpy as np

from rsfphate import RSFPhate, make_donut_survival, to_survival_array


def test_survival_array_builder():
    y = to_survival_array([1.0, 2.0], [True, False])
    assert y.dtype.names == ("event", "time")
    assert y.shape == (2,)


def test_rsfphate_smoke():
    X, time, event, _ = make_donut_survival(
        n_samples=80,
        censoring_fraction=0.10,
        random_state=7,
    )
    model = RSFPhate(
        n_clusters=2,
        n_estimators=20,
        phate_n_components=2,
        random_state=7,
    )
    labels = model.fit_predict(X, time, event)

    assert labels.shape == (80,)
    assert model.embedding_.shape == (80, 2)
    assert model.proximity_.shape == (80, 80)
    assert model.smoothed_proximity_.shape == (80, 80)
    assert np.isfinite(model.proximity_).all()
