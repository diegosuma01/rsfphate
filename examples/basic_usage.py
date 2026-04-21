"""Minimal example for the public RSF-PHATE package."""

from rsfphate import RSFPhate, make_donut_survival


def main():
    X, time, event, truth = make_donut_survival(
        n_samples=400,
        censoring_fraction=0.10,
        random_state=42,
    )

    model = RSFPhate(
        n_clusters=2,
        n_estimators=100,
        diffusion_time=3.0,
        random_state=42,
    )
    labels = model.fit_predict(X, time, event)

    print("Embedding shape:", model.embedding_.shape)
    print("Proximity shape:", model.proximity_.shape)
    print("Smoothed proximity shape:", model.smoothed_proximity_.shape)
    print("First 10 predicted labels:", labels[:10])
    print("First 10 ground-truth labels:", truth[:10])


if __name__ == "__main__":
    main()
