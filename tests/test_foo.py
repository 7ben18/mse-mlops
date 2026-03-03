from mse_mlops.train import choose_indices


def test_choose_indices_fraction():
    indices = list(range(100))
    result = choose_indices(indices, fraction=0.5, max_samples=None, seed=42)
    assert len(result) == 50


def test_choose_indices_max_samples():
    indices = list(range(100))
    result = choose_indices(indices, fraction=1.0, max_samples=10, seed=42)
    assert len(result) == 10
