import numpy as np


def test_mse_calculation_exactness():
    np.random.seed(42)
    X_test = np.random.randn(100, 29)
    predictions = np.random.randn(100, 29)

    mse_baseline = np.mean(np.power(X_test - predictions, 2), axis=1)
    mse_optimized = np.mean((X_test - predictions) ** 2, axis=1)

    np.testing.assert_allclose(mse_baseline, mse_optimized, rtol=1e-12, atol=1e-12)


def test_benchmark_script_runs():
    import benchmark_mse
    benchmark_mse.run_benchmark()
