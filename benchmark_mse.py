import time
import numpy as np


def run_benchmark():
    # Test dataset size in fraud_detection notebook (20% of 284,807 transactions = 56,962 rows, 29 features)
    n_samples = 56962
    n_features = 29

    np.random.seed(42)
    X_test = np.random.randn(n_samples, n_features)
    predictions = np.random.randn(n_samples, n_features)

    iterations = 500

    # 1. Baseline: np.power(..., 2)
    start_time = time.perf_counter()
    for _ in range(iterations):
        mse_baseline = np.mean(np.power(X_test - predictions, 2), axis=1)
    baseline_total = time.perf_counter() - start_time
    baseline_avg_ms = (baseline_total / iterations) * 1000

    # 2. Optimized: Vectorized inline squaring ((X_test - predictions) ** 2)
    start_time = time.perf_counter()
    for _ in range(iterations):
        mse_opt = np.mean((X_test - predictions) ** 2, axis=1)
    optimized_total = time.perf_counter() - start_time
    optimized_avg_ms = (optimized_total / iterations) * 1000

    # Verify numerical correctness
    np.testing.assert_allclose(mse_baseline, mse_opt, rtol=1e-12, atol=1e-12)

    speedup = baseline_avg_ms / optimized_avg_ms
    pct_reduction = ((baseline_avg_ms - optimized_avg_ms) / baseline_avg_ms) * 100

    print("=== Performance Benchmark Results (MSE Calculation) ===")
    print(
        f"Data shape: {n_samples:,} rows x {n_features} features ({iterations} iterations)"
    )
    print(f"Baseline (np.power)   : {baseline_avg_ms:.4f} ms per run")
    print(f"Optimized ((A - B)**2): {optimized_avg_ms:.4f} ms per run")
    print(f"Speedup                : {speedup:.2f}x faster")
    print(f"Latency Reduction      : {pct_reduction:.2f}%")


if __name__ == "__main__":
    run_benchmark()
