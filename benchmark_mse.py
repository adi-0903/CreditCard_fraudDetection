import time
import numpy as np


def run_benchmark():
    # Test dataset size in fraud_detection notebook (20% of 284,807 = 56,962 samples, 29 features)
    n_samples = 56962
    n_features = 29

    np.random.seed(42)
    X_test = np.random.randn(n_samples, n_features)
    predictions = np.random.randn(n_samples, n_features)

    iterations = 500

    # 1. Baseline: np.power
    start_time = time.perf_counter()
    for _ in range(iterations):
        mse_baseline = np.mean(np.power(X_test - predictions, 2), axis=1)
    baseline_total = time.perf_counter() - start_time
    baseline_avg_ms = (baseline_total / iterations) * 1000

    # 2. Optimized: ** 2 (dedicated square fast-path)
    start_time = time.perf_counter()
    for _ in range(iterations):
        mse_optimized = np.mean((X_test - predictions) ** 2, axis=1)
    optimized_total = time.perf_counter() - start_time
    optimized_avg_ms = (optimized_total / iterations) * 1000

    # Verify correctness
    assert np.allclose(mse_baseline, mse_optimized), "Results do not match!"

    speedup = baseline_avg_ms / optimized_avg_ms
    pct_reduction = ((baseline_avg_ms - optimized_avg_ms) / baseline_avg_ms) * 100

    print("=== MSE Performance Benchmark Results ===")
    print(f"Data size: {n_samples:,} rows x {n_features} features ({iterations} iterations)")
    print(f"Baseline (np.power) : {baseline_avg_ms:.4f} ms per run")
    print(f"Optimized ((...)**2): {optimized_avg_ms:.4f} ms per run")
    print(f"Speedup             : {speedup:.2f}x faster")
    print(f"Latency Reduction   : {pct_reduction:.2f}%")


if __name__ == '__main__':
    run_benchmark()
