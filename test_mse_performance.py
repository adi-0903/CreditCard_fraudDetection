import time
import unittest
import numpy as np

class TestMSEPerformance(unittest.TestCase):
    def test_mse_calculation_correctness_and_performance(self):
        # Test dataset dimension matching fraud_detection.ipynb test set (56,962 rows x 29 features)
        n_samples = 56962
        n_features = 29

        np.random.seed(42)
        X_test = np.random.randn(n_samples, n_features)
        predictions = np.random.randn(n_samples, n_features)

        # 1. Baseline: np.power
        mse_baseline = np.mean(np.power(X_test - predictions, 2), axis=1)

        # 2. Optimized: (A - B) ** 2
        mse_optimized = np.mean((X_test - predictions) ** 2, axis=1)

        # Verify identical output
        np.testing.assert_allclose(mse_baseline, mse_optimized)

        # Benchmark over 500 iterations
        iterations = 500

        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = np.mean(np.power(X_test - predictions, 2), axis=1)
        baseline_total_ms = (time.perf_counter() - start_time) * 1000 / iterations

        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = np.mean((X_test - predictions) ** 2, axis=1)
        optimized_total_ms = (time.perf_counter() - start_time) * 1000 / iterations

        speedup = baseline_total_ms / optimized_total_ms
        reduction_pct = ((baseline_total_ms - optimized_total_ms) / baseline_total_ms) * 100

        print("\n=== MSE Calculation Performance Benchmark ===")
        print(f"Dataset shape      : {n_samples:,} x {n_features} ({iterations} iterations)")
        print(f"Baseline (np.power): {baseline_total_ms:.4f} ms per run")
        print(f"Optimized ((A-B)**2): {optimized_total_ms:.4f} ms per run")
        print(f"Speedup            : {speedup:.2f}x faster")
        print(f"Latency Reduction  : {reduction_pct:.2f}%")

if __name__ == "__main__":
    unittest.main()
