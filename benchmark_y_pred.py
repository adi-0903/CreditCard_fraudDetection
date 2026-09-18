import time
import numpy as np
import pandas as pd

def run_benchmark():
    # Test dataset size in fraud_detection notebook (20% of 284,807 transactions = ~56,962 rows)
    n_samples = 56962
    threshold = 2.9

    # Generate representative reconstruction error values
    np.random.seed(42)
    error_df = pd.DataFrame({
        'reconstruction_error': np.random.exponential(scale=1.5, size=n_samples)
    })

    iterations = 500

    # 1. Baseline: List comprehension
    start_time = time.perf_counter()
    for _ in range(iterations):
        y_pred_list = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    baseline_total = time.perf_counter() - start_time
    baseline_avg_ms = (baseline_total / iterations) * 1000

    # 2. Optimized: Vectorized NumPy operation
    start_time = time.perf_counter()
    for _ in range(iterations):
        y_pred_vec = (error_df.reconstruction_error.values > threshold).astype(int)
    optimized_total = time.perf_counter() - start_time
    optimized_avg_ms = (optimized_total / iterations) * 1000

    # Verify correctness
    assert np.array_equal(y_pred_list, y_pred_vec), "Results do not match!"

    speedup = baseline_avg_ms / optimized_avg_ms
    pct_reduction = ((baseline_avg_ms - optimized_avg_ms) / baseline_avg_ms) * 100

    print("=== Performance Benchmark Results ===")
    print(f"Data size: {n_samples:,} rows ({iterations} iterations)")
    print(f"Baseline (List Comprehension) : {baseline_avg_ms:.4f} ms per run")
    print(f"Optimized (Vectorized .astype): {optimized_avg_ms:.4f} ms per run")
    print(f"Speedup                        : {speedup:.2f}x faster")
    print(f"Latency Reduction              : {pct_reduction:.2f}%")

if __name__ == '__main__':
    run_benchmark()
