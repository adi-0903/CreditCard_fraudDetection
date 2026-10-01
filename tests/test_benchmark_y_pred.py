import pytest
from benchmark_y_pred import run_benchmark

def test_run_benchmark(capsys):
    """Test that run_benchmark executes without error and outputs expected benchmark results."""
    run_benchmark()
    captured = capsys.readouterr()
    assert "=== Performance Benchmark Results ===" in captured.out
    assert "Baseline (List Comprehension)" in captured.out
    assert "Optimized (Vectorized .astype)" in captured.out
    assert "Speedup" in captured.out
