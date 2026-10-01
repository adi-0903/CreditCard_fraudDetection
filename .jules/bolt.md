## 2026-09-18 - Optimize Element-wise Squaring in NumPy arrays
**Learning:** Using `np.power(arr, 2)` incurs general power routine overhead in C/NumPy compared to direct exponentiation `(arr) ** 2` or `np.square(arr)`. Direct exponentiation is ~1.8x to 2x faster for MSE calculations on large arrays (~56,962 rows x 29 features).
**Action:** Replace `np.power(..., 2)` with `(...) ** 2` or `np.square(...)` when calculating MSE or element-wise squaring on NumPy arrays.
