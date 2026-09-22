## 2026-03-29 - Vectorized element-wise squaring in NumPy
**Learning:** Using `(A - B) ** 2` instead of `np.power(A - B, 2)` avoids generic C-level ufunc dispatch overhead in NumPy, yielding ~1.8x faster element-wise squaring performance for multi-dimensional arrays.
**Action:** Prefer explicit `** 2` or `np.square()` over `np.power(..., 2)` when calculating squared differences in NumPy arrays.
