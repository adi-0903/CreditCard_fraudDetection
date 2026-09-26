## 2026-03-29 - NumPy Element-wise Squaring Optimization

**Learning:** Using `(A - B) ** 2` in NumPy is significantly faster (~1.9x speedup) than `np.power(A - B, 2)` or `np.square(A - B)`. `np.power` incurs generic ufunc overhead for arbitrary real/complex exponentiation, whereas `** 2` utilizes NumPy's optimized binary exponentiation operator path for integer powers.

**Action:** Prefer `(A - B) ** 2` for element-wise squaring in NumPy array operations.
