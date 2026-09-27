## 2026-03-31 - NumPy Exponentiation Overhead for Squaring
**Learning:** `np.power(A - B, 2)` invokes generic binary exponentiation ufuncs in NumPy, which is roughly ~2x slower than element-wise squaring `(A - B) ** 2`.
**Action:** Replace `np.power(..., 2)` with direct `** 2` operator when calculating squared differences (like MSE) in NumPy array operations.
