# Bolt's Journal

## 2025-05-10 - NumPy Exponentiation Overhead
**Learning:** `np.power(A - B, 2)` incurs generic C-level exponentiation loop overhead in NumPy compared to element-wise squaring `(A - B) ** 2`, resulting in ~2.4x execution time difference.
**Action:** Prefer `(A - B) ** 2` or direct multiplication over `np.power(..., 2)` when calculating squared differences in NumPy array operations.
