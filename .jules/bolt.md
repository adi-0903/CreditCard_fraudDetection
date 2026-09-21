## 2026-03-31 - Fast NumPy Squaring vs `np.power`
**Learning:** `np.power(arr, 2)` calls C-level generic power functions which incur overhead compared to native Python exponentiation `arr ** 2` or `np.square(arr)` on NumPy arrays. Using `(A - B)**2` instead of `np.power(A - B, 2)` delivers ~2x speedup for element-wise squaring in array computations.
**Action:** Prefer `arr ** 2` or `np.square(arr)` over `np.power(arr, 2)` for element-wise squaring in NumPy.
