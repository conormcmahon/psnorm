"""GPU/CPU array-backend selection for the elementwise-heavy IR-MAD math
(irmad.py, normalize.py's classify step). The numeric core is written
against a generic `xp` (array module) that is either `numpy` (always
available) or `cupy` (optional, GPU-resident arrays with a numpy-compatible
API) -- everything outside this module stays backend-agnostic by taking
`xp`/`device` as explicit parameters rather than importing numpy or cupy
directly.

Small, fixed-size linear algebra (the B x B canonical-correlation eigensolve
in irmad._solve_canonical_correlation, B = number of bands) deliberately
*never* moves to GPU: B is tiny (4-8), so a GPU solve there would be
dominated by kernel-launch latency rather than compute. That step -- and
`scipy.stats.chi2.sf`, which has no reliably-available GPU equivalent --
always runs on host, on small (B x B, or (n_pixels,)) arrays converted via
`to_host`. Only the large (n_pixels, n_bands) per-pixel work stays resident
on whichever device `device` resolves to.
"""

from __future__ import annotations

import numpy as np


def gpu_available() -> bool:
    """True if CuPy is importable *and* reports at least one visible,
    usable CUDA device. Cheap to call repeatedly (no persistent state)."""
    try:
        import cupy
    except ImportError:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        # Any CUDA-runtime-level failure (no driver, driver/toolkit
        # mismatch, permissions) means "not usable", not "crash the caller".
        return False


def resolve_device(device: str = "auto") -> str:
    """Resolve a user-facing `device` request to an actual "cpu" or "gpu".

    "auto" (the default) picks "gpu" if one is usable, else "cpu" --
    silently, since this is the whole point of "auto". An explicit "gpu"
    request that can't be satisfied raises instead of silently downgrading:
    a caller who asked for GPU explicitly likely wants to know their setup
    is broken, not silently get CPU-speed results. An explicit "cpu"
    request always succeeds (matches the pre-GPU-support behavior exactly).
    """
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"device must be 'auto', 'cpu', or 'gpu', got {device!r}")
    if device == "cpu":
        return "cpu"
    if device == "gpu":
        if not gpu_available():
            raise RuntimeError(
                "device='gpu' was requested explicitly, but no usable GPU was "
                "found (CuPy not installed, or no CUDA device visible/usable). "
                "Install cupy matching your CUDA toolkit, or pass device='auto' "
                "or device='cpu' to run on CPU instead."
            )
        return "gpu"
    return "gpu" if gpu_available() else "cpu"  # auto


def get_array_module(resolved_device: str):
    """The array module (`numpy` or `cupy`) for an already-*resolved*
    device ("cpu"/"gpu" -- pass through resolve_device first, not "auto")."""
    if resolved_device == "cpu":
        return np
    if resolved_device == "gpu":
        import cupy
        return cupy
    raise ValueError(f"resolved_device must be 'cpu' or 'gpu', got {resolved_device!r} (did you forget resolve_device()?)")


def to_host(array) -> np.ndarray:
    """`array` as a plain host/NumPy array regardless of which backend
    produced it -- a no-op for NumPy arrays, `cupy.asnumpy` for CuPy ones."""
    module_root = type(array).__module__.split(".")[0]
    if module_root == "cupy":
        import cupy
        return cupy.asnumpy(array)
    return np.asarray(array)
