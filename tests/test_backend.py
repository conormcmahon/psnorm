import numpy as np
import pytest

from psnorm import backend


def test_resolve_device_explicit_cpu_always_succeeds():
    assert backend.resolve_device("cpu") == "cpu"


def test_resolve_device_invalid_string_raises():
    with pytest.raises(ValueError):
        backend.resolve_device("tpu")


def test_resolve_device_auto_falls_back_to_cpu_when_no_gpu(monkeypatch):
    monkeypatch.setattr(backend, "gpu_available", lambda: False)
    assert backend.resolve_device("auto") == "cpu"


def test_resolve_device_auto_picks_gpu_when_available(monkeypatch):
    monkeypatch.setattr(backend, "gpu_available", lambda: True)
    assert backend.resolve_device("auto") == "gpu"


def test_resolve_device_explicit_gpu_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(backend, "gpu_available", lambda: False)
    with pytest.raises(RuntimeError, match="no usable GPU"):
        backend.resolve_device("gpu")


def test_resolve_device_explicit_gpu_succeeds_when_available(monkeypatch):
    monkeypatch.setattr(backend, "gpu_available", lambda: True)
    assert backend.resolve_device("gpu") == "gpu"


def test_gpu_available_false_when_cupy_not_installed():
    # This sandbox has no cupy installed and no GPU -- gpu_available() must
    # degrade to False, not raise, so resolve_device("auto") always has a
    # working CPU fallback available in exactly this environment.
    assert backend.gpu_available() is False


def test_get_array_module_cpu_is_numpy():
    assert backend.get_array_module("cpu") is np


def test_get_array_module_rejects_unresolved_auto():
    with pytest.raises(ValueError):
        backend.get_array_module("auto")


def test_to_host_is_noop_for_numpy_arrays():
    arr = np.array([1.0, 2.0, 3.0])
    out = backend.to_host(arr)
    np.testing.assert_array_equal(out, arr)
    assert isinstance(out, np.ndarray)
