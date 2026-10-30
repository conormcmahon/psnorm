import os

import pytest

from psnorm import pipeline


def _flaky(item, state_dir):
    """Fails the first time it is called for an odd item, as a transient
    I/O error would; module-level so spawned workers can import it."""
    marker = os.path.join(state_dir, f"seen_{item}")
    if item % 2 and not os.path.exists(marker):
        open(marker, "w").close()
        raise OSError(9, "Bad file descriptor")
    return item * 10


def _always_fails(item, state_dir):
    raise OSError(9, "Bad file descriptor")


@pytest.fixture(autouse=True)
def no_delay(monkeypatch):
    monkeypatch.setattr(pipeline, "RETRY_DELAY_S", 0)


@pytest.mark.parametrize("workers", [1, 2])
def test_transient_errors_are_retried(tmp_path, workers):
    seen = []
    results = pipeline._map_parallel(_flaky, list(range(6)), workers, str(tmp_path),
                                     on_result=lambda item, r: seen.append(item), log=lambda m: None)
    assert results == [0, 10, 20, 30, 40, 50]
    assert sorted(seen) == list(range(6))    # every item reported once


def test_recovery_command_runs_before_retry(tmp_path, monkeypatch):
    flag = tmp_path / "recovered"
    monkeypatch.setenv("PSNORM_IO_RECOVERY_CMD", f"touch {flag}")
    pipeline._map_parallel(_flaky, [1], 1, str(tmp_path), log=lambda m: None)
    assert flag.exists()


def test_persistent_error_is_raised(tmp_path):
    with pytest.raises(OSError, match="Bad file descriptor"):
        pipeline._map_parallel(_always_fails, [1, 2], 2, str(tmp_path), retries=1, log=lambda m: None)
