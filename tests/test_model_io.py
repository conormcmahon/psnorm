import numpy as np

from psnorm import model_io
from psnorm.irmad import IrMadFit


def _make_fit():
    return IrMadFit(
        rho=np.array([0.1, 0.4, 0.8, 0.95]),
        A=np.arange(16, dtype=float).reshape(4, 4),
        B=np.arange(16, dtype=float).reshape(4, 4) * -1.0,
        means1=np.array([100.0, 200.0, 300.0, 400.0]),
        means2=np.array([105.0, 205.0, 305.0, 405.0]),
        sigma=np.sqrt(2.0 * (1.0 - np.array([0.1, 0.4, 0.8, 0.95]))),
        n_iterations=7,
        converged=True,
    )


def test_save_and_load_irmad_fit_round_trips(tmp_path):
    fit = _make_fit()
    path = str(tmp_path / "fit.json")
    model_io.save_irmad_fit(fit, path)

    loaded = model_io.load_irmad_fit(path)
    np.testing.assert_array_equal(loaded.rho, fit.rho)
    np.testing.assert_array_equal(loaded.A, fit.A)
    np.testing.assert_array_equal(loaded.B, fit.B)
    np.testing.assert_array_equal(loaded.means1, fit.means1)
    np.testing.assert_array_equal(loaded.means2, fit.means2)
    np.testing.assert_array_equal(loaded.sigma, fit.sigma)
    assert loaded.n_iterations == fit.n_iterations
    assert loaded.converged == fit.converged


def test_irmad_fit_is_valid_true_for_saved_fit(tmp_path):
    path = str(tmp_path / "fit.json")
    model_io.save_irmad_fit(_make_fit(), path)
    assert model_io.irmad_fit_is_valid(path)


def test_irmad_fit_is_valid_false_for_missing_or_malformed(tmp_path):
    assert not model_io.irmad_fit_is_valid(str(tmp_path / "missing.json"))

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not valid json")
    assert not model_io.irmad_fit_is_valid(str(bad_path))
