import numpy as np
import pywt

from src.features.wavelet import WaveletFeatures as WF


def test_db7_cannot_decompose_a_20_sample_window():
    """
    Pins the reason db2 (not Atzori's db7) was chosen - see
    config/settings.py's WAVELET docstring and src/features/report.py.
    """
    assert pywt.dwt_max_level(20, pywt.Wavelet("db7").dec_len) == 0


def test_db2_permits_a_2level_decomposition_of_a_20_sample_window():
    assert pywt.dwt_max_level(20, pywt.Wavelet("db2").dec_len) == 2


def test_mdwt_output_shape_matches_level_plus_one_bands():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(5, 3, 20))  # 5 windows, 3 channels, 20 samples

    bands = WF.mdwt(w, "db2", 2)

    assert bands.shape == (5, 3, 3)  # level=2 -> 3 bands: A2, D2, D1


def test_mdwt_bands_are_non_negative_energies():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(5, 3, 20))

    bands = WF.mdwt(w, "db2", 2)

    assert np.all(bands >= 0)  # sum of absolute values


def test_band_names_match_level():
    assert WF.band_names(2) == ["A2", "D2", "D1"]
    assert WF.band_names(1) == ["A1", "D1"]


def test_mdwt_uses_symmetric_boundary_mode_explicitly():
    """
    Pins mode="symmetric" as an explicit choice, not an implicit library
    default - see src/features/wavelet.py's mdwt() docstring. If a future
    PyWavelets release changed its default mode, this test would still
    pass (mdwt() no longer depends on the default) while a competing
    "asymmetric" comparison below demonstrates the two modes really do
    produce different output on this input, so the pin is meaningful.
    """
    rng = np.random.default_rng(0)
    w = rng.normal(size=(3, 2, 20))

    explicit_symmetric = np.stack(
        [np.sum(np.abs(band), axis=-1) for band in pywt.wavedec(w, "db2", level=2, axis=-1, mode="symmetric")],
        axis=-1,
    )
    asymmetric = np.stack(
        [np.sum(np.abs(band), axis=-1) for band in pywt.wavedec(w, "db2", level=2, axis=-1, mode="periodization")],
        axis=-1,
    )

    assert np.array_equal(WF.mdwt(w, "db2", 2), explicit_symmetric)
    assert not np.array_equal(explicit_symmetric, asymmetric)


def test_mdwt_on_a_constant_window_has_zero_detail_energy():
    """A flat window has no detail (high-frequency) content - its detail
    bands should be all zero, unlike its approximation band."""
    w = np.full((1, 1, 20), 3.0)

    bands = WF.mdwt(w, "db2", 2)
    band_names = WF.band_names(2)

    for name, value in zip(band_names, bands[0, 0]):
        if name.startswith("D"):
            assert np.isclose(value, 0.0), f"expected zero detail energy for {name}"
        else:
            assert value > 0
