import numpy as np

from src.features.time_domain import TimeDomainFeatures as TDF


def test_rms_matches_hand_computed_value():
    w = np.array([[[1.0, -1.0, 2.0, -2.0]]])  # (1 window, 1 channel, 4 samples)
    assert np.isclose(TDF.rms(w)[0, 0], np.sqrt((1 + 1 + 4 + 4) / 4))


def test_mav_matches_hand_computed_value():
    w = np.array([[[1.0, -1.0, 2.0, -2.0]]])
    assert np.isclose(TDF.mav(w)[0, 0], 1.5)


def test_wl_matches_hand_computed_value():
    w = np.array([[[1.0, -1.0, 2.0, -2.0]]])
    # |diff| = |-2|, |3|, |-4| = 2 + 3 + 4 = 9
    assert np.isclose(TDF.wl(w)[0, 0], 9.0)


def test_iemg_equals_window_length_times_mav():
    """On a fixed-length window IEMG is exactly window_len * MAV - see the
    collinearity note in src/features/report.py."""
    w = np.array([[[1.0, -1.0, 2.0, -2.0]]])
    assert np.isclose(TDF.iemg(w)[0, 0], 4 * TDF.mav(w)[0, 0])


def test_mcr_is_zero_on_non_negative_signal_without_demeaning():
    """
    Confirms the design rationale in src/features/time_domain.py's module
    docstring: a plain crossing test on DB1's non-negative raw envelope
    is structurally 0.
    """
    non_negative = np.array([[[0.1, 0.2, 0.05, 0.3, 0.15]]])
    a, b = non_negative[..., :-1], non_negative[..., 1:]
    assert np.sum((a * b) < 0) == 0


def test_mcr_is_nonzero_after_window_demeaning():
    """The regression this feature exists to prevent: after subtracting
    the window's own mean, a non-negative signal produces real crossings."""
    non_negative = np.array([[[0.1, 0.2, 0.05, 0.3, 0.15]]])
    assert TDF.mcr(non_negative)[0, 0] > 0


def test_mcr_matches_hand_computed_value():
    w = np.array([[[1.0, -1.0, 2.0, -2.0]]])  # mean 0, already demeaned
    # products of consecutive samples: 1*-1, -1*2, 2*-2 -> all negative -> 3 crossings
    assert TDF.mcr(w)[0, 0] == 3


def test_ssc_matches_hand_computed_value():
    w = np.array([[[1.0, -1.0, 2.0, -2.0]]])
    # diff = [-2, 3, -4]; consecutive products: -2*3, 3*-4 -> both negative -> 2
    assert TDF.ssc(w)[0, 0] == 2


def test_ssc_is_invariant_to_affine_normalization():
    """SSC tests the sign of the first difference, which an affine
    transform with positive scale (z-scoring) cannot flip - confirmed
    bit-exact identical on real data during design validation."""
    w = np.array([[[1.0, -1.0, 2.0, -2.0, 0.5]]])
    scaled = w * 3.0 + 5.0  # positive-scale affine transform, like z-scoring

    assert np.array_equal(TDF.ssc(w), TDF.ssc(scaled))


def test_hist_counts_sum_to_window_length():
    edges = [-0.5, 0.0, 0.5, 1.5]
    w = np.array([[[-2.0, -0.3, 0.2, 0.6, 2.0]]])  # 5 samples

    counts = TDF.hist(w, edges)

    assert counts.shape == (1, 1, 5)
    assert counts.sum() == 5


def test_hist_places_values_in_expected_bins():
    edges = [-0.5, 0.0, 0.5, 1.5]
    # one value strictly in each of the 5 bins: (-inf,-0.5], (-0.5,0], (0,0.5], (0.5,1.5], (1.5,inf)
    w = np.array([[[-2.0, -0.3, 0.2, 0.6, 2.0]]])

    counts = TDF.hist(w, edges)

    assert np.array_equal(counts[0, 0], [1, 1, 1, 1, 1])


def test_time_domain_functions_vectorize_over_multiple_windows_and_channels():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(7, 4, 20))  # 7 windows, 4 channels, 20 samples

    assert TDF.rms(w).shape == (7, 4)
    assert TDF.mav(w).shape == (7, 4)
    assert TDF.wl(w).shape == (7, 4)
    assert TDF.iemg(w).shape == (7, 4)
    assert TDF.mcr(w).shape == (7, 4)
    assert TDF.ssc(w).shape == (7, 4)
    assert TDF.hist(w, [-0.5, 0.0, 0.5, 1.5]).shape == (7, 4, 5)
