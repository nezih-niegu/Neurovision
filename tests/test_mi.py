"""Estimator checks. For a bivariate Gaussian the true MI is known in closed
form, MI = -0.5*log2(1 - r^2), which gives us something real to test against."""

import numpy as np
import pytest

from neurovision.core import (binned_mi_matrix, copnorm, effective_n,
                             gcmi_matrix, gcmi_null_matrix)

N = 20000


def gaussian_pair(r, seed=0, n=N):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = r * a + np.sqrt(1 - r ** 2) * rng.standard_normal(n)
    return np.vstack([a, b])


@pytest.mark.parametrize("r", [0.0, 0.3, 0.6, 0.9])
def test_gcmi_matches_analytic_mi(r):
    truth = -0.5 * np.log2(1 - r ** 2)
    assert gcmi_matrix(gaussian_pair(r))[0, 1] == pytest.approx(truth, abs=0.02)


@pytest.mark.parametrize("r", [0.3, 0.6, 0.9])
def test_binned_is_close_but_biased_low(r):
    truth = -0.5 * np.log2(1 - r ** 2)
    est = binned_mi_matrix(gaussian_pair(r), bins=16)[0, 1]
    assert est == pytest.approx(truth, abs=0.12)
    assert est <= truth + 0.02


def test_gcmi_invariant_to_monotone_transforms():
    x = gaussian_pair(0.8)
    y = np.vstack([np.exp(x[0]), x[1] ** 3])
    assert gcmi_matrix(x)[0, 1] == pytest.approx(gcmi_matrix(y)[0, 1], abs=1e-9)


def test_independent_channels_give_near_zero():
    rng = np.random.default_rng(3)
    assert np.nanmax(gcmi_matrix(rng.standard_normal((8, N)))) < 0.01


def test_matrix_is_symmetric_with_nan_diagonal():
    m = gcmi_matrix(np.random.default_rng(4).standard_normal((6, 5000)))
    assert np.allclose(m, m.T, equal_nan=True)
    assert np.isnan(np.diag(m)).all()


def test_copnorm_gives_normal_marginals():
    z = copnorm(np.random.default_rng(5).exponential(size=(3, 5000)))
    assert np.abs(z.mean(axis=1)).max() < 0.02
    assert np.abs(z.std(axis=1) - 1).max() < 0.05


def test_surrogates_match_the_floor_for_uncoupled_autocorrelated_noise():
    """Independent AR(1) channels still show non-zero MI; the surrogate null
    should sit at that same level, leaving no excess."""
    rng = np.random.default_rng(6)
    n, n_ch = 15000, 5
    x = np.zeros((n_ch, n))
    for i in range(n_ch):
        e = rng.standard_normal(n)
        for t in range(1, n):
            x[i, t] = 0.98 * x[i, t - 1] + e[t]
    excess = np.nanmean(gcmi_matrix(x) - gcmi_null_matrix(x, 10, rng=0))
    assert abs(excess) < 0.01


def test_surrogates_leave_excess_when_channels_are_coupled():
    x = gaussian_pair(0.7, n=8000)
    assert np.nanmean(gcmi_matrix(x) - gcmi_null_matrix(x, 10, rng=0)) > 0.3


def test_effective_n_shrinks_with_autocorrelation():
    rng = np.random.default_rng(7)
    white = rng.standard_normal((2, 10000))
    ar = np.cumsum(rng.standard_normal((2, 10000)), axis=1)
    assert effective_n(white) > 8000
    assert effective_n(ar) < 500


# --------------------------------------------------------------------------
# KSG (k-nearest-neighbour) estimator and windowed analysis
# --------------------------------------------------------------------------
from neurovision.core import (ksg_mi_matrix, reject_windows, window_bounds,
                              windowed_mi)

WIN = 150  # 600 ms at 250 Hz


@pytest.mark.parametrize("r", [0.0, 0.6, 0.9])
def test_ksg_recovers_analytic_mi_in_a_600ms_window(r):
    """Averaged over repeats, KSG should land on the true value even though a
    single 600 ms window holds only 150 samples."""
    rng = np.random.default_rng(11)
    est = []
    for _ in range(60):
        a = rng.standard_normal(WIN)
        b = r * a + np.sqrt(1 - r ** 2) * rng.standard_normal(WIN)
        est.append(ksg_mi_matrix(np.vstack([a, b]), rng=rng)[0, 1])
    assert np.mean(est) == pytest.approx(-0.5 * np.log2(1 - r ** 2), abs=0.04)


def test_ksg_sees_non_monotone_dependence_that_gcmi_misses():
    """The reason to pay for KSG: y = x^2 is a perfect dependence with zero
    rank correlation, so the copula estimator reports nothing."""
    rng = np.random.default_rng(12)
    a = rng.standard_normal(4000)
    x = np.vstack([a, a ** 2 + 0.25 * rng.standard_normal(4000)])
    assert ksg_mi_matrix(x, rng=0)[0, 1] > 1.0
    assert gcmi_matrix(x)[0, 1] < 0.01


def test_ksg_is_symmetric_with_nan_diagonal():
    m = ksg_mi_matrix(np.random.default_rng(13).standard_normal((5, 300)), rng=0)
    assert np.allclose(m, m.T, equal_nan=True)
    assert np.isnan(np.diag(m)).all()


def test_window_bounds_are_contiguous_and_non_overlapping_by_default():
    b = window_bounds(30000, 250.0, 0.6)
    assert len(b) == 200
    assert (b[:, 1] - b[:, 0] == WIN).all()
    assert (b[1:, 0] == b[:-1, 1]).all()


def test_window_step_controls_overlap():
    assert len(window_bounds(30000, 250.0, 0.6, step_s=0.3)) == 399


def test_windowed_mi_returns_one_matrix_per_window():
    rng = np.random.default_rng(14)
    x = rng.standard_normal((6, WIN * 12))
    stack, bounds = windowed_mi(x, 250.0, 0.6, estimator="ksg", rng=0)
    assert stack.shape == (12, 6, 6)
    assert len(bounds) == 12


def test_windowed_mi_tracks_a_coupling_change_partway_through():
    """Coupling switches on halfway; windowed MI should show it, and the
    whole-recording average should sit between the two regimes."""
    rng = np.random.default_rng(15)
    n = WIN * 20
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    b[n // 2:] = 0.9 * a[n // 2:] + np.sqrt(1 - 0.81) * b[n // 2:]
    stack, _ = windowed_mi(np.vstack([a, b]), 250.0, 0.6, estimator="ksg", rng=0)
    early = np.nanmean(stack[:10, 0, 1])
    late = np.nanmean(stack[10:, 0, 1])
    assert late - early > 0.8


def test_rejection_drops_injected_artefact_windows_only():
    rng = np.random.default_rng(16)
    x = rng.standard_normal((8, 30000))
    b = window_bounds(30000, 250.0, 0.6)
    assert reject_windows(x, b, z=6.0).all()
    for w in (12, 77, 150):
        x[:, w * WIN:(w + 1) * WIN] *= 25
    keep = reject_windows(x, b, z=6.0)
    assert set(np.where(~keep)[0]) == {12, 77, 150}
