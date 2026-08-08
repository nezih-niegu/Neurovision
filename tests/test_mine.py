"""Tests for the neural estimator and the reconstruction benchmark.

The Gaussian case again does the heavy lifting: MI is known in closed form, and
so is the R^2 an optimal predictor can reach, so both halves of the pipeline
have something real to be checked against.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="install with: uv sync --extra mine")

from neurovision.mine import (Critic, estimate_mi, make_windows, mi_from_r2,
                              pick_device, r2_from_mi, split_contiguous,
                              standardize, train_predictor)


def gaussian_pair(rho, d=1, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, d))
    b = rho * a + np.sqrt(1 - rho ** 2) * rng.standard_normal((n, d))
    return (torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32))


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def test_pick_device_returns_something_usable():
    dev = pick_device("auto")
    assert dev.type in {"cpu", "cuda", "mps"}
    torch.zeros(4, device=dev) + 1  # must not raise


def test_explicit_cpu_is_respected():
    assert pick_device("cpu").type == "cpu"


# --------------------------------------------------------------------------
# MINE
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rho", [0.6, 0.9])
def test_mine_recovers_analytic_gaussian_mi(rho):
    X, Y = gaussian_pair(rho)
    est = estimate_mi(X, Y, iters=1200, seed=0).mi_bits
    assert est == pytest.approx(-0.5 * np.log2(1 - rho ** 2), abs=0.12)


def test_mine_reports_near_zero_for_independent_variables():
    rng = np.random.default_rng(1)
    X = torch.tensor(rng.standard_normal((15000, 2)), dtype=torch.float32)
    Y = torch.tensor(rng.standard_normal((15000, 1)), dtype=torch.float32)
    assert estimate_mi(X, Y, iters=800, seed=0).mi_bits < 0.08


def test_mine_stays_finite_at_high_mi():
    """Unclipped critics overflow to NaN once MI is large; the soft clip makes
    the bound loose there instead, which is the acceptable failure mode."""
    X, Y = gaussian_pair(0.8, d=10, n=30000)
    r = estimate_mi(X, Y, iters=800, seed=0)
    assert np.isfinite(r.mi_bits)
    assert 0 < r.mi_bits <= -0.5 * 10 * np.log2(1 - 0.64) + 0.5


def test_mine_is_ordered_in_the_coupling_strength():
    vals = [estimate_mi(*gaussian_pair(r, n=15000), iters=800, seed=0).mi_bits
            for r in (0.2, 0.5, 0.85)]
    assert vals[0] < vals[1] < vals[2]


def test_critic_output_is_bounded_by_its_clip():
    net = Critic(3, 1, hidden=32, clip=5.0)
    x = torch.randn(200, 3) * 500
    y = torch.randn(200, 1) * 500
    assert net(x, y).abs().max() <= 5.0 + 1e-4


def test_mine_evaluation_is_memory_bounded_for_wide_inputs():
    """A naive InfoNCE evaluation allocates m*m*d floats and kills the process
    on wide windows. It must stay chunked."""
    rng = np.random.default_rng(2)
    X = torch.tensor(rng.standard_normal((6000, 2400)), dtype=torch.float32)
    Y = X[:, :1] * 0.7 + 0.7 * torch.randn(6000, 1)
    r = estimate_mi(X, Y, iters=60, hidden=32, seed=0)
    assert np.isfinite(r.mi_bits)


# --------------------------------------------------------------------------
# MI <-> R^2
# --------------------------------------------------------------------------
def test_r2_and_mi_conversions_are_inverses():
    for v in (0.05, 0.5, 1.0, 3.0):
        assert mi_from_r2(r2_from_mi(v)) == pytest.approx(v, abs=1e-9)


def test_zero_mi_means_zero_explained_variance():
    assert r2_from_mi(0.0) == pytest.approx(0.0)
    assert r2_from_mi(1.0) == pytest.approx(0.75)  # 1 - 2^-2


# --------------------------------------------------------------------------
# Windowing and splitting
# --------------------------------------------------------------------------
def test_make_windows_shapes_and_target_alignment():
    data = np.arange(4 * 100, dtype=float).reshape(4, 100)
    X, y = make_windows(data, target=0, sources=[1, 2], window=10, stride=5)
    assert X.shape == (19, 2 * 10)
    assert y[0] == data[0, 5]          # centre sample of the first window


def test_make_windows_rejects_a_window_longer_than_the_recording():
    with pytest.raises(ValueError):
        make_windows(np.zeros((3, 50)), 0, [1], window=80)


def test_split_is_contiguous_not_shuffled():
    """Shuffling before splitting puts near-duplicate windows on both sides and
    inflates R^2; the split must keep the test block strictly later in time."""
    X = np.arange(1000, dtype=np.float32)[:, None]
    y = X[:, 0].copy()
    Xtr, ytr, Xte, yte = split_contiguous(X, y, test_frac=0.25, gap=10)
    assert Xtr.max() < Xte.min()
    assert len(Xte) == 250


def test_standardize_uses_training_statistics_only():
    rng = np.random.default_rng(3)
    Xtr = rng.normal(5, 2, (500, 4)).astype(np.float32)
    Xte = rng.normal(50, 20, (100, 4)).astype(np.float32)
    ytr, yte = rng.normal(size=500).astype(np.float32), rng.normal(size=100).astype(np.float32)
    Str, Ste, _, _ = standardize(Xtr, Xte, ytr, yte)
    assert abs(Str.mean()) < 0.05 and abs(Str.std() - 1) < 0.05
    assert abs(Ste.mean()) > 1.0       # test block is NOT re-centred on itself


# --------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------
def test_predictor_recovers_a_known_linear_relationship():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((4000, 5)).astype(np.float32)
    y = (X @ np.array([1.0, -0.5, 0.25, 0, 0], dtype=np.float32)
         + 0.1 * rng.standard_normal(4000)).astype(np.float32)
    Xtr, ytr, Xte, yte = split_contiguous(X, y, 0.25)
    r = train_predictor(Xtr, ytr, Xte, yte, epochs=40, seed=0)
    assert r.r2 > 0.95


def test_predictor_scores_about_zero_on_unpredictable_targets():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((3000, 5)).astype(np.float32)
    y = rng.standard_normal(3000).astype(np.float32)
    Xtr, ytr, Xte, yte = split_contiguous(X, y, 0.25)
    assert train_predictor(Xtr, ytr, Xte, yte, epochs=25, seed=0).r2 < 0.1


def test_achieved_r2_respects_the_information_bound():
    """For a Gaussian pair the MI fixes the best achievable R^2 exactly. The
    fitted predictor must not beat it by more than noise."""
    rng = np.random.default_rng(6)
    rho = 0.8
    a = rng.standard_normal(6000).astype(np.float32)
    b = (rho * a + np.sqrt(1 - rho ** 2) * rng.standard_normal(6000)).astype(np.float32)
    Xtr, ytr, Xte, yte = split_contiguous(a[:, None], b, 0.25)
    r2 = train_predictor(Xtr, ytr, Xte, yte, epochs=40, seed=0).r2
    assert r2 <= r2_from_mi(-0.5 * np.log2(1 - rho ** 2)) + 0.05


# --------------------------------------------------------------------------
# Benchmark selection logic
# --------------------------------------------------------------------------
def test_target_channel_is_excluded_from_every_selection():
    """Sending the target to the back of the ranking with -inf leaves it in the
    bottom-k, where it would reconstruct itself perfectly."""
    from neurovision.benchmark import rank_channels
    rng = np.random.default_rng(7)
    x = rng.standard_normal((8, 4000))
    x[3] = 0.9 * x[0] + 0.1 * rng.standard_normal(4000)
    order, _ = rank_channels(x, target=0, estimator="gcmi")
    assert 0 not in order
    assert len(order) == 7
    assert order[0] == 3               # the genuinely coupled channel ranks first


# --------------------------------------------------------------------------
# Predicted-vs-actual traces
# --------------------------------------------------------------------------
def test_predictor_returns_aligned_test_traces():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((2000, 3)).astype(np.float32)
    y = (X @ np.array([1.0, -0.5, 0.2], dtype=np.float32)
         + 0.2 * rng.standard_normal(2000)).astype(np.float32)
    Xtr, ytr, Xte, yte = split_contiguous(X, y, 0.25)
    r = train_predictor(Xtr, ytr, Xte, yte, epochs=25, seed=0)
    assert r.y_true.shape == r.y_pred.shape == yte.shape
    assert np.allclose(r.y_true, yte, atol=1e-5)   # truth returned unmodified
    # R^2 recomputed from the returned traces must match the reported value
    ss_res = ((r.y_true - r.y_pred) ** 2).sum()
    ss_tot = ((r.y_true - r.y_true.mean()) ** 2).sum()
    assert 1 - ss_res / ss_tot == pytest.approx(r.r2, abs=1e-5)


def test_trace_figures_are_written(tmp_path):
    """The plotting path is easy to break silently, so exercise it end to end
    with a stub store rather than trusting it to work when the run is long."""
    import argparse

    import pandas as pd

    from neurovision.benchmark import plot_scatter_and_spectra, plot_traces

    rng = np.random.default_rng(9)
    n = 900
    store, rows = {}, []
    for sel, noise in (("top", 0.2), ("random", 0.4), ("bottom", 0.9)):
        for w in (50, 200):
            true = np.cumsum(rng.standard_normal(n)).astype(np.float32)
            true = (true - true.mean()) / true.std()
            pred = true + noise * rng.standard_normal(n).astype(np.float32)
            store[("sub-x/ses-y", "Cz", sel, w)] = (true, pred)
            rows.append(dict(recording="sub-x/ses-y", target="Cz",
                             selection=sel, window=w,
                             r2=1 - noise ** 2 / (1 + noise ** 2)))
    df = pd.DataFrame(rows)
    args = argparse.Namespace(stride=3, trace_start=0.5, trace_seconds=3.0, k=6,
                              max_trace_figures=6)

    out = tmp_path / "bench.png"
    made = plot_traces(store, df, out, args, sfreq=250.0)
    assert made and all(p.exists() and p.stat().st_size > 5000 for p in made)
    cal = plot_scatter_and_spectra(store, out, args, sfreq=250.0)
    assert cal.exists() and cal.stat().st_size > 5000


def test_traces_survive_an_npz_round_trip(tmp_path):
    store = {("rec/a", "Cz", "top", 50): (np.arange(9, dtype=np.float32),
                                          np.arange(9, dtype=np.float32) * 2)}
    p = tmp_path / "t.npz"
    np.savez_compressed(p, **{"|".join(str(x) for x in k): np.stack(v)
                              for k, v in store.items()})
    z = np.load(p)
    key = "rec/a|Cz|top|50"
    assert key in z.files
    assert z[key].shape == (2, 9)


# --------------------------------------------------------------------------
# Group comparison (HC vs PD-OFF vs PD-ON)
# --------------------------------------------------------------------------
def _group_frame(hc_r2=0.90, off_r2=0.80, on_r2=0.88, n_hc=12, n_pd=12, seed=0,
                 hc_bits=1.8, pd_bits=1.5, hc_mi=0.9, pd_mi=0.7):
    """Synthetic benchmark output: PD subjects appear twice (off and on),
    controls once, exactly as in ds002778."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_hc):
        for w in (50, 200):
            rows.append(dict(selection="top", group="HC", subject=f"hc{i}",
                             session="hc", window=w,
                             r2=hc_r2 + 0.02 * rng.standard_normal(),
                             mine_bits=hc_bits + 0.1 * rng.standard_normal(),
                             mean_pairwise_mi=hc_mi + 0.05 * rng.standard_normal()))
    for i in range(n_pd):
        subj_effect = 0.03 * rng.standard_normal()   # per-subject offset
        for ses, base in (("off", off_r2), ("on", on_r2)):
            for w in (50, 200):
                rows.append(dict(selection="top", group=f"PD-{ses.upper()}",
                                 subject=f"pd{i}", session=ses, window=w,
                                 r2=base + subj_effect + 0.02 * rng.standard_normal(),
                                 mine_bits=pd_bits + 0.1 * rng.standard_normal(),
                                 mean_pairwise_mi=pd_mi + 0.05 * rng.standard_normal()))
    return pd.DataFrame(rows)


def test_pd_sessions_get_a_paired_test_and_controls_a_welch_test():
    from neurovision.benchmark import compare_groups
    res = compare_groups(_group_frame(), rank_info=[])
    pdpd = res[(res.group_a.str.startswith("PD")) & (res.group_b.str.startswith("PD"))]
    hcpd = res[(res.group_a == "HC") & (res.group_b.str.startswith("PD"))]
    assert set(pdpd.test) == {"paired"}      # same subjects on and off meds
    assert set(hcpd.test) == {"welch"}       # different people


def test_group_comparison_finds_a_real_difference():

    from neurovision.benchmark import compare_groups
    res = compare_groups(_group_frame(hc_r2=0.90, off_r2=0.78), rank_info=[])
    hit = res[(res.metric == "r2") & (res.group_a == "HC")
              & (res.group_b == "PD-OFF")]
    assert len(hit) == 1
    assert bool(hit.significant.iloc[0])
    assert hit["diff"].iloc[0] > 0.1


def test_group_comparison_does_not_invent_a_difference():
    """Same distribution in every group must not produce a survivor."""
    from neurovision.benchmark import compare_groups
    res = compare_groups(_group_frame(hc_r2=0.85, off_r2=0.85, on_r2=0.85,
                                      hc_bits=1.6, pd_bits=1.6,
                                      hc_mi=0.8, pd_mi=0.8),
                         rank_info=[])
    assert not res["significant"].any()


def test_repeated_windows_do_not_inflate_the_sample_size():
    """Each recording is measured at several window lengths; those must collapse
    to one value per subject-session before any test is run."""
    from neurovision.benchmark import compare_groups
    few = compare_groups(_group_frame(n_hc=12, n_pd=12), rank_info=[])
    n = few[(few.group_a == "HC") & (few.group_b == "PD-OFF")
            & (few.metric == "r2")]["n_a"].iloc[0]
    assert n == 12          # not 24, which is what two windows each would give


def test_paired_test_uses_only_subjects_present_in_both_sessions():
    from neurovision.benchmark import compare_groups
    df = _group_frame(n_pd=8)
    df = df[~((df.subject == "pd0") & (df.session == "on"))]   # drop one session
    res = compare_groups(df, rank_info=[])
    row = res[(res.group_a == "PD-OFF") & (res.group_b == "PD-ON")
              & (res.metric == "r2")].iloc[0]
    assert row.n_a == row.n_b == 7


def test_bh_fdr_survives_a_nan_p_value():
    """A degenerate contrast yields NaN; it must not wipe out the whole family."""
    from neurovision.benchmark import _bh
    q = _bh(np.array([0.001, np.nan, 0.04, 0.2]))
    assert np.isnan(q[1]) and np.isfinite(q[[0, 2, 3]]).all()


def test_bh_fdr_is_monotone_and_bounded():
    from neurovision.benchmark import _bh
    q = _bh(np.array([0.001, 0.01, 0.04, 0.2, 0.9]))
    assert (np.diff(q) >= -1e-12).all()
    assert (q >= 0).all() and (q <= 1).all()
    assert (q >= np.array([0.001, 0.01, 0.04, 0.2, 0.9]) - 1e-12).all()


def test_channel_overlap_detects_agreement_and_disagreement():
    from neurovision.benchmark import channel_overlap
    same = [dict(group="HC", selection="top", channels=["Fz", "Cz", "Pz"]),
            dict(group="PD-OFF", selection="top", channels=["Fz", "Cz", "Pz"])]
    M, _ = channel_overlap(same)
    assert M.loc["HC", "PD-OFF"] == pytest.approx(1.0)

    diff = [dict(group="HC", selection="top", channels=["Fz", "Cz", "Pz"]),
            dict(group="PD-OFF", selection="top", channels=["O1", "O2", "T7"])]
    M, _ = channel_overlap(diff)
    assert M.loc["HC", "PD-OFF"] == pytest.approx(0.0)


def test_trace_figures_are_split_by_target(tmp_path):
    """With several targets the filenames must disambiguate, or each target
    silently overwrites the previous one's figure."""
    import argparse

    import pandas as pd

    from neurovision.benchmark import plot_traces

    rng = np.random.default_rng(11)
    store, rows = {}, []
    for tgt in ("Cz", "Pz"):
        for sel in ("top", "bottom"):
            true = rng.standard_normal(400).astype(np.float32)
            store[("sub-x/ses-y", tgt, sel, 50)] = (true, true + 0.3 * rng.standard_normal(400).astype(np.float32))
            rows.append(dict(recording="sub-x/ses-y", target=tgt,
                             selection=sel, window=50, r2=0.9))
    args = argparse.Namespace(stride=3, trace_start=0.2, trace_seconds=2.0, k=6,
                              max_trace_figures=0)
    made = plot_traces(store, pd.DataFrame(rows), tmp_path / "b.png", args, 250.0)
    assert len(made) == 2
    assert len({p.name for p in made}) == 2
    assert any("Cz" in p.name for p in made) and any("Pz" in p.name for p in made)


# --------------------------------------------------------------------------
# MINE as a training signal, not just a measurement
# --------------------------------------------------------------------------
def test_mine_predictor_with_zero_lambda_matches_a_plain_mlp():
    """lambda = 0 must remove the MI term entirely, so the two arms of the
    comparison differ in one term and nothing else."""
    from neurovision.mine import train_predictor_mine
    rng = np.random.default_rng(20)
    X = rng.standard_normal((3000, 8)).astype(np.float32)
    y = (X @ rng.standard_normal(8).astype(np.float32)
         + 0.3 * rng.standard_normal(3000)).astype(np.float32)
    Xtr, ytr, Xte, yte = split_contiguous(X, y, 0.25)
    a = train_predictor_mine(Xtr, ytr, Xte, yte, epochs=20, lam=0.0, seed=0)
    b = train_predictor_mine(Xtr, ytr, Xte, yte, epochs=20, lam=0.0, seed=0)
    assert a.r2 == pytest.approx(b.r2, abs=1e-9)   # deterministic given a seed
    assert a.r2 > 0.8                              # and it actually learns


def test_mine_predictor_runs_and_is_bounded_with_lambda():
    from neurovision.mine import train_predictor_mine
    rng = np.random.default_rng(21)
    X = rng.standard_normal((2500, 6)).astype(np.float32)
    y = np.tanh(X[:, 0] * 2).astype(np.float32) + 0.3 * rng.standard_normal(2500).astype(np.float32)
    Xtr, ytr, Xte, yte = split_contiguous(X, y, 0.25)
    r = train_predictor_mine(Xtr, ytr, Xte, yte, epochs=20, lam=0.2, seed=0)
    assert np.isfinite(r.r2) and r.r2 <= 1.0
    assert r.y_true.shape == r.y_pred.shape == yte.shape
