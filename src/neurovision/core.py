"""
mi_core.py
==========
BIDS discovery, EEG preprocessing, and pairwise mutual-information estimators.

MNE is imported lazily inside the preprocessing functions, so the estimators
here can be used (and unit-tested) with nothing but numpy + scipy.

Estimators
----------
gcmi_matrix        Gaussian-copula MI (Ince et al. 2017, HBM 38:1541).
                   Rank-transform each channel to a standard normal marginal,
                   then MI = -0.5*log(1 - r^2) on the transformed data.
                   Fast (one correlation matrix for all pairs), robust to
                   monotone nonlinearities, and a *lower bound* on true MI.
binned_mi_matrix   Plug-in histogram MI with equipopulated (quantile) bins
                   and an optional Miller-Madow bias correction.

All MI values are returned in bits, with NaN on the diagonal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.special import psi, ndtri

LN2 = np.log(2.0)

BANDS = {
    "broadband": (1.0, 45.0),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# --------------------------------------------------------------------------
# BIDS discovery
# --------------------------------------------------------------------------
@dataclass
class Recording:
    subject: str      # e.g. "hc1", "pd5"
    session: str      # e.g. "hc", "off", "on"  ("" if no session layer)
    group: str        # e.g. "HC", "PD-OFF", "PD-ON"
    path: str

    @property
    def label(self) -> str:
        return f"sub-{self.subject}" + (f"/ses-{self.session}" if self.session else "")


def _infer_group(subject: str, session: str) -> str:
    s = subject.lower()
    if s.startswith("hc"):
        return "HC"
    if s.startswith("pd"):
        return f"PD-{session.upper()}" if session else "PD"
    return session.upper() or "OTHER"


_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "src", "results",
              "derivatives", "code", "sourcedata", ".datalad", "tests"}


def _looks_like_bids(d: Path) -> bool:
    return ((d / "dataset_description.json").exists()
            or any(p.is_dir() for p in d.glob("sub-*")))


def find_bids_root(start=None, max_depth: int = 3) -> Path | None:
    """Breadth-first search at or below `start` for a BIDS dataset root.

    Lets the CLI work with no --bids-root when the data already sits inside the
    project folder, wherever the user happened to unpack it.
    """
    start = Path(start or Path.cwd()).resolve()
    queue = [(start, 0)]
    while queue:
        d, depth = queue.pop(0)
        if _looks_like_bids(d):
            return d
        if depth < max_depth:
            try:
                kids = sorted(x for x in d.iterdir() if x.is_dir())
            except OSError:
                continue
            queue += [(k, depth + 1) for k in kids
                      if k.name not in _SKIP_DIRS and not k.name.startswith(".")]
    return None


def find_recordings(bids_root, task: str = "rest", ext: str = "bdf") -> list[Recording]:
    """Walk a BIDS tree and return every <task> EEG recording it contains."""
    root = Path(bids_root)
    patterns = [f"sub-*/ses-*/eeg/*task-{task}_eeg.{ext}",
                f"sub-*/eeg/*task-{task}_eeg.{ext}"]
    recs: list[Recording] = []
    seen: set[str] = set()
    for pat in patterns:
        for f in sorted(root.glob(pat)):
            if str(f) in seen:
                continue
            seen.add(str(f))
            sub = re.search(r"sub-([A-Za-z0-9]+)", f.name)
            ses = re.search(r"ses-([A-Za-z0-9]+)", f.name)
            sub = sub.group(1) if sub else f.parts[-3]
            ses = ses.group(1) if ses else ""
            recs.append(Recording(sub, ses, _infer_group(sub, ses), str(f)))
    return recs


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def montage_channels(raw, montage_name: str = "biosemi32"):
    """Channels of `raw` that exist in the named standard montage, in raw order."""
    import mne
    try:
        mont = mne.channels.make_standard_montage(montage_name)
    except ValueError:
        return []
    names = {c.lower(): c for c in mont.ch_names}
    return [ch for ch in raw.ch_names if ch.lower() in names]


def load_and_preprocess(
    path,
    picks=None,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    notch: float | None = 60.0,
    sfreq: float = 250.0,
    tmin: float = 5.0,
    duration: float | None = 120.0,
    montage_name: str = "biosemi32",
    reference: str = "average",   # "average" | "csd" | "none"
    verbose: str = "ERROR",
):
    """Load one .bdf and return (data (n_ch, n_times), ch_names, info_dict).

    Order matters: notch -> band-pass -> resample -> re-reference -> crop.
    """
    import mne

    raw = mne.io.read_raw_bdf(path, preload=True, verbose=verbose)

    # ---- channel selection -------------------------------------------------
    if picks is not None:
        keep = [ch for ch in picks if ch in raw.ch_names]
        missing = [ch for ch in picks if ch not in raw.ch_names]
    else:
        keep = montage_channels(raw, montage_name)
        missing = []
        if not keep:  # unknown naming scheme: fall back to typed EEG channels
            keep = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    raw.pick(keep)

    try:
        raw.set_montage(montage_name, on_missing="ignore")
    except Exception:
        pass

    # ---- filtering ---------------------------------------------------------
    if notch:
        nyq = raw.info["sfreq"] / 2.0
        freqs = np.arange(notch, min(nyq, h_freq * 2 + notch), notch)
        freqs = freqs[freqs < nyq - 1]
        if len(freqs):
            raw.notch_filter(freqs, verbose=verbose)
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose=verbose)

    # ---- resample ----------------------------------------------------------
    if sfreq and raw.info["sfreq"] > sfreq:
        raw.resample(sfreq, verbose=verbose)

    # ---- reference ---------------------------------------------------------
    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose=verbose)
    elif reference == "csd":
        raw.set_eeg_reference("average", projection=False, verbose=verbose)
        raw = mne.preprocessing.compute_current_source_density(raw, verbose=verbose)

    # ---- crop --------------------------------------------------------------
    total = raw.times[-1]
    start = min(tmin, max(total - 1.0, 0.0))
    stop = total if duration is None else min(start + duration, total)
    raw.crop(start, stop)

    data = raw.get_data()
    meta = {
        "sfreq": float(raw.info["sfreq"]),
        "n_samples": int(data.shape[1]),
        "duration_s": float(data.shape[1] / raw.info["sfreq"]),
        "original_duration_s": float(total),
        "missing_channels": missing,
    }
    return data, list(raw.ch_names), meta


def band_filter(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Band-pass an already-loaded (n_ch, n_times) array."""
    import mne
    return mne.filter.filter_data(
        np.asarray(data, dtype=float), sfreq, band[0], band[1],
        fir_design="firwin", verbose="ERROR",
    )


def sensor_positions_2d(ch_names, montage_name: str = "biosemi32"):
    """Azimuthal-equidistant 2-D projection of montage positions, in [-1, 1]."""
    import mne
    try:
        mont = mne.channels.make_standard_montage(montage_name)
    except ValueError:
        return None
    pos3d = mont.get_positions()["ch_pos"]
    lut = {k.lower(): v for k, v in pos3d.items()}
    xyz = np.array([lut.get(ch.lower(), [np.nan] * 3) for ch in ch_names], float)
    if np.isnan(xyz).all():
        return None
    xyz = xyz - np.nanmean(xyz, axis=0)
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    rho = np.pi / 2.0 - np.arcsin(np.clip(xyz[:, 2] / np.where(r == 0, 1, r), -1, 1))
    rho = rho / np.nanmax(rho)
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


# --------------------------------------------------------------------------
# Mutual information
# --------------------------------------------------------------------------
def copnorm(x: np.ndarray) -> np.ndarray:
    """Gaussian-copula rank transform along the last axis."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    ranks = np.argsort(np.argsort(x, axis=-1), axis=-1) + 1.0
    return ndtri(ranks / (n + 1.0))


def _bias_term(n_samples: float) -> float:
    """Miller-type correction for the 1-D vs 1-D Gaussian entropy estimate (nats)."""
    n = float(n_samples)
    if n < 8:
        return 0.0
    return 0.5 * (psi((n - 1) / 2.0) - psi((n - 2) / 2.0))


def gcmi_matrix_from_z(z: np.ndarray, n_eff: float | None = None,
                       bias_correct: bool = True) -> np.ndarray:
    """All-pairs Gaussian-copula MI (bits) from already copula-normalised data."""
    n_ch, n = z.shape
    zc = z - z.mean(axis=-1, keepdims=True)
    zc /= zc.std(axis=-1, ddof=1, keepdims=True)
    r = (zc @ zc.T) / (n - 1.0)
    np.clip(r, -1 + 1e-12, 1 - 1e-12, out=r)
    mi = -0.5 * np.log1p(-r ** 2)                       # nats
    if bias_correct:
        mi = mi - _bias_term(n_eff if n_eff else n)
    mi = np.maximum(mi, 0.0) / LN2                      # bits
    np.fill_diagonal(mi, np.nan)
    return mi


def gcmi_matrix(x: np.ndarray, n_eff: float | None = None,
                bias_correct: bool = True) -> np.ndarray:
    """All-pairs Gaussian-copula MI (bits) for x of shape (n_channels, n_times)."""
    return gcmi_matrix_from_z(copnorm(x), n_eff=n_eff, bias_correct=bias_correct)


def gcmi_null_matrix(x: np.ndarray, n_surrogates: int = 20, rng=None,
                     n_eff: float | None = None,
                     bias_correct: bool = True) -> np.ndarray:
    """Mean MI under circular-shift surrogates.

    Each channel is rolled by an independent random lag, which destroys
    between-channel coupling while preserving each channel's own spectrum and
    autocorrelation. The result is the MI floor you would get from
    autocorrelated noise alone -- subtract it to read 'excess' MI.
    """
    rng = np.random.default_rng(rng)
    z = copnorm(x)
    n_ch, n = z.shape
    lo, hi = max(n // 20, 1), n - max(n // 20, 1)
    acc = np.zeros((n_ch, n_ch))
    for _ in range(n_surrogates):
        zs = np.empty_like(z)
        for i in range(n_ch):
            zs[i] = np.roll(z[i], int(rng.integers(lo, hi)))
        acc += np.nan_to_num(gcmi_matrix_from_z(zs, n_eff, bias_correct))
    out = acc / float(n_surrogates)
    np.fill_diagonal(out, np.nan)
    return out


def ksg_mi_matrix(x: np.ndarray, k: int = 4, jitter: float = 1e-10,
                  rng=None) -> np.ndarray:
    """All-pairs Kraskov-Stogbauer-Grassberger MI (estimator 1), in bits.

    For each point, find the radius of its k-th nearest neighbour in the joint
    2-D space under the max-norm, then count how many points fall strictly
    within that radius on each marginal axis:

        I = psi(k) + psi(N) - <psi(n_x) + psi(n_y)>

    Unlike the Gaussian-copula estimator this makes no assumption about the
    shape of the dependence, which is what you want in short windows where the
    relationship may not be monotone. It costs far more: the joint neighbour
    search runs through a KD-tree, and the marginal counts through binary
    search on pre-sorted axes, giving O(N log N) per pair rather than O(N^2).

    A tiny jitter breaks ties, as Kraskov et al. recommend for data that has
    been quantised (EEG comes off the amplifier in discrete steps).
    """
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=float)
    n_ch, n = x.shape
    if n <= k + 1:
        return np.full((n_ch, n_ch), np.nan)
    if jitter:
        rng = np.random.default_rng(rng)
        scale = np.std(x, axis=1, keepdims=True)
        scale[scale == 0] = 1.0
        x = x + jitter * scale * rng.standard_normal(x.shape)

    xs = np.sort(x, axis=1)                       # marginal axes, sorted once
    out = np.full((n_ch, n_ch), np.nan)
    base = psi(k) + psi(n)
    for a in range(n_ch):
        for b in range(a + 1, n_ch):
            z = np.column_stack((x[a], x[b]))
            eps = cKDTree(z, balanced_tree=False).query(z, k=k + 1, p=np.inf)[0][:, k]
            n_x = (np.searchsorted(xs[a], x[a] + eps, "left")
                   - np.searchsorted(xs[a], x[a] - eps, "right"))
            n_y = (np.searchsorted(xs[b], x[b] + eps, "left")
                   - np.searchsorted(xs[b], x[b] - eps, "right"))
            mi = base - np.mean(psi(n_x) + psi(n_y))
            out[a, b] = out[b, a] = mi / LN2
    return out


# --------------------------------------------------------------------------
# Windowed analysis
# --------------------------------------------------------------------------
def window_bounds(n_samples: int, sfreq: float, window_s: float,
                  step_s: float | None = None) -> np.ndarray:
    """Start/stop sample index for each analysis window."""
    win = int(round(window_s * sfreq))
    step = win if step_s is None else max(int(round(step_s * sfreq)), 1)
    if win < 8 or n_samples < win:
        return np.zeros((0, 2), dtype=int)
    starts = np.arange(0, n_samples - win + 1, step)
    return np.column_stack([starts, starts + win])


def reject_windows(x: np.ndarray, bounds: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Boolean keep-mask for windows whose peak amplitude is not an outlier.

    A window is dropped when any channel's peak amplitude sits more than z
    robust SDs (1.4826 * MAD) above the median peak *across windows*. Comparing
    windows against each other rather than against a fixed sample-wise cutoff
    keeps the criterion honest whatever the recording's overall scale.
    """
    if z <= 0 or len(bounds) == 0:
        return np.ones(len(bounds), dtype=bool)
    peaks = np.array([np.abs(x[:, s:e]).max(axis=1) for s, e in bounds])
    med = np.median(peaks, axis=0)
    mad = 1.4826 * np.median(np.abs(peaks - med), axis=0)
    mad[mad == 0] = np.inf
    return ~(peaks > med + z * mad).any(axis=1)


def windowed_mi(x: np.ndarray, sfreq: float, window_s: float = 0.6,
                step_s: float | None = None, estimator: str = "ksg",
                k: int = 4, bins: int = 16, reject_z: float = 0.0,
                rng=None) -> tuple[np.ndarray, np.ndarray]:
    """MI matrix for every window. Returns (stack (n_win, n_ch, n_ch), bounds).

    Each window is estimated independently and the windows are only combined
    afterwards, so a pair's coupling is compared window by window rather than
    pooled into one long estimate.
    """
    x = np.asarray(x, dtype=float)
    bounds = window_bounds(x.shape[1], sfreq, window_s, step_s)
    keep = reject_windows(x, bounds, reject_z)
    bounds = bounds[keep]
    if len(bounds) == 0:
        return np.zeros((0, x.shape[0], x.shape[0])), bounds

    rng = np.random.default_rng(rng)
    stack = np.empty((len(bounds), x.shape[0], x.shape[0]))
    for i, (s, e) in enumerate(bounds):
        seg = x[:, s:e]
        if estimator == "ksg":
            stack[i] = ksg_mi_matrix(seg, k=k, rng=rng)
        elif estimator == "gcmi":
            stack[i] = gcmi_matrix(seg)
        else:
            stack[i] = binned_mi_matrix(seg, bins=bins)
    return stack, bounds


def windowed_null(x: np.ndarray, sfreq: float, window_s: float = 0.6,
                  n_windows: int = 20, estimator: str = "ksg", k: int = 4,
                  bins: int = 16, rng=None) -> np.ndarray:
    """MI floor for windowed analysis, from independently rolled channels.

    Only `n_windows` randomly drawn windows are used -- the floor is a property
    of the window length and the data's autocorrelation, not of any particular
    moment, so there is no point paying for all of them.
    """
    rng = np.random.default_rng(rng)
    bounds = window_bounds(x.shape[1], sfreq, window_s, None)
    if len(bounds) == 0:
        return np.full((x.shape[0], x.shape[0]), np.nan)
    pick = rng.choice(len(bounds), size=min(n_windows, len(bounds)), replace=False)
    n_ch, n = x.shape
    lo, hi = max(n // 20, 1), n - max(n // 20, 1)
    acc = []
    for i in pick:
        s, e = bounds[i]
        rolled = np.empty((n_ch, e - s))
        for c in range(n_ch):
            rolled[c] = np.roll(x[c], int(rng.integers(lo, hi)))[s:e]
        if estimator == "ksg":
            acc.append(ksg_mi_matrix(rolled, k=k, rng=rng))
        elif estimator == "gcmi":
            acc.append(gcmi_matrix(rolled))
        else:
            acc.append(binned_mi_matrix(rolled, bins=bins))
    return np.nanmean(np.array(acc), axis=0)


def binned_mi_matrix(x: np.ndarray, bins: int = 16,
                     miller_madow: bool = True) -> np.ndarray:
    """All-pairs plug-in MI (bits) using equipopulated bins."""
    x = np.asarray(x, dtype=float)
    n_ch, n = x.shape
    d = np.empty((n_ch, n), dtype=np.int64)
    qs = np.linspace(0, 100, bins + 1)[1:-1]
    for i in range(n_ch):
        d[i] = np.searchsorted(np.percentile(x[i], qs), x[i], side="right")

    h_marg = np.empty(n_ch)
    for i in range(n_ch):
        p = np.bincount(d[i], minlength=bins).astype(float) / n
        nz = p > 0
        h_marg[i] = -(p[nz] * np.log(p[nz])).sum()
        if miller_madow:
            h_marg[i] += (nz.sum() - 1) / (2.0 * n)

    mi = np.full((n_ch, n_ch), np.nan)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            p = np.bincount(d[i] * bins + d[j], minlength=bins * bins).astype(float) / n
            nz = p > 0
            h_joint = -(p[nz] * np.log(p[nz])).sum()
            if miller_madow:
                h_joint += (nz.sum() - 1) / (2.0 * n)
            mi[i, j] = mi[j, i] = max(h_marg[i] + h_marg[j] - h_joint, 0.0) / LN2
    return mi


def effective_n(x: np.ndarray) -> float:
    """Crude AR(1) effective sample size, N * (1-rho)/(1+rho), median over channels.

    EEG samples are far from independent, so the nominal N badly overstates the
    information available. Use this when you need honest error bars.
    """
    x = np.asarray(x, dtype=float)
    xc = x - x.mean(axis=-1, keepdims=True)
    denom = (xc ** 2).sum(axis=-1)
    rho = (xc[:, 1:] * xc[:, :-1]).sum(axis=-1) / np.where(denom == 0, 1, denom)
    rho = np.clip(rho, 0.0, 0.999)
    return float(np.median(x.shape[-1] * (1 - rho) / (1 + rho)))


ESTIMATORS = {"gcmi": gcmi_matrix, "ksg": ksg_mi_matrix, "binned": binned_mi_matrix}


def estimator(name: str):
    return ESTIMATORS[name]
