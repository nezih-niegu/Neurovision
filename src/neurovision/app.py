#!/usr/bin/env python
"""
mi_app.py
=========
Interactive explorer for the cached MI matrices.

    uv run neurovision app

Deselect recordings or channels in the sidebar and every view updates. The
group average is just a mean over the cached per-recording matrices, so it is
instant.
"""

from __future__ import annotations

import io
import json
import warnings
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Circle
from scipy import stats

from neurovision import core as mc

st.set_page_config(page_title="EEG mutual information", layout="wide")
CMAP = "viridis"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def default_results_path() -> str:
    from neurovision.cli import resolve_results
    argv = sys.argv
    if "--results" in argv:
        return argv[argv.index("--results") + 1]
    return str(resolve_results(None))


@st.cache_data(show_spinner="Loading cached matrices…")
def load_results(path: str):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    for k in ("bands", "ch_names", "subjects", "sessions", "groups", "labels", "paths"):
        d[k] = [str(v) for v in d[k]]
    return d


@st.cache_data(show_spinner="Reading .bdf…")
def load_raw_preview(path: str, channels: tuple, montage: str, reference: str,
                     sfreq: float, notch: float, tmin: float, duration: float):
    return mc.load_and_preprocess(
        path, picks=list(channels), sfreq=sfreq, notch=notch or None,
        tmin=tmin, duration=duration, montage_name=montage, reference=reference)


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def draw_head(ax, r=1.15):
    ax.add_patch(Circle((0, 0), r, fill=False, lw=1.4, color="0.35", zorder=1))
    ax.plot([-0.12 * r, 0, 0.12 * r], [r * 0.99, r * 1.16, r * 0.99],
            color="0.35", lw=1.4, zorder=1)                       # nose
    for s in (-1, 1):
        ax.add_patch(Circle((s * r, 0), 0.09 * r, fill=False, lw=1.4,
                            color="0.35", zorder=1))              # ears
    ax.set_xlim(-1.35 * r, 1.35 * r)
    ax.set_ylim(-1.3 * r, 1.35 * r)
    ax.set_aspect("equal")
    ax.axis("off")


def matrix_figure(m, names, title, vmin=None, vmax=None, cbar_label="MI (bits)",
                  cmap=CMAP, center_zero=False):
    n = len(names)
    size = max(4.5, min(11, 0.32 * n + 2.6))
    fig, ax = plt.subplots(figsize=(size + 1.4, size))
    if center_zero:
        lim = np.nanmax(np.abs(m)) or 1.0
        vmin, vmax, cmap = -lim, lim, "RdBu_r"
    im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.045, label=cbar_label)
    fig.tight_layout()
    return fig


def network_figure(m, names, pos, keep_pct, node_vals):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    draw_head(ax)
    iu = np.triu_indices(len(names), 1)
    w = m[iu]
    finite = w[np.isfinite(w)]
    if finite.size:
        thr = np.percentile(finite, 100 - keep_pct)
        lo, hi = finite.min(), finite.max()
        span = (hi - lo) or 1.0
        cmap = plt.get_cmap("plasma")
        order = np.argsort(w)
        for k in order:
            v = w[k]
            if not np.isfinite(v) or v < thr:
                continue
            i, j = iu[0][k], iu[1][k]
            f = (v - lo) / span
            ax.plot(pos[[i, j], 0], pos[[i, j], 1], lw=0.4 + 3.4 * f,
                    color=cmap(f), alpha=0.28 + 0.62 * f, zorder=2,
                    solid_capstyle="round")
    ax.scatter(pos[:, 0], pos[:, 1], s=120, c=node_vals, cmap=CMAP,
               edgecolor="white", lw=1.0, zorder=3)
    for k, nm in enumerate(names):
        ax.text(pos[k, 0], pos[k, 1] - 0.085, nm, ha="center", va="top",
                fontsize=6.5, zorder=4)
    fig.tight_layout()
    return fig


def fig_download(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download figure (PNG)", buf.getvalue(), name, "image/png")


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * pv.size / (np.arange(pv.size) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.title("Resting-state EEG — pairwise mutual information")

path = st.sidebar.text_input("Results file", default_results_path())
if not Path(path).exists():
    st.warning(f"`{path}` not found. Run `uv run neurovision precompute` first, "
               "then point this box at its output.")
    st.stop()

R = load_results(path)
bands, ch_all, labels = R["bands"], R["ch_names"], R["labels"]
groups, subjects = R["groups"], R["subjects"]
pos = R.get("pos")
has_null = "mi_null" in R
has_sd = "mi_sd" in R
has_tc = "mi_timecourse" in R
has_win = "mi_windows" in R
cfg = R["meta"]["args"]
win_s = float(cfg.get("window") or 0)
n_windows = R.get("n_windows")

st.sidebar.header("Data")
band = st.sidebar.selectbox("Frequency band", bands, index=0)
bi = bands.index(band)

subtract_null = False
if has_null:
    subtract_null = st.sidebar.checkbox(
        "Subtract surrogate floor", value=True,
        help="Circular-shift surrogates keep each channel's own autocorrelation "
             "but destroy coupling. Subtracting them leaves the MI that genuinely "
             "reflects shared information.")
pool_sessions = st.sidebar.checkbox(
    "Average sessions within subject first", value=False,
    help="Stops subjects with two sessions (PD on/off) counting twice in the mean.")

# ---- recordings ----------------------------------------------------------
st.sidebar.header("Recordings")
opt_labels = [f"{l}  [{g}]" for l, g in zip(labels, groups)]
if "rec_sel" not in st.session_state:
    st.session_state.rec_sel = list(opt_labels)


def set_recs(vals):
    st.session_state.rec_sel = list(vals)


c1, c2 = st.sidebar.columns(2)
c1.button("Select all", on_click=set_recs, args=(opt_labels,),)
c2.button("Clear", on_click=set_recs, args=([],))
gcols = st.sidebar.columns(min(len(set(groups)), 3))
for k, g in enumerate(sorted(set(groups))):
    gcols[k % len(gcols)].button(
        g, on_click=set_recs,
        args=([o for o, gg in zip(opt_labels, groups) if gg == g],))
sel_rec_labels = st.sidebar.multiselect("Included recordings", opt_labels,
                                        key="rec_sel")
rec_idx = np.array([opt_labels.index(s) for s in sel_rec_labels], dtype=int)

# ---- channels ------------------------------------------------------------
st.sidebar.header("Channels")
if "ch_sel" not in st.session_state:
    st.session_state.ch_sel = list(ch_all)


def set_chs(vals):
    st.session_state.ch_sel = list(vals)


c3, c4 = st.sidebar.columns(2)
c3.button("All channels", on_click=set_chs, args=(ch_all,),)
c4.button("First 16", on_click=set_chs, args=(ch_all[:16],),)
sel_chs = st.sidebar.multiselect("Included channels", ch_all, key="ch_sel")
ch_idx = np.array([ch_all.index(c) for c in sel_chs], dtype=int)

if len(rec_idx) == 0 or len(ch_idx) < 2:
    st.info("Select at least one recording and two channels.")
    st.stop()

# --------------------------------------------------------------------------
# Assemble the selected stack
# --------------------------------------------------------------------------
stack = R["mi"][bi][np.ix_(rec_idx, ch_idx, ch_idx)]
sd_stack = (R["mi_sd"][bi][np.ix_(rec_idx, ch_idx, ch_idx)] if has_sd
            else np.full_like(stack, np.nan))
if subtract_null:
    stack = stack - R["mi_null"][bi][np.ix_(rec_idx, ch_idx, ch_idx)]
sel_groups = [groups[i] for i in rec_idx]
sel_subs = [subjects[i] for i in rec_idx]
sel_labels = [labels[i] for i in rec_idx]

if pool_sessions:
    uniq, pooled, pooled_sd, pg = [], [], [], []
    for s in dict.fromkeys(sel_subs):
        m = np.array([i for i, x in enumerate(sel_subs) if x == s])
        uniq.append(f"sub-{s}")
        pooled.append(np.nanmean(stack[m], axis=0))
        pooled_sd.append(np.nanmean(sd_stack[m], axis=0))
        pg.append(sel_groups[m[0]].split("-")[0])
    stack, sd_stack = np.array(pooled), np.array(pooled_sd)
    sel_labels, sel_groups = uniq, pg

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

with np.errstate(invalid="ignore"):
    avg = np.nanmean(stack, axis=0)
    sem = (np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
           if stack.shape[0] > 1 else np.zeros_like(avg))
    win_sd = np.nanmean(sd_stack, axis=0)
names = [ch_all[i] for i in ch_idx]
pos_sel = pos[ch_idx] if pos is not None else None
unit = "excess MI (bits)" if subtract_null else "MI (bits)"

iu = np.triu_indices(len(names), 1)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Recordings", stack.shape[0])
m2.metric("Channels", len(names))
if win_s and n_windows is not None:
    tot = int(n_windows[bi][rec_idx].sum())
    st.caption(
        f"{cfg.get('estimator', '?').upper()} estimator, k={cfg.get('k')}, "
        f"{win_s*1000:.0f} ms windows — {tot:,} window estimates across the "
        f"{len(rec_idx)} selected recordings, {len(names)*(len(names)-1)//2} "
        f"pairs each.")
edge_vals = avg[iu]
finite_edges = np.isfinite(edge_vals)
m3.metric(f"Mean {unit.split(' (')[0]}",
          f"{np.nanmean(edge_vals):.3f} bits" if finite_edges.any() else "n/a")
if finite_edges.any():
    k = int(np.nanargmax(edge_vals))
    m4.metric("Strongest pair", f"{names[iu[0][k]]}–{names[iu[1][k]]}")
else:
    m4.metric("Strongest pair", "n/a")

tab_names = ["Average matrix", "Network", "Channel strength", "Per-recording"]
if has_tc:
    tab_names.append("MI over time")
tab_names += ["Compare groups", "Raw signal"]
T = dict(zip(tab_names, st.tabs(tab_names)))

# ---- 1. matrix -----------------------------------------------------------
with T["Average matrix"]:
    left, right = st.columns([3, 1])
    views = ["Average MI"]
    if win_s:
        views.append("Variability across windows")
    views.append("Standard error across recordings")
    with right:
        view = st.radio("Show", views, index=0)
        robust = st.checkbox("Robust colour scale (2–98 %)", value=True)
    if view.startswith("Variability"):
        m, cbar = win_sd, f"SD across windows (bits)"
    elif view.startswith("Standard error"):
        m, cbar = sem, f"SEM across recordings (bits)"
    else:
        m, cbar = avg, unit
    show_sem = view != "Average MI"
    vmin = vmax = None
    if robust and np.isfinite(m).any():
        vmin, vmax = np.nanpercentile(m, [2, 98])
    with left:
        fig = matrix_figure(
            m, names, f"{view} — {band}, n = {stack.shape[0]}",
            vmin, vmax, cbar)
        st.pyplot(fig)
        fig_download(fig, f"mi_matrix_{band}.png")

    df = pd.DataFrame(avg, index=names, columns=names)
    st.download_button("Download matrix (CSV)", df.to_csv().encode(),
                       f"mi_matrix_{band}.csv", "text/csv")
    edges = pd.DataFrame({
        "ch_a": [names[i] for i in iu[0]],
        "ch_b": [names[j] for j in iu[1]],
        "mi_bits": avg[iu],
        "sem": sem[iu],
    }).sort_values("mi_bits", ascending=False).reset_index(drop=True)
    st.markdown("**Strongest pairs**")
    st.dataframe(edges.head(20), hide_index=True)
    st.download_button("Download edge list (CSV)", edges.to_csv(index=False).encode(),
                       f"mi_edges_{band}.csv", "text/csv")

# ---- 2. network ----------------------------------------------------------
with T["Network"]:
    if pos_sel is None:
        st.info("No sensor positions were cached — rerun precompute with a "
                "montage that matches the channel names.")
    else:
        keep = st.slider("Show strongest % of connections", 1, 100, 20)
        strength = np.nanmean(avg, axis=1)
        fig = network_figure(avg, names, pos_sel, keep, strength)
        st.pyplot(fig)
        fig_download(fig, f"mi_network_{band}.png")
        st.caption("Edge colour and width encode MI; node colour is that "
                   "channel's mean MI to the others shown.")

# ---- 3. channel strength -------------------------------------------------
with T["Channel strength"]:
    strength = np.nanmean(avg, axis=1)
    order = np.argsort(strength)[::-1]
    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(6, max(3, 0.22 * len(names))))
        ax.barh([names[i] for i in order][::-1], strength[order][::-1],
                color=plt.get_cmap(CMAP)(0.55))
        ax.set_xlabel(f"mean {unit}")
        ax.set_title("Channel strength")
        fig.tight_layout()
        st.pyplot(fig)
    with c2:
        if pos_sel is not None:
            fig2, ax = plt.subplots(figsize=(5.4, 5.4))
            draw_head(ax)
            sc = ax.scatter(pos_sel[:, 0], pos_sel[:, 1], s=460, c=strength,
                            cmap=CMAP, edgecolor="white", lw=1.2, zorder=3)
            for k, nm in enumerate(names):
                ax.text(pos_sel[k, 0], pos_sel[k, 1], nm, ha="center", va="center",
                        fontsize=6, color="white", zorder=4)
            fig2.colorbar(sc, ax=ax, fraction=0.045, label=unit)
            fig2.tight_layout()
            st.pyplot(fig2)
    st.dataframe(pd.DataFrame({"channel": names, "mean_mi_bits": strength})
                 .sort_values("mean_mi_bits", ascending=False),
                 hide_index=True)

# ---- 4. per-recording ----------------------------------------------------
with T["Per-recording"]:
    per = np.array([np.nanmean(s[iu]) for s in stack])
    df = pd.DataFrame({"recording": sel_labels, "group": sel_groups,
                       "mean_mi_bits": per}).sort_values("mean_mi_bits")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.24 * len(per))))
    cols = {g: plt.get_cmap("tab10")(i) for i, g in enumerate(sorted(set(sel_groups)))}
    ax.barh(df["recording"], df["mean_mi_bits"],
            color=[cols[g] for g in df["group"]])
    ax.axvline(np.nanmean(per), ls="--", lw=1, color="0.3")
    ax.set_xlabel(f"mean {unit} over selected pairs")
    ax.set_title("Per-recording mean — dashed line is the group average")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Recordings far from the dashed line are the ones worth "
               "inspecting for artefacts before you trust the average.")
    st.dataframe(df, hide_index=True)

# ---- MI over time --------------------------------------------------------
if has_tc:
    with T["MI over time"]:
        step_s = float(cfg.get("window_step") or win_s)
        t_off = float(cfg.get("tmin") or 0)

        if has_win:
            sub = R["mi_windows"][bi][np.ix_(rec_idx, np.arange(
                R["mi_windows"].shape[2]), ch_idx, ch_idx)]
            ju = np.triu_indices(len(names), 1)
            courses = np.nanmean(sub[:, :, ju[0], ju[1]], axis=2)
            scope = "selected channels"
        else:
            courses = R["mi_timecourse"][bi][rec_idx]
            scope = "all cached channels"

        t = np.arange(courses.shape[1]) * step_s + t_off
        fig, ax = plt.subplots(figsize=(10, 4.2))
        cols = {g: plt.get_cmap("tab10")(i)
                for i, g in enumerate(sorted(set(groups[i] for i in rec_idx)))}
        for row, ri in enumerate(rec_idx):
            ax.plot(t, courses[row], lw=0.6, alpha=0.35, color=cols[groups[ri]])
        for g, c in cols.items():
            rows = [r for r, ri in enumerate(rec_idx) if groups[ri] == g]
            if rows:
                ax.plot(t, np.nanmean(courses[rows], axis=0), lw=2.4, color=c, label=g)
        ax.set_xlabel("time in recording (s)")
        ax.set_ylabel(f"mean MI over pairs (bits)")
        ax.set_title(f"Window-by-window MI — {win_s*1000:.0f} ms windows, {scope}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        fig_download(fig, f"mi_timecourse_{band}.png")
        st.caption("Thin lines are individual recordings, thick lines group means. "
                   "Flat traces mean coupling is stationary over the recording; "
                   "excursions are where something actually changed — or where an "
                   "artefact survived rejection.")

        if has_win:
            c1, c2 = st.columns(2)
            ca = c1.selectbox("Channel A", names, index=0)
            cb = c2.selectbox("Channel B", names,
                              index=min(1, len(names) - 1))
            if ca != cb:
                ia, ib = names.index(ca), names.index(cb)
                pair = sub[:, :, ia, ib]
                fig2, (axl, axr) = plt.subplots(
                    1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2, 1]})
                for row, ri in enumerate(rec_idx):
                    axl.plot(t, pair[row], lw=0.7, alpha=0.4, color=cols[groups[ri]])
                axl.plot(t, np.nanmean(pair, axis=0), lw=2.4, color="k",
                         label="mean")
                axl.set_xlabel("time (s)"); axl.set_ylabel("MI (bits)")
                axl.set_title(f"{ca} – {cb}"); axl.legend(fontsize=8)
                vals = pair[np.isfinite(pair)]
                axr.hist(vals, bins=40, color=plt.get_cmap(CMAP)(0.55))
                axr.axvline(np.nanmean(vals), color="k", ls="--", lw=1.2)
                axr.set_xlabel("MI per window (bits)")
                axr.set_title("distribution over windows")
                fig2.tight_layout()
                st.pyplot(fig2)
                st.caption(
                    f"Mean {np.nanmean(vals):.3f} bits, SD across windows "
                    f"{np.nanstd(vals):.3f}. A wide distribution with a modest "
                    "mean means the pair couples intermittently rather than "
                    "weakly-but-steadily — the two look identical once averaged.")

# ---- 5. group comparison -------------------------------------------------
with T["Compare groups"]:
    gset = sorted(set(sel_groups))
    if len(gset) < 2:
        st.info("Select recordings from at least two groups.")
    else:
        c1, c2, c3 = st.columns(3)
        ga = c1.selectbox("Group A", gset, index=0)
        gb = c2.selectbox("Group B", gset, index=1)
        alpha = c3.number_input("FDR q", 0.001, 0.5, 0.05, 0.01)
        ia = [k for k, g in enumerate(sel_groups) if g == ga]
        ib = [k for k, g in enumerate(sel_groups) if g == gb]
        if min(len(ia), len(ib)) < 2:
            st.info("Need at least two recordings per group.")
        else:
            A, B = stack[ia], stack[ib]
            diff = np.nanmean(A, axis=0) - np.nanmean(B, axis=0)
            t, p = stats.ttest_ind(A[:, iu[0], iu[1]], B[:, iu[0], iu[1]],
                                   axis=0, equal_var=False, nan_policy="omit")
            t = np.ma.filled(np.ma.asarray(t, float), np.nan)
            p = np.ma.filled(np.ma.asarray(p, float), np.nan)
            q = bh_fdr(p)
            fig = matrix_figure(diff, names,
                                f"{ga} (n={len(ia)}) − {gb} (n={len(ib)}) — {band}",
                                cbar_label=f"Δ {unit}", center_zero=True)
            st.pyplot(fig)
            n_sig = int(np.nansum(q < alpha))
            st.write(f"**{n_sig}** of {len(q)} pairs survive BH-FDR at q < {alpha}.")
            res = pd.DataFrame({
                "ch_a": [names[i] for i in iu[0]],
                "ch_b": [names[j] for j in iu[1]],
                "delta_bits": diff[iu], "t": t, "p": p, "q_fdr": q,
            }).sort_values("q_fdr")
            st.dataframe(res.head(25), hide_index=True)
            st.download_button("Download comparison (CSV)",
                               res.to_csv(index=False).encode(),
                               f"mi_{ga}_vs_{gb}_{band}.csv", "text/csv")
            st.caption("Recordings are treated as independent samples, so this "
                       "is only valid if you are not comparing two sessions of "
                       "the same people. For PD on vs off, use a paired test.")

# ---- 6. raw signal -------------------------------------------------------
with T["Raw signal"]:
    a = R["meta"]["args"]
    pick = st.selectbox("Recording", sel_labels)
    row = labels.index(pick) if pick in labels else None
    if row is None:
        st.info("Session-pooled entry — turn off pooling to preview raw data.")
    else:
        if st.button("Load and plot", type="primary"):
            data, nm, meta = load_raw_preview(
                R["paths"][row], tuple(names), a["montage"], a["reference"],
                a["sfreq"], a["notch"], a["tmin"], a["duration"] or 60.0)
            sf = meta["sfreq"]
            secs = st.session_state.get("preview_secs", 10)
            n = int(secs * sf)
            t = np.arange(n) / sf
            offs = 6 * np.nanstd(data)
            fig, ax = plt.subplots(figsize=(11, 0.28 * len(nm) + 2.4))
            for k, ch in enumerate(nm):
                ax.plot(t, data[k, :n] + k * offs, lw=0.6)
            ax.set_yticks(np.arange(len(nm)) * offs)
            ax.set_yticklabels(nm, fontsize=7)
            ax.set_xlabel("time (s)")
            ax.set_title(f"{pick} — preprocessed, {secs} s")
            fig.tight_layout()
            st.pyplot(fig)

            from scipy.signal import welch
            f, P = welch(data, fs=sf, nperseg=int(4 * sf))
            fig2, ax = plt.subplots(figsize=(7, 4))
            ax.semilogy(f, P.T, lw=0.7, alpha=0.6)
            ax.semilogy(f, P.mean(0), lw=2.2, color="k", label="mean")
            ax.set_xlim(0, 45)
            ax.set_xlabel("frequency (Hz)")
            ax.set_ylabel("PSD (V²/Hz)")
            ax.legend()
            fig2.tight_layout()
            st.pyplot(fig2)
        st.slider("Seconds to show", 2, 30, 10, key="preview_secs")

st.sidebar.caption(f"{len(labels)} recordings cached · bands: {', '.join(bands)}")
