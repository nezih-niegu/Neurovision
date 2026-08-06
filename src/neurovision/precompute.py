#!/usr/bin/env python
"""
neurovision.precompute
======================
Walk a BIDS EEG dataset, preprocess every resting-state recording, and estimate
mutual information for every channel pair.

Two modes:

* **windowed** (default) -- cut the recording into short windows, estimate MI
  independently in each one for every pair of the fully connected graph, then
  average across windows. This is what `--window 0.6` gives you: 600 ms of data
  per estimate, so coupling is compared moment by moment rather than smeared
  across the whole recording. The spread across windows is kept too, which tells
  you whether a pair is steadily coupled or only intermittently.
* **whole-recording** (`--window 0`) -- one estimate per pair from the entire
  time series.

Results are cached to a single .npz that the GUI reads.

Example
-------
    uv run neurovision precompute --window 0.6 --estimator ksg \\
        --bands broadband alpha --duration 120 --reject-z 6 --jobs 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings

import numpy as np

from neurovision import core as mc


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-root", default=None,
                   help="dataset root (contains sub-*/); auto-detected if omitted")
    p.add_argument("--out", default="mi_results.npz")
    p.add_argument("--task", default="rest")
    p.add_argument("--bands", nargs="+", default=["broadband"],
                   choices=list(mc.BANDS), help="frequency bands to compute")
    p.add_argument("--channels", nargs="+", default=None,
                   help="explicit channel list; default = montage channels common "
                        "to every recording")
    p.add_argument("--n-channels", type=int, default=None,
                   help="keep only the first N channels after selection")
    p.add_argument("--montage", default="biosemi32")
    p.add_argument("--reference", default="average",
                   choices=["average", "csd", "none"],
                   help="csd = surface Laplacian, strongly reduces volume conduction")

    g = p.add_argument_group("mutual information")
    g.add_argument("--estimator", default="ksg", choices=["ksg", "gcmi", "binned"],
                   help="ksg = Kraskov k-nearest-neighbour (default), makes no "
                        "assumption about the shape of the dependence")
    g.add_argument("--k", type=int, default=4,
                   help="neighbours for the KSG estimator; 3-6 is the usual range")
    g.add_argument("--bins", type=int, default=16, help="binned estimator only")

    g = p.add_argument_group("windowing")
    g.add_argument("--window", type=float, default=0.6,
                   help="window length in seconds (0 = whole recording)")
    g.add_argument("--window-step", type=float, default=None,
                   help="hop in seconds; default = window length (no overlap)")
    g.add_argument("--reject-z", type=float, default=6.0,
                   help="drop windows whose peak amplitude exceeds z robust SDs "
                        "of the across-window median; 0 disables")
    g.add_argument("--save-windows", action="store_true",
                   help="also cache every window's full matrix, enabling per-pair "
                        "time courses in the app (larger file)")

    g = p.add_argument_group("run control")
    g.add_argument("--sfreq", type=float, default=250.0, help="resample target (Hz)")
    g.add_argument("--notch", type=float, default=60.0,
                   help="line frequency; use 50 in Europe, 0 to skip")
    g.add_argument("--tmin", type=float, default=5.0, help="skip first N seconds")
    g.add_argument("--duration", type=float, default=120.0,
                   help="seconds analysed per recording (0 = all available)")
    g.add_argument("--surrogates", type=int, default=0,
                   help="surrogate windows (or shifts) per recording for an MI floor")
    g.add_argument("--jobs", type=int, default=0,
                   help="parallel worker processes; 0 = all cores but one")
    g.add_argument("--limit", type=int, default=None, help="debug: first N recordings")
    g.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def common_channels(recs, montage, explicit=None):
    """Channel set present in every recording, in the first recording's order."""
    import mne
    sets, order = [], None
    for r in recs:
        raw = mne.io.read_raw_bdf(r.path, preload=False, verbose="ERROR")
        names = explicit if explicit else mc.montage_channels(raw, montage)
        if not names:
            names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
        names = [n for n in names if n in raw.ch_names]
        sets.append(set(names))
        if order is None:
            order = names
    keep = set.intersection(*sets) if sets else set()
    return [c for c in order if c in keep]


# --------------------------------------------------------------------------
# Worker -- one recording, all bands. Module level so it can be pickled.
# --------------------------------------------------------------------------
def process_recording(job):
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    idx, path, label, chans, cfg = job
    rng = np.random.default_rng(cfg["seed"] + idx)
    n_c, bands = len(chans), cfg["bands"]
    out = {
        "idx": idx, "label": label,
        "mi": np.full((len(bands), n_c, n_c), np.nan),
        "sd": np.full((len(bands), n_c, n_c), np.nan),
        "null": np.full((len(bands), n_c, n_c), np.nan),
        "n_windows": np.zeros(len(bands), dtype=int),
        "timecourse": [None] * len(bands),
        "windows": [None] * len(bands),
        "n_eff": {},
    }
    try:
        data, names, meta = mc.load_and_preprocess(
            path, picks=chans, l_freq=1.0, h_freq=45.0,
            notch=cfg["notch"] or None, sfreq=cfg["sfreq"], tmin=cfg["tmin"],
            duration=cfg["duration"] or None, montage_name=cfg["montage"],
            reference=cfg["reference"])
    except Exception as exc:
        out["error"] = str(exc)
        return out

    order = [names.index(c) for c in chans]
    data = data[order]
    sf = meta["sfreq"]
    out["meta"] = meta

    for bi, band in enumerate(bands):
        lo, hi = mc.BANDS[band]
        x = data if band == "broadband" else mc.band_filter(data, sf, (lo, hi))
        out["n_eff"][band] = mc.effective_n(x)

        if cfg["window"] and cfg["window"] > 0:
            stack, bounds = mc.windowed_mi(
                x, sf, window_s=cfg["window"], step_s=cfg["window_step"],
                estimator=cfg["estimator"], k=cfg["k"], bins=cfg["bins"],
                reject_z=cfg["reject_z"], rng=rng)
            if len(stack) == 0:
                continue
            with np.errstate(invalid="ignore"):
                out["mi"][bi] = np.nanmean(stack, axis=0)
                out["sd"][bi] = (np.nanstd(stack, axis=0, ddof=1)
                                 if len(stack) > 1 else np.zeros_like(stack[0]))
            out["n_windows"][bi] = len(stack)
            iu = np.triu_indices(n_c, 1)
            out["timecourse"][bi] = np.nanmean(stack[:, iu[0], iu[1]], axis=1)
            if cfg["save_windows"]:
                out["windows"][bi] = stack.astype(np.float32)
            if cfg["surrogates"]:
                out["null"][bi] = mc.windowed_null(
                    x, sf, window_s=cfg["window"], n_windows=cfg["surrogates"],
                    estimator=cfg["estimator"], k=cfg["k"], bins=cfg["bins"], rng=rng)
        else:
            if cfg["estimator"] == "ksg":
                out["mi"][bi] = mc.ksg_mi_matrix(x, k=cfg["k"], rng=rng)
            else:
                out["mi"][bi] = mc.estimator(cfg["estimator"])(x)
            out["n_windows"][bi] = 1
            if cfg["surrogates"] and cfg["estimator"] == "gcmi":
                out["null"][bi] = mc.gcmi_null_matrix(
                    x, n_surrogates=cfg["surrogates"], rng=rng)
    return out


def main(argv=None):
    args = parse_args(argv)

    root = args.bids_root or mc.find_bids_root()
    if root is None:
        sys.exit("No BIDS dataset found nearby -- pass --bids-root explicitly.")
    print(f"Dataset: {root}")

    recs = mc.find_recordings(root, task=args.task)
    if args.limit:
        recs = recs[: args.limit]
    if not recs:
        sys.exit(f"No *task-{args.task}_eeg.bdf found under {root}")
    print(f"Found {len(recs)} recordings "
          f"({len({r.subject for r in recs})} subjects)")

    chans = common_channels(recs, args.montage, args.channels)
    if args.n_channels:
        chans = chans[: args.n_channels]
    if len(chans) < 2:
        sys.exit("Fewer than 2 usable channels after selection -- check --montage "
                 "or pass --channels explicitly.")
    print(f"Using {len(chans)} channels: {', '.join(chans)}")

    n_b, n_r, n_c = len(args.bands), len(recs), len(chans)
    n_pairs = n_c * (n_c - 1) // 2
    if args.window:
        per_rec = int((args.duration or 120) / (args.window_step or args.window))
        print(f"Windowed: {args.window*1000:.0f} ms, ~{per_rec} windows per "
              f"recording, {n_pairs} pairs, estimator={args.estimator}")
    else:
        print(f"Whole-recording estimates, {n_pairs} pairs, "
              f"estimator={args.estimator}")

    cfg = dict(bands=args.bands, estimator=args.estimator, k=args.k, bins=args.bins,
               window=args.window, window_step=args.window_step,
               reject_z=args.reject_z, save_windows=args.save_windows,
               sfreq=args.sfreq, notch=args.notch, tmin=args.tmin,
               duration=args.duration, montage=args.montage,
               reference=args.reference, surrogates=args.surrogates, seed=args.seed)
    jobs = [(i, r.path, r.label, chans, cfg) for i, r in enumerate(recs)]

    n_jobs = min(args.jobs or max((os.cpu_count() or 2) - 1, 1), len(jobs))

    if args.window:
        probe = np.random.default_rng(0).standard_normal(
            (n_c, max(int(args.window * args.sfreq), 16)))
        t = time.perf_counter()
        (mc.ksg_mi_matrix(probe, k=args.k) if args.estimator == "ksg"
         else mc.estimator(args.estimator)(probe))
        per_win = time.perf_counter() - t
        est = per_win * (per_rec + args.surrogates) * n_b * n_r / n_jobs * 1.25
        pretty = f"{est:.0f} s" if est < 90 else f"{est/60:.0f} min"
        print(f"~{per_win*1000:.0f} ms per window on this machine -> roughly "
              f"{pretty} total across {n_jobs} processes")
        if est > 3600:
            print("  (that is over an hour -- --estimator gcmi is ~100x faster "
                  "for a first look, or use --limit / a shorter --duration)")

    mi = np.full((n_b, n_r, n_c, n_c), np.nan)
    sd = np.full((n_b, n_r, n_c, n_c), np.nan)
    null = np.full((n_b, n_r, n_c, n_c), np.nan)
    n_win = np.zeros((n_b, n_r), dtype=int)
    n_eff = np.full((n_b, n_r), np.nan)
    timecourses, window_stacks, per_rec_meta = {}, {}, [None] * n_r
    t0 = time.time()
    done = 0

    def absorb(res):
        nonlocal done
        done += 1
        ri = res["idx"]
        if "error" in res:
            print(f"  [{done}/{n_r}] {res['label']:22s} FAILED ({res['error']})")
            per_rec_meta[ri] = {"label": res["label"], "error": res["error"]}
            return
        mi[:, ri], sd[:, ri], null[:, ri] = res["mi"], res["sd"], res["null"]
        n_win[:, ri] = res["n_windows"]
        for bi, band in enumerate(args.bands):
            n_eff[bi, ri] = res["n_eff"].get(band, np.nan)
            if res["timecourse"][bi] is not None:
                timecourses[(bi, ri)] = res["timecourse"][bi]
            if res["windows"][bi] is not None:
                window_stacks[(bi, ri)] = res["windows"][bi]
        per_rec_meta[ri] = {"label": res["label"], **res.get("meta", {})}
        eta = (time.time() - t0) / done * (n_r - done)
        print(f"  [{done}/{n_r}] {res['label']:22s} {res['n_windows'][0]:4d} win  "
              f"mean MI = {np.nanmean(res['mi'][0]):.3f} bits   "
              f"ETA {eta/60:.1f} min")

    print(f"Running on {n_jobs} process{'es' if n_jobs > 1 else ''}...")
    if n_jobs == 1:
        for job in jobs:
            absorb(process_recording(job))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(process_recording, j) for j in jobs]
            for fut in as_completed(futs):
                absorb(fut.result())

    # ---- pack ------------------------------------------------------------
    payload = dict(
        mi=mi, mi_sd=sd, n_windows=n_win, n_eff=n_eff,
        bands=np.array(args.bands), ch_names=np.array(chans),
        subjects=np.array([r.subject for r in recs]),
        sessions=np.array([r.session for r in recs]),
        groups=np.array([r.group for r in recs]),
        labels=np.array([r.label for r in recs]),
        paths=np.array([r.path for r in recs]),
        meta=np.array(json.dumps({"args": vars(args), "recordings": per_rec_meta})),
    )
    if args.surrogates:
        payload["mi_null"] = null
    pos = mc.sensor_positions_2d(chans, args.montage)
    if pos is not None:
        payload["pos"] = pos
    if timecourses:
        w_max = max(len(v) for v in timecourses.values())
        tc = np.full((n_b, n_r, w_max), np.nan, dtype=np.float32)
        for (bi, ri), v in timecourses.items():
            tc[bi, ri, :len(v)] = v
        payload["mi_timecourse"] = tc
    if window_stacks:
        w_max = max(v.shape[0] for v in window_stacks.values())
        ws = np.full((n_b, n_r, w_max, n_c, n_c), np.nan, dtype=np.float32)
        for (bi, ri), v in window_stacks.items():
            ws[bi, ri, :v.shape[0]] = v
        payload["mi_windows"] = ws
        print(f"Per-window matrices: {ws.nbytes/1e6:.0f} MB before compression")

    np.savez_compressed(args.out, **payload)
    ok = int(np.isfinite(mi[0]).any(axis=(1, 2)).sum())
    print(f"\nSaved {args.out}  ({ok}/{n_r} recordings, "
          f"{(time.time()-t0)/60:.1f} min elapsed)")
    print("Next:  uv run neurovision app")


if __name__ == "__main__":
    main()
