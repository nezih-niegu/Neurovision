#!/usr/bin/env python
"""
pairwise_distance.py
====================
Reconstruct every channel from every *single* other channel, and regress the
result on inter-electrode distance.

    uv run python pairwise_distance.py --recordings 12

This is the direct test of the paper's central claim. The per-target benchmark
lets you observe that a target's selected channels happen to be its neighbours;
this measures how much of the MI graph is distance, as a number. With 16
channels it yields 16 x 15 = 240 ordered pairs per recording, each with a
mutual information, an inter-electrode distance, and a reconstruction R^2.

Three quantities come out of it:

* corr(MI, distance)  -- how much the edge weight itself is geometry
* corr(R^2, distance) -- how much reconstructability is geometry
* partial corr(MI, R^2 | distance) -- whether MI predicts reconstruction at all
  once distance is held constant. This is the number that decides whether the
  MI graph carries information beyond the montage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from neurovision import core as mc
from neurovision import mine as mi_nn
from neurovision.progress import Progress


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-root", default=None)
    p.add_argument("--out", default="results/pairwise")
    p.add_argument("--task", default="rest")
    p.add_argument("--recordings", type=int, default=12,
                   help="how many recordings; 0 = all. Every recording costs "
                        "n_channels*(n_channels-1) model fits, so start small.")
    p.add_argument("--window", type=int, default=25,
                   help="window in samples; 25 = 100 ms at 250 Hz")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--n-channels", type=int, default=16)
    p.add_argument("--montage", default="biosemi32")
    p.add_argument("--reference", default="average",
                   choices=["average", "csd", "none"])
    p.add_argument("--band", default="broadband", choices=list(mc.BANDS))
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--sfreq", type=float, default=250.0)
    p.add_argument("--notch", type=float, default=60.0)
    p.add_argument("--tmin", type=float, default=5.0)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--gap", type=int, default=500)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def partial_corr(x, y, z):
    """Correlation of x and y with the linear effect of z removed."""
    x, y, z = map(np.asarray, (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(x)


def main(argv=None):
    args = parse_args(argv)
    dev = mi_nn.pick_device(args.device)
    print(f"Device: {mi_nn.device_report(dev)}")

    root = args.bids_root or mc.find_bids_root()
    if root is None:
        sys.exit("No BIDS dataset found -- pass --bids-root.")
    recs = mc.find_recordings(root, task=args.task)
    if args.recordings:
        recs = recs[: args.recordings]
    if not recs:
        sys.exit("No recordings found.")

    n_ch = args.n_channels
    print(f"{len(recs)} recordings x {n_ch}x{n_ch - 1} = "
          f"{len(recs) * n_ch * (n_ch - 1)} single-channel reconstructions")

    rows = []
    t0 = time.time()
    for ri, rec in enumerate(recs):
        try:
            data, names, meta = mc.load_and_preprocess(
                rec.path, picks=None, sfreq=args.sfreq, notch=args.notch or None,
                tmin=args.tmin, duration=args.duration or None,
                montage_name=args.montage, reference=args.reference)
        except Exception as exc:
            print(f"  {rec.label}: FAILED ({exc})")
            continue
        data, names = data[:n_ch], names[:n_ch]
        if args.band != "broadband":
            data = mc.band_filter(data, meta["sfreq"], mc.BANDS[args.band])

        pos = mc.sensor_positions_2d(names, args.montage)
        import mne
        p3 = mne.channels.make_standard_montage(args.montage).get_positions()["ch_pos"]
        lut = {k.lower(): np.asarray(v, float) for k, v in p3.items()}
        xyz = np.array([lut.get(c.lower(), [np.nan] * 3) for c in names])
        D = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)

        cut = int(data.shape[1] * (1 - args.test_frac)) - args.gap
        MI = mc.gcmi_matrix(data[:, :cut])          # ranking MI, train block only

        bar = Progress(n_ch * (n_ch - 1), f"{rec.label} pairs")
        for t in range(n_ch):
            for src in range(n_ch):
                if src == t:
                    continue
                X, y = mi_nn.make_windows(data, t, [src], args.window, args.stride)
                Xtr, ytr, Xte, yte = mi_nn.split_contiguous(
                    X, y, args.test_frac, args.gap)
                Xtr, Xte, ytr, yte = mi_nn.standardize(Xtr, Xte, ytr, yte)
                r = mi_nn.train_predictor(
                    Xtr, ytr, Xte, yte, device=dev, epochs=args.epochs,
                    hidden=args.hidden, layers=2, seed=args.seed)
                rows.append(dict(recording=rec.label, group=rec.group,
                                 subject=rec.subject, session=rec.session,
                                 target=names[t], source=names[src],
                                 mi_bits=float(MI[t, src]),
                                 distance=float(D[t, src]), r2=r.r2))
                bar.update(note=f"{names[t]}<-{names[src]}")
        bar.close()
        eta = (time.time() - t0) / (ri + 1) * (len(recs) - ri - 1)
        print(f"  [{ri+1}/{len(recs)}] {rec.label:22s} [{rec.group:6s}]  "
              f"recording ETA {eta/60:.1f} min")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)

    print("\n" + "=" * 74)
    print(f"PAIRWISE GEOMETRY TEST  ({len(df)} target-source pairs)")
    print("=" * 74)
    r1, p1 = stats.pearsonr(df.distance, df.mi_bits)
    r2_, p2 = stats.pearsonr(df.distance, df.r2)
    r3, p3_ = stats.pearsonr(df.mi_bits, df.r2)
    print(f"  MI    vs distance : r = {r1:+.3f}  (p = {p1:.2e})   "
          f"-> {100*r1**2:.0f}% of edge-weight variance is distance")
    print(f"  R^2   vs distance : r = {r2_:+.3f}  (p = {p2:.2e})   "
          f"-> {100*r2_**2:.0f}% of reconstruction variance is distance")
    print(f"  MI    vs R^2      : r = {r3:+.3f}  (p = {p3_:.2e})")
    rp, pp, n = partial_corr(df.mi_bits, df.r2, df.distance)
    print(f"\n  partial corr(MI, R^2 | distance) = {rp:+.3f}  "
          f"(p = {pp:.2e}, n = {n})")
    print("  This is the decisive number. If it collapses towards zero, the MI\n"
          "  graph adds nothing beyond electrode spacing. If it stays high, MI\n"
          "  carries information the montage does not.")

    print("\n  By group:")
    for g, d in df.groupby("group"):
        rg, _ = stats.pearsonr(d.distance, d.mi_bits)
        rpg, _, _ = partial_corr(d.mi_bits, d.r2, d.distance)
        print(f"    {g:8s} MI-distance r = {rg:+.3f} | "
              f"partial corr(MI, R^2 | dist) = {rpg:+.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        ax[0].scatter(df.distance, df.mi_bits, s=5, alpha=0.15, rasterized=True)
        ax[0].set_xlabel("inter-electrode distance")
        ax[0].set_ylabel("MI (bits)")
        ax[0].set_title(f"Edge weight vs distance (r = {r1:.2f})")
        ax[1].scatter(df.distance, df.r2, s=5, alpha=0.15, color="#1b7837",
                      rasterized=True)
        ax[1].set_xlabel("inter-electrode distance")
        ax[1].set_ylabel("$R^2$")
        ax[1].set_title(f"Reconstruction vs distance (r = {r2_:.2f})")
        ax[2].scatter(df.mi_bits, df.r2, s=5, alpha=0.15, color="#7570b3",
                      rasterized=True)
        ax[2].set_xlabel("MI (bits)")
        ax[2].set_ylabel("$R^2$")
        ax[2].set_title(f"MI vs reconstruction (r = {r3:.2f}, "
                        f"partial {rp:.2f})")
        fig.tight_layout()
        fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {out.with_suffix('.csv')}, {out.with_suffix('.png')}")
    except Exception as exc:
        print(f"\nWrote {out.with_suffix('.csv')} (plot skipped: {exc})")
    return df


if __name__ == "__main__":
    main()
