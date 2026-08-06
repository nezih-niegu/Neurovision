#!/usr/bin/env python
"""
neurovision.benchmark
=====================
Does mutual information tell you which channels can reconstruct a hidden one?

The experiment, for each held-out target channel:

1. Split the recording **contiguously** into train and test blocks.
2. Rank every other channel by its MI to the target, computed on the *training
   block only*.
3. Pick k channels three ways -- highest MI, lowest MI, and random -- and for
   each, train an MLP to predict the target's value from a window of those
   channels. Score by R^2 on the test block.
4. Estimate I(window of selected channels ; target) with MINE, and compare the
   measured R^2 against the Gaussian prediction R^2 = 1 - 2^(-2I).

Step 2 is where this kind of benchmark usually goes wrong: ranking channels on
the same data you evaluate on leaks the answer, and every selection looks good.
Ranking on the training block only makes "high MI predicts better" a real
out-of-sample claim.

Sweeping `--windows` shows how much temporal context the predictor needs, and
whether the MI advantage survives as the window grows.

Example
-------
    uv run neurovision mine --target Cz --k 6 --windows 25 50 100 150 300 \\
        --device auto --recordings 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from neurovision import core as mc
from neurovision import mine as mi_nn


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-root", default=None)
    p.add_argument("--out", default="benchmark")
    p.add_argument("--task", default="rest")
    p.add_argument("--recordings", type=int, default=0,
                   help="how many recordings to run; 0 = all of them")
    p.add_argument("--groups", nargs="+", default=None,
                   help="restrict to these groups, e.g. HC PD-OFF PD-ON")
    p.add_argument("--target", default=None,
                   help="channel to hide; default = the highest-MI channel")
    p.add_argument("--all-targets", action="store_true",
                   help="repeat the whole benchmark once per channel, hiding "
                        "each in turn. Preprocessing is shared across targets, "
                        "so this costs far less than N separate runs.")
    p.add_argument("--targets", nargs="+", default=None,
                   help="explicit list of channels to use as targets")
    p.add_argument("--k", type=int, default=6, help="channels per selection")
    p.add_argument("--windows", type=int, nargs="+",
                   default=[25, 50, 100, 150, 300],
                   help="window lengths in SAMPLES (150 = 600 ms at 250 Hz)")
    p.add_argument("--selections", nargs="+",
                   default=["top", "random", "bottom"],
                   choices=["top", "random", "bottom", "all", "nearest", "single"],
                   help="nearest = the k spatially closest electrodes (geometry "
                        "baseline); single = the single highest-MI channel")
    p.add_argument("--stride", type=int, default=2,
                   help="hop between windows; >1 reduces near-duplicate rows")
    p.add_argument("--band", default="broadband", choices=list(mc.BANDS))
    p.add_argument("--rank-estimator", default="gcmi",
                   choices=["gcmi", "ksg", "binned", "pearson", "distance"],
                   help="edge weight used to rank channels, computed on the "
                        "training block only. pearson and distance are baselines: "
                        "if either ranks as well as MI, the graph adds nothing")
    p.add_argument("--predictor", default="mlp", choices=["mlp", "ridge"],
                   help="ridge is a linear baseline; if it matches the MLP, the "
                        "reconstruction needs no nonlinearity")

    g = p.add_argument_group("trace plots")
    g.add_argument("--trace-seconds", type=float, default=6.0,
                   help="seconds of held-out signal to draw in the trace figures")
    g.add_argument("--trace-start", type=float, default=2.0,
                   help="offset into the test block where the excerpt starts")
    g.add_argument("--no-traces", action="store_true",
                   help="skip the predicted-vs-actual figures")
    g.add_argument("--max-trace-figures", type=int, default=6,
                   help="cap on trace figures written; 0 = no cap. Matters with "
                        "--all-targets, which would otherwise emit one per "
                        "recording per target")

    g = p.add_argument_group("data")
    g.add_argument("--montage", default="biosemi32")
    g.add_argument("--reference", default="average", choices=["average", "csd", "none"])
    g.add_argument("--n-channels", type=int, default=16)
    g.add_argument("--sfreq", type=float, default=250.0)
    g.add_argument("--notch", type=float, default=60.0)
    g.add_argument("--tmin", type=float, default=5.0)
    g.add_argument("--duration", type=float, default=180.0)
    g.add_argument("--test-frac", type=float, default=0.25)
    g.add_argument("--gap", type=int, default=500,
                   help="samples discarded between train and test blocks")

    g = p.add_argument_group("models")
    g.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--hidden", type=int, default=256)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--mine-iters", type=int, default=1500,
                   help="MINE training steps; 0 skips the MI estimation")
    g.add_argument("--mine-batch", type=int, default=512)
    g.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def pearson_matrix(x: np.ndarray) -> np.ndarray:
    """|Pearson r| between every channel pair, as an edge weight.

    The baseline that matters most. Gaussian-copula MI is a monotone function
    of *rank* correlation, so if plain linear correlation ranks channels equally
    well, the information-theoretic framing buys nothing and the paper should
    say so.
    """
    m = np.abs(np.corrcoef(x))
    np.fill_diagonal(m, np.nan)
    return m


def distance_matrix(ch_names, montage_name="biosemi32") -> np.ndarray:
    """Negated inter-electrode distance, so that larger means 'closer'.

    Used both as a pure-geometry edge weight and to build the NEAREST-k
    selection. If picking the k physically closest electrodes matches picking
    the k highest-MI ones, the graph is not adding anything to the montage.
    """
    import mne
    pos = mne.channels.make_standard_montage(montage_name).get_positions()["ch_pos"]
    lut = {k.lower(): np.asarray(v, float) for k, v in pos.items()}
    xyz = np.array([lut.get(c.lower(), [np.nan] * 3) for c in ch_names])
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(d, np.nan)
    return -d


def rank_channels(x_train: np.ndarray, target: int, estimator: str,
                  ch_names=None, montage="biosemi32") -> np.ndarray:
    """Channel indices ordered by decreasing edge weight to `target`."""
    if estimator == "ksg":
        m = mc.ksg_mi_matrix(x_train, k=4, rng=0)
    elif estimator == "binned":
        m = mc.binned_mi_matrix(x_train)
    elif estimator == "pearson":
        m = pearson_matrix(x_train)
    elif estimator == "distance":
        m = distance_matrix(ch_names, montage)
    else:
        m = mc.gcmi_matrix(x_train)
    others = np.array([i for i in range(m.shape[0]) if i != target])
    col = np.nan_to_num(m[target][others], nan=-np.inf)
    # The target must be dropped from the pool, not merely sent to the back of
    # it: with -inf it lands last and silently becomes part of the bottom-k
    # selection, which would then "predict" itself perfectly.
    return others[np.argsort(col)[::-1]], m


def run_cell(data, target, sources, window, args, dev, seed):
    """One (selection, window) cell: build windows, fit, score, estimate MI."""
    X, y = mi_nn.make_windows(data, target, list(sources), window, args.stride)
    Xtr, ytr, Xte, yte = mi_nn.split_contiguous(X, y, args.test_frac, args.gap)
    Xtr, Xte, ytr, yte = mi_nn.standardize(Xtr, Xte, ytr, yte)

    if args.predictor == "ridge":
        pred = mi_nn.fit_ridge(Xtr, ytr, Xte, yte)
    else:
        pred = mi_nn.train_predictor(
            Xtr, ytr, Xte, yte, device=dev, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, hidden=args.hidden, seed=seed)

    traces = None if args.no_traces else (pred.y_true, pred.y_pred)
    row = dict(window=window, window_ms=1000 * window / args.sfreq,
               n_train=len(Xtr), n_test=len(Xte), d_in=X.shape[1],
               r2=pred.r2, r2_train=pred.r2_train, rmse=pred.rmse,
               epochs=pred.epochs_run, fit_s=pred.seconds,
               mi_implied_by_r2=mi_nn.mi_from_r2(pred.r2))

    if args.mine_iters:
        res = mi_nn.estimate_mi(
            Xtr, ytr[:, None], device=dev, iters=args.mine_iters,
            batch_size=args.mine_batch, hidden=args.hidden, seed=seed)
        row.update(mine_bits=res.mi_bits, mine_bits_train=res.mi_bits_train,
                   mine_infonce_bits=res.mi_infonce_bits,
                   mine_ceiling_bits=res.infonce_ceiling_bits,
                   r2_implied_by_mine=mi_nn.r2_from_mi(res.mi_bits),
                   mine_s=res.seconds)
    return row, traces


def main(argv=None):
    args = parse_args(argv)
    dev = mi_nn.pick_device(args.device)
    print(f"Device: {mi_nn.device_report(dev)}")

    root = args.bids_root or mc.find_bids_root()
    if root is None:
        sys.exit("No BIDS dataset found nearby -- pass --bids-root explicitly.")
    recs = mc.find_recordings(root, task=args.task)
    if args.groups:
        want = {g.upper() for g in args.groups}
        recs = [r for r in recs if r.group.upper() in want]
    if args.recordings:
        recs = recs[: args.recordings]
    if not recs:
        sys.exit(f"No *task-{args.task}_eeg.bdf found under {root}")
    by_group = {}
    for r in recs:
        by_group.setdefault(r.group, []).append(r.subject)
    print(f"Dataset: {root}  ({len(recs)} recordings)")
    for g, subs in sorted(by_group.items()):
        print(f"   {g:8s} {len(subs):3d} recordings, {len(set(subs))} subjects")

    # ---- pass 1: preprocess everything, then fix ONE target for all groups --
    # Comparing groups only means something if every recording reconstructs the
    # same channel from a comparable task. Letting each recording choose its own
    # target (the old default) would confound group with target identity.
    print("\nLoading and preprocessing...")
    loaded = []
    for rec in recs:
        try:
            data, names, meta = mc.load_and_preprocess(
                rec.path, picks=None, sfreq=args.sfreq, notch=args.notch or None,
                tmin=args.tmin, duration=args.duration or None,
                montage_name=args.montage, reference=args.reference)
        except Exception as exc:
            print(f"   {rec.label}: FAILED ({exc})")
            continue
        if args.n_channels:
            data, names = data[: args.n_channels], names[: args.n_channels]
        if args.band != "broadband":
            data = mc.band_filter(data, meta["sfreq"], mc.BANDS[args.band])
        if len(names) < args.k + 2:
            sys.exit(f"Need at least {args.k + 2} channels, have {len(names)}")
        cut = int(data.shape[1] * (1 - args.test_frac)) - args.gap
        loaded.append((rec, data, names, meta, mc.gcmi_matrix(data[:, :cut])))
    if not loaded:
        sys.exit("Every recording failed to load.")

    names0 = loaded[0][2]
    if args.targets:
        targets = [names0.index(t) for t in args.targets if t in names0]
        missing = [t for t in args.targets if t not in names0]
        if missing:
            print(f"   (ignoring unknown channels: {', '.join(missing)})")
    elif args.all_targets:
        targets = list(range(len(names0)))
    elif args.target in names0:
        targets = [names0.index(args.target)]
    else:
        strength = np.nanmean([np.nanmean(m, axis=1) for _, _, _, _, m in loaded],
                              axis=0)
        targets = [int(np.nanargmax(strength))]
    if not targets:
        sys.exit("No valid target channels.")
    print(f"Targets ({len(targets)}): {', '.join(names0[t] for t in targets)}")

    n_cells = len(targets) * len(loaded) * len(args.selections) * len(args.windows)
    print(f"{n_cells} cells to fit"
          + ("" if args.mine_iters else "  (MINE disabled)"))

    rows, rank_info, trace_store = [], [], {}
    for ti, target in enumerate(targets):
        if len(targets) > 1:
            print(f"\n{'='*70}\nTARGET {ti+1}/{len(targets)}: {names0[target]}\n{'='*70}")
        run_target(target, loaded, args, dev, rows, rank_info, trace_store)

    return finish(rows, rank_info, trace_store, args, dev, targets, names0)


def run_target(target, loaded, args, dev, rows, rank_info, trace_store):
    """One target channel, every recording. Appends to the shared accumulators."""
    import numpy as np
    for ri, (rec, data, names, meta, _) in enumerate(loaded):
        cut = int(data.shape[1] * (1 - args.test_frac)) - args.gap
        train_block = data[:, :cut]
        order, mi_mat = rank_channels(train_block, target, args.rank_estimator,
                                      ch_names=names, montage=args.montage)
        dist_mat = distance_matrix(names, args.montage)
        rng = np.random.default_rng(args.seed + ri)

        picks = {}
        if "top" in args.selections:
            picks["top"] = order[: args.k]
        if "bottom" in args.selections:
            picks["bottom"] = order[-args.k:]
        if "random" in args.selections:
            pool = [i for i in range(len(names)) if i != target]
            picks["random"] = rng.choice(pool, args.k, replace=False)
        if "all" in args.selections:
            picks["all"] = np.array([i for i in range(len(names)) if i != target])
        if "nearest" in args.selections:
            d = dist_mat[target].copy()
            others = np.array([i for i in range(len(names)) if i != target])
            picks["nearest"] = others[np.argsort(-d[others])][: args.k]
        if "single" in args.selections:
            picks["single"] = order[:1]

        print(f"\n[{ri+1}/{len(loaded)}] {rec.label} [{rec.group}] — "
              f"target {names[target]}, {len(names)} channels, "
              f"{meta['duration_s']:.0f}s")
        for tag, sel in picks.items():
            mis = mi_mat[target][sel]
            print(f"   {tag:6s}: {', '.join(names[i] for i in sel)}  "
                  f"(train MI {np.nanmean(mis):.3f} bits)")
            rank_info.append(dict(recording=rec.label, group=rec.group,
                                  subject=rec.subject, target=names[target],
                                  selection=tag,
                                  channels=[names[i] for i in sel],
                                  mean_pairwise_mi=float(np.nanmean(mis))))

        for tag, sel in picks.items():
            for w in args.windows:
                if w > train_block.shape[1] // 4:
                    continue
                row, traces = run_cell(data, target, sel, w, args, dev,
                                       args.seed + ri)
                if traces is not None:
                    trace_store[(rec.label, names[target], tag, w)] = traces
                row.update(recording=rec.label, group=rec.group,
                           subject=rec.subject, session=rec.session,
                           edge_weight=args.rank_estimator,
                           predictor=args.predictor,
                           target=names[target], selection=tag, k=len(sel),
                           mean_pairwise_mi=float(np.nanmean(mi_mat[target][sel])))
                rows.append(row)
                extra = (f"  MINE {row['mine_bits']:.2f} b"
                         if "mine_bits" in row else "")
                print(f"   {tag:6s} w={w:4d} ({row['window_ms']:6.0f} ms)  "
                      f"R2 = {row['r2']:+.3f}{extra}")

def finish(rows, rank_info, trace_store, args, dev, targets, names0):
    """Aggregate, test, plot and write. Shared by single- and multi-target runs."""
    import json
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = out.with_suffix(".csv")
    df.to_csv(csv, index=False)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"args": vars(args), "device": mi_nn.device_report(dev),
                   "selections": rank_info}, f, indent=2)

    multi = df["target"].nunique() > 1

    print("\n" + "=" * 74)
    print("MEAN R^2 BY SELECTION AND WINDOW"
          + (f"  (pooled over {df['target'].nunique()} targets)" if multi else ""))
    print("=" * 74)
    piv = df.pivot_table(index="window_ms", columns="selection", values="r2")
    print(piv.round(3).to_string())

    if multi:
        print("\nBY TARGET (averaged over windows and recordings)")
        bt = df.pivot_table(index="target", columns="selection", values="r2")
        if "top" in bt and "bottom" in bt:
            bt["advantage"] = bt["top"] - bt["bottom"]
            bt = bt.sort_values("advantage", ascending=False)
        print(bt.round(3).to_string())
        if "top" in bt and "bottom" in bt:
            n_ok = int((bt["advantage"] > 0).sum())
            print(f"\nhigh-MI beats low-MI for {n_ok}/{len(bt)} targets; "
                  f"advantage ranges {bt['advantage'].min():+.3f} to "
                  f"{bt['advantage'].max():+.3f}")

    if "top" in piv.columns and "bottom" in piv.columns:
        gap = (piv["top"] - piv["bottom"]).mean()
        print(f"\nmean R^2 advantage of high-MI over low-MI channels: {gap:+.3f}")
    if "mine_bits" in df.columns:
        print("\nMINE vs measured prediction (Gaussian identity R^2 = 1 - 2^-2I):")
        cmp = df.groupby("selection")[
            ["mine_bits", "r2_implied_by_mine", "r2", "mi_implied_by_r2"]].mean()
        print(cmp.round(3).to_string())

    # ---- group comparison -------------------------------------------------
    stats_df = None
    if df["group"].nunique() > 1:
        print("\n" + "=" * 74)
        print("BY GROUP  (one value per subject-session, averaged over windows)")
        print("=" * 74)
        per = (df.groupby(["selection", "group", "subject", "session"])
               [[c for c in ("r2", "mine_bits", "mean_pairwise_mi") if c in df]]
               .mean().reset_index())
        print(per.groupby(["selection", "group"])
              .agg(n=("subject", "size"),
                   **{c: (c, "mean") for c in ("r2", "mean_pairwise_mi")
                      if c in per})
              .round(3).to_string())

        stats_df = compare_groups(df, rank_info)
        if len(stats_df):
            print("\nGROUP CONTRASTS  (BH-FDR across the whole family)")
            show = stats_df[["selection", "metric", "group_a", "group_b", "test",
                             "n_a", "n_b", "diff", "cohens_d", "p", "q_fdr",
                             "significant"]]
            print(show.round(4).to_string(index=False))
            n_sig = int(stats_df["significant"].sum())
            print(f"\n{n_sig} of {len(stats_df)} contrasts survive FDR at q < 0.05.")
            if n_sig == 0:
                print("No group difference survives correction. With ~15 subjects "
                      "per group that is weak evidence of absence, not evidence "
                      "of no effect -- check the per-subject scatter in "
                      "*_groups.png before concluding anything.")
            stats_df.to_csv(out.with_name(out.stem + "_group_stats.csv"),
                            index=False)
        else:
            print("\nNo contrast had at least 3 subjects per group, so no test "
                  "was run. The descriptive table above is all this sample "
                  "supports.")

        M, consensus = channel_overlap(rank_info)
        if M is not None:
            print("\nDo the groups agree on WHICH channels are informative?")
            print("Jaccard overlap of each group's consensus top-k:")
            print(M.round(2).to_string())
            for g, chans in consensus.items():
                print(f"   {g:8s}: {', '.join(sorted(chans))}")

    try:
        plot(df, out.with_suffix(".png"), args)
        extra = []
        if df["group"].nunique() > 1:
            extra.append(plot_groups(df, out.with_suffix(".png"), args))
        if trace_store:
            # one figure per (recording, target); with many targets this is
            # hundreds of files, so cap it unless explicitly asked for more
            keep = trace_store
            cap = args.max_trace_figures
            if cap and len({(k[0], k[1]) for k in trace_store}) > cap:
                seen, keep = [], {}
                for k, v in trace_store.items():
                    if (k[0], k[1]) not in seen:
                        if len(seen) >= cap:
                            continue
                        seen.append((k[0], k[1]))
                    keep[k] = v
                print(f"(limiting trace figures to {cap} recording-target pairs; "
                      f"--max-trace-figures 0 for all)")
            extra += plot_traces(keep, df, out.with_suffix(".png"),
                                 args, args.sfreq)
            c = plot_scatter_and_spectra(trace_store, out.with_suffix(".png"),
                                         args, args.sfreq)
            if c:
                extra.append(c)
            npz = out.with_name(out.stem + "_traces.npz")
            np.savez_compressed(npz, **{
                "|".join(str(p) for p in k): np.stack(v)
                for k, v in trace_store.items()})
            extra.append(npz)
        made = ", ".join(str(x) for x in [csv, out.with_suffix(".json"),
                                          out.with_suffix(".png"), *extra])
        print(f"\nWrote {made}")
    except Exception as exc:
        print(f"\nWrote {csv} and {out.with_suffix('.json')} (plots skipped: {exc})")
    return df


def compare_groups(df, rank_info, alpha=0.05):
    """Group-level statistics, printed and returned as a DataFrame.

    Two different designs live in this dataset and they need different tests:

    * **PD-OFF vs PD-ON** is *within subject* -- the same person recorded twice
      on and off medication -- so it gets a paired test on the per-subject
      difference. Treating those as independent samples would throw away the
      pairing and badly understate the medication effect.
    * **HC vs PD** is between subject, so it gets Welch's t-test, which does not
      assume the groups share a variance.

    Sample sizes here are small (order 15 per group in ds002778), so effect
    sizes are reported alongside p-values: with n this size a non-significant
    result is weak evidence of absence, and a significant one deserves a look
    at the per-subject scatter before being believed.
    """
    import pandas as pd
    from scipy import stats

    out = []
    metrics = [m for m in ("r2", "mine_bits", "mean_pairwise_mi") if m in df]
    groups = sorted(df["group"].dropna().unique())

    for sel in sorted(df["selection"].unique()):
        d = df[df.selection == sel]
        for metric in metrics:
            # collapse to one value per subject-session first, so a recording
            # measured at five window lengths does not count five times
            per = (d.groupby(["group", "subject", "session"])[metric]
                   .mean().reset_index())
            desc = per.groupby("group")[metric].agg(["count", "mean", "std"])

            for a, b in [(x, y) for i, x in enumerate(groups)
                         for y in groups[i + 1:]]:
                A, B = per[per.group == a], per[per.group == b]
                paired = (a.startswith("PD") and b.startswith("PD")
                          and set(A.subject) & set(B.subject))
                if paired:
                    common = sorted(set(A.subject) & set(B.subject))
                    va = A.set_index("subject").loc[common, metric].to_numpy()
                    vb = B.set_index("subject").loc[common, metric].to_numpy()
                    if len(common) < 3:
                        continue
                    t, p = stats.ttest_rel(va, vb)
                    diff = va - vb
                    sd = diff.std(ddof=1)
                    d_eff = float(diff.mean() / sd) if sd > 1e-12 else np.nan
                    kind, n_a, n_b = "paired", len(common), len(common)
                else:
                    va = A[metric].to_numpy()
                    vb = B[metric].to_numpy()
                    if min(len(va), len(vb)) < 3:
                        continue
                    t, p = stats.ttest_ind(va, vb, equal_var=False)
                    sp = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)
                    # a vanishing pooled SD makes d explode to meaningless
                    # magnitudes rather than saying anything about effect size
                    d_eff = (float((va.mean() - vb.mean()) / sp)
                             if sp > 1e-12 else np.nan)
                    kind, n_a, n_b = "welch", len(va), len(vb)

                out.append(dict(
                    selection=sel, metric=metric, group_a=a, group_b=b,
                    test=kind, n_a=n_a, n_b=n_b,
                    mean_a=float(va.mean()), mean_b=float(vb.mean()),
                    diff=float(va.mean() - vb.mean()),
                    cohens_d=d_eff, t=float(t), p=float(p)))

    res = pd.DataFrame(out)
    if len(res):
        # one FDR correction over the whole family of tests, not per metric
        res["q_fdr"] = _bh(res["p"].to_numpy())
        res["significant"] = res["q_fdr"].fillna(1.0) < alpha
    return res


def _bh(p):
    """Benjamini-Hochberg step-up FDR, ignoring NaN p-values.

    NaNs occur legitimately here -- a contrast between two identical vectors
    gives no test statistic -- and without this they would propagate through
    the cumulative minimum and wipe out every q-value in the family.
    """
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def channel_overlap(rank_info):
    """How much do the groups agree about *which* channels carry information?

    A group difference in R^2 says the target got easier or harder to
    reconstruct. It does not say the informative channels moved. This measures
    the second thing: the Jaccard overlap between each group's most commonly
    selected top-k and every other group's.
    """
    import pandas as pd
    from collections import Counter

    tops = {}
    for r in rank_info:
        if r["selection"] != "top":
            continue
        tops.setdefault(r["group"], Counter()).update(r["channels"])
    if len(tops) < 2:
        return None, {}

    k = max(len(r["channels"]) for r in rank_info if r["selection"] == "top")
    consensus = {g: {c for c, _ in cnt.most_common(k)} for g, cnt in tops.items()}
    gs = sorted(consensus)
    M = pd.DataFrame(index=gs, columns=gs, dtype=float)
    for a in gs:
        for b in gs:
            u = consensus[a] | consensus[b]
            M.loc[a, b] = len(consensus[a] & consensus[b]) / len(u) if u else np.nan
    return M, consensus


def plot_groups(df, path, args):
    """Per-subject values by group, for every metric that varies across them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in ("r2", "mine_bits", "mean_pairwise_mi") if m in df]
    sels = sorted(df["selection"].unique())
    fig, axes = plt.subplots(len(metrics), len(sels), squeeze=False,
                             figsize=(3.7 * len(sels), 3.1 * len(metrics)))
    labels = {"r2": "test $R^2$", "mine_bits": "MINE (bits)",
              "mean_pairwise_mi": "mean pairwise MI (bits)"}
    groups = sorted(df["group"].dropna().unique())
    palette = {g: plt.get_cmap("tab10")(i) for i, g in enumerate(groups)}

    for i, metric in enumerate(metrics):
        for j, sel in enumerate(sels):
            ax = axes[i][j]
            d = df[df.selection == sel]
            per = (d.groupby(["group", "subject", "session"])[metric]
                   .mean().reset_index())
            data = [per[per.group == g][metric].to_numpy() for g in groups]
            bp = ax.boxplot(data, positions=range(len(groups)), widths=0.55,
                            showfliers=False, patch_artist=True)
            for patch, g in zip(bp["boxes"], groups):
                patch.set_facecolor(palette[g])
                patch.set_alpha(0.35)
            for x, (g, vals) in enumerate(zip(groups, data)):
                jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.22
                ax.scatter(x + jitter, vals, s=22, color=palette[g],
                           edgecolor="white", lw=0.5, zorder=3)
            # join the same PD subject across medication states
            pd_groups = [g for g in groups if g.startswith("PD")]
            if len(pd_groups) == 2:
                a, b = (per[per.group == g].set_index("subject")[metric]
                        for g in pd_groups)
                xa, xb = groups.index(pd_groups[0]), groups.index(pd_groups[1])
                for sub in set(a.index) & set(b.index):
                    ax.plot([xa, xb], [a[sub], b[sub]], color="0.5", lw=0.7,
                            alpha=0.7, zorder=2)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, fontsize=8)
            ax.set_ylabel(labels.get(metric, metric), fontsize=9)
            if i == 0:
                ax.set_title(f"{sel} channels", fontsize=10)
            ax.tick_params(labelsize=8)

    fig.suptitle("Group differences per subject "
                 "(grey lines join the same PD subject on and off medication)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = path.with_name(f"{path.stem}_groups.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_traces(trace_store, df, path, args, sfreq):
    """Grid of held-out signal: true channel against its reconstruction.

    One figure per recording; rows are selections, columns window lengths.
    Traces are in standardised units (the target was z-scored on training
    statistics), so the vertical scale is comparable across every panel.

    Note that at R^2 ~ 0.95 the prediction sits almost on top of the truth, and
    panels differing by 0.09 in R^2 look near-identical by eye. The R^2 in each
    panel title, and the residual spectrum in the calibration figure, are what
    actually separate the selections.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {"top": "#1b7837", "random": "#7570b3", "bottom": "#d95f02",
            "all": "#333333"}
    keys = list(trace_store)
    recs = list(dict.fromkeys((k[0], k[1]) for k in keys))
    sels = [s for s in ("top", "random", "bottom", "all")
            if any(k[2] == s for k in keys)]
    wins = sorted({k[3] for k in keys})
    written = []

    for rec, target in recs:
        fig, axes = plt.subplots(len(sels), len(wins), squeeze=False,
                                 figsize=(3.6 * len(wins), 2.6 * len(sels)),
                                 sharex="col", sharey=True)
        for i, sel in enumerate(sels):
            for j, w in enumerate(wins):
                ax = axes[i][j]
                item = trace_store.get((rec, target, sel, w))
                if item is None:
                    ax.axis("off")
                    continue
                y_true, y_pred = item
                # one row per window, so the effective step is `stride` samples
                fs_eff = sfreq / max(args.stride, 1)
                a = int(args.trace_start * fs_eff)
                b = a + int(args.trace_seconds * fs_eff)
                a, b = min(a, max(len(y_true) - 2, 0)), min(b, len(y_true))
                t = np.arange(b - a) / fs_eff
                yt, yp = y_true[a:b], y_pred[a:b]

                ax.plot(t, yt, lw=1.1, color="0.15", label="actual")
                ax.plot(t, yp, lw=1.1, color=cols.get(sel), alpha=0.9,
                        label="predicted")

                r2 = df[(df.recording == rec) & (df.target == target)
                        & (df.selection == sel) & (df.window == w)]["r2"]
                r2 = float(r2.iloc[0]) if len(r2) else np.nan
                ax.set_title(f"{sel} · {1000*w/sfreq:.0f} ms · $R^2$={r2:.3f}",
                             fontsize=9)
                if j == 0:
                    ax.set_ylabel("z-scored amplitude", fontsize=8)
                if i == len(sels) - 1:
                    ax.set_xlabel("time in held-out block (s)", fontsize=8)
                ax.tick_params(labelsize=7)
                if i == 0 and j == 0:
                    ax.legend(fontsize=7, ncol=2, loc="upper right",
                              framealpha=0.85)

        fig.suptitle(f"{rec} — reconstructing {target} from {args.k} channels",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        tag = f"{rec.replace('/', '-')}_{target}"
        out = path.with_name(f"{path.stem}_traces_{tag}.png")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def plot_scatter_and_spectra(trace_store, path, args, sfreq):
    """Two views the time-domain overlay hides.

    Left: predicted against actual, sample by sample. Systematic compression
    towards the mean shows up as a slope below the identity line, which is the
    signature of a model hedging rather than tracking.

    Right: power spectra of truth, prediction and residual. A predictor can
    score well on R^2 while reproducing only the low frequencies, and that is
    invisible in both the R^2 number and the overlaid traces.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    cols = {"top": "#1b7837", "random": "#7570b3", "bottom": "#d95f02",
            "all": "#333333"}
    sels = [s for s in ("top", "random", "bottom", "all")
            if any(k[2] == s for k in trace_store)]
    if not sels:
        return None
    w_ref = sorted({k[3] for k in trace_store})[len(set(k[3] for k in trace_store)) // 2]
    fs_eff = sfreq / max(args.stride, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for sel in sels:
        yt = np.concatenate([v[0] for k, v in trace_store.items()
                             if k[2] == sel and k[3] == w_ref])
        yp = np.concatenate([v[1] for k, v in trace_store.items()
                             if k[2] == sel and k[3] == w_ref])
        if yt.size == 0:
            continue
        idx = np.random.default_rng(0).choice(len(yt), min(3000, len(yt)),
                                              replace=False)
        axes[0].scatter(yt[idx], yp[idx], s=4, alpha=0.25, color=cols.get(sel),
                        label=sel, rasterized=True)
        slope = float(np.polyfit(yt, yp, 1)[0])
        axes[0].plot([], [], " ", label=f"   slope {slope:.2f}")

        f, P = welch(yt, fs=fs_eff, nperseg=int(min(4 * fs_eff, len(yt) // 4)))
        _, Pp = welch(yp, fs=fs_eff, nperseg=int(min(4 * fs_eff, len(yt) // 4)))
        _, Pr = welch(yt - yp, fs=fs_eff, nperseg=int(min(4 * fs_eff, len(yt) // 4)))
        if sel == sels[0]:
            axes[1].semilogy(f, P, lw=2.2, color="0.15", label="actual")
        axes[1].semilogy(f, Pp, lw=1.4, color=cols.get(sel), label=f"{sel} pred")
        axes[1].semilogy(f, Pr, lw=1.0, ls="--", color=cols.get(sel),
                         alpha=0.8, label=f"{sel} residual")

    lim = 4.0
    axes[0].plot([-lim, lim], [-lim, lim], ls="--", color="0.4", lw=1)
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_xlabel("actual (z-scored)")
    axes[0].set_ylabel("predicted")
    axes[0].set_title(f"Sample-by-sample calibration ({1000*w_ref/sfreq:.0f} ms)")
    axes[0].legend(fontsize=7, markerscale=2)

    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("power")
    axes[1].set_title("What the predictor reproduces, and what it misses")
    axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out = path.with_name(f"{path.stem}_calibration.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot(df, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_mine = "mine_bits" in df.columns
    fig, axes = plt.subplots(1, 3 if has_mine else 2,
                             figsize=(15 if has_mine else 10, 4.2))
    cols = {"top": "#1b7837", "random": "#7570b3", "bottom": "#d95f02",
            "all": "#333333"}

    ax = axes[0]
    for tag, g in df.groupby("selection"):
        m = g.groupby("window_ms")["r2"].agg(["mean", "std", "count"])
        se = m["std"] / np.sqrt(m["count"].clip(lower=1))
        ax.errorbar(m.index, m["mean"], yerr=se, marker="o", lw=2, capsize=3,
                    label=tag, color=cols.get(tag))
    ax.axhline(0, color="0.6", lw=1, ls=":")
    ax.set_xlabel("window (ms)")
    ax.set_ylabel("test $R^2$")
    ax.set_title("Reconstruction vs window length")
    ax.legend(title="channels by MI", fontsize=8)

    ax = axes[1]
    for tag, g in df.groupby("selection"):
        ax.scatter(g["mean_pairwise_mi"], g["r2"], s=26, alpha=0.75,
                   label=tag, color=cols.get(tag))
    if len(df) > 2 and df["mean_pairwise_mi"].std() > 0:
        r = np.corrcoef(df["mean_pairwise_mi"], df["r2"])[0, 1]
        ax.set_title(f"Pairwise MI vs $R^2$  (r = {r:.2f})")
    else:
        ax.set_title("Pairwise MI vs $R^2$")
    ax.set_xlabel("mean pairwise MI of selected channels (bits)")
    ax.set_ylabel("test $R^2$")
    ax.legend(fontsize=8)

    if has_mine:
        ax = axes[2]
        lim = max(df["r2"].max(), df["r2_implied_by_mine"].max(), 0.1) * 1.1
        ax.plot([0, lim], [0, lim], ls="--", color="0.5", lw=1,
                label="$R^2 = 1-2^{-2I}$")
        for tag, g in df.groupby("selection"):
            ax.scatter(g["r2_implied_by_mine"], g["r2"], s=26, alpha=0.75,
                       label=tag, color=cols.get(tag))
        ax.set_xlabel("$R^2$ implied by MINE")
        ax.set_ylabel("$R^2$ actually achieved")
        ax.set_title("Does the information bound hold?")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
