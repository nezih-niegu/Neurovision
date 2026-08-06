#!/usr/bin/env python
"""
merge_targets.py
================
Combine the per-target benchmark runs and produce the analyses that only become
available once every channel has served as a target.

    uv run python merge_targets.py --dir results/targets

Three things this does that a single-target run cannot:

1. **Distance analysis.** With one target you can observe that the selected
   channels happen to be its neighbours. With sixteen you can regress
   reconstruction quality on inter-electrode distance across every target, and
   report how much of the MI graph is explained by geometry alone. This is the
   quantitative form of the paper's central claim.

2. **Correct group statistics.** Sixteen targets times 27 contrasts is 432
   tests, and correcting over all of them is both punitive and wrong: the
   targets are sixteen measurements of the same subjects, not independent
   experiments. This script averages across targets *within subject first*,
   leaving the original 27 contrasts computed on a less noisy per-subject
   value.

3. **Target-wise heterogeneity.** Whether the MI-selection advantage holds
   everywhere on the scalp, or only for posterior targets, is itself a result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results/targets",
                   help="directory holding the per-target CSVs")
    p.add_argument("--out", default="results/all_targets",
                   help="prefix for merged outputs")
    p.add_argument("--montage", default="biosemi32")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
def load(dirpath: Path) -> pd.DataFrame:
    files = sorted(dirpath.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs in {dirpath}. Run run_all_targets.sh first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Merged {len(files)} targets, {len(df)} rows, "
          f"{df.recording.nunique()} recordings, {df.target.nunique()} targets")
    missing = set(df.target.unique()) ^ set(f.stem for f in files)
    if missing:
        print(f"  note: filenames and recorded targets differ for {sorted(missing)}; "
              f"the 'target' column is authoritative")
    return df


def electrode_distances(channels, montage_name="biosemi32"):
    """Pairwise 3-D distances between electrodes, in montage units."""
    import mne
    pos = mne.channels.make_standard_montage(montage_name).get_positions()["ch_pos"]
    lut = {k.lower(): np.asarray(v, float) for k, v in pos.items()}
    xyz = np.array([lut.get(c.lower(), [np.nan] * 3) for c in channels])
    return pd.DataFrame(
        np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1),
        index=channels, columns=channels)


# --------------------------------------------------------------------------
def target_table(df):
    """Mean R^2 per target and selection, with the MI advantage."""
    t = (df.pivot_table(index="target", columns="selection", values="r2")
         .assign(advantage=lambda x: x["top"] - x["bottom"])
         .sort_values("advantage", ascending=False))
    print("\n" + "=" * 74)
    print("BY TARGET CHANNEL  (mean test R^2 over recordings and windows)")
    print("=" * 74)
    print(t.round(3).to_string())
    hold = (t["top"] > t["random"]) & (t["random"] > t["bottom"])
    print(f"\nfull ordering top > random > bottom holds for {hold.sum()}/{len(t)} targets")
    print(f"MI advantage ranges {t.advantage.min():.3f} to {t.advantage.max():.3f} "
          f"(mean {t.advantage.mean():.3f})")
    return t


def distance_analysis(df, montage):
    """How much of reconstruction quality is explained by electrode distance?"""
    print("\n" + "=" * 74)
    print("DISTANCE ANALYSIS")
    print("=" * 74)
    chans = sorted(set(df.target.unique()))
    D = electrode_distances(chans, montage)
    if D.isna().all().all():
        print("Montage positions unavailable; skipping.")
        return None

    # Mean distance from the target to its selected channels is not stored in
    # the CSV (only the channels' mean MI is), so we approximate the geometric
    # story at target level: does a target's reconstructability track how close
    # its nearest neighbours are?
    near = {c: np.sort(D.loc[c, D.columns != c].to_numpy())[:6].mean() for c in chans}
    t = df.pivot_table(index="target", columns="selection", values="r2")
    t = t.loc[[c for c in t.index if c in near]]
    t["mean_dist_6nn"] = [near[c] for c in t.index]

    for sel in ("top", "random", "bottom"):
        if sel not in t:
            continue
        m = t[[sel, "mean_dist_6nn"]].dropna()
        if len(m) < 4:
            continue
        r, p = stats.pearsonr(m["mean_dist_6nn"], m[sel])
        print(f"  R^2({sel:6s}) vs mean distance to 6 nearest electrodes: "
              f"r = {r:+.3f}  (p = {p:.3f}, n = {len(m)})")
    print("\n  A strongly negative correlation means targets in dense regions of "
          "the montage\n  are reconstructed better -- i.e. the benchmark is "
          "measuring electrode spacing.")
    print("  For the sharper test, see the note at the bottom of this output.")
    return t


def group_stats(df, alpha=0.05):
    """Group contrasts computed on per-subject values averaged across targets."""
    print("\n" + "=" * 74)
    print("GROUP CONTRASTS  (averaged across targets within subject first)")
    print("=" * 74)
    metrics = [m for m in ("r2", "mine_bits", "mean_pairwise_mi") if m in df]
    rows = []
    for sel in sorted(df.selection.unique()):
        d = df[df.selection == sel]
        for metric in metrics:
            per = (d.groupby(["group", "subject", "session"])[metric]
                   .mean().reset_index())
            groups = sorted(per.group.unique())
            for i, a in enumerate(groups):
                for b in groups[i + 1:]:
                    A, B = per[per.group == a], per[per.group == b]
                    paired = (a.startswith("PD") and b.startswith("PD")
                              and set(A.subject) & set(B.subject))
                    if paired:
                        common = sorted(set(A.subject) & set(B.subject))
                        if len(common) < 3:
                            continue
                        va = A.set_index("subject").loc[common, metric].to_numpy()
                        vb = B.set_index("subject").loc[common, metric].to_numpy()
                        t, p = stats.ttest_rel(va, vb)
                        diff = va - vb
                        sd = diff.std(ddof=1)
                        d_eff = diff.mean() / sd if sd > 1e-12 else np.nan
                        kind = "paired"
                    else:
                        va, vb = A[metric].to_numpy(), B[metric].to_numpy()
                        if min(len(va), len(vb)) < 3:
                            continue
                        t, p = stats.ttest_ind(va, vb, equal_var=False)
                        sp = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)
                        d_eff = (va.mean() - vb.mean()) / sp if sp > 1e-12 else np.nan
                        kind = "welch"
                    rows.append(dict(selection=sel, metric=metric, group_a=a,
                                     group_b=b, test=kind, n_a=len(va), n_b=len(vb),
                                     mean_a=va.mean(), mean_b=vb.mean(),
                                     diff=va.mean() - vb.mean(), cohens_d=d_eff,
                                     t=float(t), p=float(p)))
    res = pd.DataFrame(rows)
    if not len(res):
        print("Not enough subjects per group to test.")
        return res
    pv = res.p.to_numpy()
    ok = np.isfinite(pv)
    q = np.full(len(pv), np.nan)
    n = ok.sum()
    order = np.argsort(pv[ok])
    ranked = pv[ok][order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(n)
    tmp[order] = np.clip(ranked, 0, 1)
    q[ok] = tmp
    res["q_fdr"] = q
    res["significant"] = res.q_fdr.fillna(1.0) < alpha
    print(res[["selection", "metric", "group_a", "group_b", "test", "n_a", "n_b",
               "diff", "cohens_d", "p", "q_fdr", "significant"]]
          .round(4).to_string(index=False))
    print(f"\n{int(res.significant.sum())} of {len(res)} contrasts survive "
          f"FDR at q < {alpha}.")
    neg = (res[res.group_a == "HC"]["diff"] < 0).sum()
    tot = (res.group_a == "HC").sum()
    if tot:
        print(f"{neg} of {tot} control-versus-patient contrasts show higher "
              f"values in patients.")
    return res


def plots(df, tgt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    cols = {"top": "#1b7837", "random": "#7570b3", "bottom": "#d95f02"}

    order = tgt.sort_values("top").index
    x = np.arange(len(order))
    for sel in ("top", "random", "bottom"):
        if sel in tgt:
            axes[0].plot(x, tgt.loc[order, sel], marker="o", lw=1.8,
                         color=cols[sel], label=sel)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=90, fontsize=8)
    axes[0].set_ylabel("mean test $R^2$")
    axes[0].set_title("Reconstruction by target channel")
    axes[0].legend(fontsize=8)

    per = df.groupby(["target", "selection", "window_ms"]).r2.mean().reset_index()
    for sel in ("top", "random", "bottom"):
        d = per[per.selection == sel].groupby("window_ms").r2.agg(["mean", "std"])
        axes[1].errorbar(d.index, d["mean"], yerr=d["std"], marker="o", lw=2,
                         capsize=3, color=cols[sel], label=sel)
    axes[1].set_xlabel("window (ms)")
    axes[1].set_ylabel("mean test $R^2$")
    axes[1].set_title("Window trend, pooled over all targets")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = Path(f"{path}_by_target.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None):
    args = parse_args(argv)
    df = load(Path(args.dir))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.out}.csv", index=False)

    tgt = target_table(df)
    tgt.to_csv(f"{args.out}_by_target.csv")
    distance_analysis(df, args.montage)
    res = group_stats(df)
    if len(res):
        res.to_csv(f"{args.out}_group_stats.csv", index=False)

    made = [f"{args.out}.csv", f"{args.out}_by_target.csv"]
    if not args.no_plots:
        try:
            made.append(str(plots(df, tgt, args.out)))
        except Exception as exc:
            print(f"(plot skipped: {exc})")
    print("\nWrote " + ", ".join(made))

    print("\n" + "-" * 74)
    print("For the sharpest version of the geometry test, you need per-pair data\n"
          "rather than per-target summaries: rerun with --selections all --k 1,\n"
          "which reconstructs each target from every single other channel in turn.\n"
          "Regressing that R^2 on inter-electrode distance gives the fraction of\n"
          "the MI graph explained by geometry directly, with 16x15 = 240 points\n"
          "per recording instead of 16.")
    return df


if __name__ == "__main__":
    main()
