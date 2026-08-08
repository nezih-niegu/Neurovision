#!/usr/bin/env python
"""
compare_configs.py — one table across every configuration in results/.

    uv run python compare_configs.py
    uv run python compare_configs.py --results results --out results/comparison

Walks results/ for anything that looks like a sweep (a directory of per-target
CSVs) or a single-target run (a bare CSV), reads each run's own JSON to recover
what it varied, and reports them side by side.

Runs are grouped by what distinguishes them rather than by directory name, so a
folder called base_pearson and one called targets_mine are compared on the axes
that actually differ: reference, edge weight, predictor, band, and whether the
neural estimator was enabled. Directory names are only used as a fallback label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_KEYS = ["reference", "rank_estimator", "predictor", "band",
               "mine_lambda", "k", "n_channels", "duration"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/comparison")
    p.add_argument("--min-targets", type=int, default=1,
                   help="skip runs with fewer completed targets than this")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def target_csvs(d: Path) -> list[Path]:
    """Per-target result CSVs, excluding the per-target group-stats files."""
    return sorted(f for f in d.glob("*.csv")
                  if not f.name.endswith("_group_stats.csv"))


def read_config(paths: list[Path]) -> dict:
    """Configuration from the first readable sibling JSON."""
    for p in paths:
        j = p.with_suffix(".json")
        if j.exists():
            try:
                a = json.loads(j.read_text()).get("args", {})
                return {k: a.get(k) for k in CONFIG_KEYS} | {
                    "mine": bool(a.get("mine_iters", 0)),
                    "windows": ",".join(str(w) for w in a.get("windows", [])),
                }
            except Exception:
                continue
    return {}


def discover(root: Path, min_targets: int):
    """Every run under `root`: sweep directories first, then loose CSVs."""
    runs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = target_csvs(d)
        if not files:
            print(f"  skipping {d.name}/ — no result CSVs "
                  f"({len(list(d.glob('*.log')))} logs present, run incomplete)")
            continue
        if len(files) < min_targets:
            print(f"  skipping {d.name}/ — only {len(files)} targets")
            continue
        runs.append((d.name, files))

    loose = [f for f in target_csvs(root)
             if not f.name.startswith("comparison")]
    for f in loose:
        runs.append((f.stem, [f]))
    return runs


FRAMES: dict[str, pd.DataFrame] = {}


def summarise(name: str, files: list[Path]) -> dict | None:
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:
            print(f"  {name}: could not read {f.name} ({exc})")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if not {"selection", "r2"} <= set(df.columns):
        print(f"  {name}: no selection/r2 columns, skipping")
        return None
    df = df.dropna(subset=["selection", "r2"])
    FRAMES[name] = df

    cfg = read_config(files)
    means = df.groupby("selection").r2.mean()
    row = {
        "run": name,
        "reference": cfg.get("reference", "?"),
        "edge_weight": cfg.get("rank_estimator", "?"),
        # Runs predating the --predictor flag could only be the MLP.
        "predictor": cfg.get("predictor") or "mlp",
        "band": cfg.get("band", "?"),
        "mine": cfg.get("mine", False),
        "targets": df.target.nunique() if "target" in df else 1,
        "recordings": df.recording.nunique() if "recording" in df else np.nan,
        "cells": len(df),
    }
    for s in ("top", "random", "bottom", "nearest", "single"):
        row[s] = float(means[s]) if s in means else np.nan
    row["advantage"] = (row["top"] - row["bottom"]
                        if np.isfinite(row.get("bottom", np.nan)) else np.nan)
    if "mine_bits" in df:
        m = df.groupby("selection").mine_bits.mean()
        row["mine_top"] = float(m["top"]) if "top" in m else np.nan
        row["mine_neg_frac"] = float((df.mine_bits < 0).mean())
    return row


def plots(df: pd.DataFrame, out: Path):
    """Three comparisons that the printed tables cannot show compactly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 180})
    C = {"top": "#1b7837", "random": "#7570b3", "bottom": "#d95f02"}
    made = []

    # -- 1. every run, side by side -----------------------------------------
    d = df.sort_values("advantage")
    x = np.arange(len(d))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4),
                           gridspec_kw={"width_ratios": [3, 2]})
    for i, s in enumerate(("bottom", "random", "top")):
        ax[0].bar(x + (i - 1) * 0.27, d[s], 0.27, label=s, color=C[s])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(d.run, rotation=30, ha="right")
    ax[0].set_ylabel("mean test $R^2$")
    ax[0].set_title("Reconstruction by selection, per run")
    ax[0].legend(frameon=False, ncol=3)

    ax[1].barh(x, d.advantage, color="#4575b4")
    ax[1].set_yticks(x)
    ax[1].set_yticklabels(d.run)
    ax[1].set_xlabel(r"$\Delta$ = top $-$ bottom")
    ax[1].set_title("Selection advantage")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out.with_name(out.stem + f"_runs.{e}"), bbox_inches="tight")
    plt.close(fig); made.append(f"{out.stem}_runs")

    # -- 2. window trend, one line per run -----------------------------------
    have = [n for n, f in FRAMES.items() if "window_ms" in f]
    if have:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharey=True)
        cmap = plt.get_cmap("tab10")
        for k, sel in enumerate(("bottom", "random", "top")):
            for i, n in enumerate(sorted(have)):
                f = FRAMES[n]
                g = f[f.selection == sel].groupby("window_ms").r2.mean()
                if len(g) < 2:
                    axes[k].plot(g.index, g.values, "o", color=cmap(i % 10),
                                 label=n if k == 0 else None)
                else:
                    axes[k].plot(g.index, g.values, marker="o", ms=3,
                                 color=cmap(i % 10), label=n if k == 0 else None)
            axes[k].set_title(sel)
            axes[k].set_xlabel("window (ms)")
        axes[0].set_ylabel("mean test $R^2$")
        axes[0].legend(frameon=False, fontsize=6)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(out.with_name(out.stem + f"_windows.{e}"),
                        bbox_inches="tight")
        plt.close(fig); made.append(f"{out.stem}_windows")

    # -- 3. per-target advantage, multi-target runs only ---------------------
    multi = [n for n, f in FRAMES.items()
             if "target" in f and f.target.nunique() > 1]
    if multi:
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ref = None
        for i, n in enumerate(sorted(multi)):
            p = FRAMES[n].pivot_table(index="target", columns="selection",
                                      values="r2")
            adv = (p["top"] - p["bottom"])
            ref = adv.sort_values().index if ref is None else ref
            adv = adv.reindex(ref)
            ax.bar(np.arange(len(adv)) + (i - (len(multi) - 1) / 2) * 0.8 / len(multi),
                   adv.values, 0.8 / len(multi), label=n)
        ax.set_xticks(np.arange(len(ref)))
        ax.set_xticklabels(ref, rotation=90)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_ylabel(r"$\Delta$ = top $-$ bottom")
        ax.set_xlabel("target channel")
        ax.legend(frameon=False)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(out.with_name(out.stem + f"_targets.{e}"),
                        bbox_inches="tight")
        plt.close(fig); made.append(f"{out.stem}_targets")
    return made


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.results)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    print(f"Scanning {root}/")
    runs = discover(root, args.min_targets)
    rows = [r for r in (summarise(n, f) for n, f in runs) if r]
    if not rows:
        raise SystemExit("No readable runs found.")
    df = pd.DataFrame(rows)

    # A run is fully characterised by these; the directory name is only a label.
    axes = ["reference", "edge_weight", "predictor", "band", "mine"]
    df = df.sort_values(axes + ["run"]).reset_index(drop=True)

    pd.set_option("display.width", 200)
    show = ["run", "reference", "edge_weight", "predictor", "band", "mine",
            "targets", "cells", "bottom", "random", "top", "advantage"]
    print("\n" + "=" * 100)
    print("ALL RUNS")
    print("=" * 100)
    print(df[show].round(3).to_string(index=False))

    for axis, label in (("reference", "reference scheme"),
                        ("edge_weight", "edge weight"),
                        ("predictor", "predictor"),
                        ("band", "frequency band")):
        sub = df[df[axis].notna() & (df[axis] != "?")]
        if sub[axis].nunique() < 2:
            continue
        print(f"\n-- by {label} " + "-" * (60 - len(label)))
        print(sub.groupby(axis)[["bottom", "random", "top", "advantage"]]
              .agg(["mean", "size"]).round(3).to_string())
        # Averaging a 16-target sweep with a single-target run weights them
        # equally, which is rarely what you want; say so rather than let the
        # table imply the comparison is like-for-like.
        spread = sub.groupby(axis).targets.agg(["min", "max"])
        if (spread["min"] != spread["max"]).any() or sub.targets.nunique() > 1:
            print(f"     note: runs in this grouping cover different target "
                  f"counts ({sorted(int(t) for t in sub.targets.unique())}); the "
                  f"means are not "
                  f"like-for-like")

    extra = [c for c in ("nearest", "single") if c in df and df[c].notna().any()]
    if extra:
        print("\n-- baseline selections present " + "-" * 40)
        print(df.loc[df[extra].notna().any(axis=1),
                     ["run", "top", "random", "bottom"] + extra]
              .round(3).to_string(index=False))
    else:
        print("\nNo nearest/single baseline runs found; those comparisons are "
              "still outstanding.")

    if "mine_neg_frac" in df:
        m = df[df.mine_neg_frac.notna()]
        if len(m):
            print("\n-- neural estimator " + "-" * 52)
            print(m[["run", "reference", "mine_top", "mine_neg_frac"]]
                  .round(3).to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    written = [str(out.with_suffix(".csv"))]
    if not args.no_plots:
        try:
            written += [f"{out.parent/m}.{{png,pdf}}" for m in plots(df, out)]
        except Exception as exc:
            print(f"  plots skipped: {exc}")
    print("\nWrote " + ", ".join(written))

    missing = [a for a in ("pearson", "distance") if a not in set(df.edge_weight)]
    if missing:
        print(f"Edge-weight baselines not yet present: {', '.join(missing)}")
    if "ridge" not in set(df.predictor):
        print("Predictor baseline (ridge) not yet present.")
    return df


if __name__ == "__main__":
    main()
