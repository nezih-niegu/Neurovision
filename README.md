# Neurovision

Does a mutual-information edge weight on an EEG graph predict anything out of
sample? This repository builds a dependence-weighted graph over EEG channels,
hides one channel, and reconstructs it on a temporally disjoint block from
source subsets chosen by their edge weight. If high-weight sources beat
low-weight ones, the weight has predictive utility for that task — and nothing
more is claimed.

Code for the paper *Dependence-Weighted EEG Graphs Predict Held-Out Channel
Reconstruction*, analysing OpenNeuro
[ds002778](https://openneuro.org/datasets/ds002778).

## Install

```bash
uv sync                 # core: MNE, numpy, scipy, pandas, matplotlib, streamlit
uv sync --extra mine    # adds PyTorch (~1 GB), needed for `neurovision mine`
uv sync --extra dev     # pytest as well
```

Put the dataset anywhere under the project; it is located by searching for a
directory containing `dataset_description.json` or `sub-*` folders:

```
Neurovision/data/ds002778/sub-hc1/ses-hc/eeg/...
```

`data/` and `results/` are gitignored. ds002778 is redistributed by OpenNeuro,
not here.

## Run the study

```bash
JOBS=1 bash RUN_STUDY.sh main     # 16 targets x 2 references, with MINE
JOBS=4 bash RUN_STUDY.sh base     # Pearson, ridge, nearest-k, single-channel
JOBS=4 bash RUN_STUDY.sh train    # does an MI training term help? (see MINE_TRAINING.md)
JOBS=4 bash RUN_STUDY.sh band     # theta/alpha/beta, matched target
JOBS=1 bash RUN_STUDY.sh figs     # traced run + pairwise geometry test
```

Each stage skips targets that already have a CSV, so it is safe to interrupt and
rerun; it ends with a completeness check and merges every output.

**Choose `JOBS` by stage.** MINE is GPU-friendly and runs fastest in a single
process on Metal (~49 min per target). The other stages are thousands of tiny
independent fits and parallelise well on CPU workers. Workers never touch MPS:
Metal is not multi-process safe, and creating a context per worker can hang the
run or reset the machine.

## Individual tools

```bash
uv run neurovision precompute      # cache windowed MI matrices
uv run neurovision app             # Streamlit explorer for those matrices
uv run neurovision mine --help     # the reconstruction benchmark

uv run python merge_targets.py --dir results/targets_mine
uv run python compare_configs.py                 # all runs side by side, + plots
uv run python pairwise_distance.py --recordings 0  # the geometry control
```

## What the pipeline guards against

Two mistakes would make the benchmark confirm itself, and both are prevented in
code and covered by tests:

- **Selection leakage.** Ranking channels on the data used for evaluation makes
  the ranking partly a description of the test set. All ranking happens on the
  training block.
- **Temporal leakage.** EEG autocorrelation makes adjacent windows near
  duplicates, so a shuffled split puts near-copies of test windows into
  training. Splits are contiguous with a discarded guard interval.

A third is specific to the `bottom` condition: the target must be removed from
the candidate pool, not merely ranked last, or it selects itself and
reconstructs perfectly.

## Layout

```
src/neurovision/
  core.py         BIDS discovery, preprocessing, KSG / Gaussian-copula MI
  precompute.py   windowed MI matrices
  app.py          Streamlit explorer
  mine.py         MINE and InfoNCE, MLP and ridge predictors, windowing
  benchmark.py    the reconstruction benchmark
  progress.py     progress bars that degrade to plain lines in log files
  cli.py          `neurovision` entry point
RUN_STUDY.sh          every run the paper needs, in dependency order
run_all_targets.sh    sweep one configuration over all 16 targets
merge_targets.py      combine per-target runs, group statistics
compare_configs.py    compare every configuration in results/
pairwise_distance.py  reconstruct each channel from each single other channel
BASELINES.md          what each baseline answers
MINE_TRAINING.md      MINE as a training signal rather than a measurement
```

## Tests

```bash
uv run pytest        # 57 tests, no dataset required
```

Estimators are checked against closed-form ground truth: for a bivariate
Gaussian, MI is exactly `-0.5*log2(1 - r^2)`. The suite also covers the leakage
guards, the FDR correction, and the plotting paths.

## Caveats worth reading before using this

- The benchmark validates **predictive utility**, not information flow or
  cortical connectivity. Mutual information here is symmetric and zero-lag.
- Reference choice matters more than one might expect, and in both directions.
  An average reference gives every channel a shared component that *inflates*
  the floor and so *compresses* the measured contrast; a surface Laplacian
  removes it. Report which you used.
- Neural MI estimation is unreliable at high input dimension. In our runs the
  held-out DV objective is negative in 9.7% of cells under average reference and
  21.3% after CSD, concentrated at the longest windows. Treat it as a
  diagnostic, and report the failure rate against input dimension rather than a
  single mean.
- Ranking by pairwise weight ignores redundancy and synergy, so it is a
  heuristic rather than optimal subset selection.

## Citing

See `CITATION.cff`. Please also cite ds002778 and, as its authors request,
Jackson et al. (2019) and George et al. (2013).

## Licence

MIT, see `LICENSE`. The dataset is distributed separately under its own terms.
