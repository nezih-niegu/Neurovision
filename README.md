# Neurovision

Channel-wise mutual information for resting-state EEG (BIDS `.bdf` datasets such
as `ds002778`, UC San Diego Parkinson's).

Two stages. `precompute` cuts each recording into short windows, estimates MI
independently in every window for every pair of the fully connected graph, and
caches the per-recording averages; the `app` then averages those cached matrices
under whatever subject and channel selection you tick. The split matters — MI over every channel pair
for every subject takes minutes, while averaging cached matrices takes
milliseconds, and because the average is linear in the per-subject matrices,
toggling a subject on or off needs no recomputation at all.

## Layout

```
Neurovision/
├── pyproject.toml
├── .python-version              # 3.12
├── .streamlit/config.toml
├── data/                        # your BIDS dataset (found automatically)
│   └── ds002778/
├── results/                     # cached .npz lands here (gitignored)
├── RUN_STUDY.sh                 # every run the study needs, in order
├── run_all_targets.sh           # sweep one configuration over all 16 targets
├── merge_targets.py             # combine per-target runs, group stats
├── pairwise_distance.py         # the geometry control
├── BASELINES.md                 # what each baseline answers
├── src/neurovision/
│   ├── core.py                  # BIDS discovery, preprocessing, MI estimators
│   ├── precompute.py            # batch stage
│   ├── app.py                   # Streamlit explorer
│   ├── mine.py                  # MINE, predictors (MLP and ridge)
│   ├── benchmark.py             # reconstruction benchmark
│   └── cli.py                   # `neurovision` entry point
└── tests/
    ├── test_mi.py
    └── test_mine.py
```

Drop these files into your existing `Neurovision/` folder alongside the data. The
dataset is located by search, so it can sit at the project root, under `data/`,
or one level deeper — no path configuration needed.

## Setup

```bash
cd Neurovision
uv sync                 # creates .venv and resolves everything
uv sync --extra mine    # adds PyTorch, needed only for `neurovision mine`
uv sync --extra dev     # pytest + torch, for running the test suite
```

PyTorch is a ~1 GB download, so it is kept out of the base install; the KSG
pipeline and the app do not need it.

## Use

```bash
uv run neurovision precompute --window 0.6 --estimator ksg \
    --bands broadband alpha --duration 120 --reject-z 6 --surrogates 20

uv run neurovision app
```

That is the 600 ms windowed k-nearest-neighbour analysis: each recording is cut
into non-overlapping 600 ms windows, MI is estimated separately in each window
for all 496 channel pairs, and the windows are averaged afterwards. Both
commands print a runtime estimate before they start.

`precompute` writes to `results/mi_results.npz` and `app` reads from there, so
neither normally needs an argument. Both accept `--help`.

| flag | what it does |
|---|---|
| `--window 0.6` | window length in seconds; `0` = one estimate per recording |
| `--window-step 0.3` | hop between windows; default is no overlap |
| `--estimator ksg` | Kraskov kNN (default), or `gcmi` / `binned` |
| `--k 4` | neighbours for KSG; 3–6 is the usual range |
| `--reject-z 6` | drop windows whose peak amplitude is an outlier; `0` disables |
| `--save-windows` | cache every window's matrix, enabling per-pair time courses |
| `--jobs 8` | parallel worker processes; default is all cores but one |
| `--bids-root PATH` | skip auto-detection |
| `--n-channels 16` | keep only the first N channels |
| `--channels Fz Cz Pz …` | pick an explicit set instead |
| `--reference csd` | surface Laplacian — cuts volume conduction hard |
| `--estimator binned --bins 16` | histogram MI instead of Gaussian-copula |
| `--surrogates 20` | circular-shift null, so the app can show *excess* MI |
| `--limit 3` | quick trial run on the first few recordings |
| `--out PATH` / `--results PATH` | override the cache location |

Run the tests with `uv run pytest` (55 tests).

## Running the whole study

```bash
bash RUN_STUDY.sh          # everything, ~10 h with 4-way parallelism
bash RUN_STUDY.sh main     # 16 targets x 2 references, with MINE
bash RUN_STUDY.sh base     # Pearson, ridge, nearest-k, single-channel baselines
bash RUN_STUDY.sh band     # theta/alpha/beta, matched target
bash RUN_STUDY.sh figs     # traced run + pairwise geometry test
```

Each stage skips targets that already have a CSV, so it is safe to interrupt and
rerun. The script ends with a completeness check and merges every output.
See `BASELINES.md` for what each baseline is for.

## Neural MI and the reconstruction benchmark

```bash
uv sync --extra mine     # one-time: installs PyTorch
uv run neurovision mine --k 6 --windows 25 50 100 200 400 --device auto
```

Hides one channel, ranks the rest by their MI to it, and asks whether the
high-MI channels reconstruct it better than the low-MI ones. `--device auto`
picks MPS on Apple Silicon, CUDA where available, CPU otherwise.

MI is estimated with **MINE** (Belghazi et al. 2018, arXiv:1801.04062), which
trains a critic to maximise the Donsker-Varadhan bound

    I(X;Y) >= E_joint[T(x,y)] - log E_marginal[exp T(x,y)]

The reason to use a neural estimator here rather than KSG: X is a whole window
of the selected channels — 6 channels x 400 samples is 2400 dimensions — and kNN
estimators degrade badly above a handful of dimensions.

| flag | what it does |
|---|---|
| `--k 6` | channels per selection |
| `--windows 25 50 100` | window lengths **in samples** (150 = 600 ms at 250 Hz) |
| `--target Cz` | channel to hide; default is the highest-MI channel |
| `--all-targets` | repeat the benchmark once per channel, sharing preprocessing |
| `--targets Cz Pz Oz` | an explicit list of targets |
| `--max-trace-figures 6` | cap on trace figures; matters with `--all-targets` |
| `--selections top random bottom all` | which selections to compare |
| `--device mps` / `cuda` / `cpu` | override device detection |
| `--mine-iters 0` | skip MI estimation, fit predictors only |
| `--rank-estimator ksg` | which MI is used for ranking |
| `--recordings 0` | how many recordings; 0 (default) = all of them |
| `--groups HC PD-OFF` | restrict to particular groups |
| `--trace-seconds 6` | how much held-out signal the trace figures show |
| `--trace-start 2` | offset into the test block where the excerpt begins |
| `--no-traces` | skip the trace figures |

### Two design decisions that keep the result honest

**Channels are ranked on the training block only.** Ranking on the same data you
evaluate prediction on leaks the answer and makes every selection look good.

**Windows are split contiguously in time, never shuffled.** EEG is heavily
autocorrelated, so neighbouring windows are near-duplicates; a random split puts
copies of test windows into training and inflates R^2 substantially. There is
also a configurable `--gap` of discarded samples between the blocks.

### Validating the estimator

Correlated Gaussians have MI in closed form, so MINE can be checked rather than
trusted (`tests/test_mine.py`):

| dims | true MI (bits) | MINE |
|---|---|---|
| 1 | 0.068 | 0.046 |
| 1 | 0.322 | 0.291 |
| 1 | 1.198 | 1.237 |
| 5 | 3.685 | 3.432 |
| 10 | 7.370 | 6.616 |

Accurate below roughly 4 bits, then progressively **under**estimating. That is
not a fixable bug: any variational lower bound on MI needs a number of samples
exponential in the MI (McAllester & Stratos 2020). Read MINE output above ~5
bits as "large" and nothing more precise.

### Benchmark results

Three recordings, 16 channels, k = 6, 150 s each. Mean test R^2:

| window | bottom-MI | random | top-MI |
|---|---|---|---|
| 100 ms | 0.864 | 0.946 | **0.953** |
| 200 ms | 0.870 | 0.945 | **0.951** |
| 400 ms | 0.866 | 0.943 | **0.950** |
| 800 ms | 0.856 | 0.934 | **0.944** |
| 1600 ms | 0.827 | 0.919 | **0.932** |

**MI does predict reconstruction quality.** High-MI channels beat low-MI ones by
+0.089 R^2 on average, the ordering top > random > bottom holds at every window
length, and across all 45 cells the correlation between a selection's mean
pairwise MI and its achieved R^2 is r = 0.68.

**Longer windows did not help — they hurt slightly.** R^2 falls from 0.921 at
100 ms to 0.892 at 1600 ms while the input grows from 150 to 2400 dimensions.
The informative structure here is instantaneous (volume conduction is
instantaneous mixing), so extra temporal context adds parameters without adding
information. If your hypothesis is that a particular window length is optimal,
this benchmark is the thing that would show it — on this data the answer is that
the shortest window tested is already enough. Real data with genuine lagged
interactions may well behave differently, which is exactly why the sweep is
configurable.

**MINE tracks prediction but underestimates it**, as a lower bound should:

| selection | MINE (bits) | R^2 implied by MINE | R^2 achieved |
|---|---|---|---|
| bottom | 1.224 | 0.762 | 0.857 |
| random | 1.763 | 0.906 | 0.937 |
| top | 1.837 | 0.916 | 0.946 |

The ordering is right and the gap is in the safe direction. Critic overfitting
is visible and controlled: 1.96 bits on train against 1.61 on held-out data,
which is precisely why the reported number is always the held-out one.

### Looking at the reconstructions

Every run writes, alongside `benchmark.png`:

- `benchmark_traces_<recording>.png` — a grid of the held-out signal against its
  reconstruction, rows by selection and columns by window length. Bear in mind
  that at R^2 ~ 0.95 the prediction sits almost on top of the truth: panels
  differing by 0.09 in R^2 look nearly identical by eye, so read the R^2 in each
  panel title rather than trusting the overlay. The `bottom` row is the one
  where visible mismatch appears.
- `benchmark_calibration.png` — two views the overlay hides.
- `benchmark_traces.npz` — the raw arrays, keyed
  `"recording|target|selection|window"`, each of shape `(2, n_test)` as
  `[actual, predicted]`, so you can re-plot without re-running.

The calibration figure is worth reading carefully, because it exposes something
R^2 does not. Its left panel plots predicted against actual sample by sample:
`top` and `random` sit on the identity line with slope 0.92, while `bottom`
regresses to 0.78 — a model hedging towards the mean rather than tracking.

Its right panel is the more important one. All three selections reproduce the
dominant 8–20 Hz band almost exactly, and then **collapse above roughly 22 Hz**,
where predicted power falls an order of magnitude below the truth and the
residual spectrum converges on the actual spectrum. In other words an R^2 of
0.94 is being earned almost entirely by the strongest rhythm, and the
high-frequency content is essentially unpredicted. A single R^2 number cannot
show you that, and neither can the overlaid traces.

### Group comparison: HC vs PD-OFF vs PD-ON

Run with no `--recordings` limit and the benchmark covers the whole dataset,
splits it by group, and tests whether Parkinson's changes the picture:

```bash
uv run neurovision mine --k 6 --windows 50 150 300 --device auto
```

It writes `benchmark_group_stats.csv` and `benchmark_groups.png`, and prints a
descriptive table plus the contrasts.

**The target channel is fixed across every recording.** By default it is the
channel with the highest mean MI averaged over the *whole dataset*, not per
recording. This matters: the earlier per-recording default would have given HC
and PD different targets, so any group difference would be confounded with
which channel was being reconstructed.

**Sessions are handled according to the design they came from.** PD-OFF vs PD-ON
is within-subject — the same person on and off medication — so it gets a paired
test on the per-subject difference; treating those as independent would discard
the pairing and understate the medication effect. HC vs PD is between-subject
and gets Welch's t-test. Values are collapsed to one per subject-session before
any test, so a recording measured at five window lengths counts once rather than
five times. Everything is corrected with BH-FDR across the whole family of
contrasts, and Cohen's d is reported alongside p.

**Three quantities are compared, and they answer different questions:**

- `r2` — is the hidden channel harder to reconstruct in patients?
- `mine_bits` — do the selected channels carry less information about it?
- `mean_pairwise_mi` — is the pairwise coupling itself altered?

There is also a Jaccard overlap table of each group's consensus top-k. A change
in R^2 says the target got easier or harder; the overlap says whether the
*informative channels moved*, which is a different and arguably more interesting
claim. The per-subject figure joins each PD subject's on and off points with a
grey line, so a consistent medication effect is visible as parallel lines even
when the group boxes overlap.

**On statistical power.** ds002778 has roughly 15 PD and 16 HC subjects. With
that n, only large effects (d around 1) are reliably detectable, so a
non-significant contrast is weak evidence of absence rather than evidence of no
effect. Contrasts with fewer than three subjects per group are skipped rather
than reported. Treat this as hypothesis-generating.

**A confound worth stating plainly.** Everything measured here still runs
through volume conduction and the reference choice. A genuine group difference
in scalp-level MI can arise from altered cortical coupling, but equally from
group differences in skull conductivity, electrode impedance, head size, or
movement artefact. Re-running with `--reference csd` is the cheapest partial
control, since a surface Laplacian removes much of the shared field.

### Running every channel as a target

```bash
uv run neurovision mine --all-targets --mine-iters 0 --windows 25 100 400
```

Preprocessing is shared across targets, so this costs far less than N separate
runs; with 16 channels the fitting itself dominates. Disabling MINE is the single
biggest saving, since it accounts for roughly 80% of runtime in a default run.

Two things change when there is more than one target. The console prints a
per-target breakdown showing the high-MI advantage for each, which is the direct
test of whether the effect is specific to one scalp location. And the group
comparison already collapses to one value per subject-session before testing, so
averaging over targets *reduces* per-subject noise rather than multiplying the
number of contrasts --- the family stays at 27, not 27 per target.

Trace figures are capped at six recording-target pairs by default, since
otherwise a 16-target run over 46 recordings emits 736 PNGs.

### Three limits of this benchmark

**Even the worst selection scores 0.86.** Pairwise MI ranking is greedy — it
ignores redundancy and synergy, so six weakly-coupled channels can still
collectively span the sources. The top-vs-bottom gap is the signal; the absolute
level is not.

**On real data this substantially measures volume conduction.** Neighbouring
electrodes predict each other because they see the same cortical source through
the skull. Run `--reference csd` to see how much survives a surface Laplacian.

**The MI-to-R^2 link is partly an identity.** For jointly Gaussian variables
R^2 = 1 - 2^(-2I) exactly, so near-linear coupling means you are partly
rediscovering algebra. The third panel of `benchmark.png` plots achieved R^2
against MINE-implied R^2 for this reason: points on the diagonal are consistent
with the Gaussian identity, and departures from it are where genuinely nonlinear
structure would show up.

## What the app gives you

Sidebar: band, recording list (all / clear / per-group buttons), channel list,
and two toggles — subtract the surrogate floor, and average sessions within
subject before pooling. Six tabs: average MI matrix with a ranked edge list,
network graph on a head layout, per-channel strength and topography,
per-recording means for spotting artefact outliers, group comparison with
BH-FDR, and a raw-signal preview with PSD. Everything exports to CSV or PNG.

## About this dataset

`ds002778` is **32-channel BioSemi**, not 16 — the `.bdf` header also carries
EXG/Status channels, which are dropped automatically by matching against the
`biosemi32` montage. Use `--n-channels 16` if you want a 16-channel subset
deliberately. Each PD subject has two sessions (`ses-off`, `ses-on` medication)
while controls have one (`ses-hc`), so a flat average over recordings
double-weights the patients — that is what the "average sessions within subject
first" toggle is for.

## The windowed analysis

`--window 0.6` gives 150 samples per estimate at the default 250 Hz. That is a
small sample for a mutual-information estimator, and it has a specific
consequence worth understanding: **individual windows are dominated by estimator
noise, and only the average is interpretable.**

Measured on synthetic Gaussian pairs with 150 samples, KSG is essentially
unbiased — at r = 0.6 it returns 0.316 bits against a truth of 0.322 — but the
standard deviation across repeats is 0.09 bits. So a single window's estimate
for a single pair is worth very little. Averaging ~200 windows pulls that down
by roughly √200, and averaging across subjects further still. Read the matrix,
not the individual cells of the time course.

What the windowing buys you is the *distribution*. The app's "Variability across
windows" view shows the SD across windows for each pair, and with
`--save-windows` you get per-pair time courses. A pair that couples strongly in
20 % of windows and not at all in the rest has the same average as one that
couples weakly throughout — but a very different SD, and a very different story.
That distinction is invisible in a whole-recording estimate.

Windows whose peak amplitude is an outlier relative to the other windows are
dropped (`--reject-z 6`). At 600 ms resolution a single blink no longer
contaminates the whole recording, just its own window.

## Choice of estimator

The default is **KSG** (Kraskov, Stögbauer & Grassberger 2004, *Phys Rev E*
69:066138), estimator 1. For each point it finds the radius of its k-th nearest
neighbour in the joint 2-D space under the max-norm, then counts neighbours
within that radius on each marginal axis:

    I = psi(k) + psi(N) - <psi(n_x) + psi(n_y)>

It assumes nothing about the shape of the dependence, which matters in short
windows where a relationship may not be monotone. The test suite makes the point
concretely: for y = x² + noise, KSG returns 1.79 bits while the copula estimator
returns 0.000, because squaring destroys the rank correlation it relies on.

The implementation runs the joint neighbour search through a KD-tree and the
marginal counts through binary search on pre-sorted axes, giving O(N log N) per
pair rather than the O(N²) of the naive version — about 4× faster in practice,
and it agrees with brute force to 0.009 bits.

**It is still the expensive option.** Roughly 200 ms per window for 496 pairs at
N = 150, so ~40 s per recording per band single-threaded. `--jobs` parallelises
across recordings; a 40-recording dataset over two bands takes on the order of
half an hour on eight cores.

Two cheaper alternatives remain available:

- `--estimator gcmi` — Gaussian-copula MI (Ince et al. 2017, *Hum Brain Mapp* 38:1541). Rank-transform each channel to a normal marginal, then MI = -0.5*log(1 - r²). One correlation matrix gives every pair at once, so it is ~100× faster and fine for a first look, but it only sees monotone dependence and is a lower bound. At r = 0.3/0.6/0.9 it returns 0.0675/0.3265/1.1890 bits against a truth of 0.0680/0.3219/1.1980.
- `--estimator binned` — plug-in histogram with equipopulated bins and a Miller–Madow correction. Sensitive to non-monotone dependence, but noticeably more biased than KSG at the same sample size.

## Three caveats worth taking seriously

**Referencing dominates the result.** Every channel shares whatever the
reference does, which inflates MI everywhere. On a synthetic file with a common
component, no reference gave 0.030 bits mean MI and average reference gave 0.001
— a 30x swing from one preprocessing switch. Average reference is the least-bad
common choice; `--reference csd` is stronger if you care about local coupling.
Report which one you used.

**Zero-lag MI mostly measures volume conduction.** Two nearby electrodes pick up
the same cortical source through the skull, and that shows up as MI whether or
not the underlying regions interact. Expect distance to explain much of your
matrix. If the question is genuinely about interaction, compare against
distance-matched pairs, or move to a measure built to suppress it (imaginary
coherence, phase-lag index) or to lagged MI / transfer entropy.

**N is not what it looks like.** EEG samples are heavily autocorrelated: 15,000
samples of AR(1) noise carry roughly 160 independent ones. Bias correction
handles the point estimate, but any p-value computed as if samples were
independent is wildly anticonservative. The surrogate option exists for this —
circular shifts keep each channel's own spectrum and autocorrelation while
destroying coupling, so the surrogate mean is the MI floor your data would
produce with no coupling at all. Subtract it.
