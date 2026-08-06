# Baselines

Three baselines, each answering a question a reviewer will ask.

## 1. Does MI beat plain correlation as an edge weight?

```bash
uv run neurovision mine --rank-estimator pearson --k 6 --windows 25 50 100 200 400 \
    --mine-iters 0 --no-traces --out results/base_pearson
```

**Expect this to come out close.** The Gaussian-copula estimate is
`-0.5*log2(1 - r^2)` on rank-transformed data, i.e. a monotone function of rank
correlation, so it induces almost the same *ordering* as |Pearson r| whenever the
dependence is close to monotone. If the two match, say so plainly: the
contribution is the benchmark and the geometry control, not the choice of
estimator. Claiming an MI-specific advantage that the data does not show is the
fastest way to lose a reviewer.

## 2. Does the graph beat picking the nearest electrodes?

```bash
uv run neurovision mine --selections top random bottom nearest --k 6 \
    --windows 25 50 100 200 400 --mine-iters 0 --no-traces --out results/base_nearest
```

`nearest` takes the k physically closest electrodes from the montage, ignoring
the data entirely. This is the selection-level counterpart to the pairwise
distance regression: if NEAREST matches TOP, the graph adds nothing to the
montage, and the partial-correlation result would need re-examining.

Also worth running under CSD, where distance matters more:

```bash
uv run neurovision mine --selections top nearest --reference csd --k 6 \
    --windows 25 100 --mine-iters 0 --no-traces --out results/base_nearest_csd
```

## 3. Does the reconstruction need a neural network?

```bash
uv run neurovision mine --predictor ridge --k 6 --windows 25 50 100 200 400 \
    --mine-iters 0 --no-traces --out results/base_ridge
```

Closed-form ridge with the penalty chosen on a held-out slice of the training
block. Seconds per cell rather than minutes. If ridge matches the MLP, the
mapping is essentially linear and the paper should not imply a nonlinear model
was required.

## Also useful

`--selections top random bottom single` adds a single-channel selection: the
best individual channel, which bounds how much of the six-channel result comes
from one dominant neighbour.

## Cost

All of these run with `--mine-iters 0`, so each is roughly 10 min for one target
or ~2.5 h for all 16 via `run_all_targets.sh`. Ridge is faster still. For a
first pass, run each on the single PO3 target; extend to all targets only for
whichever baseline turns out to be close.

## Reading the result

The new CSV columns `edge_weight` and `predictor` record the configuration, so
the runs can be concatenated and compared directly:

```python
import glob, pandas as pd
df = pd.concat([pd.read_csv(f) for f in glob.glob('results/base_*.csv')])
print(df.pivot_table(index=['edge_weight','predictor'],
                     columns='selection', values='r2').round(3))
```
