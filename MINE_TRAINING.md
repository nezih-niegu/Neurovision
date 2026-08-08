# Does MINE help during training?

## The distinction that matters

Until now MINE was a **measurement**: the predictor was trained with squared
error, and afterwards a critic estimated `I(window ; target)`. Running with and
without `--mine-iters` therefore produced *identical* R^2 — the only difference
was whether a column got filled in. That is not a comparison.

`--predictor mine` is different. It trains the reconstruction with

    loss = MSE(head(h), y) - lambda * I_hat(h ; y)

where `h` is the hidden representation and `I_hat` comes from a critic trained
simultaneously to maximise the Donsker-Varadhan bound. The MI term shapes the
representation rather than the output, which is where it could plausibly help.

Setting `lambda = 0` recovers the plain MLP exactly — same architecture, same
optimiser, same schedule, same seed. The two arms differ in one term and nothing
else, which is what makes the comparison clean.

## Running it

```bash
uv run neurovision mine --predictor mlp  --k 6 --windows 25 50 100 200 400 \
    --mine-iters 0 --no-traces --jobs 8 --out results/train_mse

uv run neurovision mine --predictor mine --mine-lambda 0.1 --k 6 \
    --windows 25 50 100 200 400 --mine-iters 0 --no-traces --jobs 8 \
    --out results/train_mine
```

All 16 targets:

```bash
JOBS=8 PREDICTOR=mlp  OUTDIR=results/train_mse  bash run_all_targets.sh
JOBS=8 PREDICTOR=mine OUTDIR=results/train_mine bash run_all_targets.sh
```

A lambda sweep is worth doing before committing to 16 targets, since the answer
may depend on it:

```bash
for lam in 0.0 0.03 0.1 0.3 1.0; do
  uv run neurovision mine --predictor mine --mine-lambda $lam --target PO3 \
      --k 6 --windows 25 100 400 --mine-iters 0 --no-traces --jobs 8 \
      --out "results/lam_$lam"
done
```

The new `mine_lambda` column records the setting, so runs concatenate directly.

## What to expect

On one real recording (sub-hc1, 16 channels, 6 sources, 40 epochs):

| window | mlp | mine λ=0 | λ=0.1 | λ=0.3 |
|---|---|---|---|---|
| 100 ms | +0.8883 | +0.8888 | +0.8883 | +0.8862 |
| 400 ms | +0.8786 | +0.8779 | +0.8780 | +0.8729 |
| 1600 ms | +0.8393 | +0.8319 | +0.8273 | +0.8315 |

The MI term does not help here, and mildly hurts at the longest window. The
lambda = 0 column matching the plain MLP to three decimals is the sanity check
that the comparison is fair.

This is not surprising, and the reason is worth stating in the paper. Squared
error against a scalar target is already a strong, well-posed signal with
roughly 16,000 training windows; an auxiliary MI bound adds gradient variance
without adding information the MSE lacks. MI-based objectives earn their keep
where labels are scarce, high-dimensional, or where you want a representation
transferable to a task you have not specified — none of which describes this
setup.

**Report the negative result.** It converts MINE from an ornament into a tested
design choice, and it answers a question a reviewer would otherwise ask: if the
paper is about mutual information, why is the predictor trained with squared
error? Because the alternative was tried and did not help.
