"""
neurovision.mine
================
Neural mutual-information estimation (MINE) and the channel-reconstruction
benchmark built on top of it.

Two questions, kept deliberately separate:

1. **How much information do a set of channels carry about a held-out one?**
   Estimated with MINE (Belghazi et al. 2018, arXiv:1801.04062), which trains a
   critic to maximise the Donsker-Varadhan lower bound

       I(X;Y) >= E_joint[T(x,y)] - log E_marginal[exp T(x,y)]

   Here X is a whole window of the selected channels -- 16 channels x 150
   samples is 2400 dimensions -- which is exactly the regime where kNN
   estimators such as KSG fall apart and a neural critic earns its keep.

2. **How well can those channels actually predict the held-out one?**
   A separate MLP regressor, scored by R^2 on held-out data.

For jointly Gaussian variables the two are linked exactly:

       R^2 = 1 - 2^(-2 I)     (I in bits)

so plotting measured R^2 against measured MI on the same data is a real,
falsifiable check rather than a restatement.

Devices: Apple Silicon (MPS), CUDA, or CPU, picked automatically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def pick_device(requested: str = "auto") -> torch.device:
    """Resolve a device string, preferring MPS on Apple Silicon then CUDA."""
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report(dev: torch.device) -> str:
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        return f"cuda ({p.name}, {p.total_memory/1e9:.0f} GB)"
    if dev.type == "mps":
        return "mps (Apple Silicon GPU)"
    return f"cpu ({torch.get_num_threads()} threads)"


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))


# --------------------------------------------------------------------------
# MINE
# --------------------------------------------------------------------------
class Critic(nn.Module):
    """Joint critic T(x, y): concatenate, then an MLP to a scalar.

    Output is soft-clipped to +-`clip` nats with a tanh. This matters more than
    it looks: the DV bound involves exp(T), and an unclipped critic reliably
    overflows to NaN once the true MI is large. Any *bounded* critic still gives
    a valid lower bound, so clipping costs tightness, never correctness.
    """

    def __init__(self, d_x: int, d_y: int, hidden: int = 256, layers: int = 2,
                 clip: float = 15.0, dropout: float = 0.0):
        super().__init__()
        seq: list[nn.Module] = [nn.Linear(d_x + d_y, hidden), nn.ELU()]
        for _ in range(layers - 1):
            if dropout:
                seq.append(nn.Dropout(dropout))
            seq += [nn.Linear(hidden, hidden), nn.ELU()]
        seq.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*seq)
        self.clip = float(clip)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat([x, y], dim=-1)).squeeze(-1)
        return self.clip * torch.tanh(raw / self.clip)


def _log_mean_exp(t: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(t, dim=0) - np.log(t.shape[0])


@dataclass
class MineResult:
    mi_bits: float                 # held-out DV bound
    mi_bits_train: float
    mi_infonce_bits: float         # InfoNCE bound, capped at log2(batch)
    history: list = field(default_factory=list)
    seconds: float = 0.0
    n_train: int = 0
    n_test: int = 0

    @property
    def infonce_ceiling_bits(self) -> float:
        return float(np.log2(max(self.n_test, 2)))


def estimate_mi(
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device | str = "auto",
    iters: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden: int = 256,
    layers: int = 2,
    clip: float = 15.0,
    ema_decay: float = 0.99,
    test_frac: float = 0.2,
    seed: int = 0,
    log_every: int = 0,
    nce_batch: int = 256,
    eval_batch: int = 4096,
) -> MineResult:
    """Train a MINE critic and return the held-out Donsker-Varadhan bound.

    Trained with the EMA-corrected gradient of Belghazi et al. section 3.2: the
    naive minibatch gradient of log E[exp T] is biased, so the denominator is
    replaced by a running average of E[exp T]. The bias affects the *gradient*
    only -- the reported value is always computed directly from held-out data.

    The train/test split is what makes the number trustworthy. A critic
    evaluated on its own training data will happily report MI far above the
    truth by memorising which pairs were joint samples.
    """
    dev = pick_device(device) if isinstance(device, str) else device
    _seed_all(seed)

    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = X.shape[0]
    n_test = max(int(n * test_frac), 2)
    idx = torch.randperm(n)
    te, tr = idx[:n_test], idx[n_test:]
    Xtr, Ytr = X[tr].to(dev), Y[tr].to(dev)
    Xte, Yte = X[te].to(dev), Y[te].to(dev)

    net = Critic(X.shape[1], Y.shape[1], hidden, layers, clip).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n_tr = Xtr.shape[0]
    bs = min(batch_size, n_tr)
    ema = None
    history = []
    t0 = time.time()

    for it in range(iters):
        i = torch.randint(0, n_tr, (bs,), device=dev)
        j = torch.randint(0, n_tr, (bs,), device=dev)
        t_joint = net(Xtr[i], Ytr[i])
        t_marg = net(Xtr[i], Ytr[j])

        mx = t_marg.detach().max()
        exp_mean = torch.exp(t_marg - mx).mean() * torch.exp(mx)
        ema = exp_mean.detach() if ema is None else \
            ema_decay * ema + (1 - ema_decay) * exp_mean.detach()
        loss = -(t_joint.mean() - exp_mean / ema.clamp(min=1e-8))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if log_every and (it % log_every == 0 or it == iters - 1):
            with torch.no_grad():
                dv = (t_joint.mean() - _log_mean_exp(t_marg)) / np.log(2)
            history.append((it, float(dv)))

    net.eval()
    with torch.no_grad():
        def bound(Xa, Ya):
            perm = torch.randperm(Xa.shape[0], device=dev)
            tj, tm = net(Xa, Ya), net(Xa, Ya[perm])
            dv = (tj.mean() - _log_mean_exp(tm)) / np.log(2)
            # InfoNCE over the same sample. The score matrix is built row by row:
            # expanding it in one go costs m*m*d floats, which is several GB for
            # a wide window and will kill the process (or the GPU) outright.
            m = int(min(Xa.shape[0], nce_batch))
            rows = []
            for s in range(m):
                xr = Xa[s: s + 1].expand(m, -1)
                rows.append(net(xr, Ya[:m]))
            S = torch.stack(rows)                      # (m, m), row = one x
            nce = (S.diag() - torch.logsumexp(S, 1) + np.log(m)).mean() / np.log(2)
            return float(dv), float(nce)

        e = int(eval_batch)
        dv_te, nce_te = bound(Xte[:e], Yte[:e])
        dv_tr, _ = bound(Xtr[:e], Ytr[:e])

    return MineResult(mi_bits=dv_te, mi_bits_train=dv_tr, mi_infonce_bits=nce_te,
                      history=history, seconds=time.time() - t0,
                      n_train=n_tr, n_test=min(n_test, int(nce_batch)))


# --------------------------------------------------------------------------
# Channel reconstruction
# --------------------------------------------------------------------------
class Predictor(nn.Module):
    """MLP mapping a window of source channels to the target channel's value."""

    def __init__(self, d_in: int, hidden: int = 256, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        seq: list[nn.Module] = [nn.Linear(d_in, hidden), nn.ELU()]
        for _ in range(layers - 1):
            seq += [nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ELU()]
        seq.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class PredictResult:
    r2: float
    r2_train: float
    rmse: float
    epochs_run: int
    seconds: float
    y_true: np.ndarray | None = None   # held-out target, in standardised units
    y_pred: np.ndarray | None = None


class TrunkHead(nn.Module):
    """Shared representation h = trunk(x), scalar readout y_hat = head(h).

    Splitting the predictor this way is what makes an MI training term
    meaningful: the critic scores (h, y), so the auxiliary objective acts on the
    representation rather than on the scalar output, where it would be almost
    redundant with the squared error.
    """

    def __init__(self, d_in: int, hidden: int = 256, d_rep: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_rep), nn.ELU())
        self.head = nn.Linear(d_rep, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head(h).squeeze(-1), h


def train_predictor_mine(
    Xtr, ytr, Xte, yte,
    device="auto", epochs: int = 60, batch_size: int = 256, lr: float = 1e-3,
    hidden: int = 256, d_rep: int = 32, dropout: float = 0.1,
    lam: float = 0.1, critic_hidden: int = 128, critic_lr: float = 1e-3,
    clip: float = 15.0, patience: int = 10, seed: int = 0,
) -> "PredictResult":
    """Reconstruction trained with an auxiliary MI term on the representation.

        loss = MSE(head(h), y) - lam * I_hat(h ; y)

    where I_hat is the Donsker-Varadhan bound from a critic trained
    simultaneously to maximise it. Setting lam = 0 recovers the plain MLP
    exactly, which is what makes the comparison clean: the two arms differ in
    one term and nothing else -- same architecture, same optimiser, same
    schedule, same seed.

    The critic is soft-clipped as elsewhere, and its gradient is not propagated
    into the trunk when the critic itself is being updated, so the two
    objectives do not fight over the same step.
    """
    dev = pick_device(device) if isinstance(device, str) else device
    _seed_all(seed)

    Xtr = torch.as_tensor(Xtr, dtype=torch.float32).to(dev)
    ytr = torch.as_tensor(ytr, dtype=torch.float32).to(dev)
    Xte = torch.as_tensor(Xte, dtype=torch.float32).to(dev)
    yte = torch.as_tensor(yte, dtype=torch.float32).to(dev)

    n_val = max(int(0.15 * Xtr.shape[0]), 1)
    Xva, yva, Xfit, yfit = Xtr[:n_val], ytr[:n_val], Xtr[n_val:], ytr[n_val:]

    net = TrunkHead(Xtr.shape[1], hidden, d_rep, dropout).to(dev)
    critic = Critic(d_rep, 1, critic_hidden, layers=2, clip=clip).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    n = Xfit.shape[0]
    bs = min(batch_size, n)
    best, best_state, bad, ran = np.inf, None, 0, 0
    t0 = time.time()

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for s0 in range(0, n, bs):
            b = perm[s0: s0 + bs]
            xb, yb = Xfit[b], yfit[b]

            if lam > 0:
                # 1. critic step: maximise the bound on the current (detached) h
                with torch.no_grad():
                    _, h = net(xb)
                yb2 = yb[:, None]
                perm_b = torch.randperm(h.shape[0], device=dev)
                tj = critic(h, yb2)
                tm = critic(h, yb2[perm_b])
                loss_c = -(tj.mean() - _log_mean_exp(tm))
                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                opt_c.step()

            # 2. predictor step
            pred, h = net(xb)
            loss = nn.functional.mse_loss(pred, yb)
            if lam > 0:
                yb2 = yb[:, None]
                perm_b = torch.randperm(h.shape[0], device=dev)
                bound = critic(h, yb2).mean() - _log_mean_exp(critic(h, yb2[perm_b]))
                loss = loss - lam * bound
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        ran = ep + 1
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(net(Xva)[0], yva))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pte = net(Xte)[0]
        ptr = net(Xfit)[0]

        def r2(pred, true):
            ss_tot = float(((true - true.mean()) ** 2).sum())
            return (1.0 - float(((true - pred) ** 2).sum()) / ss_tot
                    if ss_tot > 0 else float("nan"))

        return PredictResult(r2(pte, yte), r2(ptr, yfit),
                             float(torch.sqrt(((yte - pte) ** 2).mean())), ran,
                             time.time() - t0,
                             y_true=yte.detach().cpu().numpy(),
                             y_pred=pte.detach().cpu().numpy())


def fit_ridge(Xtr, ytr, Xte, yte, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100)):
    """Closed-form ridge baseline, with the penalty picked on a held-out slice
    of the training block.

    If this matches the MLP, the reconstruction is essentially linear and the
    neural predictor is unnecessary -- worth knowing before claiming that a
    nonlinear model was required.
    """
    import time
    t0 = time.time()
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.float64).ravel()
    yte = np.asarray(yte, dtype=np.float64).ravel()

    n_val = max(int(0.15 * len(Xtr)), 1)
    Xv, yv, Xf, yf = Xtr[:n_val], ytr[:n_val], Xtr[n_val:], ytr[n_val:]
    G, b = Xf.T @ Xf, Xf.T @ yf
    eye = np.eye(Xf.shape[1])
    best, w_best = np.inf, None
    for a in alphas:
        w = np.linalg.solve(G + a * eye, b)
        v = float(((yv - Xv @ w) ** 2).mean())
        if v < best:
            best, w_best = v, w

    def r2(pred, true):
        ss_tot = float(((true - true.mean()) ** 2).sum())
        return 1.0 - float(((true - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    pte, ptr = Xte @ w_best, Xf @ w_best
    return PredictResult(r2(pte, yte), r2(ptr, yf),
                         float(np.sqrt(((yte - pte) ** 2).mean())), 1,
                         time.time() - t0,
                         y_true=yte.astype(np.float32),
                         y_pred=pte.astype(np.float32))


def train_predictor(
    Xtr, Ytr, Xte, Yte,
    device: torch.device | str = "auto",
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 256,
    layers: int = 2,
    dropout: float = 0.1,
    patience: int = 10,
    seed: int = 0,
) -> PredictResult:
    """Fit the regressor and score it by R^2 on held-out data.

    R^2 is computed against the *test set's own* variance, so 0 means "no better
    than predicting the test mean" and negative values are possible and
    meaningful -- they say the model is actively worse than that baseline.
    """
    dev = pick_device(device) if isinstance(device, str) else device
    _seed_all(seed)

    Xtr = torch.as_tensor(Xtr, dtype=torch.float32).to(dev)
    Ytr = torch.as_tensor(Ytr, dtype=torch.float32).to(dev)
    Xte = torch.as_tensor(Xte, dtype=torch.float32).to(dev)
    Yte = torch.as_tensor(Yte, dtype=torch.float32).to(dev)

    n_val = max(int(0.15 * Xtr.shape[0]), 1)
    Xva, Yva = Xtr[:n_val], Ytr[:n_val]
    Xfit, Yfit = Xtr[n_val:], Ytr[n_val:]

    net = Predictor(Xtr.shape[1], hidden, layers, dropout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = Xfit.shape[0]
    bs = min(batch_size, n)
    best, best_state, bad, ran = np.inf, None, 0, 0
    t0 = time.time()

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            b = perm[s: s + bs]
            loss = nn.functional.mse_loss(net(Xfit[b]), Yfit[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        ran = ep + 1
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(net(Xva), Yva))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pte, ptr = net(Xte), net(Xfit)

        def r2(pred, true):
            ss_res = float(((true - pred) ** 2).sum())
            ss_tot = float(((true - true.mean()) ** 2).sum())
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        rmse = float(torch.sqrt(((Yte - pte) ** 2).mean()))
        return PredictResult(r2(pte, Yte), r2(ptr, Yfit), rmse, ran,
                             time.time() - t0,
                             y_true=Yte.detach().cpu().numpy(),
                             y_pred=pte.detach().cpu().numpy())


# --------------------------------------------------------------------------
# Windowing for the supervised task
# --------------------------------------------------------------------------
def make_windows(data: np.ndarray, target: int, sources: list[int],
                 window: int, stride: int = 1):
    """Build (X, y) where X is a flattened window of `sources` and y the target
    channel's value at that window's centre.

    Windows are cut with a stride and split *contiguously* by the caller, never
    shuffled before splitting: EEG samples are heavily autocorrelated, so
    neighbouring windows are near-duplicates and a random split would put
    near-copies of test windows into training. That single mistake inflates R^2
    dramatically and is the most common way this kind of benchmark goes wrong.
    """
    n_t = data.shape[1]
    if window > n_t:
        raise ValueError(f"window {window} exceeds {n_t} available samples")
    starts = np.arange(0, n_t - window + 1, stride)
    centre = starts + window // 2
    src = data[np.asarray(sources)]
    X = np.stack([src[:, s: s + window].ravel() for s in starts]).astype(np.float32)
    y = data[target, centre].astype(np.float32)
    return X, y


def split_contiguous(X, y, test_frac: float = 0.25, gap: int = 0):
    """Contiguous train/test split with an optional gap between the blocks."""
    n = len(X)
    cut = int(n * (1 - test_frac))
    return X[: max(cut - gap, 1)], y[: max(cut - gap, 1)], X[cut:], y[cut:]


def standardize(Xtr, Xte, ytr, yte):
    """Z-score using training statistics only."""
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    my, sy = float(ytr.mean()), float(ytr.std()) or 1.0
    return ((Xtr - mu) / sd, (Xte - mu) / sd,
            (ytr - my) / sy, (yte - my) / sy)


def r2_from_mi(mi_bits: float) -> float:
    """R^2 implied by MI under a joint-Gaussian assumption: R^2 = 1 - 2^(-2I)."""
    return float(1.0 - 2.0 ** (-2.0 * max(mi_bits, 0.0)))


def mi_from_r2(r2: float) -> float:
    """Inverse of the above; the MI a given R^2 implies for Gaussians."""
    r2 = min(max(r2, 0.0), 1 - 1e-12)
    return float(-0.5 * np.log2(1.0 - r2))
