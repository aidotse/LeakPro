"""Exp 1 - x_k reachability (validation gate for sensitive-attribute mode).

Question this answers
---------------------
The sensitive-attribute attack assumes the generator's latent space is rich
enough that SOME z reproduces a real member's *known* (non-sensitive) features
x_k. If it cannot, the recovered sensitive part x_s is meaningless and the whole
sensitive mode is blocked - so this is a go/no-go check to run BEFORE building
the full attack.

What it does
------------
1. Build a small tabular dataset (continuous features + a pseudo_label column).
2. Train a conditional CustomCTGAN on it.
3. For real rows, designate the first K continuous columns as "known" (x_k) and
   the rest as "sensitive" (x_s).
4. Optimize z (white-box, differentiable path) to make the generated record
   match x_k, then measure the residual on x_k.
5. PASS if the known-feature residual is driven small; FAIL otherwise.

Notes
-----
- Runnable as a self-contained smoke test on SYNTHETIC data. To run the REAL
  validation, swap `make_synthetic_data()` for your dataset (keep `pseudo_label`
  as the last, discrete column) and set KNOWN_COLS accordingly.
- Designed against `utils/CTGAN_extended.py::CustomCTGAN`.
- Not executed in this environment - run inside the `leakpro_py311` conda env
  (needs `ctgan`, `torch`). Tune the small TODOs (epochs, K) to your data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.CTGAN_extended import CustomCTGAN

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
SEED = 0
N_ROWS = 2000
N_CONT_FEATURES = 8          # continuous features
N_CLASSES = 4                # pseudo_label cardinality
N_KNOWN = 4                  # first N_KNOWN continuous cols are "known" (x_k)

DIM_Z = 32                   # latent dim (tabular: small, per design doc)
GEN_EPOCHS = 150             # TODO: raise for real data
BATCH_SIZE = 200

N_TARGETS = 64               # how many real rows to test reachability on
Z_ITERS = 800                # latent-optimization iterations per target
Z_LR = 5e-2

# Pass criterion: median relative residual on known features must be below this
# (residual measured in units of each feature's std, so it is scale-free).
PASS_THRESHOLD = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
def make_synthetic_data() -> tuple[pd.DataFrame, list[str]]:
    """Continuous features + a discrete `pseudo_label` (last column).

    Each class is a Gaussian blob so the generator has clear class structure to
    learn - a fair best case for the reachability check.
    """
    rng = np.random.default_rng(SEED)
    centers = rng.normal(0, 4, size=(N_CLASSES, N_CONT_FEATURES))
    labels = rng.integers(0, N_CLASSES, size=N_ROWS)
    X = centers[labels] + rng.normal(0, 1, size=(N_ROWS, N_CONT_FEATURES))

    cols = [f"f{i}" for i in range(N_CONT_FEATURES)]
    df = pd.DataFrame(X, columns=cols)
    df["pseudo_label"] = labels.astype(int)   # MUST be last + discrete
    return df, cols


# --------------------------------------------------------------------------- #
# A tiny differentiable target model                                           #
# --------------------------------------------------------------------------- #
class TinyTarget(nn.Module):
    """Minimal pytorch_tabular-style model: takes a {"continuous": ...} batch,
    returns {"logits": ...}. Only needed so CustomCTGAN.fit() can run; the
    reachability test itself does not depend on its quality.
    """

    def __init__(self, n_cont: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_cont, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, batch: dict) -> dict:
        return {"logits": self.net(batch["continuous"].float())}


# --------------------------------------------------------------------------- #
# Train the conditional generator                                              #
# --------------------------------------------------------------------------- #
def train_generator(df: pd.DataFrame) -> CustomCTGAN:
    gen = CustomCTGAN(
        embedding_dim=DIM_Z,
        num_classes=N_CLASSES,
        batch_size=BATCH_SIZE,
        epochs=GEN_EPOCHS,
        pac=10,
        verbose=True,
        cuda=torch.cuda.is_available(),
    )
    target = TinyTarget(N_CONT_FEATURES, N_CLASSES).to(device)

    # Self-contained losses (avoid coupling to gan_losses signatures):
    #   WGAN-style generator loss, cross-entropy inversion loss.
    def gen_criterion(y_fake):          # noqa: ANN001, ANN202
        return -y_fake.mean()

    def inv_criterion(logits, y):       # noqa: ANN001, ANN202
        return torch.nn.functional.cross_entropy(logits, y)

    gen.fit(
        train_data=df,
        target_model=target,
        num_classes=N_CLASSES,
        inv_criterion=inv_criterion,
        gen_criterion=gen_criterion,
        dis_criterion=None,             # unused by fit() (WGAN D-step is internal)
        n_iter=GEN_EPOCHS,
        n_dis=1,
        alpha=0.1,
        discrete_columns=("pseudo_label",),
        use_inv_loss=True,              # keeps inv_loss in the graph (fit asserts this)
    )
    gen.eval()
    return gen


# --------------------------------------------------------------------------- #
# The reachability test                                                        #
# --------------------------------------------------------------------------- #
def reachability(gen: CustomCTGAN, df: pd.DataFrame, cont_cols: list[str]) -> None:
    """Optimize z so the generated record matches each target's known features."""
    # Sample real targets
    sample = df.sample(n=min(N_TARGETS, len(df)), random_state=SEED)
    y = torch.tensor(sample["pseudo_label"].to_numpy(), dtype=torch.long, device=device)
    x_true = torch.tensor(sample[cont_cols].to_numpy(), dtype=torch.float32, device=device)  # [B, n_cont]

    known_idx = list(range(N_KNOWN))
    x_k = x_true[:, known_idx]                                  # [B, K]

    # Per-feature std for scale-free residuals
    feat_std = torch.tensor(df[cont_cols].std(ddof=0).to_numpy(), dtype=torch.float32, device=device)
    feat_std = feat_std.clamp(min=1e-6)

    # Optimize one z per target (batched)
    z = torch.randn(len(sample), gen.dim_z, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=Z_LR)

    for it in range(Z_ITERS):
        fakeact, _ = gen.forward_fakeact_with_labels(z, y)     # differentiable
        x_cont, _, _ = gen._pack_for_gandalf(fakeact)          # [B, n_cont], schema order
        gen_known = x_cont[:, known_idx]
        # scale-free MSE on known features
        loss = (((gen_known - x_k) / feat_std[known_idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 100 == 0:
            print(f"  iter {it + 1:4d}  known-feature loss (std units): {loss.item():.4f}")

    # --- Report ---
    with torch.no_grad():
        fakeact, _ = gen.forward_fakeact_with_labels(z, y)
        x_cont, _, _ = gen._pack_for_gandalf(fakeact)
        # relative residual per known feature, per row
        rel = ((x_cont[:, known_idx] - x_k).abs() / feat_std[known_idx])
        rel_per_row = rel.mean(dim=1).cpu().numpy()
        median_res = float(np.median(rel_per_row))

        # bonus context: residual on the *sensitive* (unknown) cols vs truth
        sens_idx = list(range(N_KNOWN, len(cont_cols)))
        if sens_idx:
            sens_rel = ((x_cont[:, sens_idx] - x_true[:, sens_idx]).abs()
                        / feat_std[sens_idx]).mean().item()
        else:
            sens_rel = float("nan")

    print("\n================ Exp 1 results ================")
    print(f"targets tested            : {len(sample)}")
    print(f"known features (x_k)       : {known_idx}")
    print(f"median rel. residual (x_k) : {median_res:.4f}  (std units; lower = better)")
    print(f"pass threshold             : {PASS_THRESHOLD}")
    print(f"sensitive residual (x_s)   : {sens_rel:.4f}  (context only, not the gate)")
    verdict = "PASS - x_k reachable; sensitive mode is viable" if median_res < PASS_THRESHOLD \
        else "FAIL - latent cannot reproduce known features; rethink generator / dim_z"
    print(f"VERDICT                    : {verdict}")
    print("===============================================")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device: {device}")
    df, cont_cols = make_synthetic_data()
    print(f"data: {df.shape[0]} rows, {len(cont_cols)} continuous features, {N_CLASSES} classes")
    print("training conditional generator ...")
    gen = train_generator(df)
    print("running x_k reachability optimization ...")
    reachability(gen, df, cont_cols)


if __name__ == "__main__":
    main()
