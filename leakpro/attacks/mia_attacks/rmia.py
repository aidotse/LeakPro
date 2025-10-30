"""Implementation of the RMIA attack."""
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

##### some helper functions used in the vectorized code of rmia #####

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid.

    Parameters
    ----------
    x : np.ndarray
        Input logits.

    Returns
    -------
    np.ndarray
        Sigmoid(x), computed stably in float64.

    """
    x = x.astype(np.float64, copy=False)
    pos = x >= 0
    neg = ~pos
    z = np.zeros_like(x, dtype=np.float64)
    z[pos] = np.exp(-x[pos])
    z[neg] = np.exp(x[neg])
    out = np.empty_like(x, dtype=np.float64)
    out[pos] = 1.0 / (1.0 + z[pos])
    out[neg] = z[neg] / (1.0 + z[neg])
    return out


def _logit_from_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert probabilities to logits log(p/(1-p)).

    Parameters
    ----------
    p : np.ndarray
        Probabilities in [0, 1].
    eps : float, optional
        Numerical clamp for stability.

    Returns
    -------
    np.ndarray
        Logits.

    """
    p = np.clip(p, eps, 1.0 - eps).astype(np.float64, copy=False)
    return np.log(p) - np.log1p(-p)


def _estimate_prior_probs(
    shadow_logits: np.ndarray,
    shadow_inmask: np.ndarray,
    use_out_only: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the marginal p(x) from shadow models.

    Parameters
    ----------
    shadow_logits : np.ndarray
        Array of shape (n, m) with one binary logit per (sample, shadow-model).
    shadow_inmask : np.ndarray
        Boolean array of shape (n, m): True if that shadow trained on the sample.
    use_out_only : bool, optional
        If True, compute OUT-only mean; otherwise mean over all shadows.
    eps : float, optional
        Numerical clamp for stability.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        prior : (n,) estimated p(x)
        counts : (n,) number of references used per sample

    """
    probs = _sigmoid(shadow_logits.astype(np.float64))
    ref = np.where(shadow_inmask, np.nan, probs) if use_out_only else probs
    with np.errstate(all="ignore"):
        prior = np.nanmean(ref, axis=1)
        counts = np.sum(~np.isnan(ref), axis=1).astype(np.int64)

    # Fallback: if no OUT references, use all-model mean
    need_fallback = counts == 0
    if np.any(need_fallback):
        with np.errstate(all="ignore"):
            prior_all = np.nanmean(probs, axis=1)
        prior[need_fallback] = prior_all[need_fallback]
        counts[need_fallback] = np.sum(~np.isnan(probs[need_fallback]), axis=1)

    prior = np.clip(prior, eps, 1.0 - eps)
    return prior, counts


def vectorized_rmia_score(  # noqa: PLR0915
    shadow_logits: np.ndarray,
    shadow_inmask: np.ndarray,
    target_logits: np.ndarray,
    target_inmask: Optional[np.ndarray] = None,  # unused; kept for API parity
    z_sample_size: int = 512,
    gamma: float = 1.0,
    rng: Optional[np.random.Generator | int] = None,
    use_all_z: bool = False,
    batch_points: int = 1024,
    eps: float = 1e-12,
    scale: float = 1.0,
    z_chunk: int = 2048,
    offline_a: Optional[float] = None,
) -> np.ndarray:
    """Compute RMIA scores in a vectorized manner.

    Parameters
    ----------
    shadow_logits : np.ndarray
        (n, m) array of binary logits per (sample, shadow model).
    shadow_inmask : np.ndarray
        (n, m) boolean IN-mask (True if that shadow trained on the sample).
    target_logits : np.ndarray
        (n,) binary logits for the target model.
    target_inmask : Optional[np.ndarray], optional
        Ignored; present for API compatibility.
    z_sample_size : int, optional
        Per-point Z-subsample size when use_all_z is False.
    gamma : float, optional
        Decision margin for the pairwise LR test.
    rng : Optional[np.random.Generator | int], optional
        RNG or seed for Z sampling.
    use_all_z : bool, optional
        If True, compare against all Z with streaming chunks.
    batch_points : int, optional
        Number of audit points processed per batch.
    eps : float, optional
        Numerical clamp.
    scale : float, optional
        Multiplicative prior scaling (unused if offline_a is provided).
    z_chunk : int, optional
        Chunk size for streaming Z when use_all_z is True.
    offline_a : Optional[float], optional
        If provided, apply 0.5*((1+a)*p_out + (1-a)) correction.

    Returns
    -------
    np.ndarray
        (n,) RMIA scores in [0, 1].

    """
    del target_inmask  # silence ARG001 (unused)

    n, m = shadow_logits.shape
    assert shadow_inmask.shape == (n, m)
    assert target_logits.shape == (n,)

    # p_theta: target true-label prob per sample
    p_theta = _sigmoid(target_logits.astype(np.float64))
    p_theta = np.clip(p_theta, eps, 1.0 - eps)

    # prior: OUT-only mean (fallback to all if no OUT)
    prior, _ = _estimate_prior_probs(shadow_logits, shadow_inmask, use_out_only=True, eps=eps)

    # offline linear correction (paper): 0.5 * ((1+a)*p_out + (1-a))
    if offline_a is not None:
        a = float(offline_a)
        prior = 0.5 * ((1.0 + a) * prior + (1.0 - a))
    else:
        prior = prior * float(scale)

    prior = np.clip(prior, eps, 1.0 - eps)

    # RNG
    if isinstance(rng, (int, np.integer)) or rng is None:
        rng = np.random.default_rng(rng)

    # Z selection
    if use_all_z:
        pz_all = p_theta
        prz_all = prior
    else:
        k = min(z_sample_size, max(1, n - 1))
        z_samples = np.empty((n, k), dtype=np.int64)
        for i in range(n):
            pool = np.arange(n - 1, dtype=np.int64)
            idxs = rng.choice(pool, size=k, replace=(k > (n - 1)))
            idxs = np.where(idxs >= i, idxs + 1, idxs)
            z_samples[i, :] = idxs

    # Scores
    scores = np.zeros(n, dtype=np.float64)
    i0 = 0
    while i0 < n:
        i1 = min(n, i0 + batch_points)
        p_x = p_theta[i0:i1][:, None]  # (b,1)
        pr_x = prior[i0:i1][:, None]   # (b,1)

        if use_all_z:
            # Memory-safe: stream Z in chunks so we never materialize (b x n)
            b = i1 - i0
            votes = np.zeros(b, dtype=np.int64)
            total = 0
            s = 0
            while s < n:
                t = min(n, s + z_chunk)
                rz = pz_all[s:t] / prz_all[s:t]          # (t-s,)
                likelihood_ratio = (p_x / pr_x) / rz[None, :]  # (b, t-s)
                # Exclude self-comparisons
                for j in range(b):
                    idx = i0 + j
                    if s <= idx < t:
                        likelihood_ratio[j, idx - s] = -np.inf
                votes += np.count_nonzero(float(gamma) <= likelihood_ratio, axis=1)  # SIM300-friendly
                total += (t - s)
                s = t
            denom = max(1, total - 1)  # minus self
            scores[i0:i1] = votes / float(denom)
        else:
            idx = z_samples[i0:i1, :]     # (b,k)
            p_z = p_theta[idx]            # (b,k)
            pr_z = prior[idx]             # (b,k)
            lrxz = (p_x / pr_x) / (p_z / pr_z)
            votes = (lrxz >= float(gamma)).mean(axis=1)
            scores[i0:i1] = votes

        i0 = i1

    return scores

class AttackRMIA(AbstractMIA):
    """Implementation of the RMIA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the RMIA attack."""

        num_shadow_models: int = Field(default=1,
                                       ge=1,
                                       description="Number of shadow models")
        temperature: float = Field(default=2.0,
                                   ge=0.0,
                                   description="Softmax temperature")
        training_data_fraction: float = Field(default=0.5,
                                              ge=0.0,
                                              le=1.0,
                                              description="Part of available attack data to use for shadow models")
        online: bool = Field(default=False,
                             description="Online vs offline attack")
        # Parameters to be used with optuna
        z_data_sample_fraction: float = Field(default=0.5,
                                                ge=0.0,
                                                le=1.0,
                                                description="Part of available attack data to use for estimating p(z)",)
        gamma: float = Field(default=2.0,
                        ge=0.0,
                        description="Parameter to threshold LLRs",
                        json_schema_extra = {"optuna": {"type": "float", "low": 0.1, "high": 10, "log": True}})
        offline_a: float = Field(default=0.33,
                                 ge=0.0,
                                 le=1.0,
                                 description="Parameter to estimate the marginal p(x)",
                                 json_schema_extra = {"optuna": {"type": "float", "low": 0.0, "high": 1.0,"enabled_if": lambda model: not model.online}})  # noqa: E501

        # Vectorized-path options (minimal & safe defaults)
        vectorized: bool = Field(
            default=False,
            description="Enable the vectorized RMIA fast-path (keeps classic path when False)."
        )

        vec_batch_points: int = Field(
            default=1024,
            ge=1,
            description=(
                "Number of audit points to process per inner batch in the vectorized path. "
                "Lower this if you have limited memory; keep larger for speed."
            ),
        )

        vec_use_all_z: bool = Field(
            default=False,
            description=(
                "If True, each audit point is compared to the entire Z population (may be memory heavy). "
                "When False, a per-point Z subsample is used (derived from z_data_sample_fraction)."
            ),
        )

        vec_z_chunk: int = Field(
            default=2048,
            ge=1,
            description=(
                "Chunk size used to stream the Z set when vec_use_all_z=True. "
                "Reduces peak memory by processing Z in slices of this size."
            ),
        )


        @model_validator(mode="after")
        def check_num_shadow_models_if_online(self) -> Self:
            """Check if the number of shadow models is at least 2 when online is True.

            Returns
            -------
                Config: The attack configuration.

            Raises
            ------
                ValueError: If online is True and the number of shadow models is less than 2.

            """
            if self.online and self.num_shadow_models < 2:
                raise ValueError("When online is True, num_shadow_models must be >= 2")
            return self

    def __init__(self:Self,
                 handler: MIAHandler,
                 configs: dict
                 ) -> None:
        """Initialize the RMIA attack.

        Args:
        ----
            handler (MIAHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        logger.info("Configuring the RMIA attack")
        # Initializes the pydantic object using the user-provided configs
        # This will ensure that the user-provided configs are valid
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)

        # Call the parent class constructor. It will check the configs.
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        self.shadow_models = []
        self.epsilon = 1e-6
        self.shadow_models = None
        self.shadow_model_indices = None

        self.load_for_optuna = False
        self.attack_cache_folder_path = ShadowModelHandler().attack_cache_folder_path
        self.bayesian_optimization = False

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "RMIA attack"
        reference_str = "Zarifzadeh, Sajjad, Philippe Cheng-Jie Marc Liu, and Reza Shokri. \
            Low-Cost High-Power Membership Inference by Boosting Relativity. (2023)."
        summary_str = "The RMIA attack is a membership inference attack based on the output logits of a black-box model."
        detailed_str = "The attack is executed according to: \
            1. A fraction of the population is sampled to compute the likelihood LR_z of p(z|theta) to p(z) for the target model.\
            2. The ratio is used to compute the likelihood ratio LR_x of p(x|theta) to p(x) for the target model. \
            3. The ratio LL_x/LL_z is viewed as a random variable (z is random) and used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def _prepare_shadow_models(self:Self) -> None:

        # Shadow models are trained on all data points in pairs.
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = True,
                                                                    include_test_indices = True)
        # train shadow models
        logger.info(f"Check for {self.num_shadow_models} shadow models (dataset: {len(self.attack_data_indices)} points)")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = self.num_shadow_models,
            shadow_population = self.attack_data_indices,
            training_fraction = self.training_data_fraction,
            online = True)
        # load shadow models
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        if self.online is False:
            self.out_indices = ~ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T

    def prepare_attack(self:Self) -> None:
        """Prepare Data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """
        logger.info("Preparing shadow models for RMIA attack")

        # If we already have one run, we dont need to check for shadow models as logits are stored
        if not self.load_for_optuna:
            self._prepare_shadow_models()

            self.ground_truth_indices = self.handler.get_labels(self.audit_dataset["data"])
            self.logits_theta = ShadowModelHandler().load_logits(name="target")
            self.logits_shadow_models = []
            for indx in self.shadow_model_indices:
                self.logits_shadow_models.append(ShadowModelHandler().load_logits(indx=indx))

        # collect the softmax output of the correct class
        n_attack_points = self.z_data_sample_fraction * len(self.handler.population) # points to compute p(z)

        # pick random indices sampled from the attack data
        z_indices = np.random.choice(self.attack_data_indices, size=int(n_attack_points), replace=False)
        z_labels = self.handler.get_labels(z_indices)

        p_z_given_theta = softmax_logits(self.logits_theta, self.temperature)[z_indices,z_labels]  # noqa: E501
        p_z_given_theta = np.atleast_2d(p_z_given_theta)

        # collect the softmax output of the correct class for each shadow model
        sm_logits_shadow_models = [softmax_logits(x, self.temperature) for x in self.logits_shadow_models]
        p_z_given_shadow_models = np.array([x[z_indices,z_labels] for x in sm_logits_shadow_models])  # noqa: E501

        # evaluate the marginal p(z)
        if self.online is True:
            p_z = np.mean(p_z_given_shadow_models, axis=0, keepdims=True)
        else:
            # create a mask that checks, for each point, if it was in the training set
            p_z = np.mean(p_z_given_shadow_models, axis=0)
            p_z = 0.5*((self.offline_a + 1) * p_z + (1-self.offline_a))

        self.ratio_z = p_z_given_theta / (p_z + self.epsilon)

    def _run_attack(self:Self) -> None:
        logger.info("Running RMIA online attack")

        # collect the softmax output of the correct class
        n_audit_points = len(self.ground_truth_indices)
        p_x_given_theta = softmax_logits(self.logits_theta, self.temperature)[np.arange(n_audit_points),self.ground_truth_indices]
        p_x_given_theta = np.atleast_2d(p_x_given_theta)

        # run points through shadow models, colelct logits and compute p(x)
        sm_shadow_models = [softmax_logits(x, self.temperature) for x in self.logits_shadow_models]
        p_x_given_shadow_models = np.array([x[np.arange(n_audit_points),self.ground_truth_indices] for x in sm_shadow_models])

        if self.online is True:
            p_x = np.mean(p_x_given_shadow_models, axis=0, keepdims=True)
        else:
            # compute the marginal p(x) from P_out and p_in where p_in = a*p_out+b
            masked_values = np.where(self.out_indices, p_x_given_shadow_models, np.nan)
            p_x_out = np.nanmean(masked_values, axis=0)
            p_x = 0.5*((self.offline_a + 1) * p_x_out + (1-self.offline_a))

        # compute the ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_theta / (p_x + self.epsilon)

        # ===== Vectorized fast-path (audit-as-Z) =====
        if self.configs.vectorized:
            # Build inputs your helper expects.
            # Target probs (N,) → logits
            target_probs_true = p_x_given_theta.ravel()              # (N,)
            target_logits = _logit_from_prob(target_probs_true)

            # Shadow probs (M, N) → transpose to (N, M) then logits
            shadow_probs_true = p_x_given_shadow_models.T            # (N, M)
            shadow_logits = _logit_from_prob(shadow_probs_true)

            # Shadow in-mask: mark samples that each shadow model trained on.
            # offline: we have out_indices -> derive IN mask
            # online : no per-shadow in/out info -> treat *all* as OUT (i.e., IN mask all False)
            # Shadow in-mask:
            shadow_inmask = (~self.out_indices).T if not self.online else np.zeros_like(p_x_given_shadow_models.T, dtype=bool)

            # Optional: membership mask for ROC (not used by scorer)
            target_inmask = np.zeros_like(target_probs_true, dtype=bool)
            if "in_members" in self.audit_dataset:
                target_inmask[self.audit_dataset["in_members"]] = True

            # Derive z_sample_size from existing fraction
            n_targets = target_logits.shape[0]
            z_sample_size = max(1, min(n_targets - 1, int(self.z_data_sample_fraction * n_targets)))

            # Call your vectorized scorer; stream when using all-Z to be memory-safe
            scores = vectorized_rmia_score(
                shadow_logits=shadow_logits,
                shadow_inmask=shadow_inmask,
                target_logits=target_logits,
                target_inmask=target_inmask,
                z_sample_size=z_sample_size,
                gamma=float(self.gamma),
                rng=None,
                use_all_z=bool(self.vec_use_all_z),
                batch_points=int(self.vec_batch_points),
                eps=1e-12,
                scale=1.0,
                z_chunk=int(self.vec_z_chunk),
                offline_a=float(self.offline_a) if not self.online else None,
            )

            # Convert scores to the same shape your classic code produces
            score = scores.reshape(1, -1)

            # Pick out in/out signals (same as classic path)
            in_members = self.audit_dataset["in_members"]
            out_members = self.audit_dataset["out_members"]
            self.in_member_signals = score[0, in_members].reshape(-1, 1)
            self.out_member_signals = score[0, out_members].reshape(-1, 1)
            return
        # ===== end vectorized =====

        # for each x, compute the score
        score = np.zeros((1, n_audit_points))
        for i in range(n_audit_points):
            likelihoods = ratio_x[0,i] / self.ratio_z
            score[0, i] = np.mean(likelihoods > self.gamma)

        # pick out the in-members and out-members signals
        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]
        self.in_member_signals = score[0,in_members].reshape(-1,1)
        self.out_member_signals = score[0,out_members].reshape(-1,1)

    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """
        # perform the attack
        self._run_attack()

        # set true labels for being in the training dataset
        true_labels = np.concatenate([np.ones(len(self.in_member_signals)), np.zeros(len(self.out_member_signals)),])
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # Ensure we use the stored quantities from now
        self.load_for_optuna = True

        # Save the results
        return MIAResult.from_full_scores(true_membership=true_labels,
                                          signal_values=signal_values,
                                          result_name="RMIA",
                                          metadata=self.configs.model_dump())

    def reset_attack(self: Self, config:BaseModel) -> None:
        """Reset attack to initial state."""

        # Assign the new configuration parameters to the object
        self.configs = config
        for key, value in config.model_dump().items():
            setattr(self, key, value)

        # new hyperparameters have been set, let's prepare the attack again
        self.prepare_attack()
### unit test
if __name__ == "__main__":
    print("[RMIA] self-test comparing classic (looped) vs vectorized paths")

    rng = np.random.default_rng(42)
    N, M = 80, 5
    gamma = 2.0
    offline_a = 0.33

    # synthetic logits
    target_logits = rng.normal(size=N)
    shadow_logits = rng.normal(size=(N, M))
    shadow_inmask = rng.random((N, M)) < 0.5

    # ----- reference (non-vectorized) implementation -----
    def _naive_scores(
        target_logits: np.ndarray,
        shadow_logits: np.ndarray,
        inmask: np.ndarray,
        gamma: float,
        offline_a: float,
    ) -> np.ndarray:
        p_theta = _sigmoid(target_logits)
        prior, _ = _estimate_prior_probs(shadow_logits, inmask, use_out_only=True)
        prior = 0.5 * ((1 + offline_a) * prior + (1 - offline_a))
        num_points = len(p_theta)
        scores = np.zeros(num_points, dtype=np.float64)
        for i in range(num_points):
            rx = p_theta[i] / prior[i]
            pool = np.delete(np.arange(num_points), i)
            rz = p_theta[pool] / prior[pool]
            votes = np.count_nonzero((rx / rz) >= gamma)
            scores[i] = votes / float(len(pool))
        return scores

    ref_scores = _naive_scores(target_logits, shadow_logits, shadow_inmask,
                               gamma=gamma, offline_a=offline_a)

    # ----- vectorized version: use_all_z=True to match the above -----
    vec_scores = vectorized_rmia_score(
        shadow_logits=shadow_logits,
        shadow_inmask=shadow_inmask,
        target_logits=target_logits,
        z_sample_size=max(1, int(0.2 * N)),
        gamma=gamma,
        rng=42,
        use_all_z=True,          # full-Z to replicate reference
        batch_points=16,
        z_chunk=20,              # stress chunking
        offline_a=offline_a,
    )

    # ----- compare -----
    diff = np.abs(ref_scores - vec_scores)
    print("First 10 ref vs vec scores:") # noqa: T201
    for i in range(min(10, N)):
        print(f"  i={i:02d}: ref={ref_scores[i]:.6f}, vec={vec_scores[i]:.6f}") # noqa: T201
    print(f"Mean(abs diff) = {diff.mean():.3e},  Max(abs diff) = {diff.max():.3e}") # noqa: T201

    # strict check
    assert np.allclose(ref_scores, vec_scores, atol=1e-12), \
        "Vectorized RMIA results differ from reference!"

    print("[RMIA] Self-test passed ✔  vectorized == classic within tolerance") # noqa: T201
