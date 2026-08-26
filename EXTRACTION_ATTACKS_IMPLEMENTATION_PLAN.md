# Carlini and SIDE extraction implementation plan

## Scope

Add image training-data extraction as a distinct LeakPro attack family. MIA and MINV are architectural references
only: extraction keeps their `handler -> factory -> prepare_attack -> run_attack -> result.save` lifecycle without
reusing their classifier-specific attack logic.

Included:

- Carlini et al. conditional black-box extraction and unconditional reference audit;
- SIDE's white-box time-dependent classifier-guidance branch;
- deterministic configuration, bounded-memory scoring, persisted candidates, traces, and stress tests.

Excluded for now:

- DoRI and Webster one-step extraction, which are different attacks;
- GAN adaptation;
- SIDE's Stable Diffusion LoRA branch, whose published description does not determine a unique generic
  implementation.

## Mathematical contract

Let an image contain `D` scalar pixel-channel values. LeakPro uses normalized L2 distance

$$
d_2(x,y)=\sqrt{\frac{1}{D}\sum_{p=1}^{D}(x_p-y_p)^2}.
$$

Carlini distances and persisted candidates use `[0,1]` coordinates. SIDE L2 evaluation uses the configured target
range (`[0,1]` or `[-1,1]`) so paper bands remain meaningful; persisted SIDE candidates use `[0,1]`. Custom
similarity scorers receive `[0,1]` tensors and own their preprocessing contract.

### Carlini conditional attack

Split each image into corresponding tiles `b` in a grid `B` and define

$$
d_{\mathrm{tile}}(x,y)
=\max_{b\in B}\sqrt{\frac{1}{|b|}\sum_{p\in b}(x_p-y_p)^2}.
$$

For `N` generations made with one condition, construct the graph

$$
A_{ij}=\mathbf 1\!\left[d_{\mathrm{tile}}(x_i,x_j)\leq \tau_{\mathrm{tile}}\right],
\qquad A_{ii}=0.
$$

Find an exact maximum clique `C`. A condition qualifies when

$$
|C|\geq m,
$$

with paper defaults `N=500`, a `4 x 4` grid, and `m=10`. The graph threshold
`tau_tile` is not published and remains a recorded engineering parameter. The paper specifies clique qualification
and condition ranking, but not which image represents a qualifying clique. This implementation uses the medoid policy

$$
i^*=\arg\min_{i\in C}\frac{1}{|C|}\sum_{j\in C}d_{\mathrm{tile}}(x_i,x_j),
$$

and rank qualifying conditions by their mean unordered within-clique distance

$$
s(C)=\frac{2}{|C|(|C|-1)}\sum_{i<j,\ i,j\in C}d_{\mathrm{tile}}(x_i,x_j).
$$

When authorized training references `T` are available, final L2 verification is

$$
\min_{z\in T}d_2(x_{i^*},z)\leq 0.15.
$$

Clique qualification and reference verification are separate claims.

### Carlini unconditional attack

Let `x_hat=argmin_{z in T} d_2(x,z)` be generation `x`'s nearest training record, and let `S_(x_hat)` contain the
`k` closest records in `T` to `x_hat`, including `x_hat` itself as the literal closest element. Compute

$$
R(x)=
\frac{d_2(x,\hat{x})}
{\alpha\,\frac{1}{k}\sum_{y\in S_{\hat{x}}}d_2(\hat{x},y)}.
$$

The paper settings are `alpha=0.5`, `k=50`, and one million generations. A candidate qualifies when
`R(x) <= tau_R`; the implementation defaults to `tau_R=1`. It then deduplicates by nearest reference, retaining
the lowest-ratio candidate. Direct L2 verification remains a separate field.

### SIDE

Generate a synthetic bank `S={x_i}` and extract features `z_i=F(x_i)`. Following Algorithm 1, ordinary
feature-space K-means solves

$$
\min_{\mu_1,\ldots,\mu_K}\sum_i\min_k\lVert z_i-\mu_k\rVert_2^2.
$$

For cluster `C_k`, normalized centroid `mu_k`, and paper threshold `tau=0.5`, cohesion is

$$
h_k=\frac{1}{|C_k|}\sum_{z_i\in C_k}\frac{z_i^\top\mu_k}{\lVert z_i\rVert_2\lVert\mu_k\rVert_2},
\qquad C_k\text{ is retained if }h_k\geq\tau.
$$

After filtering, every synthetic sample is relabeled by its nearest retained centroid. For target schedule
`alpha_bar_t`, forward noising must use the target's exact operation

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

Train the time-dependent classifier with

$$
\mathcal L_{\mathrm{cls}}(\phi)
=\mathbb E_{(x_0,y),t,\epsilon}\left[-\log p_\phi(y\mid x_t,t)\right].
$$

The paper does not determine the classifier epoch count or one concrete timestep-injection architecture. LeakPro
therefore records the configured epochs and its timestep-conditioned ResNet as implementation choices.

Classifier guidance supplies

$$
g_\phi(x_t,t,c)=\lambda\nabla_{x_t}\log p_\phi(c\mid x_t,t).
$$

The adapter must insert `g_phi` with the sign and scaling expected by the target's score, noise, velocity, or
denoised-sample parameterization. The attack does not assume these parameterizations are interchangeable.

For an optional higher-is-more-similar score, let `b_i=max_j s(x_i,r_j)` be generation `i`'s best match. For L2,
use `b_i=min_j d_2(x_i,r_j)` instead. For band `B`, `N_G` generations, and reference set `T`, report

$$
\mathrm{AMS}_B=\frac{1}{N_G}\sum_{i=1}^{N_G}\mathbf 1\!\left[b_i\in B\right],
$$

$$
\mathrm{UMS}_B=\frac{1}{N_G}\sum_{j=1}^{|T|}\mathbf 1\!\left[\exists i:s(x_i,r_j)\in B\right].
$$

For L2 UMS, replace `s` with `d_2`. Thus AMS uses one best-match value per candidate, while UMS counts every
reference reached by an in-band pair. The score function and orientation must be recorded; normalized L2 and SSCD
similarity are not interchangeable.

The deferred Stable Diffusion branch would optimize LoRA parameters `Delta theta` with

$$
\mathcal L_{\mathrm{LoRA}}
=\mathbb E\left[\left\lVert\epsilon-\epsilon_{\theta+\Delta\theta}(x_t,t,y)\right\rVert_2^2\right],
$$

but it will not be implemented until a named checkpoint, target-module list, conditioning contract, and behavioral
oracle are available.

## LeakPro structure

```text
audit.yaml
  -> LeakProConfig
  -> ExtractionHandler
  -> AttackScheduler
  -> AttackFactoryExtraction
     -> AttackCarliniExtraction
     -> AttackSIDEExtraction
  -> prepare_attack()
  -> run_attack()
  -> ExtractionResult.save()
```

`AbstractExtractionInputHandler` exposes only the required generator boundary: sampling, optional conditions and
references, SIDE features, optional classifier construction, and optional pairwise evaluation. It does not require
the model metadata, criterion, train/test indices, or public classifier dataset used by MIA and MINV.

`ExtractionTargetConfig.fingerprint` is required and must identify the checkpoint plus provider implementation and
target-specific callbacks. LeakPro combines it with hashed conditions and reference tensors. The combined audit
fingerprint enters the result ID and provenance, preventing identical parameters on different targets from sharing
an output path.

## Execution phases

- [x] Inspect MIA and MINV factories, handlers, schemas, lifecycle, results, and tests.
- [x] Add extraction schemas, handler, factory, scheduler dispatch, and package export.
- [x] Implement Carlini conditional and unconditional paths.
- [x] Implement SIDE classifier guidance, surrogate labels, uniform seeded cluster selection, and optional metrics.
- [x] Add finite-value, shape, device, authorization, and lifecycle guards.
- [x] Persist compact JSON metadata, compressed candidate arrays, and phase traces through staged, rollback-safe saves.
- [x] Fingerprint conditions instead of persisting raw prompt or label values.
- [x] Reject result collisions unless `overwrite_results=true` is explicit.
- [x] Reject extraction PDF requests before attack execution; extraction PDF rendering is not implemented.
- [x] Add numerical, graph, attack, result, fault-injection, and public-path tests.
- [x] Run repository lint and focused MIA/MINV regression tests.
- [x] Execute moderate non-divisible-block and tiled-distance stress profiles.
- Release gate: obtain fresh independent review of the exact final diff.
- [ ] Run real checkpoint/data reproductions before making empirical extraction claims.

## Trace and stress contract

Each result records attack ID, phase, elapsed time, sampler-call count, generated-image count, retained clusters,
guidance-call count, and candidate count. A sampler call may contain many denoiser evaluations; the generic adapter
cannot report internal model calls without target-specific instrumentation. The trace never logs image pixels or raw
conditions. Result configuration carries the seed and attack thresholds; the real-run manifest must additionally
record checkpoint, code, dependency, and dataset hashes.

Stress profiles:

| Profile | Check | Failure signal |
|---|---|---|
| Numerical | zero distance, exact copy, near threshold, non-finite tensor | wrong orientation or silent NaN |
| Graph | sparse/dense seeded graphs against brute force | missed or nondeterministic clique |
| Blocking | non-divisible candidate/reference blocks against dense result | boundary loss or wrong top-k |
| Lifecycle | run-before-prepare, authorization false, missing SIDE operations | query before validation |
| Labels | deterministic K-means, cluster collapse, uniform seeded guidance labels | unstable or missing diversity |
| Scaling | increasing generations, references, image size, and graph density | unexpected superlinear memory |
| Persistence | interrupted/duplicate save and path-unsafe identifier | partial or escaped output |

Complexity expectations:

- conditional Carlini distance construction is `O(N^2 D)` per condition with `O(N^2)` graph memory; exact maximum
  clique is exponential in the worst case;
- unconditional Carlini scoring is `O(|T|^2 D + G |T| D)`: reference-neighborhood means are computed once, then
  reused for every generation; block streaming avoids quadratic distance-matrix memory;
- SIDE feature clustering is approximately `O(I N_syn K F)` for K-means iterations `I` and feature width `F`;
- SIDE classifier training is `O(E N_syn)` classifier passes plus target forward-noising;
- SIDE guided generation requires a classifier gradient at every reverse-diffusion step invoked by the adapter.

## Existing implementation evidence

No official Carlini attack code or public SIDE repository was located in the completed 2026-08-26 search. The two
closest community Carlini implementations found were unlicensed and incomplete. OpenAI Improved Diffusion is useful
as a target stack, not as attack code. The official Webster one-step repository implements a different extraction
method. These resources remain behavioral references only; no external attack code is copied into LeakPro.

## Plan evaluation

The plan is internally coherent because threat model, input contract, metric orientation, and result claim are
separated. Its main risk is not ordinary software correctness but paper-scale fidelity: target-specific timestep
mapping, guidance parameterization, feature/checkpoint choice, graph-threshold calibration, reference coverage, and
compute budget can change empirical results.

Release gate for framework code: focused tests, lint, public-path trace, regression tests, and independent diff
approval. Release gate for research claims: exact model/data provenance, paper-scale budgets, peak-resource records,
threshold calibration separated from evaluation, and manual inspection of verified matches. Until the second gate
passes, the implementation is an attack-capable framework integration, not a reproduction of either paper's reported
extraction rate.
