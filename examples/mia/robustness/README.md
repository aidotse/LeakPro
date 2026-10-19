# Model-level robustness

Run one configured membership inference attack on the original target and then
on distinct shadows drawn without replacement from the same trained model pool.
The original target joins the references when a shadow becomes the target. The
model being attacked is always excluded from its own references.

This continues Rickard Brännvall's [robustness PR #351](https://github.com/aidotse/LeakPro/pull/351).
The wrapper adapts its cached-signal model selection and pointwise ROC percentile
aggregation. The original commits remain ancestors of this branch. Attack-specific
vectorization and output-heavy experimental notebooks are outside this change.

## Run

First initialize `LeakPro` with your existing data, target model, training metadata,
and input handlers, as in the CIFAR example. Then run:

```python
from leakpro.attacks.robustness import Robustness

handler = leakpro.handler
runner = Robustness(
    handler=handler,
    attack_name="lira",
    attack_config={
        "num_shadow_models": 16,
        "training_data_fraction": len(handler.train_indices) / handler.population_size,
        "online": True,
    },
    n_runs=9,  # Original target plus eight distinct shadow targets.
    num_shadow_models=16,
    seed=42,
)
runs = runner.run()
original = runs[0]["result_object"]
print(original.roc_auc)
print(runner.summary)
print(runner.output_dir)
```

The wrapper uses the normal MIA factory and each attack's `prepare_attack()` and
`run_attack()` methods. All model selection, membership handling, cache alignment,
and aggregation live in the wrapper; there are no per-attack adapter classes.

## Requirements and limits

- Target and shadow training settings must agree: architecture, initialization
  parameters, optimizer, loss, epochs, batch size, and number of training records.
  The user's training callback must follow the same procedure used for the target.
- Shadows train once on independently sampled, target-sized subsets of the original
  audit-record universe. Compatible models from the wrapper's dedicated
  `robustness/pool` cache are reused. Ordinary shadow caches do not record their
  sampling strategy and are kept separate. The wrapper validates cached membership
  and never rewrites existing model metadata to claim compatibility.
- `n_runs` includes the original target and must be between 2 and the shadow count
  plus 1. Subsequent targets are selected without replacement using the seed.
- Reference-based attacks must request the entire fixed reference pool and accept
  its training records under their normal data restrictions. Incompatible requests
  raise an error instead of training substitute references. An attack requiring
  disjoint auxiliary training data, such as a trajectory attack in that setting,
  cannot use this interchangeable pool unchanged.
- Attacks without shadow references can still rotate targets. Their auxiliary-data
  requirements remain in force. Attack-specific classifier fitting or distillation
  may still be needed for each target.
- The wrapper currently accepts attacks returning one `MIAResult`, using a PyTorch
  `MIAHandler`. It does not support simultaneous audits in the same process because
  LeakPro's auxiliary-model handlers are singletons.
- A seed controls target order, new pool training, and attack preparation through
  LeakPro's existing seed utility. Reused model weights retain their original training
  outcomes. Bitwise GPU determinism is not promised.

## Results

Each run saves the normal attack result in its own output directory. `summary.json`
records pool model hashes, record IDs, selected targets and references, per-run
member counts, and links to saved attack results. Attack filtering can leave
different eligible record counts in different rotations; those counts are retained.
Model ID `-1` denotes the original target; other IDs identify the cached shadows.

The original target is reported separately. The remaining results are summarized
with mean AUC and 2.5th/97.5th percentiles where AUC exists. Full ROC results also
receive mean TPR and pointwise TPR percentiles on a common FPR grid. Repeated FPR
values use their highest TPR before interpolation.

These are empirical summaries across a finite, shared model pool. They are not
calibrated confidence intervals for the original target, simultaneous confidence
bands, or additional independent training outcomes. Low-FPR resolution still
depends on the number of evaluated nonmembers.

## Relationship to bootstrap sampling

PR #351's helper accepted `replace=False`, but its supplied bootstrap notebooks
used the default `replace=True`. This wrapper implements target rotation without
reference resampling. Bootstrap reference draws would be a separate selection
method, not a prerequisite for this implementation.

For the research background, see Andreas Söderman's 2025 master's thesis,
[*Reliable Privacy Risk Testing for AI Systems*](https://ltu.diva-portal.org/smash/record.jsf?pid=diva2%3A1974044),
supervised by Rickard Brännvall, especially sections 3.1 and 4.3. Its findings about
leave-one-out and bootstrap bands are preliminary and attack-dependent. Brännvall's
2026 [BaVarIA paper](https://arxiv.org/html/2603.11799) also evaluates rotated targets
and notes the dependence introduced by a shared model pool.
