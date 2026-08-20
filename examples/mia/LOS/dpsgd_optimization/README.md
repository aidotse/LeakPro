# DP-SGD privacy-utility optimization on MIMIC-III length-of-stay — pending migration

The LOS DP-SGD optimization examples (logistic regression and GRU-D) are **not yet
migrated** to the current, integrated pipeline and have been removed. The earlier
LR/GRU-D scripts scored the frontier with a hand-rolled membership attack rather
than LeakPro's own RMIA, and swept a fixed set of points rather than optimizing;
both are fixed in the reference example.

See [`examples/mia/cifar/dpsgd_optimization`](../../cifar/dpsgd_optimization/README.md)
for the reference implementation: Optuna-driven Bayesian optimization over the
DP-SGD knobs, with the privacy axis measured by the real
`leakpro.attacks.mia_attacks.rmia` attack through the LeakPro pipeline.

To bring LOS back, mirror the CIFAR example:

- a `dp_handler.py` providing the LOS `UserDataset` plus one DP-SGD training
  function for the LR (or GRU-D) target, reading the noise multiplier and
  clipping norm directly from `dpsgd_dic.pkl`;
- a `run_optimization.py` that builds the population, and for each proposed config
  trains the target, saves it in the LeakPro target-folder layout, and calls
  `leakpro.optimization.run_rmia_audit`;
- utility measured as AUC instead of accuracy in the objective.
