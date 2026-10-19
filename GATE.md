# Robustness implementation verification

Scope: continue PR #351 at `561b812788eecdc07b989bb8c581aac40dac26a3`, preserving its
commits and authorship; implement a minimal MIA target-rotation wrapper with its
own selection, membership handling, and summaries, without per-attack adapters.

Upstream integration: merged `origin/main` at `971cdb76966d8daec9bb08e55e3ad399c19ad1bc`
in commit `679d6e15`. Existing attack calculations and current upstream fixes are retained.
Selection and aggregation adapt the original PR's `sampling.py`; the obsolete
experimental notebooks and attack-specific optimization changes are absent from
the final comparison with main.

Acceptance: original target first; distinct seeded shadow targets; original target
included in subsequent reference pools; target excluded from its own pool;
compatible model and signal cache reuse; correct per-target membership; unchanged
attack settings; explicit failure for incompatible reference requirements;
original shared handler state restored on success and failure; empirical summaries
and persisted per-run provenance; independent review of the final candidate.

Environment: Python 3.12.12, PyTorch 2.10.0, NumPy 1.26.4, pytest 9.0.2. CPU checks
use the existing `leakpro` conda environment, `LEAKPRO_DEVICE=cpu`, and `MPLBACKEND=Agg`.

Implementation: `Robustness` owns model selection, handler contexts, record-aligned
cached logits, and summaries. A fixed-pool hook reuses normal shadow-handler access;
distillation now reuses the existing singleton decorator so both handlers can be
restored after a rotation. No attack-specific adapter classes or scoring changes.

Observed checks before final review: 19 new tests passed (including a saved-input
LeakPro workflow and real LiRA, RMIA, BASE, and binary population attacks); 60
existing attack/model-handler tests passed. Ruff 0.11.2 passed the exact workflow
command: `ruff check leakpro --exclude examples,leakpro/tests,leakpro/webapp`.
An exploratory lint command that omitted the webapp exclusion reported existing
webapp findings; that directory is excluded by the repository's CI workflow.

The first independent review requested changes for fractional training-size
rounding: 15 training records out of 22 became 14. The new regression failed on
that candidate, then passed after preserving the integer count. A genuinely
incompatible fractional request is still rejected. The combined affected suite
reported `77 passed, 124 warnings in 8.44s`; raw output is in
`/tmp/robustness-rounding-final-tests.log`. An isolated mutation that retained a
shadow target in its own reference pool was rejected by the LiRA integration test.

Limits: CPU/synthetic-data verification; full CIFAR/GPU training and hosted full
CI are not claimed. The wrapper retains attacks' reference-data restrictions and
accepts one MIAResult per run. Percentile summaries are empirical, not calibrated
coverage claims. Publication to GitHub has not been requested.

A second independent review confirmed the training-count fix and passed an
additional cache test with permuted, noncontiguous records, shuffled training,
missing cached logits, and all five target rotations. It requested changes for
ROC interpolation without the origin when top scores tie. New mixed-tie and
all-tie tests failed before adding the origin to each curve and the shared grid;
the existing upper-TPR tie test also passes. The corrected combined suite reported
`79 passed, 124 warnings in 8.35s`; raw output is in
`/tmp/robustness-roc-final-tests.log`. CI Ruff passed again.

Status: final candidate checks and independent review pending.
