# Security policy

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository:
<https://github.com/aidotse/LeakPro/security/advisories/new>. Reports go only to
the maintainers. Please allow time to patch before public disclosure.

## Trust model

LeakPro is a library and a local tool run by the **operator auditing their own
models and data**. Its security boundary is the operator's machine and account,
not the inputs themselves. Concretely:

- **Model and dataset files are trusted.** The target model, its metadata and
  pickled or joblib datasets selected in `audit.yaml` or passed to a handler are
  deserialised with full `pickle`/`torch.load`/`joblib` semantics, because a
  PyTorch checkpoint or a custom dataset class cannot be reconstructed any other
  way. A file from a third party (model zoo, Kaggle, a vendor) can execute code
  when loaded. Convert such files to a passive format yourself (`.npz`,
  Parquet, CSV, a plain `state_dict` loaded with `weights_only=True`) before
  pointing LeakPro at them.
- **Handler code is trusted.** The input handler and model architecture are the
  operator's own Python and run as the operator.
- **The audit output directory is private to the operator.** Attacks cache
  intermediate results under `audit.output_dir` and reuse them on the next run:
  shadow and distillation checkpoints, attack classifiers, feature arrays and
  the TS2Vec representation model. Those caches are written and read only in
  pickle-free formats (`.npz` with `allow_pickle=False`, `torch.load` with
  `weights_only=True`), so a tampered cache file fails to load rather than
  executing. This holds only on `torch>=2.6`, which the package requires; on
  older releases `weights_only=True` could be bypassed (CVE-2025-32434).
  Shadow and distillation training metadata is stored as JSON and validated
  against its schema on read. Metadata files from older versions
  (`attack_objects/shadow_model/metadata_<n>.pkl`) are ignored, never loaded,
  and their models are retrained. Keep the output directory on storage that
  only the operator can write.
- **Example data preparation scripts run with the operator's credentials.** The
  MIMIC-III build recipes and extraction scripts under `examples/mia/LOS/` are
  vendored third-party code that take a database password on the command line
  and in `make` variables. Run them in a private shell and do not capture their
  output into shared logs or CI.

For the web application, which adds a network boundary, see
[`leakpro/webapp/SECURITY.md`](leakpro/webapp/SECURITY.md).
