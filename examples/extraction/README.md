# Diffusion extraction example

The [`cifar10`](cifar10) directory follows the same layout as the MIA and model-inversion examples. Use
[`train_config.yaml`](cifar10/train_config.yaml) to select the target data, model, and training profile. The selected
[`audit.yaml`](cifar10/audit.yaml) or [`audit_demonstration.yaml`](cifar10/audit_demonstration.yaml) owns every Carlini
and SIDE parameter, including generation budgets. Then run
[`main.ipynb`](cifar10/main.ipynb). The notebook downloads CIFAR-10, trains or reloads an unconditional DDPM, runs both
attacks, visualizes nearest training references, and saves the resolved config and run manifest under the ignored
`target/` and `leakpro_output/` directories.

The default smoke profile runs the complete workflow with reduced budgets. It is not evidence of paper-scale
extraction performance.

The notebook needs torchvision in addition to the extraction dependencies:

```bash
pip install -e '.[extraction]' torchvision
```

The toy provider exercises Carlini and SIDE through LeakPro without downloads or a GPU:

```python
from leakpro import LeakPro
from toy_extraction_handler import ToyExtractionHandler

results = LeakPro(ToyExtractionHandler, "toy_audit.yaml").run_audit()
```

Run the snippet from this directory. Outputs are written to `/tmp/leakpro_extraction_toy`.

For a real model, replace `ToyExtractionHandler.get_diffusion_adapter()` with either:

- `OpenAIDiffusionAdapter` for an Improved Diffusion or Guided Diffusion compatible model and diffusion object; plain
  Improved Diffusion supports Carlini sampling, while SIDE requires a Guided Diffusion-style loop accepting `cond_fn`; or
- `CallableDiffusionAdapter` backed by target-specific sampling, forward-noising, timestep mapping, and guided reverse
  sampling callables.

Carlini conditional mode needs conditions. Carlini unconditional mode needs authorized training references. SIDE
also needs a frozen feature extractor and white-box diffusion operations. Record the exact checkpoint, scheduler,
preprocessing, reference-set hash, dependency versions, and paper-omitted thresholds for any real audit.

SIDE moves classifier inputs to `compute_device` and the classifier parameter dtype, then returns guidance gradients
to the reverse sampler's original device and dtype. The adapter's forward-noising operation must accept tensors on
`compute_device`; cross-device guidance transfers can be expensive and should be recorded in paper-scale runs.

Conditions must have a stable canonical form: strings, finite scalars, bytes, nested lists or tuples, string-keyed
dictionaries, Pydantic models, NumPy arrays, and torch tensors are supported. Custom objects are rejected; encode them
into one of these forms so audit identities do not depend on process-specific object representations.

Set `target.fingerprint` to a stable identifier for that checkpoint/provider bundle. Results with the same audit
identity are not overwritten unless `overwrite_results: true` is explicit; the toy config enables it only so the
example is rerunnable. Raw conditions are represented in saved metadata by fingerprints, not prompt text.

The toy result validates control flow only; it is not evidence of paper-scale extraction performance.

Extraction attack instances are one-shot. Create a new `LeakPro` instance for a repeated audit, including an explicit
overwrite run, so classifiers, counters, and execution traces cannot carry over between results.
