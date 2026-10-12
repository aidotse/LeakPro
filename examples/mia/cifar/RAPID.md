# RAPID

RAPID is the offline membership inference attack from He et al.,
[Is Difficulty Calibration All We Need? Towards More Practical Membership
Inference Attacks](https://arxiv.org/abs/2409.00426), ACM CCS 2024.
The [authors' implementation](https://github.com/T0hsakar1n/RAPID) is the
reference for the loss convention and MLP architecture.
LeakPro implements the algorithm independently; it does not copy the upstream code.

From this directory, prepare the dataset and target model with the existing
`cifar_main.ipynb` workflow, then use `audit_rapid.yaml` for the audit:

```python
from leakpro import LeakPro
from cifar_data_handler import CifarDataHandler
from cifar_model_handler import CifarModelHandler

audit = LeakPro(CifarDataHandler, "audit_rapid.yaml", model_handler=CifarModelHandler)
results = audit.run_audit(create_pdf=False, use_optuna=False)
```

The population must contain auxiliary samples outside both the target's training
and test indices. The example's model paths must match the target you trained.
Install LeakPro with its MIA dependencies (`pip install -e '.[mia]'` from the
repository root). This attack accepts multiclass logits and single-logit binary
classifiers, using unweighted cross entropy and binary cross entropy respectively.

LeakPro splits the auxiliary pool into disjoint shadow and reference pools.
One shadow model is trained on `training_data_fraction` of the shadow pool;
the full shadow pool supplies known IN/OUT labels to fit the scorer. Each of the
`num_reference_models` trains on an independently sampled fraction of the
reference pool. Both pools must leave at least one training and one held-out point.
By default each pool gets half the auxiliary samples and each model trains on half
its pool. For paper-scale comparisons, provide pools large enough to match the
target's training size and use the target's training recipe.

For each model, losses are averaged across `query_num` queries. The scorer's two
features are `[original loss, original loss - mean reference loss]`, matching the
authors' positive-loss convention. The paper uses negative-loss scores, so both
feature signs are reversed relative to its Equations 3 and 7. The scorer uses
linear layers of widths 2, 512, 256, 128, and 1, ReLU hidden activations, dropout
0.2 after its first two hidden layers, Xavier initialization, BCE-with-logits loss,
and Adam at learning rate 0.0002 without a scheduler, matching `mia_attack.py`.

`augment_strength` is optional and requires a dataset that implements that setting,
such as `CifarDataHandler.UserDataset`. `easy` enables CIFAR crop/flip augmentation.
When omitted, queries use the dataset's existing transforms. Repeating a
deterministic dataset yields the same loss and adds no augmentation benefit.
Query loaders must preserve index order and yield every requested sample; use
`drop_last=False` and `in_order=True` when using multiple workers.
Normalization parameters are inherited from the population. Loss computation
uses the shared `functional.loss` signal in its stable logit-space mode.

The random seed inherits `audit.random_seed` unless overridden in the RAPID entry.
Auxiliary model training, persistence, and reuse use `ShadowModelHandler`.
Cached models are eligible only if their training indices belong to the requested
pool. The scorer is fitted again on each preparation; query and scorer RNGs are
reset after auxiliary model loading so cache hits do not change their random stream.
Results and reports use the standard `MIAResult` and audit output paths.

Set `use_attack` to `lira` or `rmia` to train RAPID on
their model inputs or scores. `use_attack_output: scores` preserves the default
`[original loss, selected attack score]` combination. For example:

```yaml
- attack: rapid
  use_attack: lira
  use_attack_config:
    var_calculation: fixed
    signal: rescaled_logits
  num_reference_models: 4
```

In scores mode, the existing LiRA/RMIA scoring implementations run first on the
shadow model's data to supply classifier training features, then on the target
audit data to supply prediction features. Target membership labels serve only
evaluation and row alignment. The intermediate
`MIAResult` objects are available as `attack.source_results["shadow"]` and
`attack.source_results["target"]`; scores are realigned to the queried sample order
before entering the classifier. These source results are not produced in logits mode.

Set `use_attack_output: logits` to calculate RAPID's usual
`[original loss, original loss - mean OUT-reference loss]` features directly from
the aligned logits queried for the source attack. RAPID averages these loss
features over `query_num` queries. It does not need to run the source's scoring
method or consume a source result. This keeps the existing two-feature classifier.
For example, to reuse online RMIA inputs:

```yaml
- attack: rapid
  use_attack: rmia
  use_attack_output: logits
  use_attack_config:
    online: true
  num_reference_models: 4
```

`use_attack_config` accepts the selected attack's scoring options, such as LiRA's
`var_calculation` and `signal`, or RMIA's `gamma`, `temperature`, `offline_a`, and
`attack_data_fraction`. RAPID's `num_reference_models` and `training_data_fraction`
control the shared references. Scoring options affect scores mode; logits mode
uses RAPID's unweighted classification loss. RMIA samples its comparison population
from the reference pool. In scores mode, original losses retain RAPID's query
averaging and source scores follow the selected attack's own logit-based calculation.
`use_attack` defaults to `None`, so leaving the field out runs ordinary RAPID.
YAML `null`, `None`, and `RAPID` select the same behavior. Selector names are
case-insensitive and stored canonically, so equivalent fallbacks share identities.

Both modes accept `use_attack_config.online: true`. Offline references train on
the disjoint auxiliary reference pool. Online references use LeakPro's balanced
sampling over the auxiliary and audit data, with IN/OUT membership read from model
training metadata. Every point must have both IN and OUT references; invalid
coverage raises an error rather than dropping points. Logits mode calibrates each
point using only its OUT references. Online reference training assumes access to
the queried records and their class labels, without access to their membership in
the target model's training set.

The paper's [Section 8](https://arxiv.org/html/2409.00426v2#S8) describes this
combination with offline LiRA. The RMIA combination is an implementation extension
using the same classifier interface; no reproduced performance is claimed.

The scorer trains only on the auxiliary shadow model's membership for a fixed
`mia_classifier_epochs` (default 150). Unlike the
reference evaluation script, this implementation does not select its best epoch
using target membership labels. Those labels are used only to evaluate the final
membership probabilities. The paper's additional LiRA-like IN/OUT RAPID variant
is distinct from reusing an online source and is outside this implementation.
Small tests validate the
algorithm and integration, not the paper's reported attack performance.
