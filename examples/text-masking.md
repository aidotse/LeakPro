# Text Masking

Text masking (named entity recognition / anonymization) models are trained on sensitive text corpora to identify and redact personal information. Models trained on such data carry a risk of leaking the underlying sensitive records.

## Supported Attacks

| Attack | Supported | Example |
|--------|-----------|---------|
| Membership Inference (MIA) | ✅ | *private repo* |
| Model Inversion (MInvA) | ❌ | — |
| Gradient Inversion (GIA) | ✅ | [examples/gia/pii_inverting_masked_text/](gia/pii_inverting_masked_text/) |
| Synthetic Data Attacks (SynA) | ✅ | [examples/synthetic_data/syn_text_pii_scanner_example.ipynb](synthetic_data/syn_text_pii_scanner_example.ipynb) |

## Notes

<!-- Add any additional context, results, or references here -->
