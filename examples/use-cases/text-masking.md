# Text Masking

Text masking and named entity recognition (NER) models play a key role in automating anonymization pipelines, helping organizations handle sensitive text responsibly. LeakPro allows teams to stress-test these models before deployment — quantifying residual privacy risks and validating that the anonymization process holds up against adversarial scrutiny, supporting compliance and building trust with data providers.

## Supported Attacks

| Attack | Supported | Example |
|--------|-----------|---------|
| Membership Inference (MIA) | ✅ | *private repo* |
| Model Inversion (MInvA) | ❌ | — |
| Gradient Inversion (GIA) | ✅ | [examples/gia/pii_inverting_masked_text/](../gia/pii_inverting_masked_text/) |
| Synthetic Data Attacks (SynA) | ✅ | [examples/synthetic_data/syn_text_pii_scanner_example.ipynb](../synthetic_data/syn_text_pii_scanner_example.ipynb) |

## Notes

<!-- Add any additional context, results, or references here -->
