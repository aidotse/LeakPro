# Camera Surveillance

Machine learning models for person re-identification and face recognition enable important safety and operational capabilities. Responsible deployment of these models requires understanding what information they retain about training individuals. LeakPro provides a structured way to audit privacy exposure ahead of deployment, helping teams demonstrate due diligence and make evidence-based decisions about model sharing and access control.

## Supported Attacks

| Attack | Supported | Example |
|--------|-----------|---------|
| Membership Inference (MIA) | ✅ | [examples/mia/celebA_HQ/](../mia/celebA_HQ/) |
| Model Inversion (MInvA) | ❌ | — |
| Gradient Inversion (GIA) | ✅ | [examples/gia/inverting_celebA_1_image/](../gia/inverting_celebA_1_image/) |
| Synthetic Data Attacks (SynA) | ❌ | — |

## Notes

<!-- Add any additional context, results, or references here -->
