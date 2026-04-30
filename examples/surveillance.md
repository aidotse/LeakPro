# Camera Surveillance

Machine learning models deployed in camera surveillance systems (e.g., person re-identification, face recognition) are trained on sensitive image data. These models are at risk of leaking biometric information about individuals in the training data.

## Supported Attacks

| Attack | Supported | Example |
|--------|-----------|---------|
| Membership Inference (MIA) | ✅ | [examples/mia/celebA_HQ/](mia/celebA_HQ/) |
| Model Inversion (MInvA) | ❌ | — |
| Gradient Inversion (GIA) | ✅ | [examples/gia/inverting_celebA_1_image/](gia/inverting_celebA_1_image/) |
| Synthetic Data Attacks (SynA) | ❌ | — |

## Notes

<!-- Add any additional context, results, or references here -->
