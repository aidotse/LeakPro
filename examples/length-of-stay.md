# Length-of-Stay Prediction

Predicting hospital length-of-stay (LOS) for patients is a critical task in clinical decision-making and resource planning. Models trained on sensitive patient records (e.g., MIMIC-III) carry a significant risk of leaking private health information.

## Supported Attacks

| Attack | Supported | Example |
|--------|-----------|---------|
| Membership Inference (MIA) | ✅ | [examples/mia/LOS/](mia/LOS/) |
| Model Inversion (MInvA) | ❌ | — |
| Gradient Inversion (GIA) | ✅ | *private repo* |
| Synthetic Data Attacks (SynA) | ✅ | *private repo* |

## Dataset

MIMIC-III clinical database. Preprocessing scripts and setup instructions are provided in [examples/mia/LOS/](mia/LOS/).

## Notes

<!-- Add any additional context, results, or references here -->
