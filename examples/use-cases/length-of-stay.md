# Length-of-Stay Prediction

Predicting hospital length-of-stay (LOS) is a high-value task in clinical decision-making and resource planning. Deploying ML models in healthcare settings requires a clear understanding of privacy exposure before sharing or releasing a model. LeakPro enables practitioners to proactively quantify privacy risks and take informed steps — such as applying differential privacy or limiting what the model exposes — so that models can be shared with confidence.

## Supported Attacks

| Attack | Supported | Example |
|--------|-----------|---------|
| Membership Inference (MIA) | ✅ | [examples/mia/LOS/](../mia/LOS/) |
| Model Inversion (MInvA) | ❌ | — |
| Gradient Inversion (GIA) | ✅ | *private repo* |
| Synthetic Data Attacks (SynA) | ✅ | *private repo* |

## Dataset

MIMIC-III clinical database. Preprocessing scripts and setup instructions are provided in [examples/mia/LOS/](../mia/LOS/).

## Notes

<!-- Add any additional context, results, or references here -->
