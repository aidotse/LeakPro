# Inverting Gradients on Masked PII Text (Longformer NER)

This example applies the **Inverting Gradients** attack (Geiping et al., 2020) to a **text/NER model**, showing that gradient leakage is a risk beyond image classifiers: it can also reconstruct **personally identifiable information (PII)** from the gradients of a named-entity-recognition model trained on text documents.

## Goal

The target model is trained to detect masked PII spans (label `"MASK"`) in text using a Longformer backbone (`OneHotBERT`, built on `allenai/longformer-base-4096`). The attack reconstructs the client's private document from the gradient it would send during federated training, demonstrating that even text data behind a masking/NER task is not safe from gradient inversion.

## Setup

- **Model:** `OneHotBERT`, a Longformer-based token classifier ([longformer_model.py](longformer_model.py)), with a single `"MASK"` label.
- **Data:** 1 PII document, loaded via [data_/pii_data.py](data_/pii_data.py), tokenized with `LongformerTokenizerFast`.
- **Attack:** `InvertingGradients` (`leakpro.attacks.gia_attacks.invertinggradients_text` — the text-specific variant of the attack, using `GiaNERExtension` to adapt gradient matching to token sequences), tuned via Optuna over 5 trial documents.
- **Training:** [train.py](train.py) supplies the meta-train function used to simulate the client's local training step.

## How to Run

```bash
python main.py
```

## Credits

- Geiping, J., et al. (2020). [Inverting gradients - How easy is it to break privacy in federated learning?](https://arxiv.org/abs/2003.14053) NeurIPS.
