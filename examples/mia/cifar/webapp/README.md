# LeakPro Webapp: CIFAR-10 Example

Files for running a CIFAR-10 membership inference audit through the LeakPro webapp.

## Setup

```bash
pip install -e ".[webapp]" opacus
cd leakpro/webapp/frontend && npm install
```

Build the population file first by running the data cells of `../cifar_main.ipynb`. They write
`examples/mia/cifar/data/cifar10.pkl`.

## Running

**Backend** (from the repo root):
```bash
uvicorn leakpro.webapp.backend.main:app --host 127.0.0.1 --port 8000
```

The backend prints an API token at startup; paste it into the UI when prompted, or set
`LEAKPRO_WEBAPP_TOKEN` to pin your own. Keep the bind address on `127.0.0.1`: the backend loads
user model files and runs user-supplied Python by design. See
[`leakpro/webapp/SECURITY.md`](../../../../leakpro/webapp/SECURITY.md).

**Frontend** (from `leakpro/webapp/frontend/`):
```bash
npm run dev
```

Open the URL npm prints (usually `http://localhost:5173`).

## Files

| File | Step | Required |
|------|------|----------|
| `cifar_data_handler.py` | Step 1: Dataset | **Mandatory.** Tells the webapp how to load `cifar10.pkl` |
| `cifar_model_architecture.py` | Step 2: Architecture | Optional. WideResNet-28-2, the model `cifar_main.ipynb` audits; skip to use the default ResNet-18 |
| `cifar_model_handler.py` | Step 2: Architecture | Optional. Training loop with cosine LR and DP-SGD support; skip to use the built-in default |

Use these files rather than `../target_model_class.py` and `../cifar_model_handler.py`. The webapp
trains the first `nn.Module` subclass it finds in the architecture file, and `target_model_class.py`
imports torchvision's `BasicBlock` before defining its models, so the webapp would try to train a
single residual block. The notebook's training handler also holds out 10% of the training set for
early stopping. Those records are labelled members in the metadata but never trained on, which
mislabels membership in the audit.

## Runtime

A full audit on the 60,000-image population trains every shadow model on as much data as the target.
For a quick run, keep the number of shadow models and epochs low in the attack step.
