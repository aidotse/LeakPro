# LeakPro Webapp — CelebA-HQ Example

Example files for running a CelebA-HQ membership inference audit through the LeakPro webapp.

## Setup

Install extra dependencies into your LeakPro conda environment:

```bash
conda activate <your_env>
pip install fastapi "uvicorn[standard]" python-multipart opacus
cd leakpro/webapp/frontend && npm install
```

## Running

**Backend** (from repo root):
```bash
uvicorn leakpro.webapp.backend.main:app --reload --port 8000
```

**Frontend** (from `leakpro/webapp/frontend/`):
```bash
npm run dev
# To use a different backend port:
BACKEND_PORT=8001 npm run dev
```

Open the URL shown by npm in your browser (usually `http://localhost:5173`).

## Files

| File | Step | Required |
|------|------|----------|
| `celebA_data_handler.py` | Step 1 — Dataset | **Mandatory** — tells LeakPro how to load your `.pkl` population file |
| `model_architecture.py` | Step 2 — Architecture | Optional — pretrained ResNet-18 for CelebA; skip to use the default (random init ResNet-18) |
| `celebA_model_handler.py` | Step 2 — Architecture | Optional — custom train/eval loop with DP-SGD support; skip to use the built-in default |

## Data

The CelebA-HQ population `.pkl` must be pre-built using the notebook `main_celebA_hq.ipynb` before running the webapp.
