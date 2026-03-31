import os
import sys
import warnings
import yaml

# Suppres warnings, pytorch_tabular is very verbose
warnings.filterwarnings("ignore")
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.insert(0,project_root)

from leakpro import LeakPro
from examples.minv.mimic.mimic_plgmi_handler import Mimic_InputHandler
config_path = "audit.yaml"

# Initialize the LeakPro object
leakpro = LeakPro(Mimic_InputHandler, config_path)

# Run the audit
results = leakpro.run_audit()


######################################
import re
import plotly.io as pio

with open("audit.yaml", "r") as f:   # <-- change to your yaml path
    cfg = yaml.safe_load(f)

atk = cfg["audit"]["attack_list"][0]   # first (plgmi) attack block

def fmt_float(x: float) -> str:
    # compact float: keep up to 6 sig figs, strip trailing zeros/dot
    s = f"{float(x):.6g}"
    return s.rstrip("0").rstrip(".") if "." in s else s

def safe(name: str) -> str:
    return re.sub(r"[^-\w\.]+", "_", name)

outdir = (
    f"genlr_{fmt_float(atk['gen_lr'])}"
    f"__dislr_{fmt_float(atk['dis_lr'])}"
    f"__bs_{atk['batch_size']}"
    f"__n_dis_{atk['n_dis']}"
    f"__alpha_{fmt_float(atk['alpha'])}"
)

# outdir = os.path.join(cfg["audit"]["output_dir"], safe(folder_name))
os.makedirs(outdir, exist_ok=True)

######################################
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# -------- settings --------
os.makedirs(outdir, exist_ok=True)
# --------------------------

# Load the DataFrame from the pickle file
with open("GAN_losses.pkl", "rb") as f:
    df = pickle.load(f)

# 1) Inversion + CE
ax1 = df.plot(x="Epoch", y=["Inversion Loss", "Conditioning Loss (CE)"], title="GAN Losses")
fig1 = ax1.get_figure()
fig1.savefig(os.path.join(outdir, "gan_losses.png"), dpi=200, bbox_inches="tight")
plt.close(fig1)

# 2) Accuracy
ax2 = df.plot(x="Epoch", y=["Accuracy"], title="Accuracy")
fig2 = ax2.get_figure()
fig2.savefig(os.path.join(outdir, "accuracy.png"), dpi=200, bbox_inches="tight")
plt.close(fig2)

# 3) Inversion / Generator / Discriminator Loss
ax3 = df.plot(x="Epoch", y=["Inversion Loss", "Generator Loss", "Discriminator Loss"],
              title="Inversion-Generator-Dis Loss")
fig3 = ax3.get_figure()
fig3.savefig(os.path.join(outdir, "losses_combo.png"), dpi=200, bbox_inches="tight")
plt.close(fig3)

######################################
import os
import re
import plotly.io as pio

# Make sure kaleido is installed: pip install -U kaleido

# Directory to save plots (set this variable)
os.makedirs(outdir, exist_ok=True)

def safe(name: str) -> str:
    """Make a safe filename from a column name."""
    return re.sub(r'[^-\w\. ]', '_', name).strip().replace(' ', '_')

# Save numerical plots
for col, fig in results[0].numerical_plots.items():
    filename = f"{safe(col)}_num.png"
    fig.write_image(os.path.join(outdir, filename), scale=2)

# Save categorical plots
for col, fig in results[0].categorical_plots.items():
    filename = f"{safe(col)}_cat.png"
    fig.write_image(os.path.join(outdir, filename), scale=2)
######################################
# ================== COLLECT ARTIFACTS ==================
from pathlib import Path
import shutil
import glob

cwd = Path.cwd()
outpath = Path(outdir)  # already defined above

# 1) Copy the current config (do NOT remove original)
src_cfg = cwd / "audit.yaml"
if src_cfg.exists():
    shutil.copy2(src_cfg, outpath / "audit.yaml")

# 2) Move model/checkpoint + losses (CUT them)
for fname in ("ctgan.pkl", "GAN_losses.pkl"):
    src = cwd / fname
    if src.exists():
        dst = outpath / fname
        try:
            # if destination exists, replace it
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
        except Exception as e:
            print(f"[WARN] Could not move {fname}: {e}")
######################################
