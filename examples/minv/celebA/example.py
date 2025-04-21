import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

from leakpro import LeakPro
from examples.minv.celebA.celebA_plgmi_handler import CelebA_InputHandler
config_path = "audit.yaml"

# Initialize the LeakPro object
leakpro = LeakPro(CelebA_InputHandler, config_path)

# Run the audit
results = leakpro.run_audit(return_results=True)