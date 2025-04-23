import os
import sys

"""
This script runs the LeakPro audit on the CelebA dataset using the LeakPro library.
It requires that:
- target_model is trained and saved in the specified directory
- the audit configuration file is present in the same directory as this script
- the CelebA private and public .pkl dataset is available in the specified directory
"""

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

from leakpro import LeakPro
from examples.minv.celebA.celebA_plgmi_handler import CelebA_InputHandler
config_path = "audit.yaml"

# Initialize the LeakPro object
leakpro = LeakPro(CelebA_InputHandler, config_path)

# Run the audit
results = leakpro.run_audit(return_results=True)