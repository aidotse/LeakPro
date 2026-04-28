import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.insert(0, project_root)

from data_handler import IndividualizedInputHandler
from leakpro import LeakPro


def main(audit_config_path: str):
    # Prepare leakpro object
    leakpro = LeakPro(IndividualizedInputHandler, audit_config_path)

    # Run the audit 
    mia_results = leakpro.run_audit(create_pdf=False, use_optuna=False)
    
    # Print results
    for result in mia_results:
        print(result)


if __name__ == "__main__":
    audit_config_path = "audit.yaml"
    main(audit_config_path)