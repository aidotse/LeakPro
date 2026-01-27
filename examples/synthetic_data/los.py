import os
import sys
import pandas as pd

sys.path.append("../..")

from leakpro.synthetic_data_attacks.anonymeter.evaluators.singling_out_evaluator import SinglingOutEvaluator
from leakpro.synthetic_data_attacks.leaky_synthesizer import LeakySynthesizer
from leakpro.synthetic_data_attacks.plots import plot_singling_out
from leakpro.synthetic_data_attacks.singling_out_utils import load_singling_out_results, singling_out_risk_evaluation

ori = pd.read_csv("/nfs/home/mirmus/new_projects/LeakPro/examples/synthetic_data/datasets/static_data.csv")
syn = pd.read_csv("/nfs/home/mirmus/new_projects/LeakPro/examples/synthetic_data/datasets/dpctgan.csv")

common_columns = ori.columns.intersection(syn.columns)
ori = ori[common_columns]

syn.drop("Unnamed: 0", axis=1, inplace=True)

def clean_string(value):
    if isinstance(value, str):
        value = value.replace("\\", "\\\\")  # Escape backslashes
        value = value.replace("'", "")      # Remove single quotes
        value = value.replace('"', "")      # Remove double quotes
        value = value.strip()               # Trim leading/trailing spaces
    return value

for col in ori.select_dtypes(include=['object']).columns:
    ori[col] = ori[col].apply(clean_string)

for col in syn.select_dtypes(include=['object']).columns:
    syn[col] = syn[col].apply(clean_string)

from sklearn.model_selection import train_test_split
ori, control = train_test_split(ori, test_size=0.5, random_state=42)

synthesizer = LeakySynthesizer(training_set=ori, control_set=control)
synthetic_data = synthesizer.generate_leaky_data(
        leak_fraction=0.8,
        random_state=13
    )

sin_out_res = singling_out_risk_evaluation(
    dataset = "los_syn_10_diverse_modified",
    ori = ori,
    syn = syn,
    n_cols=2, 
    n_attacks = 10,
    verbose = True,
    save_results_json = True,
    max_attempts = 200_000,
    max_per_combo= 50,
    sample_size_per_combo= 10,
    max_rounds_no_progress= 100,
    use_medians = False,
    use_tree=True,
    tree_params={
        'min_samples_leaf': 1,
        'max_depth': None,           # let the tree grow
        'random_state': 42
    }
)