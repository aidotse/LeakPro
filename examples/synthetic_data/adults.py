import os
import sys

import pandas as pd

sys.path.append("../..")

from leakpro.synthetic_data_attacks.anonymeter.evaluators.singling_out_evaluator import SinglingOutEvaluator
from leakpro.synthetic_data_attacks.leaky_synthesizer import LeakySynthesizer
from leakpro.synthetic_data_attacks.plots import plot_singling_out
from leakpro.synthetic_data_attacks.singling_out_utils import load_singling_out_results, singling_out_risk_evaluation

bucket_url = "https://storage.googleapis.com/statice-public/anonymeter-datasets/"

ori = pd.read_csv(os.path.join(bucket_url, "adults_train.csv"))
syn = pd.read_csv(os.path.join(bucket_url, "adults_syn_ctgan.csv"))
control = pd.read_csv(os.path.join(bucket_url, "adults_control.csv"))

synthesizer = LeakySynthesizer(training_set=ori, control_set=syn)
synthetic_data = synthesizer.generate_leaky_data(
        leak_fraction=0.5,
        random_state=13
    )    

sin_out_res = singling_out_risk_evaluation(
    dataset = "adults_syn_leaked_diverse_modified_report_20k",
    ori = ori,
    syn = syn,
    n_attacks = 20_000,
    verbose = True,
    save_results_json = True,
    max_attempts = 10_000,
    max_per_combo= 5,
    sample_size_per_combo= 2,
    max_rounds_no_progress= 10,
    use_medians = False
)