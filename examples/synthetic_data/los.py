import os
import sys
import pandas as pd

sys.path.append("../..")

from leakpro.synthetic_data_attacks.anonymeter.evaluators.singling_out_evaluator import SinglingOutEvaluator
from leakpro.synthetic_data_attacks.plots import plot_singling_out
from leakpro.synthetic_data_attacks.singling_out_utils import load_singling_out_results, singling_out_risk_evaluation

from typing import Optional
import pandas as pd
import numpy as np

class LeakySynthesizer:
    """
    A class to implement a leaky synthesizer for privacy analysis using pre-split datasets.
    
    The LeakySynthesizer takes pre-split training and control datasets, treating the control
    set as the release set. It creates a leaky dataset by replacing a specified fraction
    of records in the release set with records from the training set.
    
    When generating the leaky data, the leak fraction (fl) determines how many records
    from the release set are replaced with training records. When fl=0, the output
    remains identical to the release set (maximum privacy), when fl=1, all release
    records are replaced with training records (privacy violation).
    
    Attributes:
        training_set (pd.DataFrame): The pre-split training dataset
        release_set (pd.DataFrame): The pre-split control dataset used as release set
    
    Example:
        >>> synthesizer = LeakySynthesizer(training_data, control_data)
        >>> leaky_data = synthesizer.generate_leaky_data(0.3)
    """

    def __init__(
        self, 
        training_set: pd.DataFrame, 
        control_set: pd.DataFrame
    ) -> None:
        """
        Initialize the LeakySynthesizer with pre-split training and control datasets.
        
        Args:
            training_set: The pre-split training dataset
            control_set: The pre-split control dataset to be used as release set
        
        Raises:
            ValueError: If either dataset is empty or None
            ValueError: If datasets have different columns
        """
        if training_set is None or training_set.empty:
            raise ValueError("training_set cannot be None or empty")
        if control_set is None or control_set.empty:
            raise ValueError("control_set cannot be None or empty")
        if not training_set.columns.equals(control_set.columns):
            raise ValueError("training_set and control_set must have identical columns")

        self.training_set = training_set
        self.release_set = control_set.copy()
    
    def generate_leaky_data(
        self, 
        leak_fraction: float, 
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate leaky data by replacing release set records with training records.
        
        This method modifies a copy of the release set by replacing a specified fraction
        of its records with randomly selected records from the training set. The number
        of records remains constant, only the content is modified based on the leak fraction.
        
        Args:
            leak_fraction: Proportion of records to replace with training data (0 to 1)
            random_state: Seed for random number generation
        
        Returns:
            A DataFrame containing the modified release set with leaked training records
        
        Raises:
            ValueError: If leak_fraction is not between 0 and 1
        """
        if not 0 <= leak_fraction <= 1:
            raise ValueError("leak_fraction must be between 0 and 1")
            
        rng = np.random.RandomState(random_state)
        leaky_data = self.release_set.copy()
        
        if leak_fraction == 0:
            return leaky_data
            
        num_records_to_replace = int(len(leaky_data) * leak_fraction)
        
        indices_to_replace = rng.choice(
            len(leaky_data), 
            size=num_records_to_replace, 
            replace=False
        )
        
        replacement_records = self.training_set.sample(
            n=num_records_to_replace,
            random_state=rng,
            replace=True
        )
        
        leaky_data.iloc[indices_to_replace] = replacement_records.values
        
        return leaky_data
    
    def get_sets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieve the training and release (control) sets used by the synthesizer.
        
        This method provides access to the training set and control set (used as release set)
        for further analysis or validation of the synthetic data generation process.
        
        Returns:
            A tuple containing (training_set, release_set)
        """
        return self.training_set, self.release_set

    @property
    def training_size(self) -> int:
        """
        Get the number of records in the training set.
        
        Returns:
            The number of records in the training set
        """
        return len(self.training_set)

    @property
    def release_size(self) -> int:
        """
        Get the number of records in the release (control) set.
        
        Returns:
            The number of records in the release set
        """
        return len(self.release_set)
    

import pandas as pd
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