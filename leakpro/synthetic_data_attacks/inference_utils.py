"""Inference risk util functions."""
import itertools
import math
import random
from typing import List, Union

from leakpro.synthetic_data_attacks.anonymeter.evaluators.inference_evaluator import InferenceEvaluator
from leakpro.synthetic_data_attacks.utils import load_res_json_file, save_res_json_file
from pandas import DataFrame
from pydantic import BaseModel


class InferenceResults(BaseModel):
    """Class that holds results of inference risk calculations.

    Parameters
    ----------
    res_cols: List[str]
        List of results columns names
    res: List[List[Union[int,float]]]
        Array containing results from EvaluationResults plus length of auxiliar columns.
    aux_cols: List[str]]
        List containing results auxiliary columns.
    secrets: List[str]]
        List containing results secrets.

    """

    res_cols: List[str]
    res: List[List[Union[int,float]]]
    aux_cols: List[List[str]]
    secrets: List[str]

def get_inference_prefix(*, worst_case_flag: bool) -> str:
    """Auxiliary function to get inference prefix used in json results filename."""
    prefix = "inference_"
    if worst_case_flag:
        return prefix + "worst_case"
    return prefix + "base_case"

def get_n_random_combinations(*, cols: List[str], k: int, n: int) -> List[List[str]]:
    """Auxiliary function to get a sample of n random combinations of col in cols of size k, without repetition."""
    # Placeholder sample to return
    sample = []
    while len(sample)<n:
        # Get one sample
        sample_ = sorted(random.sample(cols, k))
        # Append to sample
        if sample_ not in sample:
            sample.append(sample_)
    return sample

def get_samples_length_subsets_cols(*, cols: List[str], n_samples: int) -> List[List[str]]:
    """Auxiliary function get_samples_length_subsets_cols.

    Function returns, for all lengths of subsets of columns, a sample of combinations
    with size minimum between nr of combinations and size of n_samples.
    """
    # Placeholder all_samples to return
    all_samples = []
    # Iterate thorugh lengths of columns
    for i in range(1, len(cols)+1):
        # Get nr of combinations
        nr_combs = math.comb(len(cols), i)
        if nr_combs>n_samples:
            #Nr combs greater than n_samples, get sample of combinations
            sample = get_n_random_combinations(cols=cols, k=i, n=n_samples)
        else:
            #Nr combs less than n_samples, get all combinations
            sample = [list(comb) for comb in itertools.combinations(cols, i)]
        # Add to all_samples
        all_samples += sample
    return all_samples

def get_progress(*, num: float, div: List) -> float:
    """Auxiliary function used in inference_risk_evaluation to report progress."""
    return round((num)/len(div)*100, 2)

def inference_risk_evaluation(
    ori: DataFrame,
    syn: DataFrame,
    worst_case_flag: bool = True,
    n_samples: int = 30,
    dataset: str = "test",
    verbose: bool = False,
    save_results_json: bool = False,
    **kwargs: dict
) -> InferenceResults:
    """Perform a full inference risk evaluation.

    Full evaluation can be a worst case scenario, where each column acts as secret
    against rest of columns as auxiliary information; or a base case scenario,
    where a sample of rest of columns is used as auxiliary information.

    Parameters
    ----------
    ori : pd.DataFrame
        Dataframe containing original data.
    syn : pd.DataFrame
        Dataframe containing synthetic data.
        It has to have the same columns as df_ori.
    worst_case_flag: bool
        If True, evaluation against worst case scenario performed.
    n_samples: int
        Nr of samples to take for each length of auxiliary columns and each secret.
        Used only in base case scenario evaluation.
    dataset: str
        Name of dataframes, used when verbose True and when saving results to json.
    verbose: bool, default is False
        If True, prints progress of evaluation.
    save_results_json: bool, default is False
        If True, saves results and combinations to json file.
    kwargs: dict
        Other keyword arguments for InferenceEvaluator.

    Returns
    -------
    InferenceResults
        InferenceResults with results.

    """
    if verbose:
        print(f"\nRunning inference risk evaluation each column against rest of columns for `{dataset}`") # noqa: T201
    #Placeholder variables
    res = []
    secrets = []
    aux_cols_all = []
    #Iterate through columns as secrets
    for i, secret in enumerate(ori.columns):
        #Get rest of columns as aux_cols
        aux_cols = [col for col in ori.columns if col != secret]
        #Get aux_cols_samples depending on worst_case_flag
        aux_cols_samples = [aux_cols] if worst_case_flag else get_samples_length_subsets_cols(cols=aux_cols, n_samples=n_samples)
        #Get prefix
        prefix = get_inference_prefix(worst_case_flag=worst_case_flag)
        for j, aux_col_sample in enumerate(aux_cols_samples):
            #Instantiate inference evaluator and evaluate
            evaluator = InferenceEvaluator(
                ori = ori,
                syn = syn,
                aux_cols = aux_col_sample,
                secret = secret,
                **kwargs
            )
            evaluator.evaluate()
            #Pack results and append
            res_ = evaluator.results.pack_results()
            res_.append(len(aux_col_sample)) #Adding len of aux_cols
            res.append(res_)
            #Append to secrets and aux_cols_all
            secrets.append(secret)
            aux_cols_all.append(aux_col_sample)
            if verbose and j>0 and j%100==0:
                sub_progress = get_progress(num=j+1, div=aux_cols_samples)/100
                progress = get_progress(num=i+sub_progress, div=ori.columns)
                print(f"inference_risk_evaluation progress: {progress}%") # noqa: T201
        if verbose:
            progress = get_progress(num=i+1, div=ori.columns)
            print(f"Finished {secret} on inference_risk_evaluation, progress: {progress}%") # noqa: T201
    #Set results column names
    res_cols = evaluator.results.res_cols.copy()
    res_cols.append("nr_aux_cols")
    #Instantiate InferenceResults
    inf_res = InferenceResults(
        res_cols = res_cols,
        res = res,
        aux_cols = aux_cols_all,
        secrets = secrets
    )
    #Save results to json
    if save_results_json:
        save_res_json_file(
            prefix = prefix,
            dataset = dataset,
            res = inf_res.model_dump()
        )
    return inf_res

def load_inference_results(*, dataset: str, worst_case_flag: bool = True) -> InferenceResults:
    """Function to load and return inference results from given dataset."""
    prefix = get_inference_prefix(worst_case_flag=worst_case_flag)
    return InferenceResults(**load_res_json_file(prefix=prefix, dataset=dataset))
