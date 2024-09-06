"""Linkability risk util functions."""
import itertools
import math
from typing import List, Union

import numpy as np
from leakpro.synthetic_data_attacks.anonymeter.evaluators.linkability_evaluator import LinkabilityEvaluator
from leakpro.synthetic_data_attacks.utils import load_res_json_file, save_res_json_file
from pandas import DataFrame
from pydantic import BaseModel


def aux_assert_input_values_get_combs_2_buckets(*, cols: List, buck1_nr: int, buck2_nr: int) -> None:
    """Aux function to assert input values of functions get_nr_all_combs_2_buckets and get_all_combs_2_buckets."""
    assert buck1_nr>0
    assert buck2_nr>0
    assert buck1_nr >= buck2_nr
    assert len(cols) >= 2
    assert buck1_nr+buck2_nr<=len(cols)

def get_nr_all_combs_2_buckets(*, cols: List[str], buck1_nr: int, buck2_nr: int) -> int:
    """Function to get number of all combinations from a list of columns divided in 2 buckets.

    Buckets will have length buck1_nr/buck2_nr respectively
    Note: buck1_nr must be great or equal than buck2_nr
    """
    #Assert input values
    aux_assert_input_values_get_combs_2_buckets(cols=cols, buck1_nr=buck1_nr, buck2_nr=buck2_nr)
    #Set dividend
    dividend = 1
    if buck1_nr == buck2_nr:
        dividend = 2
    #Calculate number of combinations in respective bucket and return
    nr_combs_buck1 = math.comb(len(cols), buck1_nr)
    nr_combs_buck2 = math.comb(len(cols)-buck1_nr, buck2_nr)
    return int(nr_combs_buck1*nr_combs_buck2/dividend)

def get_all_combs_2_buckets(*,
    cols: List[str],
    buck1_nr: int,
    buck2_nr: int
) -> List[List[List[str]]]:
    """Function to get all combinations from a list of columns divided in 2 buckets.

    Buckets will have length buck1_nr/buck2_nr respectively
    Note: buck1_nr must be great or equal than buck2_nr
    """
    #Assert input values
    aux_assert_input_values_get_combs_2_buckets(cols=cols, buck1_nr=buck1_nr, buck2_nr=buck2_nr)
    #Placeholder all_combs to return
    all_combs = []
    #Iterate to get combinations bucket1
    for comb_buck1 in itertools.combinations(cols, buck1_nr):
        #Get remaining cols
        remain_cols = [col for col in cols if col not in comb_buck1]
        #Iterate to get combinations bucket2
        for comb_buck2 in itertools.combinations(remain_cols, buck2_nr):
            #Sort combinations
            comb_buck1 = sorted(comb_buck1) # noqa: PLW2901
            comb_buck2 = sorted(comb_buck2) # noqa: PLW2901
            #Assemble combination
            comb = [comb_buck1, comb_buck2]
            #If comb not in all_combs, append
            if comb not in all_combs:
                if buck1_nr!=buck2_nr:
                    all_combs.append(comb)
                else:
                    #Check swapped combs not in all_combs
                    comb_ = [comb_buck2, comb_buck1]
                    if comb_ not in all_combs:
                        all_combs.append(comb)
    return all_combs

def get_n_sample_combinations(*,
    cols: List[str],
    buck1_nr_arr: np.ndarray,
    buck2_nr_arr: np.ndarray
) -> List[List[List[str]]]:
    """Function to get a sample (non-repeated) of combinations from a list of columns divided in 2 buckets (np.arrays).

    Buckets arrays will have length of buck1_nr/buck2_nr respectively, in each entry
    Combinations sample length (n) returned will be length of arrays
    Note: buck1_nr must be great or equal than buck2_nr
    """
    #Assert bucket arrays shapes
    assert len(buck1_nr_arr.shape) == 1
    assert buck1_nr_arr.shape[0] > 0
    assert buck1_nr_arr.shape == buck2_nr_arr.shape
    #Set rng
    rng = np.random.default_rng()
    #Placeholder for combinations sample to return
    combs_sample = []
    #np.array from cols
    cols = np.array(cols)
    #Iterate through entries in buckX_nr_arr to append to combs_sample
    for buck1_nr, buck2_nr in zip(buck1_nr_arr, buck2_nr_arr):
        #Assert input values
        aux_assert_input_values_get_combs_2_buckets(cols=cols, buck1_nr=buck1_nr, buck2_nr=buck2_nr)
        #While comb not appended, get new comb and try to append
        comb_appended = False
        while not comb_appended:
            #Get combination of bucket 1 and sort
            comb_buck1 = rng.choice(cols, size=buck1_nr, replace=False)
            comb_buck1.sort()
            #Calculate remaining columns
            remain_cols = np.setdiff1d(cols, comb_buck1)
            #Get combination of bucket2 and sort
            comb_buck2 = rng.choice(remain_cols, size=buck2_nr, replace=False)
            comb_buck2.sort()
            #Assemble combination sample
            comb_sample = [comb_buck1.tolist(), comb_buck2.tolist()]
            #Check if comb_sample not in combs_sample to append
            if comb_sample not in combs_sample:
                if buck1_nr != buck2_nr:
                    combs_sample.append(comb_sample)
                    comb_appended = True
                else:
                    #Check swapped comb_sample not in combs_sample
                    comb_sample_ = [comb_sample[1], comb_sample[0]]
                    if comb_sample_ not in combs_sample:
                        combs_sample.append(comb_sample)
                        comb_appended = True
                    else:
                        #Following print is to let user know possibility of never ending looking for combinations
                        print("comb_sample was in combs_sample in get_n_random_combinations_linkability!") # noqa: T201
    return combs_sample

def linkability_combinations_samples(*, cols: List[str], n_samples: int = 300) -> List[List[List[str]]]:
    """Function to get samples of combinations from a list of columns, for linkability evaluator.

    Columns will be randomly placed in 2 buckets, and for the range of the sum of the length of the buckets
    (starting from 2 to len(cols)), a sample of n_samples of combinations will be returned.
    """
    #Set cols
    cols = sorted(cols)
    #Set number of cols
    n_cols = len(cols)
    #Placeholder for combinations to return
    combs = []
    #Iterate through range of the number of columns (sum of length of buckets) to add to combs
    for n_col in range(2, n_cols+1):
        #Calculate buck1_nr_low and buck1_nr_high
        buck1_nr_low = math.ceil(n_col/2)
        buck1_nr_high = n_col - 1
        #Calculate buck1_nr_arr
        if buck1_nr_high-buck1_nr_low == 0:
            buck1_nr_arr = buck1_nr_low * np.ones((n_samples,), dtype="int")
        else:
            buck1_nr_arr = np.random.choice(np.arange(buck1_nr_low, buck1_nr_high+1), size=(n_samples,))
        #Calculate bucket 1 unique values and counts and iterate through them
        u_buck1_vals, u_buck1_counts = np.unique(buck1_nr_arr, return_counts=True)
        for u_buck1_val, u_buck1_count in zip(u_buck1_vals, u_buck1_counts):
            #Get possible number of combinations
            nr_possible_combs = get_nr_all_combs_2_buckets(
                cols = cols,
                buck1_nr = u_buck1_val,
                buck2_nr = n_col-u_buck1_val
            )
            if nr_possible_combs<=u_buck1_count:
                #Append all combinations
                combs += get_all_combs_2_buckets(
                    cols = cols,
                    buck1_nr = u_buck1_val,
                    buck2_nr = n_col-u_buck1_val
                )
            else:
                #Get index of u_buck1_val
                idx = np.where(np.equal(buck1_nr_arr, u_buck1_val))
                #Append sample of combinations
                combs += get_n_sample_combinations(
                    cols = cols,
                    buck1_nr_arr = buck1_nr_arr[idx],
                    buck2_nr_arr = n_col-buck1_nr_arr[idx]
                )
    return combs

class LinkabilityResults(BaseModel):
    """Class that holds results of linkability_risk_evaluation.

    Parameters
    ----------
    res_cols: List[str]
        List of results columns names
    res: List[List[Union[int,float]]]
        Array containing results from EvaluationResults plus length of auxiliar columns.
    aux_cols: List[str]]
        List containing results auxiliary columns.

    """

    res_cols: List[str]
    res: List[List[Union[int,float]]]
    aux_cols: List[List[List[str]]]

def linkability_risk_evaluation(
    ori: DataFrame,
    syn: DataFrame,
    n_samples: int = 300,
    dataset: str = "test",
    verbose: bool = False,
    save_results_json: bool = False,
    **kwargs: dict
) -> LinkabilityResults:
    """Perform a full linkability risk evaluation.

    For full evaluation, n_samples of combinations will be used as auxiliary columns
    and evaluated with length of sample in range of (2,len(columns)).

    Parameters
    ----------
    ori : pd.DataFrame
        Dataframe containing original data.
    syn : pd.DataFrame
        Dataframe containing synthetic data.
        It has to have the same columns as df_ori.
    n_samples: int, default is 300
        Number of samples for each sum of auxiliary columns lengths
    dataset: str
        Name of dataframes, used when verbose True and when saving results to json.
    verbose: bool, default is False
        If True, prints progress of evaluation.
    save_results_json: bool, default is False
        If True, saves results and combinations to json file.
    kwargs: dict
        Other keyword arguments for LinkabilityEvaluator.

    Returns
    -------
    LinkabilityResults
        LinkabilityResults with results.

    """
    if verbose:
        print(f"\nRunning linkability risk evaluation for `{dataset}` with len(columns)={len(ori.columns)}") # noqa: T201
    #Get combinations
    combs = linkability_combinations_samples(cols=ori.columns, n_samples=n_samples)
    #Placeholder for results
    res = []
    #Iterate throught combinations and append to results
    for i, comb in enumerate(combs):
        if verbose and i%100==0:
            print("Evaluating linkability combination (and total)", i, len(combs)) # noqa: T201
        #Instantiate linkability evaluator and evaluate
        evaluator = LinkabilityEvaluator(
            ori = ori,
            syn = syn,
            aux_cols = comb,
            **kwargs
        )
        evaluator.evaluate()
        #Pack results and append
        res_ = evaluator.results.pack_results()
        res_.append(len(comb[0]+comb[1])) #Adding len of aux_cols
        res.append(res_)
    #Set results column names
    res_cols = evaluator.results.res_cols.copy()
    res_cols.append("nr_aux_cols")
    #Instantiate LinkabilityResults
    link_full_res = LinkabilityResults(
        res_cols = res_cols,
        res = res,
        aux_cols = combs
    )
    #Save results to json
    if save_results_json:
        save_res_json_file(
            prefix = "linkability",
            dataset = dataset,
            res = link_full_res.model_dump()
        )
    return link_full_res

def load_linkability_results(*, dataset: str) -> LinkabilityResults:
    """Function to load and return linkability results from given dataset."""
    return LinkabilityResults(**load_res_json_file(prefix="linkability", dataset=dataset))
