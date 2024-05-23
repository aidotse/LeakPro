"""Inference risk util functions."""
from typing import List, Union

from pandas import DataFrame
from pydantic import BaseModel

from leakpro.synthetic_data_attacks.anonymeter.evaluators.inference_evaluator import InferenceEvaluator
from leakpro.synthetic_data_attacks.utils import load_res_json_file, save_res_json_file


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

def inference_risk_evaluation_each_against_rest_columns(
    ori: DataFrame,
    syn: DataFrame,
    dataset: str = "test",
    verbose: bool = False,
    save_results_json: bool = False,
    **kwargs: dict
) -> InferenceResults:
    """Perform an inference risk evaluation for each column as secret against rest of columns as auxiliary information.

    Parameters
    ----------
    ori : pd.DataFrame
        Dataframe containing original data.
    syn : pd.DataFrame
        Dataframe containing synthetic data.
        It has to have the same columns as df_ori.
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
    InferenceResults
        InferenceResults with results.

    """
    if verbose:
        print(f"\nRunning inference_risk_evaluation_each_against_rest_columns for `{dataset}`") # noqa: T201
    #Placeholder variables
    res = []
    secrets = []
    aux_cols_all = []
    #Iterate through columns as secrets
    for i, secret in enumerate(ori.columns):
        #Get rest of columns as aux_cols
        aux_cols = [col for col in ori.columns if col != secret]
        #Instantiate inference evaluator and evaluate
        evaluator = InferenceEvaluator(
            ori = ori,
            syn = syn,
            aux_cols = aux_cols,
            secret = secret,
            **kwargs
        )
        evaluator.evaluate()
        if verbose:
            progress = round((i+1)/len(ori.columns)*100, 2)
            print(f"Finished {secret} on ir_eval_each_against_rest_columns, progress: {progress}%") # noqa: T201
        #Pack results and append
        res_ = evaluator.results.pack_results()
        res_.append(len(aux_cols)) #Adding len of aux_cols
        res.append(res_)
        #Append to secrets and aux_cols_all
        secrets.append(secret)
        aux_cols_all.append(aux_cols)
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
            prefix = "inference_each_against_rest",
            dataset = dataset,
            res = inf_res.model_dump()
        )
    return inf_res

def load_inference_results(*, dataset: str) -> InferenceResults:
    """Function to load and return inference results from given dataset."""
    return InferenceResults(**load_res_json_file(prefix="inference_each_against_rest", dataset=dataset))
