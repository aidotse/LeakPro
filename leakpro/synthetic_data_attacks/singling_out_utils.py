"""Singling-out risk util functions."""
import json
import multiprocessing as mp
import os
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional, Self, Tuple, Union

from pandas import DataFrame
from pydantic import BaseModel

from leakpro.synthetic_data_attacks.anonymeter.evaluators.singling_out_evaluator import SinglingOutEvaluator
from leakpro.synthetic_data_attacks.utils import load_res_json_file, save_res_json_file


class SinglingOutResults(BaseModel):
    """Class that holds results of singling out risk calculations.

    Parameters
    ----------
    res_cols: List[str]
        List of results columns names
    res: List[List[Union[int,float]]]
        Array containing results from EvaluationResults plus number of columns used in query predicates.

    """

    res_cols: List[str]
    res: List[List[Union[int,float]]]
    prefix: str
    dataset: str

    def save(self:Self, path:str = "../leakpro_output/results/", name: str = "singling_out", config:dict = None) -> None: # noqa: ARG002
        """Save method for SinglingOutResults."""

        id = f"{self.prefix}"+f"_{self.dataset}"

        # Data to be saved
        data = {
            "resulttype": self.__class__.__name__,
            "resultname": name,
            "res": self.model_dump(),
            "id": id,
        }

        # Check if path exists, otherwise create it.
        for _ in range(3):
            if os.path.exists(path):
                break
            path = "../"+path

        # If no result folder can be found
        if not os.path.exists(path):
            os.makedirs("../../leakpro_output/results/")

        # Save the results to a file
        if not os.path.exists(f"{path}/{name}/{id}"):
            os.makedirs(f"{path}/{name}/{id}")

        with open(f"{path}/{name}/{id}/data.json", "w") as f:
            json.dump(data, f)

        from leakpro.synthetic_data_attacks.plots import plot_singling_out
        plot_singling_out(sin_out_res=SinglingOutResults(res=self.res,
                                                         res_cols=self.res_cols,
                                                         prefix=self.prefix,
                                                         dataset=self.dataset),
                          show=False,
                          save=True,
                          save_name=f"{path}/{name}/{id}/{self.prefix}",
                        )

    @staticmethod
    def load(data: dict) -> None:
        """Load method for SinglingOutResults."""
        return SinglingOutResults(res=data["res"]["res"],
                                  res_cols=data["res"]["res_cols"],
                                  dataset=data["res"]["dataset"],
                                  prefix=data["res"]["prefix"]
                                  )

    def plot(self:Self,
            high_res_flag:bool = False,
            show:bool = True,
            save:bool = False,
            save_path:str = "./",
            save_name:str = "fig.png",
        ) -> None:
        """Plot method for SinglingOutResults."""
        from leakpro.synthetic_data_attacks.plots import plot_singling_out
        plot_singling_out(sin_out_res=SinglingOutResults(res=self.res,
                                                         res_cols=self.res_cols,
                                                         prefix=self.prefix,
                                                         dataset=self.dataset),
                        high_res_flag=high_res_flag,
                        show = show,
                        save = save,
                        save_name = f"{save_path}/{save_name}",
                        )

    @staticmethod
    def create_results(
            results: list,
            save_dir: str = "./",
        ) -> str:
        """Result method for SinglingOutResults."""
        latex = ""

        def _latex(
                save_dir: str,
                save_name: str,
            ) -> str:
            """Latex method for SinglingOutResults."""

            filename = f"{save_dir}/{save_name}.png"
            return f"""
            \\subsection{{{" ".join(save_name.split("_"))}}}
            \\begin{{figure}}[ht]
            \\includegraphics[width=0.8\\textwidth]{{{filename}}}
            \\caption{{Original}}
            \\end{{figure}}
            """
        for res in results:
            res.plot(show=False, save=True, save_path=save_dir, save_name=res.prefix)
            latex += _latex(save_dir=save_dir, save_name=res.prefix)
        return latex

def check_for_int_value(*, x: int) -> None:
    """Auxiliary function to check a given integer value."""
    assert isinstance(x, int)
    assert x>0

def get_singling_out_suffix(*, n_cols: Optional[int]) -> str:
    """Auxiliary function to get singling-out suffix."""
    suffix = "all"
    if n_cols is not None:
        check_for_int_value(x=n_cols)
        suffix = str(n_cols)
    return suffix

def get_singling_out_prefix(*, n_cols: Optional[int]) -> str:
    """Auxiliary function to get singling-out prefix used in json results filename."""
    return "singling_out_n_cols_" + get_singling_out_suffix(n_cols=n_cols)

def aux_singling_out_risk_evaluation(**kwargs: Any) -> Tuple[Optional[Union[int, float]], Optional[str]]: # noqa: ANN401
    """Auxiliary function to perform a singling out risk evaluation for given kwargs."""
    #Pop verbose
    verbose = kwargs.pop("verbose")
    #Get n_cols
    n_cols = kwargs["n_cols"]
    #Return non if n'_cols==2
    #Note: this is because n_cols==2 takes A LOT of time. Seems algorithm is not good for predicates with len==2
    if n_cols == 2:
        return None, None
    #Instantiate singling-out evaluator and evaluate
    evaluator = SinglingOutEvaluator(**kwargs)
    evaluator.evaluate()
    #Pack results and append
    res = evaluator.results.pack_results()
    res.append(n_cols) #Adding n_cols
    #Set results column names
    res_cols = evaluator.results.res_cols.copy()
    res_cols.append("n_cols")
    if verbose:
        print(f"Finished aux_singling_out_risk_evaluation for n_cols: {n_cols}") # noqa: T201
    return res, res_cols

def aux_apply_kwargs_to_fun(fun: Callable, kwargs: Dict) -> Any: # noqa: ANN401
    """Auxiliary function that executes passed fun with given kwargs."""
    return fun(**kwargs)

def starmap_with_kwargs(pool: mp.Pool, fn: Callable, kwargs_list: List[Dict]) -> List:
    """Auxiliary wrapper function for mp.Pool.starmap function, that enables passing kwargs to it."""
    args_for_starmap = zip(repeat(fn), kwargs_list)
    return pool.starmap(aux_apply_kwargs_to_fun, args_for_starmap)

def singling_out_risk_evaluation(
    ori: DataFrame,
    syn: DataFrame,
    n_cols: Optional[int] = None,
    dataset: str = "test",
    verbose: bool = False,
    save_results_json: bool = False,
    **kwargs: dict
) -> SinglingOutResults:
    """Perform an individual/full singling-out risk evaluation.

    Individual evaluation is for given n_cols to be used as len of predicates.
    Full evaluation is for all number of cols (from 1 to len(ori.columns)) to be used as len of predicates.

    Parameters
    ----------
    ori : pd.DataFrame
        Dataframe containing original data.
    syn : pd.DataFrame
        Dataframe containing synthetic data.
        It has to have the same columns as df_ori.
    n_cols: Optional[int], default is None
        If None, performs a full evaluation. Otherwise performs individual evaluation for given n_cols
    dataset: str
        Name of dataframes, used when verbose True and when saving results to json.
    verbose: bool, default is False
        If True, prints progress of evaluation.
    save_results_json: bool, default is False
        If True, saves results and combinations to json file.
    kwargs: dict
        Other keyword arguments for SinglingOutEvaluator.

    Returns
    -------
    SinglingOutResults
        SinglingOutResults with results.

    """
    if verbose:
        suffix = get_singling_out_suffix(n_cols=n_cols)
        print(f"\nRunning singling out risk evaluation for `{dataset}` with n_cols {suffix}") # noqa: T201
    if n_cols is not None:
        check_for_int_value(x=n_cols)
        if n_cols == 2:
            raise ValueError("Parameter `n_cols` must be different than 2.")
        #Run individual aux_singling_out_risk_evaluation
        res, res_cols = aux_singling_out_risk_evaluation(
            ori = ori,
            syn = syn,
            n_cols = n_cols,
            verbose = verbose,
            **kwargs
        )
        #Repack res
        res = [res]
    else:
        #Construct kwargs_list
        kwargs_list = []
        for i in range(len(ori.columns)):
            kwargs_t = {
                "ori": ori,
                "syn": syn,
                "n_cols": i+1,
                "verbose": verbose
            }
            kwargs_t.update(kwargs)
            kwargs_list.append(kwargs_t)
        #Set nr_processes
        max_processes = mp.cpu_count() - 2
        nr_processes = min(max_processes, len(kwargs_list))
        if verbose:
            print("Nr. of processors to use for singling out evaluation:", nr_processes) # noqa: T201
        # Create multiprocessing pool and run aux_singling_out_risk_evaluation for each kwargs
        with mp.Pool(processes=nr_processes) as pool:
            res_ = starmap_with_kwargs(pool, aux_singling_out_risk_evaluation, kwargs_list)
        # Repack res and res_cols
        res = [i[0] for i in res_ if i[0] is not None]
        res_cols = res_[0][1]
    #Instantiate SinglingOutResults
    sin_out_res = SinglingOutResults(
        res_cols = res_cols,
        res = res,
        prefix = get_singling_out_prefix(n_cols=n_cols),
        dataset = dataset,
    )
    #Save results to json
    if save_results_json:
        save_res_json_file(
            prefix = get_singling_out_prefix(n_cols=n_cols),
            dataset = dataset,
            res = sin_out_res.model_dump()
        )
    return sin_out_res

def load_singling_out_results(*, dataset: str, n_cols: Optional[int] = None) -> SinglingOutResults:
    """Function to load and return singling-out results from given dataset."""
    prefix = get_singling_out_prefix(n_cols=n_cols)
    return SinglingOutResults(**load_res_json_file(prefix=prefix, dataset=dataset))
