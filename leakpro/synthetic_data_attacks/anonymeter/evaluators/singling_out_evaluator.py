# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Privacy evaluator that measures the singling out risk."""
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict

from leakpro.synthetic_data_attacks.anonymeter.stats.confidence import EvaluationResults
from leakpro.utils.import_helper import Self

#Set rng
rng = np.random.default_rng()

def escape_quotes(*, input: str) -> str:
    """Function escapes quotes for given input string."""
    return input.replace('"', '\\"').replace("'", "\\'")

def safe_query_elements(*, query: str, df: pd.DataFrame) -> List[int]:
    """Return elements indexes in df satisfying a given query."""
    try:
        return df.query(query, engine="python").index.to_list()
    except Exception as ex:
        raise Exception(f"Query {query} failed with {ex}.") # noqa: B904

def query_equality_expression(col: str, val: Any, dtype: np.dtype) -> str: # noqa: ANN401
    """Function returns a type-aware query equality expression for given col, val and dtype."""
    if is_datetime64_any_dtype(dtype):
        query = f"{col} == '{val}'"
    elif isinstance(val, str):
        query = f"{col} == '{escape_quotes(input=val)}'"
    else:
        query = f"{col} == {val}"
    return query

def random_operator(*, data_type: str) -> str:
    """Function returns a random operator for given input data_type."""
    if data_type == "categorical":
        ops = ["==", "!="]
    elif data_type == "boolean":
        ops = ["", "not "]
    elif data_type == "numerical":
        ops = ["==", "!=", ">", "<", ">=", "<="]
    else:
        raise ValueError(f"Unknown `data_type`: {data_type}")
    return rng.choice(ops)

def random_query(*, unique_values: Dict[str, List[Any]], cols: List[str]) -> str:
    """Function returns a random query for given columns."""
    query = []
    for col in cols:
        #Get values and choose a value
        values = unique_values[col]
        val = rng.choice(values)
        #NaNs
        if pd.isna(val):
            sub_query = f"{random_operator(data_type='boolean')}{col}.isna()"
        #Boolean
        elif is_bool_dtype(values):
            sub_query = f"{random_operator(data_type='boolean')}{col}"
        #Categorical
        elif isinstance(values, pd.CategoricalDtype):
            sub_query = f"{col} {random_operator(data_type='categorical')} {val}"
        #Numerical
        elif is_numeric_dtype(values):
            sub_query = f"{col} {random_operator(data_type='numerical')} {val}"
        #Categorical with escape quotes
        elif isinstance(val, str):
            sub_query = f"{col} {random_operator(data_type='categorical')} '{escape_quotes(input=val)}'"
        #Categorical
        else:
            sub_query = f"{col} {random_operator(data_type='categorical')} '{val}'"
        #Append to query
        query.append(sub_query)
    return " & ".join(query)

def random_queries(*, df: pd.DataFrame, n_queries: int, n_cols: int) -> List[str]:
    """Function returns n_queries random queries each with size n_cols on df as input."""
    #Get unique_values
    unique_values = {col: df[col].unique() for col in df.columns}
    #Placeholder queries
    queries = []
    #Iterate through range n_queries
    for _ in range(n_queries):
        #Get random columns
        random_cols = rng.choice(df.columns, size=n_cols, replace=False).tolist()
        #Get random query
        query = random_query(unique_values=unique_values, cols=random_cols)
        #Append to queries
        queries.append(query)
    return queries

class UniqueSinglingOutQueries(BaseModel):
    """
    Collection of unique queries that single out records in a DataFrame.

    Attributes:
        df (pd.DataFrame): The DataFrame to evaluate queries against.
        sorted_queries_set (Set[str]): A set of sorted query strings to ensure uniqueness.
        queries (List[str]): List of unique queries.
        idxs (List[int]): Indices of records singled out by the queries.
        count (int): Total count of unique queries.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    df: pd.DataFrame
    sorted_queries_set: Set[str] = set()
    queries: List[str] = []
    idxs: List[int] = []
    count: int = 0
    len_passed_queries: int = 0

    def evaluate_queries(self: Self, *, queries: List[str]) -> Self:
        """Evaluate queries on self.df.

        Function returns UniqueSinglingOutQueries instance with unique singling out queries and count.
        """
        #Reset parameters
        self.sorted_queries_set = set()
        self.queries = []
        self.idxs = []
        self.count = 0
        self.len_passed_queries = len(queries)
        #Iterate through queries
        for query in queries:
            self.check_and_append(query=query)
        return self

    def check_and_append(self: Self, *, query: str,) -> None:
        """Add a singling out query to the collection.

        A query singles out if it returns one record from self.df.
        Only queries which element's index has not been added,
        or that are not already in the collection
        are appended.

        Parameters
        ----------
        query : str
            query expression to be added.

        """
        #Get indexes for query
        idxs = safe_query_elements(query=query, df=self.df)
        #Check for 1 record, idxs and sorted query
        if (len(idxs) == 1) and (idxs[0] not in self.idxs):
            #Get sorted_query
            sorted_query = "".join(sorted(query))
            if sorted_query not in self.sorted_queries_set:
                #Append to sorted_queries, queries, idxs and count
                self.sorted_queries_set.add(sorted_query)
                self.queries.append(query)
                self.idxs.append(idxs[0])
                self.count += 1

    def append_only(self, *, query: str) -> None:
        """
        Add a query to the collection if it is unique, without evaluating its result.

        Parameters:
            query (str): The query to add.
        """
        sorted_query = "".join(sorted(query))
        if sorted_query not in self.sorted_queries_set:
            self.sorted_queries_set.add(sorted_query)
            self.queries.append(query)
            self.count += 1

def naive_singling_out_attack(*,
    ori: pd.DataFrame,
    syn: pd.DataFrame,
    n_attacks: int,
    n_cols: int
) -> UniqueSinglingOutQueries:
    """Naive singling-out attack function.

    Function returns a UniqueSinglingOutQueries object, with succesful singling out queries and count.
    """
    #Get random queries and evaluate
    queries = random_queries(df=syn, n_queries=n_attacks, n_cols=n_cols)
    return UniqueSinglingOutQueries(df=ori).evaluate_queries(queries=queries)

def univariate_singling_out_queries(*, df: pd.DataFrame, n_queries: int) -> List[str]:
    """Function to generate singling out queries from univariate rare attributes.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe from which queries will be generated.
    n_queries: int
        Number of queries to generate.

    Returns
    -------
    List[str]
        The singling out queries.

    """
    #Placeholder candidate queries
    can_queries = []
    #Iterate through df.columns
    for col in df.columns:
        #NaN
        if df[col].isna().sum() == 1:
            can_queries.append(f"{col}.isna()")
        #Numeric types extreme values
        if pd.api.types.is_numeric_dtype(df.dtypes[col]):
            values = df[col].dropna().sort_values()
            if len(values) > 0:
                can_queries.extend([
                    f"{col} <= {values.iloc[0]}",
                    f"{col} >= {values.iloc[-1]}"
                ])
        #Rare values
        counts = df[col].value_counts()
        rare_values = counts[counts == 1]
        if len(rare_values) > 0:
            can_queries.extend([
                query_equality_expression(col=col, val=val, dtype=df.dtypes[col]) for val in rare_values.index
            ])
    #Shuffle candidate queries
    rng.shuffle(can_queries)
    #Instantiate queries
    queries = UniqueSinglingOutQueries(df=df)
    for can_query in can_queries:
        queries.check_and_append(query=can_query)
        if queries.count == n_queries:
            break
    return queries.queries

def query_from_record(*,
    record: pd.Series,
    dtypes: pd.Series,
    columns: List[str],
    medians: Optional[pd.Series]
) -> str:
    """Function that constructs and returns a query from the attributes in a record."""
    #Placeholder query
    query = []
    #Iterate through columns
    for col in columns:
        #NaN
        if pd.isna(record[col]):
            item = ".isna()"
        #Boolean
        elif is_bool_dtype(dtypes[col]):
            item = f"== {record[col]}"
        #Numeric
        elif is_numeric_dtype(dtypes[col]):
            if medians is None:
                operator = rng.choice([">=", "<="])
            elif record[col] > medians[col]:
                operator = ">="
            else:
                operator = "<="
            item = f"{operator} {record[col]}"
        #Categorical
        elif isinstance(dtypes[col], pd.CategoricalDtype) and is_numeric_dtype(dtypes[col].categories.dtype):
            item = f"== {record[col]}"
        elif isinstance(record[col], str):
            item = f"== '{escape_quotes(input=record[col])}'"
        else:
            item = f'== "{record[col]}"'
        query.append(f"{col} {item}")
    return " & ".join(query)

def multivariate_singling_out_queries(
    df: pd.DataFrame,
    n_queries: int,
    n_cols: int,
    max_attempts: Optional[int]
) -> List[str]:
    """Function that generates singling out queries from a combination of attributes.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe from which queries will be generated.
    n_queries: int
        Number of queries to generate.
    n_cols: int
        Number of columns that the attacker uses to create the
        singling out queries.
    max_attempts: int, optional.
        Maximum number of attempts that the attacker can make to generate
        the requested `n_queries` singling out queries. Parameter
        caps the total number of query generation attempts.
        If `max_attempts` is None, no limit will be imposed.

    Returns
    -------
    List[str]
        The singling out queries.

    """
    #Set medians and n_attemps
    medians = df.median(numeric_only=True)
    n_attempts = 0
    #Instantiate queries
    queries = UniqueSinglingOutQueries(df=df)
    #Construct queries
    while queries.count < n_queries:
        #Reached max attemps
        if max_attempts is not None and n_attempts >= max_attempts:
            print( # noqa: T201
                f"Reached maximum number of attempts ({max_attempts}) when generating singling out queries. "
                f"Returning {queries.count} instead of the requested {n_queries} queries. "
                "To avoid this, increase max_attempts or set it to `None` to disable limit."
            )
            return queries.queries
        #Choose record and columns
        record = df.iloc[rng.integers(df.shape[0])]
        columns = rng.choice(df.columns, size=n_cols, replace=False).tolist()
        #Construct query
        query = query_from_record(record=record, dtypes=df.dtypes, columns=columns, medians=medians)
        #Check query and append
        queries.check_and_append(query=query)
        #Add to n_attempts
        n_attempts += 1
    return queries.queries

def main_singling_out_attack(
    ori: pd.DataFrame,
    syn: pd.DataFrame,
    n_attacks: int,
    n_cols: int,
    max_attempts: Optional[int]
) -> UniqueSinglingOutQueries:
    """Main singling-out attack function.

    Function returns a UniqueSinglingOutQueries object, with succesful singling out queries and count.
    """
    #Get queries (depends on n_cols)
    if n_cols == 1:
        queries = univariate_singling_out_queries(df=syn, n_queries=n_attacks)
    else:
        queries = multivariate_singling_out_queries(
            df = syn,
            n_queries = n_attacks,
            n_cols = n_cols,
            max_attempts = max_attempts
        )
    #Warning message
    if len(queries) < n_attacks:
        print( # noqa: T201
            f"Main singling out attack generated only {len(queries)} "
            f"singling out queries out of the requested {n_attacks}. "
            "This can probably lead to an underestimate of the singling-out risk."
        )
    return UniqueSinglingOutQueries(df=ori).evaluate_queries(queries=queries)

class SinglingOutEvaluator(BaseModel):
    """Measure the singling-out risk created by a synthetic dataset.

    Singling out happens when the attacker can determine that
    there is a single individual in the dataset that has certain
    attributes (for example "zip_code == XXX and first_name == YYY")
    with high enough confidence.

    The risk is estimated comparing the number of successfull singling out
    queries to the desired number of attacks (`n_attacks`).

    Parameters
    ----------
    ori : pd.DataFrame
        Original dataframe on which the success of the singling out attacker
        attacker will be evaluated.
    syn : pd.DataFrame
        Synthetic dataframe used to generate the singling out queries.
    n_cols : int, default is 1
        Number of columns that the attacker uses to create the singling
        out queries.
    n_attacks : int, default is 2_000
        Number of singling out attacks to attempt.
    confidence_level : float, default is 0.95
        Confidence level for the error bound calculation.
    max_attempts : int or None, default is 10_000_000
        Maximum number of attempts that the attacker can make to generate
        the requested `n_attacks` singling out queries. This is useful to
        avoid excessively long running calculations. If ``max_attempts`` is None,
        no limit will be imposed.
    main_queries: UniqueSinglingOutQueries, optional
        UniqueSinglingOutQueries object holding main attack queries and count.
        Parameter will be set in evaluate method.
    naive_queries: UniqueSinglingOutQueries, optional
        UniqueSinglingOutQueries object holding naive attack queries and count.
        Parameter will be set in evaluate method.
    results: EvaluationResults, optional
        EvaluationResults object containing the success rates for the various attacks.
        Parameter will be set in evaluate method.

    """

    model_config = ConfigDict(arbitrary_types_allowed = True)
    ori: pd.DataFrame
    syn: pd.DataFrame
    n_cols: int = 1
    n_attacks: int = 2_000
    confidence_level: float = 0.95
    max_attempts: Optional[int] = 10_000_000
    #Following parameters are set in evaluate method
    main_queries: Optional[UniqueSinglingOutQueries] = None
    naive_queries: Optional[UniqueSinglingOutQueries] = None
    results: Optional[EvaluationResults] = None

    def __init__(self: Self, **kwargs: Any) -> None: # noqa: ANN401
        super().__init__(**kwargs)
        self.ori = self.ori.drop_duplicates()
        self.syn = self.syn.drop_duplicates()

    def evaluate(self: Self) -> EvaluationResults:
        """Run the singling-out attacks (main and naive) and set and return results."""
        # Main singling-out attack
        self.main_queries = main_singling_out_attack(
            ori = self.ori,
            syn = self.syn,
            n_attacks = self.n_attacks,
            n_cols = self.n_cols,
            max_attempts = self.max_attempts
        )
        # Naive singling-out attack
        self.naive_queries = naive_singling_out_attack(
            ori = self.ori,
            syn = self.syn,
            n_attacks = self.n_attacks,
            n_cols = self.n_cols
        )
        # Set results
        self.results = EvaluationResults(
            n_main_total = self.main_queries.len_passed_queries,
            n_main_success = self.main_queries.count,
            n_naive_total = self.naive_queries.len_passed_queries,
            n_naive_success = self.naive_queries.count,
            confidence_level = self.confidence_level
        )
        return self.results
