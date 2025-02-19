"""Tests for singling_out_evaluator module."""
# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import re
from typing import List

import numpy as np
import pandas as pd
import pytest

import leakpro.synthetic_data_attacks.anonymeter.evaluators.singling_out_evaluator as singl_ev
from leakpro.synthetic_data_attacks.anonymeter.stats.confidence import EvaluationResults
from leakpro.tests.tests_synthetic_data_attacks.anonymeter_tests.fixtures import get_adult


@pytest.mark.parametrize(
    ("input", "e_output"),
    [
        ("test1", "test1"),
        ('te"st1', 'te\\"st1'),
        ("te'st1", "te\\'st1"),
        ("""te's"t1""", """te\\'s\\"t1""")
    ]
)
def test_escape_quotes(input:str, e_output: str) -> None:
    """Assert results of function escape_quotes for different input values."""
    output = singl_ev.escape_quotes(input=input)
    assert output == e_output

def test_safe_query_elements() -> None:
    """Assert results of function safe_query_elements for mal-formed and well-formed input queries."""
    df = pd.DataFrame({"c1": [0, 0, 2, 1], "c2": ["a", "a", "c", "d"], "c3": [1, 2, 4, 4]})
    #Case mal-formed queries
    query = "c4==4"
    with pytest.raises(Exception, match=f"Query {query} failed with name 'c4' is not defined."):
        singl_ev.safe_query_elements(query=query, df=df)
    query = "c2#=4"
    with pytest.raises(Exception, match=f"Query {query} failed with "):
        singl_ev.safe_query_elements(query=query, df=df)
    #Case well-formed queries
    query = "c2=='e'"
    idxs = singl_ev.safe_query_elements(query=query, df=df)
    assert idxs == []
    query = "c2=='a'"
    idxs = singl_ev.safe_query_elements(query=query, df=df)
    assert idxs == [0,1]
    query = "c2=='a' & c3==1"
    idxs = singl_ev.safe_query_elements(query=query, df=df)
    assert idxs == [0]
    query = "c3==4 & (c2=='c' or c1==1)"
    idxs = singl_ev.safe_query_elements(query=query, df=df)
    assert sorted(idxs) == [2,3]

def test_query_equality_expression() -> None:
    """Assert results of function query_equality_expression for different input values."""
    query = singl_ev.query_equality_expression(col="a", val="2018-01-01", dtype=np.datetime64)
    assert query == "a == '2018-01-01'"
    query = singl_ev.query_equality_expression(col="a", val="a", dtype=np.dtype("object"))
    assert query == "a == 'a'"
    query = singl_ev.query_equality_expression(col="a", val=1, dtype=np.dtype("int64"))
    assert query == "a == 1"

@pytest.mark.parametrize(
    ("data_type", "e_error", "e_ops_list"),
    [
        ("categorical", False, ["==", "!="]),
        ("boolean", False, ["", "not "]),
        ("numerical", False, ["==", "!=", ">", "<", ">=", "<="]),
        ("error", True, None),
    ]
)
def test_random_operator(data_type: str, e_error: bool, e_ops_list: List[str]) -> None:
    """Assert results of function random_operator for different input values."""
    if not e_error:
        op = singl_ev.random_operator(data_type=data_type)
        assert op in e_ops_list
    else:
        with pytest.raises(ValueError, match=f"Unknown `data_type`: {data_type}"):
            singl_ev.random_operator(data_type=data_type)

def test_random_query() -> None:
    """Assert results of function random_query with simple input."""
    #Setup test variables
    cols = ["NaN", "boolean", "integer", "float", "string"]
    u_values = [
        [np.nan], #NaN
        [True, False], #boolean
        [1, 2, 3], #integer
        [1.0, 2.0, 3.0], #float
        ["a", "b", "c"], #string
    ]
    assert len(cols) == len(u_values)
    operators_transf_u_values = {
        "integer": (["==", "!=", ">", "<", ">=", "<="], int, u_values[2]),
        "float": (["==", "!=", ">", "<", ">=", "<="], float, u_values[3]),
        "string": (["==", "!="], str, u_values[4])
    }
    unique_values = {}
    for col, u_values_ in zip(cols, u_values):
        unique_values[col] = np.array(u_values_)
    #Run random_query
    query_all = singl_ev.random_query(unique_values=unique_values, cols=cols)
    #Assert results
    assert isinstance(query_all, str)
    query_split = query_all.split("&")
    assert len(query_split) == len(cols)
    for query, col in zip(query_split, cols):
        query_ = query.strip()
        if col == "NaN":
            q = "NaN.isna()"
            assert query_ in (q, "not "+q)
        elif col == "boolean":
            q = "boolean"
            assert query_ in (q, "not "+q)
        else:
            query_ = query_.replace("'", "")
            #Regex expression for sub_query
            pattern = r"\b([a-zA-Z]+)\s*(==|!=|>=|<=|>|<)\s*(\S+)"
            match = re.search(pattern, query_)
            #Assert column
            assert match
            assert match.group(1) == col
            #Assert operator
            ops_trans_u_values = operators_transf_u_values[col]
            assert match.group(2) in ops_trans_u_values[0]
            #Assert value
            transf_val = ops_trans_u_values[1](match.group(3))
            assert transf_val in np.array(ops_trans_u_values[2])
            assert str(transf_val) == match.group(3)

def test_random_queries() -> None:
    """Assert simplified results of function random_queries with adults simple input."""
    #Set test variables
    df = get_adult(return_ori=True, n_samples=100)
    n_queries = 10
    n_cols = 3
    #Auxiliary test variable
    def assert_column_in_query(*, query: str) -> None:
        column_in_query = False
        for col in df.columns:
            if col in query:
                column_in_query = True
                break
        assert column_in_query
    #Run random queries
    rand_queries = singl_ev.random_queries(df=df, n_queries=n_queries, n_cols=n_cols)
    #Assert results
    assert len(rand_queries) == n_queries
    for rand_query in rand_queries:
        rand_query_ = rand_query.split("&")
        assert len(rand_query_) == n_cols
        for query in rand_query_:
            assert_column_in_query(query=query)

def test_UniqueSinglingOutQueries_init() -> None: # noqa: N802
    """Assert results of initializing class UniqueSinglingOutQueries."""
    df = pd.DataFrame({"c1": [0, 0, 2, 1], "c2": ["a", "a", "c", "d"], "c3": [1, 2, 4, 4]})
    queries = singl_ev.UniqueSinglingOutQueries(df=df)
    assert isinstance(queries, singl_ev.UniqueSinglingOutQueries)
    assert queries.df.equals(df)
    assert queries.sorted_queries_set == set()
    assert queries.queries == []
    assert queries.idxs == []
    assert queries.count == 0
    assert queries.len_passed_queries == 0

@pytest.mark.parametrize(
    ("passed_queries", "e_count", "e_m_queries", "e_idxs"),
    [
        (["c2=='fuffa'"], 0, [], []), #0 total matches
        (["c1==0 and c2=='a'"], 0, [], []), #2 total matches
        (["c1==2 and c2=='c'"], 1, ["c1==2 and c2=='c'"], [2]),
        (["c1==2 and c2=='c'", "c3==1", "c2=='fuffa'"], 2, ["c1==2 and c2=='c'", "c3==1"], [0, 2]),
        (["c1==2 and c2=='c'", "c3==1", "c2=='fuffa'", "c1==1 and c3==4"], 3, ["c1==2 and c2=='c'", "c3==1", "c1==1 and c3==4"], [0, 2, 3]), # noqa: E501
        (["c1==2 and c2=='c'", "c3==1", "c2=='fuffa'", "c1==1 and c3==4", "c2=='d'", "c2=='c'"], 3, ["c1==2 and c2=='c'", "c3==1", "c1==1 and c3==4"], [0, 2, 3]) # noqa: E501 #repeated matches
    ]
)
def test_evaluate_queries(passed_queries: List[str], e_count: int, e_m_queries: List[str], e_idxs: List[int]) -> None:
    """Assert results of UniqueSinglingOutQueries.evaluate_queries method with different input values."""
    df = pd.DataFrame({"c1": [0, 0, 2, 1], "c2": ["a", "a", "c", "d"], "c3": [1, 2, 4, 4]})
    queries = singl_ev.UniqueSinglingOutQueries(df=df).evaluate_queries(queries=passed_queries)
    assert isinstance(queries, singl_ev.UniqueSinglingOutQueries)
    assert queries.count == e_count
    assert queries.queries == e_m_queries
    assert sorted(queries.idxs) == e_idxs
    assert e_count == len(queries.queries)
    assert e_count == len(queries.idxs)
    assert queries.len_passed_queries == len(passed_queries)

def test_check_and_append() -> None:
    """Assert results of UniqueSinglingOutQueries.check_and_append method with different input values."""
    #Test auxiliary function
    def aux_assert_queries_count(*,
        queries: singl_ev.UniqueSinglingOutQueries,
        queries_: List[str],
        idxs: List[int]
    ) -> None:
        assert queries.queries == queries_
        assert queries.idxs == idxs
        assert len(queries.queries) == queries.count
        assert len(queries.idxs) == queries.count
    #Set df
    df = pd.DataFrame({"c1": [1], "c2": [2]})
    #Instantiate UniqueSinglingOutQueries
    queries = singl_ev.UniqueSinglingOutQueries(df=df)
    q1 = "c1 == 2"
    queries.check_and_append(query=q1)
    aux_assert_queries_count(queries=queries, queries_=[], idxs=[])
    q1, q2 = "c1 == 1", "c2 == 2"
    queries.check_and_append(query=q1)
    queries.check_and_append(query=q1)
    aux_assert_queries_count(queries=queries, queries_=[q1], idxs=[0])
    queries.check_and_append(query=q2)
    aux_assert_queries_count(queries=queries, queries_=[q1], idxs=[0])
    #Instantiate UniqueSinglingOutQueries
    queries = singl_ev.UniqueSinglingOutQueries(df=df)
    q3, q4 = f"{q1} and {q2}", f"{q2} and {q1}"
    queries.check_and_append(query=q3)
    queries.check_and_append(query=q4)
    aux_assert_queries_count(queries=queries, queries_=[q3], idxs=[0])
    #Reset df
    df = pd.DataFrame({"c1": [1, 1], "c2": [2, 3]})
    q1 = "c1 == 1"
    #Instantiate UniqueSinglingOutQueries
    queries = singl_ev.UniqueSinglingOutQueries(df=df)
    queries.check_and_append(query=q1)
    aux_assert_queries_count(queries=queries, queries_=[], idxs=[])
    q1 = "c1 == 1 and c2 == 3"
    queries.check_and_append(query=q1)
    aux_assert_queries_count(queries=queries, queries_=[q1], idxs=[1])

def test_naive_singling_out_attack() -> None:
    """Assert function naive_singling_out_attack raises no errors with adults simple input."""
    #Set test variables
    ori = get_adult(return_ori=True, n_samples=100)
    syn = get_adult(return_ori=False, n_samples=100)
    n_attacks = 10
    n_cols = 3
    #Get queries
    queries = singl_ev.naive_singling_out_attack(ori=ori, syn=syn, n_attacks=n_attacks, n_cols=n_cols)
    assert isinstance(queries, singl_ev.UniqueSinglingOutQueries)
    if queries.count>0:
        for query in queries.queries:
            assert len(query.split("&")) == n_cols

#Following variables are for following test
col1 =  ["a", "b", "c", "d", np.nan]
col2 =  [-2, -1, 2, 1, np.nan]
e_queries_col1 = ["col1 == 'a'", "col1 == 'b'", "col1 == 'c'", "col1 == 'd'", "col1.isna()"]
e_queries_col2 = ["col2 == -2.0", "col2 == -1.0", "col2 == 2.0", "col2 == 1.0", "col2.isna()", "col2 >= 2.0", "col2 <= -2.0"]

@pytest.mark.parametrize(
    ("input_dict", "e_queries"),
    [
        ({"col1": col1}, e_queries_col1),
        ({"col2": col2}, e_queries_col2),
        ({"col1": col1, "col2": col2}, e_queries_col1+e_queries_col2)
    ]
)
def test_univariate_singling_out_queries(input_dict: dict, e_queries: List[str]) -> None:
    """Assert results of function univariate_singling_out_queries with simple input."""
    df = pd.DataFrame(input_dict)
    queries = singl_ev.univariate_singling_out_queries(df=df, n_queries=15)
    assert set(queries).issubset(set(e_queries))
    assert len(set(queries)) == 5

@pytest.mark.parametrize("max_attempts", [1, 2, 3])
def test_multivariate_singling_out_queries_max_attempts(max_attempts: int) -> None:
    """Assert results of function multivariate_singling_out_queries with max_attempts input."""
    ori = get_adult(return_ori=True, n_samples=10)
    queries = singl_ev.multivariate_singling_out_queries(df=ori, n_queries=10, n_cols=2, max_attempts=max_attempts)
    assert len(queries) <= max_attempts

def test_multivariate_singling_out_queries() -> None:
    """Assert results of function multivariate_singling_out_queries with simple input."""
    df = pd.DataFrame({"c0": ["a", "b"], "c1": [1.23, 9.87]})
    n_queries = 2
    queries = singl_ev.multivariate_singling_out_queries(df=df, n_queries=n_queries, n_cols=2, max_attempts=None)
    assert len(queries) == n_queries
    possible_queries = [
        "c0 == 'a' & c1 <= 1.23",
        "c1 <= 1.23 & c0 == 'a'",
        "c0 == 'a' & c1 >= 9.87",
        "c1 >= 9.87 & c0 == 'a'",
        "c0 == 'b' & c1 <= 1.23",
        "c1 <= 1.23 & c0 == 'b'",
        "c0 == 'b' & c1 >= 9.87",
        "c1 >= 9.87 & c0 == 'b'"
    ]
    for query in queries:
        assert query in possible_queries

def test_main_singling_out_attack() -> None:
    """Assert function main_singling_out_attack raises no errors with adults simple input."""
    #Set test variables
    ori = get_adult(return_ori=True, n_samples=100)
    syn = get_adult(return_ori=False, n_samples=100)
    n_attacks = 10
    n_cols = 3
    #Get queries
    queries = singl_ev.main_singling_out_attack(ori=ori, syn=syn, n_attacks=n_attacks, n_cols=n_cols, max_attempts=None)
    assert isinstance(queries, singl_ev.UniqueSinglingOutQueries)
    if queries.count>0:
        for query in queries.queries:
            assert len(query.split("&")) == n_cols

@pytest.mark.parametrize("n_cols", [1, 3])
def test_SinglingOutEvaluator(n_cols: int) -> None: # noqa: N802
    """Assert SinglingOutEvaluator results with adults simple input and varying n_cols."""
    ori = get_adult(return_ori=True, n_samples=10)
    syn = get_adult(return_ori=False, n_samples=10)
    #Instantiate SinglingOutEvaluator
    soe = singl_ev.SinglingOutEvaluator(ori=ori, syn=syn, n_cols=n_cols, n_attacks=5)
    assert soe.ori.equals(ori)
    assert soe.syn.equals(syn)
    assert soe.n_cols == n_cols
    assert soe.n_attacks == 5
    assert soe.confidence_level == 0.95
    assert soe.max_attempts == 10_000_000
    assert soe.main_queries is None
    assert soe.naive_queries is None
    assert soe.results is None
    #Run evaluate method
    soe.evaluate()
    assert isinstance(soe.main_queries, singl_ev.UniqueSinglingOutQueries)
    assert isinstance(soe.naive_queries, singl_ev.UniqueSinglingOutQueries)
    assert isinstance(soe.results, EvaluationResults)
    for q in soe.main_queries.queries:
        assert len(singl_ev.safe_query_elements(query=q, df=ori)) == 1
        assert len(singl_ev.safe_query_elements(query=q, df=syn)) == 1
