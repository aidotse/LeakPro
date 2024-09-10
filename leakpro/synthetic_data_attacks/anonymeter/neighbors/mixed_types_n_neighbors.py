# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Nearest neighbor search for mixed type data."""
from math import fabs, isnan
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from numba import jit

from leakpro.synthetic_data_attacks.anonymeter.preprocessing.transformations import mixed_types_transform
from leakpro.synthetic_data_attacks.anonymeter.preprocessing.type_detection import detect_consistent_col_types
from leakpro.utils.logger import logger


@jit(nopython=True, nogil=True)
def gower_distance(
    r0: npt.NDArray,
    r1: npt.NDArray,
    cat_cols_index: int
) -> float:
    r"""Distance between two records inspired by the Gower distance [1].

    To handle mixed type data, the distance is specialized for numerical (continuous)
    and categorical data. For numerical records, we use the L1 norm,
    computed after the columns have been normalized so that :math:`d(a_i, b_i)\leq 1`
    for every :math:`a_i`, :math:`b_i`. For categorical, :math:`d(a_i, b_i)` is 1,
    if the entries :math:`a_i`, :math:`b_i` differ, else, it is 0.

    Notes
    -----
    To keep the balance between numerical and categorical values, the input records
    have to be properly normalized. Their numerical part need to be scaled so that
    the difference between any two values of a column (from both dataset) is *at most* 1.

    References
    ----------
    [1]. `Gower (1971) "A general coefficient of similarity and some of its properties.
    <https://www.jstor.org/stable/2528823?seq=1>`_

    Parameters
    ----------
    r0 : npt.NDArray
        Input array of shape (D,).
    r1 : npt.NDArray
        Input array of shape (D,).
    cat_cols_index : int
        Index delimiting the categorical columns in r0/r1 if present. For example,
        ``r0[:cat_cols_index]`` are the numerical columns, and ``r0[cat_cols_index:]`` are
        the categorical ones. For a fully numerical dataset, use ``cat_cols_index =
        len(r0)``. For a fully categorical one, set ``cat_cols_index`` to 0.

    Returns
    -------
    float
        distance between the records.

    """
    dist = 0.0
    for i in range(len(r0)):
        if (isnan(r0[i]) and not isnan(r1[i])) or (not isnan(r0[i]) and isnan(r1[i])):
            dist += 1
        elif isnan(r0[i]) and isnan(r1[i]):
            pass
        elif i < cat_cols_index:
            dist += fabs(r0[i] - r1[i])
        elif r0[i] != r1[i]:
            dist += 1
    return dist

@jit(nopython=True, nogil=True)
def nearest_neighbors(
    queries: npt.NDArray,
    candidates: npt.NDArray,
    cat_cols_index: int,
    n_neighbors: int
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """For every element of ``queries``, find its nearest neighbors in ``candidates``.

    Parameters
    ----------
    queries : npt.NDArray
        Input array of shape (Nx, D).
    candidates : npt.NDArray
        Input array of shape (Ny, D).
    cat_cols_index : int
        Index delimiting the categorical columns in queries/candidates, if present.
    n_neighbors : int
        Determines the number of closest neighbors per entry to be returned.

    Returns
    -------
    idx : npt.NDArray[int64]
        Array of shape (Nx, n_neighbors). For each element in ``queries``,
        this array contains the indices of the closest neighbors in
        ``candidates``. That is, ``candidates[idx[i]]`` are the elements of
        ``candidates`` that are closer to ``queries[i]``.
    dists : npt.NDArray[float64]
        Array of shape (Nx, n_neighbors). This array containing the distances
        between the record pairs identified by idx.

    """
    idx = np.zeros((queries.shape[0], n_neighbors), dtype=np.int64)
    dists = np.zeros((queries.shape[0], n_neighbors), dtype=np.float64)
    for ix in range(queries.shape[0]):
        dist_ix = np.zeros((candidates.shape[0]), dtype=np.float64)
        for iy in range(candidates.shape[0]):
            dist_ix[iy] = gower_distance(r0=queries[ix], r1=candidates[iy], cat_cols_index=cat_cols_index)
        close_match_idx = dist_ix.argsort()[:n_neighbors]
        idx[ix] = close_match_idx
        dists[ix] = dist_ix[close_match_idx]
    return idx, dists

def mixed_type_n_neighbors(*,
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    ctypes: Optional[Dict[str, List[str]]] = None,
    n_jobs: int = -2,
    n_neighbors: int = 1,
    return_distance: bool = False
) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    """Nearest neighbor algorithm for a set of query points with mixed type data.

    To handle mixed type data, we use a distance function inspired by the Gower similarity.
    The distance is specialized for numerical (continuous) and categorical data. For
    numerical records, we use the L1 norm, computed after the columns have been
    normalized so that :math:`d(a_i, b_i) <= 1` for every :math:`a_i`, :math:`b_i`.
    For categorical, :math:`d(a_i, b_i)` is 1, if the entries :math:`a_i`, :math:`b_i`
    differ, else, it is 0.

    References
    ----------
    [1]. `Gower (1971) "A general coefficient of similarity and some of its properties.
    <https://www.jstor.org/stable/2528823?seq=1>`_

    Parameters
    ----------
    queries : pd.DataFrame
        Query points for the nearest neighbor searches.
    candidates : pd.DataFrame
        Dataset containing the records one would find the neighbors in.
    ctypes : dict, optional.
        Dictionary specifying which columns in X should be treated as
        continuous and which should be treated as categorical. For example,
        ``ctypes = {'num': ['distance'], 'cat': ['color']}`` specify the types
        of a two column dataset.
    n_jobs : int, default is -2
        Number of jobs to use. It follows joblib convention, so that ``n_jobs = -1``
        means all available cores.
    n_neighbors : int, default is 5
        Determines the number of closest neighbors per query row entry to be returned.
    return_distance : bool, default is False
        Whether or not to return the distances of the neigbors or
        just the indexes.

    Returns
    -------
    np.narray of shape (df.shape[0], n_neighbors)
        Array with the indexes of the elements of the fit dataset closer to
        each element in the query dataset.
    np.narray of shape (df.shape[0], n_neighbors)
        Array with the distances of the neighbors pairs. This is optional and
        it is returned only if ``return_distances`` is ``True``

    Note
    ----
    The search is performed in a brute-force fashion. For large datasets
    or large number of query points, the search for nearest neighbor will
    become very slow.

    """
    if n_neighbors > candidates.shape[0]:
        logger.warning(
            f"Parameter ``n_neighbors``={n_neighbors} cannot be "
            f"larger than the size of the training data {candidates.shape[0]}."
        )
        n_neighbors = candidates.shape[0]
    if ctypes is None:
        ctypes = detect_consistent_col_types(df1=queries, df2=candidates)
    queries, candidates = mixed_types_transform(
        df1=queries, df2=candidates, num_cols=ctypes["num"], cat_cols=ctypes["cat"]
    )
    cols = ctypes["num"] + ctypes["cat"]
    queries = queries[cols].values
    candidates = candidates[cols].values
    #Parallelize search for each row in queries
    with Parallel(n_jobs=n_jobs, backend="threading") as executor:
        res = executor(
            delayed(nearest_neighbors)(
                queries=queries[ii : ii + 1],
                candidates=candidates,
                cat_cols_index=len(ctypes["num"]),
                n_neighbors=n_neighbors,
            )
            for ii in range(queries.shape[0])
        )
        indexes_array, distances_array = zip(*res)
        indexes, distances = np.vstack(indexes_array), np.vstack(distances_array)
    if return_distance:
        return indexes, distances
    return indexes
