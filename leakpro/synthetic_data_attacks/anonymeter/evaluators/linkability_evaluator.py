# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Privacy evaluator that measures the linkability risk."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict

from leakpro.import_helper import Self
from leakpro.synthetic_data_attacks.anonymeter.neighbors.mixed_types_n_neighbors import mixed_type_n_neighbors
from leakpro.synthetic_data_attacks.anonymeter.stats.confidence import EvaluationResults
from leakpro.synthetic_data_attacks.anonymeter.utils import assert_x_in_bound


class LinkabilityIndexes(BaseModel):
    """Utility class to store indexes from linkability attack.

    Parameters
    ----------
    idx_0 : np.ndarray
        Array containing the result of the nearest neighbor search
        between the first original dataset and the synthetic data.
        Rows correspond to original records and the i-th column
        contains the index of the i-th closest synthetic record.
    idx_1 : np.ndarray
        Array containing the result of the nearest neighbor search
        between the second original dataset and the synthetic data.
        Rows correspond to original records and the i-th column
        contains the index of the i-th closest synthetic record.

    """

    model_config = ConfigDict(arbitrary_types_allowed = True)
    idx_0: npt.NDArray
    idx_1: npt.NDArray
    links: Optional[Dict[int, npt.NDArray]] = None
    count: Optional[int] = None

    def __init__(self: Self, **kwargs: npt.NDArray) -> None:
        super().__init__(**kwargs)
        #Assert input values
        assert len(self.idx_0.shape) > 1
        assert self.idx_0.shape[1] > 0
        assert self.idx_0.shape == self.idx_1.shape

    def find_links(self: Self, *, n_neighbors: int = 1) -> Self:
        """Find and count synthetic records that link records between the first and the second datasets.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors considered for the link search.

        Returns
        -------
        Self : LinkabilityIndexes
            Object containing indexes, links and counts.

        """
        assert_x_in_bound(
            x = n_neighbors,
            x_name = "n_neighbors",
            low_bound = 1,
            high_bound = self.idx_0.shape[1],
            inclusive_flag = True
        )
        # Calculate links and count
        self.links = {}
        self.count = 0
        for ii, (row0, row1) in enumerate(zip(self.idx_0, self.idx_1)):
            matches = np.intersect1d(row0[:n_neighbors], row1[:n_neighbors])
            if len(matches) > 0:
                self.links[ii] = matches
                self.count += 1
        return self

def main_linkability_attack(
    ori: pd.DataFrame,
    syn: pd.DataFrame,
    n_attacks: int,
    aux_cols: Tuple[List[str], List[str]],
    n_neighbors: int,
    n_jobs: int,
) -> LinkabilityIndexes:
    """Main linkability attack function.

    Function returns a LinkabilityIndexes object, with indexes being
    closest n_neighbors between first/second original dataset and synthetic data.
    """
    targets = ori.sample(n_attacks, replace=False)
    idx_0 = mixed_type_n_neighbors(queries=targets[aux_cols[0]], candidates=syn[aux_cols[0]], n_neighbors=n_neighbors, n_jobs=n_jobs) # noqa: E501
    idx_1 = mixed_type_n_neighbors(queries=targets[aux_cols[1]], candidates=syn[aux_cols[1]], n_neighbors=n_neighbors, n_jobs=n_jobs) # noqa: E501
    return LinkabilityIndexes(idx_0=idx_0, idx_1=idx_1).find_links()

def random_links(*, n_synthetic: int, n_attacks: int, n_neighbors: int) -> npt.NDArray:
    """Auxiliary function for naive_linkability_attack.

    Function returns an array with shape (n_attacks, n_neighbors),
    filled with random values chosen from range(n_synthetic).
    """
    rng = np.random.default_rng()
    return np.array(
        [rng.choice(n_synthetic, size=n_neighbors, replace=False) for _ in range(n_attacks)]
    )

def naive_linkability_attack(*, n_synthetic: int, n_attacks: int, n_neighbors: int) -> LinkabilityIndexes:
    """Naive linkability attack function.

    Function returns a LinkabilityIndexes object, with 2 randomly chosen arrays as indexes.
    """
    idx_0 = random_links(n_synthetic=n_synthetic, n_attacks=n_attacks, n_neighbors=n_neighbors)
    idx_1 = random_links(n_synthetic=n_synthetic, n_attacks=n_attacks, n_neighbors=n_neighbors)
    return LinkabilityIndexes(idx_0=idx_0, idx_1=idx_1).find_links()

class LinkabilityEvaluator(BaseModel):
    """Measure the linkability risk created by a synthetic dataset.

    The linkability risk is measured from the success of a linkability attack.
    The attack is modeled along the following scenario. The attacker posesses
    two datasets, both of which share some columns with the *original* dataset
    that was used to generate the synthetic data. Those columns will be
    referred to as *auxiliary columns*. The attacker's aim is then to use the
    information contained in the synthetic data to connect these two datasets,
    i.e. to find records that belong to the same individual.

    To model this attack, the original dataset is split vertically into two
    parts. Then we try to reconnect the two parts using the synthetic data
    by looking for the closest neighbors of the split original records in
    the synthetic data. If both splits of an original record have the same
    closest synthetic neighbor, they are linked together. The more original
    records get relinked in this manner the more successful the attack.

    A linkability risk of 1 means that every single attacked record
    could be successfully linked together. A linkability risk of 0
    means that no links were found at all.

    Parameters
    ----------
    ori : pd.DataFrame
        Dataframe containing original data.
    syn : pd.DataFrame
        Dataframe containing synthetic data.
        It has to have the same columns as df_ori.
    aux_cols : tuple of two lists of strings
        Features (columns) of data that are given to the attacker as auxiliary information.
        First/second tuple represents first/second original dataset.
    n_attacks : int, default is min(2_000, ori.shape[0]).
        Number of records to attack.
    confidence_level : float, default is 0.95
        Confidence level for the error bound calculation.
    n_neighbors : int, default is 1
        The number of closest neighbors to include in the main attack for linking.
        The default of 1 means that the linkability attack is considered
        successful only if the two original record split have the same
        synthetic record as closest neighbor.
    n_jobs : int, default is -2
        The number of parallel jobs to run for neighbors search.
    main_links: LinkabilityIndexes, optional
        LinkabilityIndexes object holding main attack links.
        Parameter will be set in evaluate method.
    naive_links: LinkabilityIndexes, optional
        LinkabilityIndexes object holding naive attack links.
        Parameter will be set in evaluate method.
    results: EvaluationResults, optional
        EvaluationResults object containing the success rates for the various attacks.
        Parameter will be set in evaluate method.

    """

    model_config = ConfigDict(arbitrary_types_allowed = True)
    ori: pd.DataFrame
    syn: pd.DataFrame
    aux_cols: Tuple[List[str], List[str]]
    n_attacks: int = 2_000
    confidence_level: float = 0.95
    n_neighbors: int = 1
    n_jobs: int = -2
    #Following parameters are set in evaluate method
    main_links: Optional[LinkabilityIndexes] = None
    naive_links: Optional[LinkabilityIndexes] = None
    results: Optional[EvaluationResults] = None

    def __init__(self: Self, **kwargs: pd.DataFrame) -> None:
        super().__init__(**kwargs)
        #Assert input values
        if self.ori.shape[0]==0 or self.syn.shape[0]==0:
            raise ValueError("ori and syn must contain rows.")
        if list(self.ori.columns) != list(self.syn.columns):
            raise ValueError("ori and syn columns must be equal.")
        if len(self.aux_cols[0])==0 or len(self.aux_cols[1])==0:
            raise ValueError("aux_cols tuple must contain 2 list with at least 1 element.")
        assert_x_in_bound(x=self.confidence_level, x_name="confidence_level")
        self.n_attacks = min(self.n_attacks, self.ori.shape[0])
        self.n_neighbors = min(self.n_neighbors, self.ori.shape[1])

    def evaluate(self: Self) -> EvaluationResults:
        """Run the linkability attacks (main and naive) and set and return results."""
        # Main linkability attack
        self.main_links = main_linkability_attack(
            ori=self.ori,
            syn=self.syn,
            n_attacks=self.n_attacks,
            aux_cols=self.aux_cols,
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
        )
        # Naive linkability attack
        self.naive_links = naive_linkability_attack(
            n_synthetic=self.syn.shape[0],
            n_attacks=self.n_attacks,
            n_neighbors=self.n_neighbors
        )
        # Set results
        self.results = EvaluationResults(
            n_total = self.n_attacks,
            n_main = self.main_links.count,
            n_naive = self.naive_links.count,
            confidence_level = self.confidence_level
        )
        return self.results
