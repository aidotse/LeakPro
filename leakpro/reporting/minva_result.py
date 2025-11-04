"""Contains the MinvResult class."""

import json
import os

import numpy as np
from leakpro.schemas import MinvResultSchema
from leakpro.utils.import_helper import Any, Self, Union
from leakpro.attacks.utils.diff_mi.setup import DiffMiConfig

class MinvResult:
    """Contains results for a MI attack."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG002
        raise RuntimeError(
            "Use one of the constructors: \"from_metrics\""
            ""
        )

    def _init_obj(self: Self, result_name: str, result_id: str) -> None:
        """Initialize the MinvResult object.

        Args:
        ----
            result_name (str): Name of the result.
            result_id (str): Unique identifier for the result.

        Returns:
        -------
            None
        """
        self.name = result_name
        self.id = result_id

    @classmethod
    def from_metrics(cls,
                     result_name: str,
                     result_id: str,
                     config: DiffMiConfig,
                     metric_dict: dict) -> Self:
        """Create MinvResult from evaluation metrics.

        Args:
        ----
            result_name (str): Name of the result.
            result_id (str): Unique identifier for the result.
            config (DiffMiConfig): Configuration used for the attack.
            metric_dict (dict): Dictionary containing the evaluation metrics.

        Returns:
        -------
            MinvResult: An instance of MinvResult with the evaluation metrics.

        """
        if metadata is None:
            metadata = {}
        obj = object.__new__(cls)
        obj._init_obj(result_name, result_id)
        obj.result = obj._make_result_object(config, metric_dict)
        return obj

    def _make_result_object(self, config: DiffMiConfig, metric_dict: dict) -> MinvResultSchema:
        """Create result metric object for top1 accuracy, top5 accuracy, and KNN."""
        return MinvResultSchema(
            result_name=self.name,
            result_id=self.id,
            config=config.model_dump(),
            metrics=metric_dict,
        )

    def _create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

    def save(self:Self, attack_obj: Any, output_dir: str) -> None:
        """Save the MinVResult to disk.
        Args.
        ----
            attack_obj: The attack object associated with this result.
            output_dir (str): The directory to save the results to.
        Returns.
        --------
            None.            
        """
        save_path = f"{output_dir}/results/{self.id}"
        self._create_dir(save_path)

        # Create directory for saving data objects
        data_obj_storage = f"{output_dir}/data_objects/"
        self._create_dir(data_obj_storage)

        def json_fallback(obj: Any) -> Any:
            """Fallback function for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Save the results to a file in data objects using attack hash
        with open(f"{data_obj_storage}{attack_obj.attack_id}.json", "w") as f:
            json.dump(self.result.model_dump(), f, default=json_fallback)

        # Store results for user output
        with open(f"{save_path}/result.txt", "w") as f:
            f.write(str(self.result.model_dump()))
