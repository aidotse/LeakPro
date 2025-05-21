"""Contains the Result classes for MIA, MiNVA, and GIA attacks."""

import json
import os

from torch import Tensor, clamp, stack
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image

from leakpro.reporting.report_utils import get_config_name, reduce_to_unique_labels
from leakpro.utils.import_helper import Any, List, Self


class GIAResults:
    """Contains results for a GIA attack."""

    def __init__(
            self: Self,
            original_data: DataLoader = None,
            recreated_data: DataLoader = None,
            psnr_score: float = None,
            ssim_score: float = None,
            data_mean: float = None,
            data_std: float = None,
            config: dict = None,
        ) -> None:

        self.original_data = original_data
        self.recreated_data = recreated_data
        self.PSNR_score = psnr_score
        self.SSIM_score = ssim_score
        self.data_mean = data_mean
        self.data_std = data_std
        self.config = config

    @staticmethod
    def load(
            data:dict
        ) -> None:
        """Load the GIAResults from disk."""

        giaresult = GIAResults()

        giaresult.original = data["original"]
        giaresult.resulttype = data["resulttype"]
        giaresult.recreated = data["recreated"]
        giaresult.id = data["id"]
        giaresult.result_config = data["result_config"]

        return giaresult

    def save(
            self: Self,
            name: str,
            path: str,
            config: dict,
            show_plot: bool = False # noqa: ARG002
        ) -> None:
        """Save the GIAResults to disk."""

        def get_gia_config(instance: Any, skip_keys: List[str] = None) -> dict:
            """Extract manually typed variables and their values from a class instance with options to skip keys."""
            if skip_keys is None:
                skip_keys = []

            cls_annotations = instance.__class__.__annotations__  # Get typed attributes
            return {
                var: getattr(instance, var)
                for var in cls_annotations
                if var not in skip_keys  # Exclude skipped keys
            }

        result_config = get_gia_config(config, skip_keys=["optimizer", "criterion"])

        # Get the name for the attack configuration
        config_name = get_config_name(result_config)
        self.id = f"{name}{config_name}"
        path = f"{path}/gradient_inversion/{self.id}"

        # Check if path exists, otherwise create it.
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")

        def extract_tensors_from_subset(dataset: Dataset) -> Tensor:
            all_tensors = []
            if isinstance(dataset, Subset):
                for idx in dataset.indices:
                    all_tensors.append(dataset.dataset[idx][0])

            else:
                for idx in range(len(dataset)):
                    all_tensors.append(dataset[idx][0])
            return stack(all_tensors)

        recreated_data = extract_tensors_from_subset(self.recreated_data.dataset)
        original_data = extract_tensors_from_subset(self.original_data.dataset)

        output_denormalized = clamp(recreated_data * self.data_std + self.data_mean, 0, 1)
        recreated = os.path.join(path, "recreated_image.png")
        save_image(output_denormalized, recreated)

        gt_denormalized = clamp(original_data * self.data_std + self.data_mean, 0, 1)
        original = os.path.join(path, "original_image.png")
        save_image(gt_denormalized, original)

        # Data to be saved
        data = {
            "resulttype": self.__class__.__name__,
            "original": original,
            "recreated": recreated,
            "result_config": result_config,
            "id": self.id,
        }

        # Save the results to a file
        with open(f"{path}/data.json", "w") as f:
            json.dump(data, f)

    @staticmethod
    def create_results(
            results: list,
            save_dir: str = "./", # noqa: ARG004
            save_name: str = "foo", # noqa: ARG004
        ) -> str:
        """Result method for GIA."""
        latex = ""
        def _latex(
                save_name: str,
                original: str,
                recreated: str
            ) -> str:
            """Latex method for GIAResults."""
            return f"""
            \\subsection{{{" ".join(save_name.split("_"))}}}
            \\begin{{figure}}[ht]
            \\includegraphics[width=0.6\\textwidth]{{{original}}}
            \\caption{{Original}}
            \\end{{figure}}

            \\begin{{figure}}[ht]
            \\includegraphics[width=0.6\\textwidth]{{{recreated}}}
            \\caption{{Recreated}}
            \\end{{figure}}
            """
        unique_names = reduce_to_unique_labels(results)
        for res, name in zip(results, unique_names):
            latex += _latex(save_name=name, original=res.original, recreated=res.recreated)
        return latex
