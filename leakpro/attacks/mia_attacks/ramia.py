"""Implementation of the RaMIA attack on top of LiRA attack."""

from typing import Literal, Callable, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm

import optuna
import os
import torch
import pickle
from leakpro.schemas import OptunaConfig, avg_tpr_at_low_fpr
from leakpro.input_handler.modality_extensions.image_extension import ImageExtension
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.boosting import Memorization
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.metrics.attack_result import MIAResult
from leakpro.signals.signal import ModelRescaledLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

# Define the TransformedDataset class at module level so it can be pickled
class TransformedDataset:
    """Dataset class for transformed samples that works with any image format."""
    
    def __init__(self, samples, labels=None):
        """Initialize with samples and optional labels.
        
        Args:
            samples: List of transformed samples (tensors or arrays)
            labels: Optional labels for the samples
        """
        self.data = samples  # Store directly without conversion
        self.targets = labels if labels is not None else np.zeros(len(samples), dtype=np.int64)
        
        # Detect what type of data we're dealing with
        sample = samples[0] if samples else None
        self._is_tensor = isinstance(sample, torch.Tensor)
        self._is_numpy = isinstance(sample, np.ndarray)
        
        # Store the data format characteristics
        if self._is_tensor:
            self._channels_first = (sample.dim() == 3 and sample.shape[0] in [1, 3, 4])
        elif self._is_numpy:
            self._channels_first = (sample.ndim == 3 and sample.shape[0] in [1, 3, 4])
        else:
            self._channels_first = False
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retrieve the sample and its label at index 'idx'.
        
        This method handles different data formats that might be used by various handlers.
        """
        sample = self.data[idx]
        label = self.targets[idx]
        
        return sample, label
    
    def get_data_format(self):
        """Return information about the data format for debugging."""
        return {
            "is_tensor": self._is_tensor,
            "is_numpy": self._is_numpy,
            "channels_first": self._channels_first,
            "sample_shape": self.data[0].shape if len(self.data) > 0 else None
        }
    
    def subset(self, indices):
        """Return a subset of the dataset based on the given indices."""
        # Handle different data formats
        if isinstance(self.data, torch.Tensor):
            new_samples = self.data[indices]
        elif isinstance(self.data, np.ndarray):
            new_samples = self.data[indices]
        else:  # Handle lists
            new_samples = [self.data[i] for i in indices]
            
        new_labels = self.targets[indices] if self.targets is not None else None
        
        return TransformedDataset(
            samples=new_samples,
            labels=new_labels
        )
    
class AttackRaMIA(AbstractMIA):
    """Implementation of the RaMIA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the RaMIA attack."""

        # RaMIA attack parameters
        num_transforms: int = Field(default=15, ge=1, le=100, description="Number of transformations to apply to each sample in a range") # Range transformation parameters
        range_num_audit_samples: int = Field(default=50, ge=5, description="Number of audit samples to use for range membership inference") # Range transformation parameters
        online: bool = Field(default=False, description="Online vs offline attack") # Attack type configuration
        transform_type: Literal["standard", "cifar", "custom"] = Field(default="cifar", description="Type of transformation to apply to images") # Image transformation configuration
        dataset_type: Literal["image", "tabular", "text"] = Field(default="image", description="Type of dataset being analyzed") # Dataset type configuration
        optuna_config: OptunaConfig = Field(default=OptunaConfig()) # Optuna configuration for hyperparameter search
        objective: Optional[Callable[[MIAResult], float]] = Field(default=None, description="Objective function for optimization")

        num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
        eval_batch_size: int = Field(default=32, ge=1, description="Batch size for evaluation")
        var_calculation: Literal["carlini", "individual_carlini", "fixed"] = Field(default="carlini", description="Variance estimation method to use [carlini, individual_carlini, fixed]")  # noqa: E501
        # memorization boosting
        memorization: bool = Field(default=False, description="Activate memorization boosting")
        use_privacy_score: bool = Field(default=False, description="Filter based on privacy score aswell as memorization score")
        memorization_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Set percentile for most vulnerable data points, use 0.0 for paper thresholds")  # noqa: E501
        min_num_memorization_audit_points: int = Field(default=10, ge=1, description="Set minimum allowed audit points after memorization")  # noqa: E501
        num_memorization_audit_points: int = Field(default=0, ge=0, description="Directly set number of most vulnerable audit data points (Overrides 'memorization_threshold')")  # noqa: E501

        @model_validator(mode="after")
        def check_num_shadow_models_if_online(self) -> Self:
            """Check if the number of shadow models is at least 2 when online is True.

            Returns
            -------
                Config: The attack configuration.

            Raises
            ------
                ValueError: If online is True and the number of shadow models is less than 2.

            """
            if self.online and self.num_shadow_models < 2:
                raise ValueError("When online is True, num_shadow_models must be >= 2")
            return self
        
        @model_validator(mode='after')
        def handle_objective(self) -> Self:
            """Set default objective if not provided."""
            if self.objective is None:
                self.objective = avg_tpr_at_low_fpr
            return self

    def __init__(self:Self,
                 handler: MIAHandler,
                 configs: dict
                 ) -> None:
        """Initialize the RaMIA attack.

        Args:
        ----
            handler (MIAHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)

        # Initializes the parent metric
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        if self.online is False and self.population_size == self.audit_size:
            raise ValueError("The audit dataset is the same size as the population dataset. \
                    There is no data left for the shadow models.")

        self.shadow_models = []
        self.signal = ModelRescaledLogits()

        # Initialize image extension for transformations if using image data
        if self.dataset_type == "image":
            self.image_extension = ImageExtension(handler)
            transform_config = configs.get("transform_config", {})
            self.transforms = self.image_extension.get_transform(self.transform_type, **transform_config)
        else:
            # For non-image data, we'll initialize transforms in prepare_attack
            self.transforms = None
            logger.warning(f"Using dataset type '{self.dataset_type}', custom transformations may be needed")

        # Set default objective function if not provided
        if self.configs.objective is None:
            self.objective = avg_tpr_at_low_fpr
        else:
            self.objective = self.configs.objective

    def description(self: Self) -> dict:
        """Return a description of the attack for documentation and reporting.
        
        Returns:
            dict: Contains title, reference, summary, and detailed description
        """
        title_str = "Range Membership Inference Attack"
        
        reference_str = "Tao J, Shokri R. Range Membership Inference Attacks"
        
        summary_str = ("RaMIA extends membership inference to test if a range of data points contains "
                       "any training samples, providing more comprehensive privacy auditing.")
        
        detailed_str = (
            "Range Membership Inference Attack (RaMIA) is designed to detect privacy leakage "
            "beyond exact matches of training data. It works by checking if a range of points "
            "(defined by transformations of a center point) contains any training data. "
            "This better captures privacy risks since similar data points often contain similar "
            "private information. RaMIA aggregates membership scores over transformed samples "
            "using a trimmed average to reduce the impact of outliers and improve attack performance."
        )
        
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str
        }
    
    def create_transformed_dataset(self: Self, num_samples: int) -> tuple:
        """Generate transformed ranges for selected audit samples, balanced between IN and OUT members.
        
        Args:
            num_samples: Number of audit samples to select for range creation
            
        Returns:
            Tuple containing:
                - transformed_samples: List of lists of transformed samples
                - true_labels: List of binary labels (1=IN, 0=OUT)
                - selected_indices: Indices from original audit dataset that were selected
        """

        # Create a boolean mask for IN members
        is_in = np.zeros(len(self.audit_data_indices), dtype=bool)
        is_in[self.in_members] = True  # Mark IN members

        # Get indices of IN and OUT members within audit_data
        in_indices = np.where(is_in)[0]
        out_indices = np.where(~is_in)[0]

        # Calculate natural class distribution
        total_audit_samples = len(self.audit_data_indices)
        in_ratio = len(in_indices) / total_audit_samples
        out_ratio = 1 - in_ratio

        # Calculate sample sizes based on natural distribution
        num_in = int(num_samples * in_ratio)
        num_out = num_samples - num_in

        # Ensure we don't exceed available samples
        num_in = min(num_in, len(in_indices))
        num_out = min(num_out, len(out_indices))

        # Adjust if total samples < requested
        if (num_in + num_out) < num_samples:
            shortfall = num_samples - (num_in + num_out)
            # Distribute remaining samples proportionally
            add_in = min(shortfall, len(in_indices) - num_in)
            add_out = min(shortfall - add_in, len(out_indices) - num_out)
            num_in += add_in
            num_out += add_out
            logger.warning(f"Adjusted sample sizes to {num_in} IN and {num_out} OUT")


        # Randomly select indices from IN and OUT members
        self.selected_in = np.random.choice(in_indices, num_in, replace=False)
        self.selected_out = np.random.choice(out_indices, num_out, replace=False)
        self.selected_indices = np.concatenate([self.selected_in, self.selected_out])
        np.random.shuffle(self.selected_indices)  # Shuffle to mix IN/OUT

        # Validation checks
        unique_indices = np.unique(self.selected_indices)
        if len(unique_indices) != len(self.selected_indices):
            raise ValueError("Duplicate indices selected in audit samples")

        # Get dataset from handler
        dataset = self.handler.population

        transformed_samples = []
        true_labels = []

        # Process each selected index
        for idx in tqdm(self.selected_indices, desc="Creating transformed ranges"):
            data_idx = self.audit_data_indices[idx]
            
            # Get image using the method from ImageExtension
            raw_sample = self.image_extension.get_data(dataset, data_idx)
            
            # Skip if we couldn't get the data
            if raw_sample is None:
                logger.warning(f"Failed to get data for index {data_idx}")
                continue
                
            # Convert to PIL image using the existing method
            sample = self.image_extension.to_pil_image(raw_sample)
            
            # Skip if conversion failed
            if sample is None:
                logger.warning(f"Failed to convert sample to PIL image for index {data_idx}")
                continue

            # Generate transformed samples
            try:
                range_samples = []
                for transform_idx in range(self.num_transforms):
                    transformed = self.transforms(sample, transform_idx)
                    range_samples.append(transformed)

                transformed_samples.append(range_samples)
                true_labels.append(int(is_in[idx]))
                logger.info(f"Successfully created transforms for index {idx}")
            except Exception as e:
                logger.error(f"Error transforming sample {idx}: {e}")
                continue

        logger.info(f"Created {len(transformed_samples)} ranges "
                    f"({sum(true_labels)} IN, {len(true_labels)-sum(true_labels)} OUT)")
        
        return transformed_samples, true_labels, self.selected_indices

    def create_transformed_dataset_pkl(self: Self, transformed_samples, selected_indices, tmp_dir="./tmp"):
        """Create a generic dataset with transformed samples that works with any image dataset type.
        
        This function creates a dataset that maintains the same interface expected by the handler
        but can work with any type of image data, not just CIFAR.
        
        Args:
            transformed_samples: List of lists of transformed tensors
            selected_indices: Indices of original selected samples
            tmp_dir: Directory to save the temporary dataset
            
        Returns:
            tuple: (path to saved dataset, path to metadata, number of samples)
        """
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Flatten all transformed samples into a single list
        flattened_samples = []
        sample_to_range_map = []
        range_to_samples_map = {}
        
        for range_idx, range_samples in enumerate(transformed_samples):
            range_to_samples_map[range_idx] = []
            for sample in range_samples:
                sample_idx = len(flattened_samples)
                flattened_samples.append(sample)
                sample_to_range_map.append(range_idx)
                range_to_samples_map[range_idx].append(sample_idx)
        
        # Create the dataset directly from our transformed samples
        # We keep them as tensors to preserve their format
        transformed_dataset = TransformedDataset(flattened_samples)
        
        # Log dataset format information for debugging
        format_info = transformed_dataset.get_data_format()
        logger.info(f"Created transformed dataset with format: {format_info}")
        
        # Save the dataset
        transformed_dataset_path = os.path.join(tmp_dir, "ramia_transformed_dataset.pkl")
        with open(transformed_dataset_path, "wb") as f:
            pickle.dump(transformed_dataset, f)
        
        logger.info(f"Saved transformed dataset with {len(flattened_samples)} samples to {transformed_dataset_path}")
        
        # Save metadata for mapping samples to ranges
        metadata = {
            "sample_to_range": sample_to_range_map,
            "range_to_samples": range_to_samples_map,
            "selected_indices": selected_indices,
            "range_membership": {i: self.range_labels[i] for i in range(len(self.range_labels))} # added to check
        }
        
        metadata_path = os.path.join(tmp_dir, "ramia_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved RaMIA metadata to {metadata_path}")
        
        return transformed_dataset_path, metadata_path, len(flattened_samples)

    def prepare_attack(self:Self)->None:
        """Prepares data to obtain metric on the target model and dataset, using signals computed on the auxiliary model/dataset.

        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and computes the logits
        for both shadow models and the target model.
        """

        # Fixed variance is used when the number of shadow models is below 32 (64, IN and OUT models)
        #       from (Membership Inference Attacks From First Principles)
        self.fix_var_threshold = 32

        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_shadow_models,
                                                                              shadow_population =  self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              online = self.online)

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        logger.info("Create masks for all IN and OUT samples")
        self.in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"])

        self.audit_data_indices = self.audit_dataset["data"]
        self.in_members = self.audit_dataset["in_members"]
        self.out_members = self.audit_dataset["out_members"]

        # Check offline attack for possible IN- sample(s)
        if not self.online:
            count_in_samples = np.count_nonzero(self.in_indices_masks)
            if count_in_samples > 0:
                logger.info(f"Some shadow model(s) contains {count_in_samples} IN samples in total for the model(s)")
                logger.info("Potential contamination in offline attack!")

        # Create a mapping from data indices to positions in the audit dataset
        # This helps us correctly map back when calculating scores
        audit_data = self.audit_dataset["data"]
        self.audit_data_index_map = {data_idx: pos for pos, data_idx in enumerate(audit_data)} # maps every data in the audit data to a position for easier tracking.
        logger.info(f"Created mapping for {len(self.audit_data_index_map)} audit data indices")

        # Generate transformed ranges
        num_samples = self.range_num_audit_samples
        self.transformed_ranges, self.range_labels, self.selected_indices = \
            self.create_transformed_dataset(num_samples)
        
        # Validate results
        if len(self.transformed_ranges) == 0:
            raise RuntimeError("No ranges generated")
        if sum(self.range_labels) == 0:
            raise ValueError("No IN members in ranges")
        
        logger.info(f"Prepared {len(self.transformed_ranges)} ranges "
                    f"({sum(self.range_labels)} IN, {len(self.range_labels)-sum(self.range_labels)} OUT)")
        
        # Store original population dataset
        self.original_population = self.handler.population
            
        # Create a generic dataset of all transformed samples
        self.transformed_dataset_path, self.metadata_path, self.num_transformed_samples = \
            self.create_transformed_dataset_pkl(self.transformed_ranges, self.selected_indices) 
        
        # Load the transformed dataset into memory
        with open(self.transformed_dataset_path, "rb") as f:
            self.transformed_dataset = pickle.load(f)

        # Set the transformed dataset as the handler's population
        self.handler.population = self.transformed_dataset
        
        # Load metadata for mapping between samples and ranges
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        # Create indices for all transformed samples
        self.transformed_indices = np.arange(self.num_transformed_samples)
        

        logger.info(f"Calculating the logits for all {self.num_shadow_models} shadow models")
        self.shadow_models_logits = np.swapaxes(self.signal(self.shadow_models,
                                                            self.handler,
                                                            self.transformed_indices,
                                                            self.eval_batch_size), 0, 1)

        # Calculate logits for the target model
        logger.info("Calculating the logits for the target model")
        self.target_logits = np.swapaxes(self.signal([self.target_model],
                                                     self.handler,
                                                     self.transformed_indices,
                                                     self.eval_batch_size), 0, 1).squeeze()
        

        # Cleanup (call this AFTER attack completes)
        self.cleanup()


    # After attack completion, restore the original handler state
    def cleanup(self):
        """Restore original handler state"""
        self.handler.population = self.original_population
        logger.info("Restored original handler state")

    def get_std(self:Self, logits: list, mask: list, is_in: bool, var_calculation: str) -> np.ndarray:
        """A function to define what method to use for calculating variance for LiRA."""

        # Fixed/Global variance calculation.
        if var_calculation == "fixed":
            return self._fixed_variance(logits, mask, is_in)

        # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        if var_calculation == "carlini":
            return self._carlini_variance(logits, mask, is_in)

        # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        #   but check IN and OUT samples individualy
        if var_calculation == "individual_carlini":
            return self._individual_carlini(logits, mask, is_in)

        return np.array([None])

    def _fixed_variance(self:Self, logits: list, mask: list, is_in: bool) -> np.ndarray:
        if is_in and not self.online:
            return np.array([None])
        return np.std(logits[mask])

    def _carlini_variance(self:Self, logits: list, mask: list, is_in: bool) -> np.ndarray:
        if self.num_shadow_models >= self.fix_var_threshold*2:
                return np.std(logits[mask])
        if is_in:
            return self.fixed_in_std
        return self.fixed_out_std

    def _individual_carlini(self:Self, logits: list, mask: list, is_in: bool) -> np.ndarray:
        if np.count_nonzero(mask) >= self.fix_var_threshold:
            return np.std(logits[mask])
        if is_in:
            return self.fixed_in_std
        return self.fixed_out_std
    
    def prepare_transformed_masks(self: Self) -> None:
        # Get the original sample's IN/OUT masks for each transformed sample
        self.sample_to_range = self.metadata["sample_to_range"]
        
        self.transformed_masks = []
        for idx in range(len(self.transformed_indices)):
            range_idx = self.sample_to_range[idx]
            pos_in_selected = range_idx  # Position in the selected_indices array #Check this later(pos_in_selected)
            
            if pos_in_selected >= len(self.selected_indices):
                logger.warning(f"Index out of bounds: {pos_in_selected} >= {len(self.selected_indices)}")
                # Use a default mask (assume all OUT)
                self.transformed_masks.append(np.zeros(len(self.shadow_models), dtype=bool))
                continue
                
            pos_in_audit = self.selected_indices[pos_in_selected]
            
            # Get masks from LiRA's in_indices_masks
            if pos_in_audit < len(self.in_indices_masks):
                self.transformed_masks.append(self.in_indices_masks[pos_in_audit])
            else:
                logger.warning(f"Audit position out of bounds: {pos_in_audit} >= {len(self.in_indices_masks)}")
                # Use a default mask (assume all OUT)
                self.transformed_masks.append(np.zeros(len(self.shadow_models), dtype=bool))
        
        self.transformed_masks = np.array(self.transformed_masks)
    
    def calculate_scores_for_transformed_samples(self: Self): 
        # Calculate membership scores for each transformed sample
        sample_scores = np.zeros(len(self.shadow_models_logits))

        # Iterate over and extract logits for IN and OUT shadow models for each audit sample
        for i, (shadow_models_logits, mask) in tqdm(enumerate(zip(self.shadow_models_logits, self.transformed_masks)),
                                                    total=len(self.shadow_models_logits),
                                                    desc="Processing audit samples"):

            # Calculate the mean for OUT shadow model logits
            out_mean = np.mean(shadow_models_logits[~mask])
            out_std = self.get_std(shadow_models_logits, ~mask, False, self.var_calculation)

            # Get the logit from the target model for the current sample
            target_logit = self.target_logits[i]

            # Calculate the log probability density function value
            pr_out = -norm.logpdf(target_logit, out_mean, out_std + 1e-30)

            pr_in = 0

            sample_scores[i] = (pr_in - pr_out)  # Append the calculated probability density value to the score list
            
            if np.isnan(sample_scores[i]):
                raise ValueError("Score is NaN")
            
        return sample_scores
    
    # def calculate_scores_for_transformed_samples(self: Self): 
    #     # Calculate membership scores for each transformed sample
    #     sample_scores = np.zeros(len(self.shadow_models_logits))

    #     # Calculate global statistics for better estimation
    #     all_shadow_logits = self.shadow_models_logits.flatten()
    #     all_transformed_masks = self.transformed_masks.flatten()

    #     # Global OUT statistics (from all shadow models)
    #     # Provides a macro-level view of how non-member samples typically behave across models.
    #     global_out_mean = np.mean(all_shadow_logits[~all_transformed_masks])
    #     global_out_std = np.std(all_shadow_logits[~all_transformed_masks])
    #     global_out_std = max(global_out_std, 1e-30)  # Ensure non-zero variance

    #     logger.info(f"Global OUT distribution: mean={global_out_mean:.4f}, std={global_out_std:.4f}")

    #     # Iterate over and extract logits for IN and OUT shadow models for each audit sample
    #     for i, (shadow_models_logits, mask) in tqdm(enumerate(zip(self.shadow_models_logits, self.transformed_masks)),
    #                                                 total=len(self.shadow_models_logits),
    #                                                 desc="Processing audit samples"):


    #         # Calculate the mean for OUT shadow model logits
    #         out_mean = np.mean(shadow_models_logits[~mask])
    #         out_std = self.get_std(shadow_models_logits, ~mask, False, self.var_calculation)

    #         # Get the logit from the target model for the current sample
    #         target_logit = self.target_logits[i]

    #         # Reference based calibration
    #         # Calculate the target model's score
    #         target_score = -norm.logpdf(target_logit, out_mean, out_std + 1e-30)

    #         # Calculate reference scores from each shadow model
    #         ref_scores = []
    #         for j in range(len(self.shadow_models)):
    #             shadow_logit = shadow_models_logits[j]
    #             # Use the same calculation for all shadow models
    #             ref_score = -norm.logpdf(shadow_logit, out_mean, out_std + 1e-30)
    #             ref_scores.append(ref_score)

    #         # Calibrate the target score against shadow model scores
    #         # This helps distinguish true membership from model similarity effects
    #         reference_baseline = np.mean(ref_scores)

    #         # Calibrated score: higher values indicate membership
    #         sample_scores[i] = target_score - reference_baseline


    #     # Check for NaN values
    #     if np.isnan(sample_scores[i]):
    #         logger.warning(f"NaN score for sample {i}, replacing with 0")
    #         sample_scores[i] = 0
            
            
    #     return sample_scores
            
    def estimate_distribution_separation(self: Self) -> None:
        """Estimate the distributions of membership scores for IN and OUT samples.
        
        This method uses the existing transformed samples to estimate the
        statistical properties of membership scores, which will be used for
        adaptive trimming thresholds.
        """
        logger.info("Estimating score distributions for adaptive trimming")
        
        # Get masks that tell us which samples come from members vs non-members
        #self.sample_to_range = self.metadata["sample_to_range"]
        
        in_scores = []
        out_scores = []
        
        # Separate scores into IN and OUT distributions based on the original samples
        for idx, score in enumerate(self.all_sample_scores):
            range_idx = self.sample_to_range[idx]
            if range_idx in self.metadata["range_membership"]:
                is_member = self.metadata["range_membership"][range_idx]
                if is_member:
                    in_scores.append(score)
                else:
                    out_scores.append(score)
        
        # Calculate distribution parameters
        self.in_mean_est = np.mean(in_scores) if in_scores else 0
        self.in_std_est = np.std(in_scores) if in_scores else 1
        self.out_mean_est = np.mean(out_scores) if out_scores else 0
        self.out_std_est = np.std(out_scores) if out_scores else 1
        
        logger.info(f"Estimated IN distribution: N({self.in_mean_est:.4f}, {self.in_std_est:.4f})")
        logger.info(f"Estimated OUT distribution: N({self.out_mean_est:.4f}, {self.out_std_est:.4f})")

    def optimize_p(self: Self) -> None:
        """Optimize the p parameter for adaptive trimming using Optuna.
        
        This parameter controls the proportion of the OUT distribution that
        determines the adaptive trimming threshold.
        """
        logger.info("Starting optimization of p parameter")
        
        # Create our objective function for Optuna
        def objective(trial):
            # Sample a value for p (the probability parameter)
            # This is the 'p' in P[LLR(x) > qs] = p for x~out
            p = trial.suggest_float("p", 0.01, 0.75)
            
            # Set the current p value
            temp_p = p
            
            # Compute range scores with this p value
            range_scores = []
            for range_idx, sample_indices in self.metadata["range_to_samples"].items():
                range_sample_scores = self.all_sample_scores[sample_indices]
                # Calculate qs based on OUT distribution and current p value
                qs = norm.ppf(1.0 - temp_p, loc=self.out_mean_est, scale=self.out_std_est)
                # Apply trimming
                sorted_scores = np.sort(range_sample_scores)
                trimmed = sorted_scores[sorted_scores <= qs]
                score = np.mean(trimmed) if len(trimmed) > 0 else 0.0
                range_scores.append(score)
            
            # Convert to numpy array
            range_scores = np.array(range_scores)
            
            # Generate simplified decision thresholds
            min_score = np.min(range_scores)
            max_score = np.max(range_scores)
            thresholds = np.linspace(min_score, max_score, 1000)
            
            # Prepare binary predictions matrix
            predictions = (range_scores.reshape(-1, 1) < thresholds.reshape(1, -1))
            
            # Create temporary MIAResult for evaluation
            result = MIAResult(
                predicted_labels=predictions.T,
                true_labels=np.array(self.range_labels, dtype=np.int32),
                predictions_proba=None,
                signal_values=range_scores,
                audit_indices=np.array(self.selected_indices)
            )
            
            # Return the objective value
            return self.objective(result)
        
        # Create and run the Optuna study
        study = optuna.create_study(
            direction=self.configs.optuna_config.direction, # Whether to maximize (e.g., higher TPR) or minimize (e.g., lower FPR)
            pruner=self.configs.optuna_config.pruner, # Kills underperforming trials early (saves compute)
            sampler=optuna.samplers.TPESampler(seed=self.configs.optuna_config.seed) # Tree-structured Parzen Estimator: Smart algorithm to suggest better p values over time
        )
        
        study.optimize(
            objective,
            n_trials=self.configs.optuna_config.n_trials
        )
        
        # Set the best p value found
        self.best_p = study.best_params["p"]
        logger.info(f"Best p parameter: {self.best_p:.4f}")

    def compute_adaptive_trimmed_average(self: Self, scores: np.ndarray) -> float:
        """Compute adaptively trimmed average of membership scores.
        
        This method uses the estimated OUT distribution and the optimized p parameter
        to determine the appropriate trimming threshold (qs). It implements the 
        P[LLR(x) > qs] = p approach shown in the handwritten notes.
        
        Args:
            scores: Array of membership scores for samples in a range
            
        Returns:
            float: Adaptively trimmed average score
        """
        n = len(scores)
        if n == 0:
            return 0.0
        
        # Calculate quantile threshold qs based on OUT distribution and p
        # For a given p (probability), find the qs such that P[LLR(x) > qs] = p for x~out
        # This means qs is the (1-p) quantile of the OUT distribution
        qs = norm.ppf(1.0 - self.best_p, loc=self.out_mean_est, scale=self.out_std_est)
        logger.info(f"Using adaptive trimming with p={self.best_p:.4f}, qs={qs:.4f}")
        
        # Sort scores and keep only those below qs
        sorted_scores = np.sort(scores)
        trimmed = sorted_scores[sorted_scores <= qs]
        
        # Return mean of trimmed scores
        if len(trimmed) == 0:
            # If all scores are above qs, just return the smallest score
            return sorted_scores[0] if len(sorted_scores) > 0 else 0.0
        
        return np.mean(trimmed)

    def run_attack(self:Self) -> MIAResult:
        """Runs the attack on the target model and dataset and assess privacy risks or data leakage.

        This method evaluates how the target model's output (logits) for a specific dataset
        compares to the output of shadow models to determine if the dataset was part of the
        model's training data or not.

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and signal values.

        """
        # n_audit_samples = self.shadow_models_logits.shape[0]
        # score = np.zeros(n_audit_samples)  # List to hold the computed probability scores for each sample

        self.prepare_transformed_masks()

        #self.fixed_in_std = self.get_std(self.shadow_models_logits.flatten(), self.transformed_masks.flatten(), True, "fixed")
        self.fixed_out_std = self.get_std(self.shadow_models_logits.flatten(), (~self.transformed_masks).flatten(), False, "fixed")

        # Calculate all membership scores for transformed samples
        self.all_sample_scores = self.calculate_scores_for_transformed_samples()

        self.estimate_distribution_separation() 
        self.optimize_p()

        range_scores = []
        for range_idx, sample_indices in self.metadata["range_to_samples"].items():
            range_sample_scores = self.all_sample_scores[sample_indices]
            trimmed_score = self.compute_adaptive_trimmed_average(range_sample_scores)
            range_scores.append(trimmed_score)
        
        # Convert to numpy array and handle NaN values
        range_scores = np.array(range_scores)

        # Generate thresholds based on range scores
        self.thresholds = np.linspace(np.min(range_scores), np.max(range_scores), 1000)  # High resolution for final evaluation

        # Create prediction matrix (num_thresholds x num_ranges)
        predictions = (range_scores.reshape(-1, 1) < self.thresholds.reshape(1, -1)).T

        # Prepare true labels (original range membership)
        true_labels = np.array(self.range_labels, dtype=np.int32)

        # Get original audit indices for result tracking
        audit_indices = np.array(self.selected_indices)

        # Create final attack result
        return MIAResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,  # Not implemented for RaMIA
            signal_values=range_scores,
            audit_indices=audit_indices,
            metadata={
                "attack_type": "RaMIA",
                "best_p": getattr(self, "best_p", None),
                "num_ranges": len(range_scores)
            }
        )


        
