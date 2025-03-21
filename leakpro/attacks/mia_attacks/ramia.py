"""Implementation of the RaMIA attack on top of LiRA attack."""

from typing import Literal, Callable, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

import optuna
import os
import torch
import pickle
import torchvision.transforms as transforms
from torch import tensor, float32, cat

# np.random.seed(1234)
# torch.manual_seed(1234)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(1234)

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
        num_transforms: int = Field(default=1, ge=1, le=100, description="Number of transformations to apply to each sample in a range") # Range transformation parameters
        range_num_audit_samples: int = Field(default=50, ge=5, description="Number of audit samples to use for range membership inference") # Range transformation parameters
        online: bool = Field(default=False, description="Online vs offline attack") # Attack type configuration
        transform_type: Literal["standard", "cifar", "custom"] = Field(default="cifar", description="Type of transformation to apply to images") # Image transformation configuration
        dataset_type: Literal["image", "tabular", "text"] = Field(default="image", description="Type of dataset being analyzed") # Dataset type configuration
        optuna_config: OptunaConfig = Field(default=OptunaConfig()) # Optuna configuration for hyperparameter search
        objective: Optional[Callable[[MIAResult], float]] = Field(default=None, description="Objective function for optimization")

        num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
        eval_batch_size: int = Field(default=32, ge=1, description="Batch size for evaluation")
        var_calculation: Literal["carlini", "individual_carlini", "fixed"] = Field(default="individual_carlini", description="Variance estimation method to use [carlini, individual_carlini, fixed]")  # noqa: E501
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
            self.range_transforms = self.image_extension.get_transform(self.transform_type, **transform_config)
        else:
            # For non-image data, we'll initialize transforms in prepare_attack
            self.range_transforms = None
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
        # Calculate natural class distribution
        total_audit_samples = len(self.audit_data_indices)
        in_ratio = len(self.in_members) / total_audit_samples

        # Calculate sample sizes based on natural distribution
        num_in = int(num_samples * in_ratio)
        num_out = num_samples - num_in

        # Ensure we don't exceed available samples
        num_in = min(num_in, len(self.in_members))
        num_out = min(num_out, len(self.out_members))

        # Adjust if total samples < requested
        if (num_in + num_out) < num_samples:
            shortfall = num_samples - (num_in + num_out)
            # Distribute remaining samples proportionally
            add_in = min(shortfall, len(self.in_members) - num_in)
            add_out = min(shortfall - add_in, len(self.in_members) - num_out)
            num_in += add_in
            num_out += add_out
            logger.warning(f"Adjusted sample sizes to {num_in} IN and {num_out} OUT")


        # Randomly select indices from IN and OUT members
        self.selected_in = np.random.choice(self.in_members, num_in, replace=False)
        self.selected_out = np.random.choice(self.out_members, num_out, replace=False)
        self.selected_indices = np.concatenate([self.selected_in, self.selected_out])
        # self.selected_indices = np.concatenate([self.in_members, self.out_members]) # Uncomment this for no transformation and comment the above three lines


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
                
            # Convert to PIL image using the existing method (need to fix this while using transformation; double normalization is performed somehow)
            sample = self.image_extension.to_pil_image(raw_sample)
            
            # Skip if conversion failed
            if sample is None:
                logger.warning(f"Failed to convert sample to PIL image for index {data_idx}")
                continue

            # Generate transformed samples
            try:
                range_samples = []
                for transform_idx in range(self.num_transforms):
                    # Apply transformations to the image
                    transformed = self.range_transforms(sample, transform_idx)
                    range_samples.append(transformed)

                    # For no transformation uncomment the below code
                    # sample = tensor(raw_sample, dtype=torch.uint8)
                    # p_sample = sample.float()
                    # p_sample /= 255.0
                    # normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    # p_sample = p_sample.permute(2, 0, 1) # <class 'torch.Tensor'>
                    # p_sample = normalize(p_sample)
                    # range_samples.append(p_sample)

                transformed_samples.append(range_samples)
                true_labels.append(1 if idx in self.in_members else 0)
                logger.info(f"Successfully created transforms for index {idx}")
            except Exception as e:
                logger.warning(f"Error transforming sample {idx}: {e}")
                continue

        logger.info(f"Created {len(transformed_samples)} ranges "
                    f"({sum(true_labels)} IN, {len(true_labels)-sum(true_labels)} OUT)")
        
        return transformed_samples, true_labels, self.selected_indices


    def create_transformed_dataset_pkl(self: Self, transformed_samples, selected_indices, true_labels, tmp_dir="./tmp"):
        """Create a generic dataset with transformed samples that works with any image dataset type.
        
        This function creates a dataset that maintains the same interface expected by the handler
        but can work with any type of image data, not just CIFAR.
        
        Args:
            transformed_samples: List of lists of transformed tensors
            selected_indices: Indices of original selected samples
            true_labels: List of binary labels (1=IN, 0=OUT) for each range
            tmp_dir: Directory to save the temporary dataset
            
        Returns:
            tuple: (path to saved dataset, path to metadata, number of samples)
        """
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Flatten all transformed samples into a single list
        flattened_samples = []
        sample_to_range_map = []
        range_to_samples_map = {}
        range_masks = []  # Moved inside to rebuild correctly
        
        # Flatten samples and create mappings
        for range_idx, range_samples in enumerate(transformed_samples):
            start = len(flattened_samples)
            flattened_samples.extend(range_samples)
            end = len(flattened_samples)
            sample_to_range_map.extend([range_idx] * len(range_samples))
            range_to_samples_map[range_idx] = list(range(start, end))
            
        # Convert to tensors and standardize format
        transformed_tensors = []
        for sample in flattened_samples:
            transformed_tensors.append(sample)
            
        # Create dataset with proper labels
        transformed_dataset = TransformedDataset(
            samples=transformed_tensors,
            labels=np.repeat(true_labels, [len(rs) for rs in transformed_samples])
        )
        
        # Save dataset
        transformed_dataset_path = os.path.join(tmp_dir, "ramia_transformed_dataset.pkl")
        with open(transformed_dataset_path, "wb") as f:
            pickle.dump(transformed_dataset, f)


        # Get mask from original audit dataset
        for range_idx, audit_idx in enumerate(selected_indices):  # Position in filtered audit dataset
            mask = self.in_indices_masks[audit_idx]  # Mask from prepare_attack()
            range_masks.append(mask)  # Append mask for this range
            
        # Prepare metadata
        metadata = {
            "sample_to_range": sample_to_range_map,
            "range_to_samples": range_to_samples_map,
            "selected_indices": selected_indices,
            "range_masks": range_masks,  # Now mask per transformed sample
            "range_membership": {i: label for i, label in enumerate(true_labels)},
            "data_format": transformed_dataset.get_data_format()
        }
        
        metadata_path = os.path.join(tmp_dir, "ramia_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
            
        return transformed_dataset_path, metadata_path, len(transformed_tensors)


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
        
        # Store original population dataset
        self.original_population = self.handler.population
        
        # Map original audit indices to subset indices
        self.original_to_subset = {
            original_idx: subset_idx
            for subset_idx, original_idx in enumerate(self.original_population.metadata["original_idx"])
        }

        # In prepare_attack after creating self.audit_data_indices:
        for idx in self.audit_dataset["data"]:
            if idx not in self.original_to_subset:
                raise ValueError(f"Audit index {idx} not found in population subset")
    
        valid_audit_indices = []
        valid_in_members = []
        valid_out_members = []

        # Track positions in the filtered audit dataset
        for pos, original_idx in enumerate(self.audit_dataset["data"]):
            if original_idx in self.original_to_subset:
                subset_idx = self.original_to_subset[original_idx]
                valid_audit_indices.append(subset_idx)
                # Check membership based on original dataset's labels
                if pos in self.audit_dataset["in_members"]:
                    valid_in_members.append(len(valid_audit_indices)-1)  # Position in filtered list
                if pos in self.audit_dataset["out_members"]:
                    valid_out_members.append(len(valid_audit_indices)-1)  # Position in filtered list
            else:
                logger.warning(f"Audit index {original_idx} not found in population. Skipping.")

        # Update audit data indices and membership
        self.audit_data_indices = np.array(valid_audit_indices)
        self.in_members = np.array(valid_in_members)
        self.out_members = np.array(valid_out_members)

        logger.info("Create masks for all IN and OUT samples")
        self.in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_data_indices)

        count_in_samples = np.count_nonzero(self.in_indices_masks)
        if count_in_samples > 0:
            logger.info(f"Some shadow model(s) contains {count_in_samples} IN samples in total for the model(s)")
            logger.info("Potential contamination in offline attack!")       

        # Generate transformed ranges
        num_samples = self.range_num_audit_samples
        # num_samples = len(self.audit_data_indices) # Uncomment this for no transformation and comment the above line
        self.transformed_ranges, self.range_labels, self.selected_indices = \
            self.create_transformed_dataset(num_samples)
        
        # Validate results
        if len(self.transformed_ranges) == 0:
            raise RuntimeError("No ranges generated")
        if sum(self.range_labels) == 0:
            raise ValueError("No IN members in ranges")
        
        logger.info(f"Prepared {len(self.transformed_ranges)} ranges "
                    f"({sum(self.range_labels)} IN, {len(self.range_labels)-sum(self.range_labels)} OUT)")
            
        # Create a generic dataset of all transformed samples
        self.transformed_dataset_path, self.metadata_path, self.num_transformed_samples = \
            self.create_transformed_dataset_pkl(self.transformed_ranges, self.selected_indices, self.range_labels) 
        
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
            
    def estimate_distribution_separation(self: Self) -> None:
        """Estimate the distributions of membership scores for IN and OUT samples.
        
        This method uses the existing transformed samples to estimate the
        statistical properties of membership scores, which will be used for
        adaptive trimming thresholds.
        """
        logger.info("Estimating score distributions for adaptive trimming")
        
        in_scores = []
        out_scores = []
        
        # Separate scores into IN and OUT distributions based on the original samples
        for idx, score in enumerate(self.all_sample_scores):
            range_idx = self.metadata["sample_to_range"][idx]
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
            p = trial.suggest_float("p", 0.01, 0.5)
            
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

    def compute_adaptive_trimmed_average(self: Self, ta_scores: np.ndarray) -> float:
        """Compute adaptively trimmed average of membership scores.
        
        This method uses the estimated OUT distribution and the optimized p parameter
        to determine the appropriate trimming threshold (qs). It implements the 
        P[LLR(x) > qs] = p approach shown in the handwritten notes.
        
        Args:
            scores: Array of membership scores for samples in a range
            
        Returns:
            float: Adaptively trimmed average score
        """
        n = len(ta_scores)
        if n == 0:
            return 0.0
        
        # Calculate quantile threshold qs based on OUT distribution and p
        # For a given p (probability), find the qs such that P[LLR(x) > qs] = p for x~out
        # This means qs is the (1-p) quantile of the OUT distribution
        qs = norm.ppf(1.0 - self.best_p, loc=self.out_mean_est, scale=self.out_std_est)
        logger.info(f"Using adaptive trimming with p={self.best_p:.4f}, qs={qs:.4f}")
        
        # Sort scores and keep only those below qs
        sorted_scores = np.sort(ta_scores)
        trimmed = sorted_scores[sorted_scores <= qs]
        return np.mean(trimmed) if len(trimmed) > 0 else sorted_scores[0]
        # return np.mean(ta_scores)

    def compute_trimmed_average(self: Self, ta_scores: np.ndarray) -> float:
        """Trim top x% of scores and return mean of remaining x%"""
        if len(ta_scores) == 0:
            return 0.0
        
        # Sort scores in ascending order
        sorted_scores = np.sort(ta_scores)
        
        # Calculate cutoff index for bottom x%
        cutoff_idx = int(len(sorted_scores) * 0.4)
        
        # Get scores to keep (bottom x%)
        trimmed_scores = sorted_scores[:cutoff_idx]
        
        return np.mean(trimmed_scores) if len(trimmed_scores) > 0 else 0.0

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
        # Expand range_masks to transformed_samples:
        self.transformed_masks = np.array([self.metadata["range_masks"][range_idx] for range_idx in self.metadata["sample_to_range"]])

        # After creating transformed_masks
        assert len(self.transformed_masks) == self.num_transformed_samples, \
            "transformed_masks length mismatch with transformed samples"
        
        # Check mask consistency for a sample range
        sample_range_idx = 0
        sample_audit_idx = self.selected_indices[sample_range_idx]
        assert np.all(self.transformed_masks[self.metadata["range_to_samples"][sample_range_idx]] 
                    == self.in_indices_masks[sample_audit_idx]), \
            "Mask mismatch between transformed samples and original audit data"

        #self.fixed_in_std = self.get_std(self.shadow_models_logits.flatten(), self.transformed_masks.flatten(), True, "fixed")
        self.fixed_out_std = self.get_std(self.shadow_models_logits.flatten(), (~self.transformed_masks).flatten(), False, "fixed")

        # Calculate all membership scores for transformed samples
        self.all_sample_scores = self.calculate_scores_for_transformed_samples()

        # Comment below two lines for no transformation
        self.estimate_distribution_separation() 
        self.optimize_p()

        range_scores = []
        for range_idx, sample_indices in self.metadata["range_to_samples"].items():
            range_sample_scores = self.all_sample_scores[sample_indices]
            
            # transformation with adaptive trimming
            trimmed_score = self.compute_adaptive_trimmed_average(range_sample_scores)
            range_scores.append(trimmed_score)

            # transformation with predefined trimming
            # trimmed_score = self.compute_trimmed_average(range_sample_scores)
            # range_scores.append(trimmed_score)

            # Uncomment below line for no transformation with no adaptive trimming or any trimming
            # range_scores.append(range_sample_scores)
        
        # Convert to numpy array and handle NaN values
        range_scores = np.array(range_scores)

        # Generate thresholds based on range scores
        self.thresholds = np.linspace(np.min(range_scores), np.max(range_scores), 1000)

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
