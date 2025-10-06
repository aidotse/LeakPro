"""Implementation of the RaMIA attack on top of LiRA attack."""

import os
from typing import Literal

import numpy as np
import torch
import umap
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from torch import Tensor, cat, stack
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.base import AttackBASE
from leakpro.attacks.utils.group_testing import GroupTestDecoder
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from leakpro.utils.save_load import hash_attack


class AttackRaMIA(AbstractMIA):
    """Implementation of the RaMIA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the RaMIA attack."""

        # RaMIA attack parameters
        num_transforms: int = Field(default=10, ge=0, le=100, description="Number of augmentations to apply to each sample")
        n_ops: int = Field(default=1, ge=0, le=10, description="Number of augmentation operations to apply per augmentation") # noqa: E501
        stealth_method: Literal["none", "bcjr", "agglomerative"] = Field(default="none", description="Stealth mode using group testing. Options are 'none', 'bcjr', or 'agglomerative'")  # noqa: E501
        groups: int = Field(default=1, ge=2, le=100, description="Number of groups to use in stealth mode (group testing)")  # noqa: E501
        groups_per_sample: int = Field(default=3, ge=2, le=10, description="Number of groups each sample is assigned to in stealth mode (group testing)")  # noqa: E501
        augment_strength: Literal["none", "easy", "medium", "strong"] = Field(default="easy", description="Strength of augmentations to apply to center point when creating range samples. Options are 'none', 'low', 'medium', or 'high'")  # noqa: E501
        # parameters used for the MIA attack
        online: bool = Field(default=False, description="Online vs offline attack")
        num_shadow_models: int = Field(default=2, ge=2, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501

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

        self.aug = handler.modality_extension
        self.aug.set_augment_strength(self.augment_strength)
        self.augment_center = False  # always augment the center point for RaMIA
        self.num_audit = 500
        self.group_score_threshold = 0.49

        # Get MIA attack object
        configs = {
            "num_shadow_models": self.configs.num_shadow_models,
            "training_data_fraction": self.training_data_fraction,
            "online": self.online,
        }
        self.mia_attack = AttackBASE(handler, configs=configs)

    def description(self: Self) -> dict:
        """Return a description of the attack for documentation and reporting.

        Returns:
            dict: Contains title, reference, summary, and detailed description

        """
        title_str = "Range Membership Inference Attack"

        reference_str = "Tao, Jiashu, and Reza Shokri. Range membership inference attacks. 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). IEEE, 2025."  # noqa: E501

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

    def create_assignment_matrix(self,
                                range_samples: list,
                                features: np.ndarray,
                                n_clusters:int=12,
                                m:int=4) -> tuple[list, np.ndarray]:
        """Perform group testing to select representative samples from the range samples.

        Args:
        ----
            range_samples (list): List of samples within the range to be tested.
            features (np.ndarray): Latent feature representations of the samples.
            n_clusters (int): Number of clusters/groups to form.
            m (int): Number of clusters each sample is assigned to.

        Returns:
        -------
            representative_samples (list): List of indices of representative samples.
            assignment_matrix (np.ndarray): Binary matrix indicating sample-group assignments.

        """

        # Step 2: Run k-means to get cluster centers
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=1234)
        kmeans.fit(features)
        centroids = kmeans.cluster_centers_

        # Step 3: Calculate distances to all centroids for each sample
        distances = euclidean_distances(features, centroids)

        # Step 4: Create assignment matrix
        n_samples = len(range_samples)
        assignment_matrix = np.zeros((n_samples, n_clusters), dtype=np.uint8) # Initialize assignment matrix

        # Assign each sample to its m closest clusters
        for i in range(n_samples):
            closest_clusters = np.argsort(distances[i])[:m] # Get m closest clusters for sample i
            assignment_matrix[i, closest_clusters] = 1

        # Step 5: Find representatives using medoids
        representative_indices = []
        cluster_to_representative = {}  # Track which sample represents each cluster
        for j in range(n_clusters):
            samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]

            if len(samples_in_cluster) > 1:
                # Find medoid
                sample_features = features[samples_in_cluster]
                within_distances = euclidean_distances(sample_features)
                sum_distances = np.sum(within_distances, axis=1)
                medoid_idx = np.argmin(sum_distances)
                representative_idx = samples_in_cluster[medoid_idx]
            else:
                representative_idx = samples_in_cluster[0]
            representative_indices.append(representative_idx)
            cluster_to_representative[j] = representative_idx  # Store mapping

        # Get representative samples
        representative_samples = [range_samples[idx] for idx in representative_indices]

        # Return representatives and assignment matrix
        return representative_samples, assignment_matrix.T

    def create_assignment_matrix_agglomerative(self,
                                range_samples: list,
                                features: np.ndarray,
                                n_clusters:int=12) -> tuple[list, np.ndarray]:
        """Perform group testing to select representative samples from the range samples using agglomerative clustering.

        Args:
        ----
            range_samples (list): List of samples within the range to be tested.
            features (np.ndarray): Latent feature representations of the samples.
            n_clusters (int): Number of clusters/groups to form.

        Returns:
        -------
            representative_samples (list): List of indices of representative samples.
            assignment_matrix (np.ndarray): Binary matrix indicating sample-group assignments.

        """

        x = np.asarray(features, dtype=np.float64)
        n, d = x.shape
        k = int(max(1, min(n_clusters, n)))  # cannot have more clusters than samples

        eps = 1e-12
        xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)
        sim = np.clip(xn @ xn.T, -1.0, 1.0)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)

        agg = AgglomerativeClustering(n_clusters=k,linkage="ward")
        labels = agg.fit_predict(dist)  # shape (N,)

        assignment_matrix = np.zeros((n,k), dtype=int)
        assignment_matrix[np.arange(n), labels] = 1

        # Find representative samples (medoids) for each cluster
        reps = []
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) == 1:
                reps.append(int(idx[0]))
                continue
            sub_distance = dist[np.ix_(idx, idx)]          # (m, m)
            # average distance of each point to rest
            mean_d = sub_distance.mean(axis=1)
            rep_local = int(idx[np.argmin(mean_d)])
            reps.append(rep_local)

        representative_samples = [range_samples[idx] for idx in reps]

        return representative_samples, assignment_matrix

    def get_latent_features(self, dataloader:DataLoader, method: str = "pca") -> np.ndarray:
        """Extract 512-d ResNet18 features, then do per-group (per-n) 2D PCA on the augmentations.

        Args:
        ----
            dataloader (DataLoader): DataLoader for the dataset to extract features from.
            method (str): Dimensionality reduction method, either 'pca' or 'umap'.

        Returns:
        -------
            np.ndarray of shape [N, 2] with rows aligned to the dataloader sample order.

        """

        # ----------------------------
        # 1) Build feature extractor
        # ----------------------------
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # keep layers up to avgpool, then flatten -> 512-d vectors
        feature_extractor = torch.nn.Sequential(*(list(m.children())[:-1]),
                                                torch.nn.Flatten(1)).eval().to(dev)
        torch.backends.cudnn.benchmark = True  # speedup for fixed-size inputs
        # ----------------------------
        # 2) Pass all data once
        # ----------------------------
        all_feats = []
        with torch.inference_mode():
            for x, _ in tqdm(dataloader, total=len(dataloader), desc="Extracting features"):
                x = x.to(dev, non_blocking=True)
                f = feature_extractor(x)
                f = torch.nn.functional.normalize(f, p=2, dim=1)           # L2 normalize in torch
                all_feats.append(f.detach().cpu())       # keep on CPU after this
        feats = torch.cat(all_feats, dim=0).numpy().astype(np.float32)   # [N, 512]

        # Preserve original order

        # ----------------------------
        # 3) global PCA
        # ----------------------------
        n_comp = 2
        if method == "pca":
            reducer = PCA(n_components=n_comp, random_state=42)
            z = reducer.fit_transform(feats)
        elif method == "umap":
            reducer = umap.UMAP(n_components=n_comp, random_state=42)
            z = reducer.fit_transform(feats)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        return z

    def prepare_attack(self:Self)->None:
        """Prepare attack on the target model and dataset."""

        self.in_indices = self.handler.train_indices[:self.num_audit]
        self.out_indices = self.handler.test_indices[:self.num_audit]
        self.audit_indices =  np.concatenate((self.in_indices, self.out_indices))
        audit_data = self.handler.get_dataset(self.audit_indices)
        audit_data.augment_strength = None
        self.audit_dataloader = DataLoader(audit_data, batch_size=1, shuffle=False)
        # Prepare MIA to be used
        self.mia_attack.prepare_attack()

    def run_attack(self:Self)-> MIAResult:  # noqa: C901, PLR0912, PLR0915
        """Run the RaMIA attack on the target model and dataset."""


        # Step 1: prepare the transformed samples
        def save_img(x, x_augs, name):  # noqa: ANN001, ANN202
            save_image(x, name+"_orig.png")
            for j, aug in enumerate(x_augs):
                save_image(aug, name + f"_{j}.png") # Already normalized

        augmented_data_dir = f"{self.handler.configs.audit.output_dir}/attack_objects/range_mia_data"
        if not os.path.exists(augmented_data_dir):
            os.makedirs(augmented_data_dir, exist_ok=True)

        hash_config = {"augment_strength": self.augment_strength,
                       "num_transforms": self.num_transforms,
                       "n_ops": self.n_ops}
        hash_str = hash_attack(hash_config, AbstractMIA.handler.target_model)

        if os.path.exists(f"{augmented_data_dir}/{hash_str}"):
            logger.info("Loading existing augmented data...")
            augmented_data = torch.load(f"{augmented_data_dir}/{hash_str}", weights_only=False)
            labels = torch.load(f"{augmented_data_dir}/labels_{hash_str}", weights_only=False)
        else:
            logger.info("Creating new augmented data...")
            augmented_data = []
            labels = Tensor()
            for i, (x, y) in tqdm(enumerate(self.audit_dataloader),
                                    total=len(self.audit_dataloader),
                                    desc="Creating transformed samples"):
                augs, op_names = self.aug.augment(x[0],
                                                    k=self.num_transforms,
                                                    num_ops=self.n_ops,
                                                    base_seed=i)
                augmented_data.append(augs)
                labels = cat((labels, y.repeat(len(augs))), dim=0)

            augmented_data = cat(augmented_data, dim=0)
            torch.save(augmented_data, f"{augmented_data_dir}/{hash_str}")
            torch.save(labels, f"{augmented_data_dir}/labels_{hash_str}")
        # create a list of what audit datapoint each transformed sample belongs to
        audit_indice_map = [x for x in self.audit_indices for _ in range(max(self.num_transforms, 1))]
        aug_dataset = self.handler.UserDataset(augmented_data, labels.int())
        aug_dataset.augment = None
        aug_dataset.erase_post_norm = None
        aug_dataloader = DataLoader(aug_dataset, batch_size=1024, shuffle=False)

        # Step 2: run MIA to get logits for target and shadow models
        latent_files = f"{augmented_data_dir}/latent_features_{hash_str}.pt"
        if os.path.exists(latent_files):
            logger.info("Loading existing latent features...")
            latent_features = torch.load(latent_files, weights_only=False).numpy()
        else:
            logger.info("Creating new latent features...")
            if self.num_transforms > 2:
                logger.info("Extracting latent features with per-group PCA...")
                latent_features = self.get_latent_features(aug_dataloader, method="umap")
                torch.save(torch.tensor(latent_features), latent_files)

        if self.stealth_method in ["bcjr"]:
            gt_decoder = GroupTestDecoder()

            # Create assignment matrix and representative samples for each group
            representative_samples = []
            assignment_matrices = []
            for i in tqdm(range(len(self.audit_indices)), "Creating group testing representatives"):
                # get all transformed samples for this audit sample
                range_samples = aug_dataset.data[i*self.num_transforms:(i+1)*self.num_transforms]
                latent_sample_features = latent_features[i*self.num_transforms:(i+1)*self.num_transforms]
                sample_representatives, assignment_matrix = self.create_assignment_matrix(range_samples,
                                                                                        latent_sample_features,
                                                                                        n_clusters=self.groups,
                                                                                        m=self.groups_per_sample)

                representative_samples.extend(sample_representatives)
                assignment_matrices.append(assignment_matrix)

            audit_indice_map = [x for x in self.audit_indices for _ in range(self.groups)]

            repr_labels = self.audit_dataloader.dataset.targets.repeat_interleave(self.groups)
            representative_samples = stack(representative_samples, dim=0)
            repr_dataset = self.handler.UserDataset(representative_samples, repr_labels.int())
            repr_dataset.augment_strength = None
            repr_dataset.erase_post_norm = None
            aug_dataloader = DataLoader(repr_dataset, batch_size=1024, shuffle=False)

            cluster_scores = self.mia_attack.score_samples(aug_dataloader, audit_indice_map)
            binary_outcomes = cluster_scores > 0  # Convert scores to binary outcomes

            aug_scores = []
            for i in tqdm(range(len(assignment_matrices)), desc="Decoding group test results"):
                assignment_matrix = assignment_matrices[i]
                test_result = binary_outcomes[i*self.groups:(i+1)*self.groups]
                llrs = gt_decoder.gt_decode(assignment_matrix, test_result)
                membership_scores = 1 / (1 + np.exp(llrs))  # Convert LLRs to probabilities
                aug_scores.extend(membership_scores)

            aug_scores = np.array(aug_scores)

        elif self.stealth_method in ["agglomerative"]:
            # Create assignment matrix and representative samples for each group
            representative_samples = []
            assignment_matrices = []
            group_cardinality = []
            for i in tqdm(range(len(self.audit_indices)), "Creating group testing representatives"):
                # get all transformed samples for this audit sample
                range_samples = aug_dataset.data[i*self.num_transforms:(i+1)*self.num_transforms]
                latent_sample_features = latent_features[i*self.num_transforms:(i+1)*self.num_transforms]
                sample_representatives, assignment_matrix = self.create_assignment_matrix_agglomerative(range_samples,
                                                            latent_sample_features,
                                                            n_clusters=self.groups)

                representative_samples.extend(sample_representatives)
                assignment_matrices.append(assignment_matrix)
                group_cardinality.append(assignment_matrix.sum(axis=0))  # number of samples in each group
            group_cardinality = np.array(group_cardinality).flatten()  # [num_audit, groups]
            audit_indice_map = [x for x in self.audit_indices for _ in range(self.groups)]

            repr_labels = self.audit_dataloader.dataset.targets.repeat_interleave(self.groups)
            representative_samples = stack(representative_samples, dim=0)
            repr_dataset = self.handler.UserDataset(representative_samples, repr_labels.int())
            repr_dataset.augment = None
            repr_dataset.erase_post_norm = None
            aug_dataloader = DataLoader(repr_dataset, batch_size=1024, shuffle=False)
            aug_scores = self.mia_attack.score_samples(aug_dataloader, audit_indice_map)

        elif self.stealth_method in ["none"]:
            aug_scores = self.mia_attack.score_samples(aug_dataloader, audit_indice_map)
        else:
            raise ValueError(f"Unknown stealth_method: {self.stealth_method}. Choose from 'none', 'bcjr', or 'agglomerative'.")

        # Step 3: Perform trimmed averaging over the scores of transformed samples
        qs = 0.4
        qe = 1
        final_scores = []
        step_size = aug_scores.shape[0] // len(self.audit_indices) # number of scores per audit sample
        for i in range(len(self.audit_indices)):
            if self.num_transforms <= 1:
                final_scores.append(aug_scores[i])
                continue
            current_scores = aug_scores[i*step_size:(i+1)*step_size]
            sorted_scores = np.sort(current_scores)

            # compute trimmed mean within qs and qe
            trim_start = int(len(sorted_scores) * qs)
            trim_end = int(len(sorted_scores) * qe)
            trimmed_mean = np.mean(sorted_scores[trim_start:trim_end])
            final_scores.append(trimmed_mean)
        range_scores = np.array(final_scores)

        true_labels = np.array([True]*len(self.in_indices) + [False]*len(self.out_indices))

        attack_name = "RaMIA" if self.stealth_method in ["none"] else f"RaMIA-{self.stealth_method}"

        return MIAResult.from_full_scores(true_membership=true_labels,
                                        signal_values=range_scores,
                                        result_name=attack_name,
                                        metadata=self.configs.model_dump())
