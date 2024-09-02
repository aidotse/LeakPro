"""Class(es) built to boost membership attacks."""

import logging

import numpy as np
from tqdm import tqdm

from leakpro.attacks.utils.utils import softmax_logits
from leakpro.import_helper import Self
from leakpro.model import PytorchModel
from leakpro.signals.signal import ModelLogits, ModelRescaledLogits
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler


class Memorization():
    """Implementation of memorization filtering.

    From the paper "Why Train More? Effective and Efficient Membership Inference via Memorization", Choi J. et al.
    """

    def __init__(
            self:Self,
            use_privacy_score: bool,
            memorization_threshold: float,
            min_num_memorization_audit_points: int,
            num_memorization_audit_points: int,
            in_indices_mask: list,
            shadow_models: list,
            target_model: PytorchModel,
            audit_data_indices: list,
            audit_data_labels: list,
            org_audit_data_length: int,
            handler: AbstractInputHandler,
            online: bool,
            logger: logging.Logger,
            batch_size: int = 32,
        ) -> None:
        """Initialize the memorization boosting.

        Args:
        ----
            use_privacy_score (bool): Use privacy score.
            memorization_threshold (float): Percentile threshold for memorization.
            min_num_memorization_audit_points (int): Minimum number of audit points allowed after memorization.
            num_memorization_audit_points (int): Number of most vulnerable audit datapoints to be used.
            in_indices_mask (list): List of masks(bool) for IN samples.
            shadow_models (list): List of shadow models.
            target_model (PytorchModel): The target model to be audited.
            audit_data_indices (list): List of all audit data indices.
            audit_data_labels (list): List of corresponding labels.
            org_audit_data_length (int): Length of the original audit dataset, before any filtering.
            handler (AbstractInputHandler): Data handler to manage data.
            online (bool): Flag if the attack is online or not.
            logger (logging.Logger): Logger for logging the memorization process.
            batch_size (int): Integer to set batch size when loading data (will effect performance).

        Returns:
        -------
            mask (list): The memorization mask that determins the most vulnerable data points.
            memorization_score (list): List of memorization scores for all data points.
            privacy_score (list): List of privacy scores for all data points.

        """
        self.use_privacy_score = use_privacy_score
        self.memorization_threshold = memorization_threshold
        self.min_num_memorization_audit_points = min_num_memorization_audit_points
        self.num_memorization_audit_points = num_memorization_audit_points
        self.in_indices_mask = in_indices_mask
        self.shadow_models = shadow_models
        self.target_model = target_model
        self.audit_data_indices = audit_data_indices
        self.audit_data_labels = audit_data_labels
        self.org_audit_data_length = org_audit_data_length
        self.handler = handler
        self.online = online
        self.logger = logger
        self.batch_size = batch_size

        self.audit_data_length = len(self.audit_data_indices)
        if self.org_audit_data_length*(1-self.memorization_threshold) < self.min_num_memorization_audit_points:
            self.logger.info("Warning!")
            self.logger.info("Memorization threshold gives less audit points than the minimum allowed")
            self.logger.info('Please adjust "memorization_threshold" or "min_num_memorization_audit_points"')
            self.memorization_threshold = 1-self.min_num_memorization_audit_points/self.org_audit_data_length
            self.logger.info(f'Setting "memorization_threshold" to {self.memorization_threshold}')

        if self.memorization_threshold != 0.0 and self.num_memorization_audit_points != 0:
            self.logger.info("Warning!")
            self.logger.info('"memorization_threshold" and "num_memorization_audit_points" are both used.')
            self.logger.info('"memorization_threshold" is by default set to 0.8, try setting it to 0.0')
            self.logger.info('Overriding "memorization_threshold" for "num_memorization_audit_points"')

        elif self.memorization_threshold != 0.0:
            self.num_memorization_audit_points = int(self.org_audit_data_length * (1-self.memorization_threshold))

        # Initialize memorization and privacy score
        self.memorization_score = np.zeros(self.audit_data_length, dtype=float)
        self.privacy_score = np.zeros(self.audit_data_length, dtype=float)

    def run(self:Self) -> list:
        """Run memorization enhancement."""

        self._memorization_score()

        if self.use_privacy_score:
            self._privacy_score()

        mem_mask, privacy_mask = self.adjust_memorization_mask()
        mask = (mem_mask & privacy_mask)

        return mask, self.memorization_score, self.privacy_score

    def _memorization_score(self:Self) -> None:
        """Run memorization score enhancement.

        Memorization enhances the attack performance by only inlucing vulnerable data points
        """

        logits_function = ModelLogits()
        logits = np.swapaxes(logits_function(self.shadow_models, self.handler, self.audit_data_indices, self.batch_size), 0, 1)

        logits = softmax_logits(logits)

        if self.online:
            for i, (logit, mask, label) in tqdm(enumerate(zip(logits, self.in_indices_mask, self.audit_data_labels)),
                                                total=len(logits),
                                                desc="Calculating memorization score",
                                                leave=False
                                                ):
                self.memorization_score[i] = np.mean(logit[mask, label]) - np.mean(logit[~mask, label])
        else:
            # Offline impementation details.
            # 1. Assumption made: The target logits will be a good approximate to the missing IN-samples
            # 2. Given that memorization scores of the target model will likely be higher for IN-samples
            #       and low for OUT-samples, the offline version does more directly impact the
            #       filtering of IN- vs. OUT-samples.
            target_logits = np.swapaxes(logits_function([self.target_model], self.handler, self.audit_data_indices,\
                                        self.batch_size), 0, 1).squeeze()
            target_logits = softmax_logits(target_logits)
            for i, (logit, target_logit, label) in tqdm(enumerate(zip(logits, target_logits, self.audit_data_labels)),
                                                total=len(logits),
                                                desc="Calculating memorization score",
                                                leave=False
                                                ):
                self.memorization_score[i] = np.mean(target_logit[label]) - np.mean(logit[:, label])

    def _privacy_score(self:Self) -> None:
        """Run privacy score enhancement.

        Privacy score enhances the attack performance by only including vulnerable data points.
        The privacy score definition can be found in "Membership Inference Attacks From First Principles" by Carlini et al.
        """

        logits_function = ModelRescaledLogits()
        logits = np.swapaxes(logits_function(self.shadow_models, self.handler, self.audit_data_indices, self.batch_size), 0, 1)
        target_logits = np.swapaxes(logits_function([self.target_model], self.handler, self.audit_data_indices, self.batch_size),\
                                    0, 1).squeeze()

        privacy_score = []

        # From "Membership Inference Attacks From First Principles", fixed variance is used when number of shadow models < 64
        if len(self.shadow_models) < 64:
            in_std, out_std = np.std(logits[self.in_indices_mask].flatten()), np.std(logits[~self.in_indices_mask].flatten())

        for (logit, target_logit, mask) in tqdm(zip(logits, target_logits, self.in_indices_mask),
                                                total=len(logits),
                                                desc="Calculating privacy score",
                                                leave=False
                                                ):
            in_mean, out_mean = np.mean(logit[mask]), np.mean(logit[~mask])

            if len(self.shadow_models) >= 64:
                in_std, out_std = np.std(logit[mask]), np.std(logit[~mask])

            if self.online:
                privacy_score.append(np.abs(in_mean-out_mean)/(in_std+out_std+1e-30))
            else:
                # Assumptions
                # 1. target_logit is a good aproximation for the missing IN-samples
                # 2. out_std ~= in_std => in_std+out_std ~= 2*out_std
                privacy_score.append(np.abs(target_logit-out_mean)/(2*out_std+1e-30))

        self.privacy_score = np.asarray(privacy_score)

    def adjust_memorization_mask(self:Self) -> list:
        """Adjust thesholds to achieve the desired amount or percentile of most vulnerable datapoints."""

        # Set starting thresholds, from the paper ("why train more...")
        mem_thrshld = 0.8
        priv_thrshld = 2.0 if self.use_privacy_score else -1

        # Check for negative memorization scores
        if (positive_mem := np.count_nonzero((self.memorization_score > 0) & (self.privacy_score >= 0)))\
                                            < self.num_memorization_audit_points:
            self.logger.info("Too many samples with negative memorization score")
            self.logger.info(f"Only {positive_mem} points with positive score, requesting {self.num_memorization_audit_points}")
            self.logger.info("Please make sure to train the models enough")
            if positive_mem > 0:
                self.logger.info(f"Returning {positive_mem} points")
                self.num_memorization_audit_points = positive_mem
            else:
                self.logger.info("Returning the 50% most vulnerable data points")
                return self.memorization_score >= np.median(self.memorization_score),\
                        self.privacy_score >= np.min(self.privacy_score)

        # Use the thresholds from the paper
        if self.memorization_threshold == 0.0:
            return self.memorization_score > mem_thrshld, self.privacy_score > priv_thrshld

        # Adjust initial thresholds if they are set too high
        while (np.count_nonzero((self.memorization_score > mem_thrshld)\
                & (self.privacy_score >= priv_thrshld)) < self.num_memorization_audit_points):
            mem_thrshld = mem_thrshld/2
            priv_thrshld = priv_thrshld/2

        # Find the thresholds corresponding to the percentile set in config
        while (np.count_nonzero((self.memorization_score > mem_thrshld)\
                & (self.privacy_score >= priv_thrshld)) > self.num_memorization_audit_points):
            mem_thrshld = 1 - (1 - mem_thrshld)/(1.001)
            priv_thrshld = priv_thrshld*1.001
        return self.memorization_score > mem_thrshld, self.privacy_score > priv_thrshld
