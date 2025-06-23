"""Utility functions for fitting and loading TS2Vec representation models."""

import os
import pickle
import re

import numpy as np
import torch
from torch import cuda, is_tensor, os
from ts2vec import TS2Vec

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.logger import logger


def get_ts2vec_model(
        handler: AbstractInputHandler,
        shadow_population_indices: np.ndarray,
        batch_size: int = 256
    ) -> TS2Vec:
    """Get a TS2Vec model fitted on the shadow population.

    The function first checks for a saved model that matches the given representation indices (shadow population).
    Any found match is loaded, else a new TS2Vec model is fit, saved, and returned.

    Args:
    ----
        handler: The input handler object.
        shadow_population_indices: List of indices in population dataset used to train the shadow models.
        batch_size: Batch size used during fitting and inference of TS2Vec.

    Returns:
    -------
        The TS2Vec representation model.

    """

    ts2vec_dir = os.path.join(handler.configs.audit.output_dir, "ts2vec")
    if not os.path.exists(ts2vec_dir):
        os.makedirs(ts2vec_dir)

    device = "cuda:0" if cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = False

    # Init TS2Vec
    model_loaded = False
    ts2vec_model = TS2Vec(
        input_dims=handler.population.targets.shape[-1],
        device=device,
        batch_size=batch_size
    )

    # Get saved ts2vec files
    files = os.listdir(ts2vec_dir)
    model_files = [f for f in files if re.match(r"representation_model_(\d+)", f)]

    # Check if representation model is available
    for model_file in model_files:

        # Get model metadata
        model_number = re.match(r"representation_model_(\d+)", model_file).group(1)
        metadata_file = f"metadata_{model_number}.pkl"
        if metadata_file in files:
            metadata_path = os.path.join(ts2vec_dir, metadata_file)
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)  # noqa: S301

            # Compare representation indices and load model if match
            if np.array_equal(np.sort(metadata["representation_indices"]), np.sort(shadow_population_indices)):
                model_path = os.path.join(ts2vec_dir, model_file)
                ts2vec_model.load(model_path)
                model_loaded = True
                logger.info("Loaded TS2Vec representation model")

    # If no matching model, train a representation model on the shadow population
    if not model_loaded:
        # Get representation data
        representation_fit_data = handler.population.targets[shadow_population_indices]
        if is_tensor(representation_fit_data):
            representation_fit_data = representation_fit_data.numpy()
        logger.info("Training TS2Vec representation model")
        ts2vec_model.fit(
            representation_fit_data
        )

        # Save representation model
        model_number = len(model_files)
        ts2vec_model.save(os.path.join(ts2vec_dir, f"representation_model_{model_number}.pkl"))

        # Save metadata
        metadata = {
            "representation_indices": shadow_population_indices
        }
        metadata_path = os.path.join(ts2vec_dir, f"metadata_{model_number}.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Representation model and metadata saved to {ts2vec_dir}")

    return ts2vec_model
