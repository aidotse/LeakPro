# This file is part of celevrhans and is released under MIT License.  # noqa: D100
# Copyright (c) 2019 Google Inc., OpenAI and Pennsylvania State University
# See https://github.com/cleverhans-lab/cleverhans?tab=MIT-1-ov-file#readme for details.
import builtins
import logging

import numpy as np
from torch import (
    Tensor,
    amax,
    cat,
    clamp,
    cuda,
    device,
    float32,
    from_numpy,
    full,
    long,
    max,
    mean,
    min,
    no_grad,
    norm,
    ones,
    rand,
    rand_like,
    randn,
    sign,
    stack,
    tensor,
    where,
    zeros,
)
from torch import any as torch_any
from torch import isnan as torch_isnan
from torch import sqrt as torch_sqrt
from torch import sum as torch_sum
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.import_helper import List, Self, Tuple


class HopSkipJumpDistance:
    """PyTorch implementation of HopSkipJumpAttack.

    HopSkipJumpAttack was originally proposed by Chen, Jordan and Wainwright.
    It is a decision-based attack that requires access to output
    labels of a model alone.
    Paper link: https://arxiv.org/abs/1904.02144
    At a high level, this attack is an iterative attack composed of three
    steps: Binary search to approach the boundary; gradient estimation;
    stepsize search. HopSkipJumpAttack requires fewer model queries than
    Boundary Attack which was based on rejective sampling.

    :param model: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor with n samples.
    :param norm: The distance to optimize. Possible values: 2 or np.inf.
    :param y_target:  A tensor of shape (n, nb_classes) for target labels.
    Required for targeted attack.
    :param image_target: A tensor of shape (n, **image shape) for initial
    target images. Required for targeted attack.
    :param initial_num_evals: initial number of evaluations for
                              gradient estimation.
    :param max_num_evals: maximum number of evaluations for gradient estimation.
    :param stepsize_search: How to search for stepsize; choices are
                            'geometric_progression', 'grid_search'.
                            'geometric progression' initializes the stepsize
                             by ||x_t - x||_p / sqrt(iteration), and keep
                             decreasing by half until reaching the target
                             side of the boundary. 'grid_search' chooses the
                             optimal epsilon over a grid, in the scale of
                             ||x_t - x||_p.
    :param num_iterations: The number of iterations.
    :param gamma: The binary search threshold theta is gamma / d^{3/2} for
                   l2 attack and gamma / d^2 for linf attack.
    :param batch_size: batch_size for model prediction.
    :param verbose: (boolean) Whether distance at each step is printed.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    def __init__(self: Self,
                 model: Module,
                 data_loader: DataLoader,
                 logger:logging.Logger,
                 norm: int =2,
                 y_target: np.ndarray =None,
                 image_target: np.ndarray =None,
                 initial_num_evals: int=100,
                 max_num_evals: int=100,
                 stepsize_search: str="geometric_progression",
                 num_iterations: int= 10,
                 gamma: float = .0,
                 constraint: int =2,
                 batch_size: int =128,
                 verbose: bool =True,
                 clip_min: int = -1,
                 clip_max: int =1,
                 ) -> None:

        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.norm = norm
        self.y_target = y_target
        self.image_target = image_target
        self.initial_num_evals = initial_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.constraint = constraint
        self.batch_size = batch_size
        self.image_shape = self.data_loader.dataset[0][0].shape
        self.verbose = verbose
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device("cuda" if cuda.is_available() else "cpu")

        self.batch_shape = (self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        self.shape = self.batch_shape[1:]
        d = int(np.prod(self.shape))

        if self.constraint == 2:
            # In the original code self.gamma / (np.sqrt(d) * d)
            self.theta = self.gamma /  d
        else:
            self.theta = self.gamma / (d * d)



    def hop_skip_jump(self: Self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the HopSkipJump attack.

        Returns
        -------
            Tuple of numpy arrays representing perturbed images and perturbation distances.

        """
        # If the attack is targeted towards a specific class, then image_target must be provided.
        if self.y_target is not None:
            assert self.image_target is not None, "Require a target image for targeted attack."
        if self.clip_min is not None and self.clip_max is not None and self.clip_min > self.clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    self.clip_min, self.clip_max
                )
            )

        # Set the model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        self.model.requires_grad_(False)

        # Getting labels from the classifier
        if self.y_target is None:
            self.model_output_labels = self.get_classifier_labels(self.data_loader)
            self.model_output_labels = self.model_output_labels.to(self.device)

        # untargeted attack without an initialized image.
        pert_imgs, perturbation_distances = self.hsja(self.data_loader,
                                                      None
                                                      )

        pert_imgs = cat(pert_imgs, 0).cpu().numpy()
        perturbation_distances = cat(perturbation_distances, 0).cpu().numpy()
        return pert_imgs, perturbation_distances


    def get_classifier_labels(self: Self, dataloader: DataLoader) -> np.ndarray:
        """Get the labels from the classifier for the given data loader."""
        labels = []
        self.model.eval()
        for _, (batch_data, _) in enumerate(tqdm(dataloader, desc="Epoch")):
            batch_data = batch_data.to(self.device)  # noqa: PLW2901

            with no_grad():
                y_sample = self.model(batch_data)
                _, labels_i = max(y_sample, 1)
            labels.append(labels_i)

        return cat(labels, 0)


    def hsja(self: Self, samples:DataLoader,
             batch_target_image: np.ndarray) -> Tuple[List, List]:
        """Compute the HopSkipJump attack for a batch of samples.

        Parameters
        ----------
        samples : iterable
            The input samples.
        batch_y_target : Any
            The target labels for the attack.
        batch_target_image : Any
            The target images for the attack.

        Returns
        -------
        Tuple of lists representing perturbed images and perturbation distances.

        """
        self.model.eval()
        # Initialize.
        if batch_target_image is None:
            perturbed, perturbed_indices = self.initialize()
        else:
            perturbed = batch_target_image


        #TODO: check if the indices are correct ( should not be sequential)
        # Ordering the perturbed images same as the samples by perturbed_indices
        perturbed_indices_tensor = tensor(perturbed_indices, dtype=long)
        ordered_perturbed = perturbed[perturbed_indices_tensor].to(self.device)

        perturbation_distances = []
        perturbed_images = []
        for i, (batch_sample, _) in enumerate(tqdm(samples, desc="Epoch")):
            active_indices = list(range(i*self.batch_size,
                                        builtins.min((i +1) * self.batch_size,  len(ordered_perturbed))))
            batch_perturbed = stack([ordered_perturbed[j] for j in active_indices])

            batch_sample = batch_sample.to(self.device)  # noqa: PLW2901
            batch_perturbed = batch_perturbed.to(self.device)

            perturbed, perturbation_distance = self.distance_batch(
                batch_sample, batch_perturbed, active_indices,i)

            perturbation_distances.append(perturbation_distance)
            perturbed_images.append(perturbed)
            if self.verbose:
                self.logger.info(f"Batch {i} ")
        return perturbed_images, perturbation_distances


    def distance_batch(self: Self,
                       batch_sample: Tensor,
                       batch_perturbed: Tensor,
                       active_indices: np.ndarray,
                        b_i: int) -> Tuple[Tensor, Tensor]:
        """Compute the distance between a batch of perturbed images and the original batch of samples.

        Parameters
        ----------
        batch_sample : Tensor
            The original batch of samples.
        batch_perturbed : Tensor
            The batch of perturbed images.
        active_indices : List[int]
            The indices of the active samples.
        b_i : int
            The batch index.

        Returns
        -------
        Tuple of the perturbed images and the perturbation distance.

        """
        # Project the initialization to the boundary.
        perturbed, dist_post_update, perturbed_label = self.binary_search_batch(
            batch_sample, batch_perturbed, active_indices)

        dist = self.compute_distance(perturbed, batch_sample)
        sum_intial_distance = torch_sum(dist, dim=0)
        if self.verbose:
            self.logger.info(f"Batch {b_i} Initial distance: {sum_intial_distance}")

        perturbation_distance = 0
        j = 0
        for j in np.arange(self.num_iterations):

            current_iteration = j + 1

            # Choose delta.
            delta = self.select_delta(
                dist_post_update,
                current_iteration
            )

            # approximate gradient.
            gradf = self.approximate_gradient(
                perturbed,
                j,
                delta,
                active_indices
            )
            update = sign(gradf) if self.constraint == np.inf else gradf

            # search step size.
            if self.stepsize_search == "geometric_progression":
                # find step size.
                epsilon = self.geometric_progression_for_stepsize(
                    perturbed, update, dist, current_iteration, active_indices, b_i
                )

                # Update the sample.
                perturbed = clamp(
                    perturbed + epsilon.view(len(epsilon),1,1,1) * update, self.clip_min, self.clip_max
                )

                # Binary search to return to the boundary.
                perturbed, dist_post_update , perturbed_label= self.binary_search_batch(
                    batch_sample, perturbed, active_indices )

            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.
                epsilons = (
                    from_numpy(np.logspace(-4, 0, num=20, endpoint=True))
                    .to(self.device)
                    .float()
                    * dist
                )
                perturbeds = (
                    perturbed + epsilons.view((20,) + (1,) * (len(self.shape) - 1)) * update
                )
                perturbeds = clamp(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed,perturbed_label = self.decision_function(perturbeds, active_indices)

                if torch_sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    perturbed, dist_post_update, perturbed_label = self.binary_search_batch(
                        batch_sample,
                        perturbeds[idx_perturbed],
                    )

            # compute new distance.
            perturbation_distance = self.compute_distance(perturbed, batch_sample)

            if j % 2 == 0 and self.verbose:
                self.logger.info(f"iteration: {j + 1}, distance {perturbation_distance}")

        return perturbed, perturbation_distance


    def decision_function(self: Self,
                          images: Tensor,
                          active_indices: np.ndarray = None ) -> Tuple[Tensor, Tensor]:
        """Compute the decision function for the given images.

        Args:
        ----
            images (Tensor): The input images.
            active_indices (np.ndarray, optional): The indices of the active images. Defaults to None.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing the decision function results and the predicted labels.

        """
        # Check if active_indices is None and set it to all indices if so
        if active_indices is None:
            active_indices = np.arange(images.size(0))

        images = clamp(images, self.clip_min, self.clip_max)
        images = images.to(float32)

        with no_grad():
            prob = self.model(images)
        perturbed_label = max(prob, dim=1)[1]
        model_output_labels_tensor = self.model_output_labels[active_indices].clone().detach()


        decitions = perturbed_label != model_output_labels_tensor if self.y_target is None else perturbed_label == self.y_target
        return decitions, perturbed_label


    def get_random_noise(self: Self) -> Tuple[np.ndarray, List[int]]:
        """Generate random noise for each data point in the dataset.

        Returns
        -------
            Tuple[np.ndarray, List[int]]: A tuple containing the generated random noises and the ordered indices.

        """
        success = np.zeros(len(self.data_loader.dataset), dtype=bool)
        passed_random_noises = zeros((len(self.data_loader.dataset), self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        active_indices = np.arange(self.batch_size)
        passed_ordered_indices = []
        num_evals = 0

        # Find a misclassified random noise.
        while not success.all():
            active_data = []
            for idx in active_indices:
                active_data.append(self.data_loader.dataset[idx])
            # Flatten the list of tuples into a list of tensors
            all_tensors = [unpack_tuple[0] for unpack_tuple in active_data]

            # Stack all tensors into a single tensor
            active_data = stack(all_tensors)

            random_noise = self.clip_min + rand_like(active_data) * (
                self.clip_max - self.clip_min
            )
            random_noise = random_noise.to(self.device)
            decision, _ = self.decision_function(random_noise, active_indices)

            for i, idx in enumerate(active_indices):
                if decision[i]:
                    success[idx] = True
                    passed_random_noises[idx] = random_noise[i]
                    passed_ordered_indices.append(idx)

            # Update active_indices with new data points
            active_indices = np.where(~success)[0][:self.batch_size]

            num_evals += 1
            if len(active_indices) == 0:
                self.logger.info(f"All data points have been successfully perturbed by random noise after {num_evals} evaluations.")
                break

        return passed_random_noises, passed_ordered_indices

    def initialize(self: Self) -> Tuple[np.ndarray, List[int]]:
        """Efficient Implementation of BlendedUniformNoiseAttack in Foolbox."""
        passed_random_noises, _ = self.get_random_noise()
        passed_random_noises = passed_random_noises.to(self.device)
        opt_intialization, opt_ordered_indices = self.binary_search_init(
                                                          passed_random_noises)

        return opt_intialization, opt_ordered_indices

    def binary_search_init(self: Self,
                           passed_random_noises: Tensor) -> Tuple[np.ndarray, List[int]]:
        """Perform binary search initialization for generating optimal noises.

        Args:
        ----
            passed_random_noises (np.ndarray): The random noises.

        Returns:
        -------
            Tuple[np.ndarray, List[int]]: A tuple containing the generated optimal noises and the ordered indices.

        """
        low  = zeros(len(self.data_loader.dataset), dtype=float).to(self.device)
        high = ones(len(self.data_loader.dataset), dtype=float).to(self.device)
        mid = zeros(len(self.data_loader.dataset), dtype=float).to(self.device)
        success = np.zeros(len(self.data_loader.dataset), dtype=bool)
        active_indices = np.arange(self.batch_size)
        passed_opt_noises = zeros((len(self.data_loader.dataset), self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        passed_opt_ordered_indices = []
        num_evals = 0


        while not success.all():

            active_data = []
            for idx in active_indices:
                active_data.append(self.data_loader.dataset[idx])
            all_tensors = [unpack_tuple[0] for unpack_tuple in active_data]
            active_data = stack(all_tensors).to(self.device)

            active_noise = passed_random_noises[active_indices]

            mid_batch = mid[active_indices]
            low_batch = low[active_indices]
            high_batch = high[active_indices]

            mid_batch = (high_batch + low_batch) / 2.0
            blended = (1 -  mid_batch.view(-1, 1, 1, 1) ) * active_noise +  mid_batch.view(-1, 1, 1, 1) * active_noise


            blended = blended.to(self.device)
            decisions, _  = self.decision_function(blended, active_indices)
            # Update high where decisions is True
            high_batch[decisions] = mid_batch[decisions].clone().detach()

            # Update low where decisions is False
            low_batch[~decisions] = mid_batch[~decisions].clone().detach()

            for i, idx in enumerate(active_indices):
                if high_batch[i] - low_batch[i] > 0.00001 : #< 1.5:
                    success[idx] = True
                    opt_noise = (1 - high_batch[i]) * active_data[i] + high_batch[i] * passed_random_noises[i]
                    passed_opt_noises[idx] = opt_noise.clone().detach()
                    passed_opt_ordered_indices.append(idx)


            # Update active_indices with new data points
            active_indices = np.where(~success)[0][:self.batch_size]

            if num_evals % 20 == 0 and self.verbose:
                # Print progress
                ratio_success = sum(success).item() / len(self.data_loader.dataset)
                self.logger.info(f"Iteration {num_evals}: {ratio_success*100:.4f}% success")

            num_evals += 1
            if len(active_indices) == 0:
                self.logger.info(f"Initialized binary search has been completed after {num_evals} evaluations.")
                break

        return passed_opt_noises, passed_opt_ordered_indices



    def compute_distance(self: Self,
                         batch_ori: Tensor,
                         batch_pert: Tensor) -> Tensor:
        """Compute the distance between the original batch and the perturbed batch.

        Args:
        ----
            batch_ori (Tensor): The original batch.
            batch_pert (Tensor): The perturbed batch.

        Returns:
        -------
            Tensor: The computed distance between the original batch and the perturbed batch.

        """
        if torch_any(torch_isnan(batch_ori)):
            raise AssertionError("NaN values found in batch_ori")
        assert not torch_any(torch_isnan(batch_pert)), "NaN values found in batch_ori"

        if self.constraint == 2:
            dist = norm(batch_ori - batch_pert, p=2,  dim=(1, 2, 3))
        elif self.constraint == np.inf:
            dist= amax(abs(batch_ori - batch_pert),  dim=(1, 2, 3))
        return dist


    def project(self: Self,
                original_image_batch: Tensor,
                perturbed_images_batch: Tensor,
                alphas: Tensor) -> Tensor:
        """Projects the perturbed images onto the epsilon ball around the original images.

        Args:
        ----
            original_image_batch (Tensor): The batch of original images.
            perturbed_images_batch (Tensor): The batch of perturbed images.
            alphas (Tensor): The scaling factor for the perturbation.

        Returns:
        -------
            Tensor: The projected images.

        Raises:
        ------
            None

        """
        alphas = alphas.to(self.device)
        alphas = alphas.view((len(original_image_batch), 1, 1, 1))
        if self.constraint == 2:
            projected = (1 - alphas) * original_image_batch + alphas * perturbed_images_batch
        elif self.constraint == np.inf:
            projected = clamp(
                perturbed_images_batch, original_image_batch - alphas, original_image_batch + alphas
            )
        return projected


    def geometric_progression_for_stepsize(self: Self,
                                           samples: Tensor,
                                           updates: Tensor,
                                           dist: Tensor,
                                           current_iteration: int,
                                           active_indices: np.ndarray, b_i:int) -> Tensor:
        """Calculates the step size for geometric progression in the HopSkipJumpDistance algorithm.

        Args:
        ----
            samples (Tensor): The input samples.
            updates (Tensor): The updates to be applied to the samples.
            dist (Tensor): The distance value.
            current_iteration (int): The current iteration of the algorithm.
            active_indices (np.ndarray): The active indices.
            b_i (int): The value of b_i.

        Returns:
        -------
            Tensor: The calculated step size for geometric progression.

        """
        batch_epsilon = dist / np.sqrt(current_iteration)
        active_indices = np.array(active_indices)

        success = np.zeros(len(samples), dtype=bool)
        batch_active_indices = np.arange(len(samples))
        num_evals = 0

        while not success.all() and num_evals <= 10:

            active_samples = samples[batch_active_indices]
            active_updates = updates[batch_active_indices]
            active_epsilon = batch_epsilon[batch_active_indices].view(len(batch_active_indices), 1, 1, 1)

            active_updateed_samples = active_samples + active_epsilon * active_updates
            decisions, _  = self.decision_function(active_updateed_samples,
                                                   active_indices[batch_active_indices])
            passed_index= []
            for i, idx in enumerate(batch_active_indices):
                if decisions[i]:
                    success[idx] = True
                    passed_index.append(idx)
                else:
                    batch_epsilon[idx] =  batch_epsilon[idx] / 2.0
            if passed_index:
                # Find the positions in batch_active_indices corresponding to the values in passed_index
                positions_to_delete = [i for i, val in enumerate(batch_active_indices) if val in passed_index]

                batch_active_indices = np.delete(batch_active_indices, positions_to_delete)
            num_evals += 1
            if self.verbose:
                self.logger.info(f"Stepsize search: {num_evals} evaluations, and failed data {len(batch_active_indices)} in {b_i}.")
        return batch_epsilon


    def select_delta(self: Self, dist_post_update: Tensor, current_iteration: int) -> Tensor:
        """Selects the delta value based on the given distance post update and current iteration.

        Args:
        ----
            dist_post_update (Tensor): The distance post update.
            current_iteration (int): The current iteration.

        Returns:
        -------
            Tensor: The selected delta value.

        Raises:
        ------
            None

        """
        d = int(np.prod(self.shape))
        if current_iteration == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
            delta_batch = full((len(dist_post_update),), delta)
        elif self.constraint == 2:
            delta_batch = np.sqrt(d) * self.theta * dist_post_update
        elif self.constraint == np.inf:
            delta_batch = d * self.theta * dist_post_update

        return delta_batch.to(self.device)


    def approximate_gradient(
        self: Self,
        samples: Tensor,
        iteration: int,
        delta: Tensor,
        active_indices: np.ndarray,
            ) -> Tensor:
        """Approximates the gradient of a function using random vectors.

        Args:
        ----
            samples (Tensor): The input samples.
            iteration (int): The current iteration number.
            delta (Tensor): The delta values for perturbation.
            active_indices (np.ndarray): The active indices for each sample.

        Returns:
        -------
            Tensor: The approximate gradient.

        Raises:
        ------
            None

        """
        # Generate random vectors.
        # Choose number of evaluations.
        num_evals = int(builtins.min(self.initial_num_evals * np.sqrt(iteration + 1),
                                        self.max_num_evals))
        approx_grad = zeros((len(samples),) + self.shape).to(self.device)

        for i, sample in enumerate(samples):
            noise_shape = [num_evals] + list(self.shape)
            if self.constraint == 2:
                rv = randn(noise_shape)
            elif self.constraint == np.inf:
                rv = -1 + rand(noise_shape) * 2

            axis = tuple(range(1, 1 + len(self.shape)))
            rv = rv / torch_sqrt(torch_sum(rv ** 2, dim=axis, keepdim=True))
            rv = rv.to(self.device)
            perturbed = sample + delta[i] * rv
            perturbed = clamp(perturbed, self.clip_min, self.clip_max)
            rv = (perturbed - sample) / delta[i]

            # query the model.
            sample_index = np.full(num_evals, active_indices[i])
            decisions, _ = self.decision_function(perturbed, sample_index)
            fval = 2.0 * decisions.view((decisions.shape[0],) +
                                        (1,) * len(self.shape)) - 1.0

            # Baseline subtraction (when fval differs)
            fval_mean = mean(fval)
            if fval_mean == 1.0:  # label changes.
                gradf = mean(rv, dim=0)
            elif fval_mean == -1.0:  # label not change.
                gradf = -mean(rv, dim=0)
            else:
                fval = fval - fval_mean
                gradf = mean(fval * rv, dim=0)

            # Get the gradient direction.
            approx_grad[i] = gradf / norm(gradf, p=2)

        return approx_grad



    def binary_search_batch(self: Self,
                            original_image_batch: Tensor, perturbed_images_batch: Tensor,
                            active_indices: np.ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform binary search to find the optimal perturbation for a batch of images.

        Args:
        ----
            original_image_batch (Tensor): The original batch of images.
            perturbed_images_batch (Tensor): The batch of perturbed images.
            active_indices (np.ndarray): The indices of the active images.

        Returns:
        -------
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the output images, the distances
            after the update, and the perturbed labels.

        Raises:
        ------
            None

        """
        # I think this should be included when stepsize_search is grid_search.
        if self.stepsize_search == "grid_search":
            dists_post_update = stack(
                [
                    self.compute_distance(original_image_batch, perturbed_image)
                    for perturbed_image in perturbed_images_batch
                ]
            )
        else:
            dists_post_update = self.compute_distance(original_image_batch, perturbed_images_batch)

        # Choose upper thresholds in binary searches based on constraint.
        if self.constraint == np.inf:
            highs = dists_post_update
            # Stopping criteria.
            thresholds = min(dists_post_update * self.theta, self.theta)
        else:
            highs = ones(len(perturbed_images_batch)).to(self.device)
            thresholds = self.theta

        lows = zeros(len(perturbed_images_batch)).to(self.device)

        while max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_images = self.project(original_image_batch, perturbed_images_batch, mids)

            # Update highs and lows based on model decisions.
            decisions, perturbed_label = self.decision_function(mid_images, active_indices)
            lows = where(decisions == 0, mids, lows)
            highs = where(decisions == 1, mids, highs)

        out_images = self.project(original_image_batch, perturbed_images_batch, highs)

        # Compute distance of the output image to select the best choice.
        if self.stepsize_search == "grid_search":
            dists = stack(
                [
                    self.compute_distance(original_image_batch, out_image)
                    for out_image in out_images
                ]
            )
            _, idx = min(dists, 0)

            dist = dists_post_update[idx]
            out_image = out_images[idx].unsqueeze(0)
            return out_image, dist, perturbed_label

        return out_images, dists_post_update, perturbed_label












