# This file is part of celevrhans and is released under MIT License.  # noqa: D100
# Copyright (c) 2019 Google Inc., OpenAI and Pennsylvania State University
# See https://github.com/cleverhans-lab/cleverhans?tab=MIT-1-ov-file#readme for details.
import builtins

import numpy as np
from torch import (
    Tensor,
    amax,
    cat,
    clamp,
    cuda,
    device,
    float32,
    long,
    max,
    mean,
    min,
    no_grad,
    norm,
    ones,
    rand,
    sign,
    stack,
    tensor,
    zeros,
)
from torch import any as torch_any
from torch import isnan as torch_isnan
from torch import sum as torch_sum
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.utils.import_helper import List, Self, Tuple
from leakpro.utils.logger import logger


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
    :param epsilon_threshold: (optional float) The threshold for epsilon.
    """

    def __init__(self: Self,
                 model: Module,
                 data_loader: DataLoader,
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
                 epsilon_threshold: float =1e-6,
                 verbose: bool =True,
                 ) -> None:

        self.model = model
        self.data_loader = data_loader
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
        self.verbose = verbose
        self.clip_min = -1
        self.clip_max = 1
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.image_shape = self.data_loader.dataset[0][0].shape
        self.batch_shape = (self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        d = int(np.prod(self.image_shape))

        if self.constraint == 2:
            self.theta = self.gamma / (np.sqrt(d) * d)
        else:
            self.theta = self.gamma / (d * d)
        self.epsilon_threshold = epsilon_threshold
        self.dataset_mean, self.dataset_std = self.get_mean_std(self.data_loader)

    def get_mean_std(self: Self, dataloader: DataLoader) -> Tuple[float, float]:
        """Compute the mean and standard deviation of the dataset."""
        transformed_data = [dataloader.dataset[i][0] for i in range(len(dataloader.dataset))]
        transformed_data = np.stack(transformed_data, axis=0)


        mean = Tensor(transformed_data.mean(axis=(0, 2, 3))).to(self.device)
        std = Tensor(transformed_data.std(axis=(0,2 , 3))).to(self.device)
        return mean, std


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
        pert_imgs, perturbation_distances = self.hsja(self.data_loader)

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


    def hsja(self: Self, samples:DataLoader) -> Tuple[List, List]:
        """Compute the HopSkipJump attack for a batch of samples.

        Parameters
        ----------
        samples : iterable
            The input samples.
        batch_y_target : Any
            The target labels for the attack.

        Returns
        -------
        Tuple of lists representing perturbed images and perturbation distances.

        """
        # Initialize the perturbed images.
        init_perturbed, init_indices = self.initialize()
        init_indices_tensor = tensor(init_indices, dtype=long)
        init_perturbed_ordered = init_perturbed[init_indices_tensor].to(self.device)


        perturbation_distances = []
        perturbed_images = []
        for i, (batch_sample, _) in enumerate(tqdm(samples, desc="Batch")):
            active_dataset_indices = list(range(i*self.batch_size,
                                        builtins.min((i +1) * self.batch_size,  len(init_perturbed_ordered))))
            init_batch_perturbed = stack([init_perturbed_ordered[j] for j in active_dataset_indices])

            batch_sample = batch_sample.to(self.device)  # noqa: PLW2901
            init_batch_perturbed = init_batch_perturbed.to(self.device)

            perturbed, perturbation_distance = self.distance_batch(batch_sample,
                                                                   init_batch_perturbed,
                                                                   np.array(active_dataset_indices),
                                                                   i)

            perturbation_distances.append(perturbation_distance)
            perturbed_images.append(perturbed)
            if self.verbose:
                logger.info(f"Batch {i} ")
        return perturbed_images, perturbation_distances

    def distance_batch(self: Self,
                       batch_sample: Tensor,
                       init_batch_perturbed: Tensor,
                       active_dataset_indices: np.ndarray,
                       b_i: int) -> Tuple[Tensor, Tensor]:
        """Compute the distance between a batch of perturbed images and the original batch of samples.

        Parameters
        ----------
        batch_sample : Tensor
            The original batch of samples.
        init_batch_perturbed : Tensor
            The batch of init perturbed images.
        active_dataset_indices : List[int]
            The indices of the active samples.
        b_i : int
            The batch index.

        Returns
        -------
        Tuple of the perturbed images and the perturbation distance.

        """
        # Find the d_0
        distance_zero = self.compute_distance(batch_sample, init_batch_perturbed)
        sum_intial_distance = torch_sum(distance_zero, dim=0)


        # Project the initialization to the boundary.
        self.current_batch_size = len(batch_sample)

        if self.verbose:
            logger.info(f"Batch {b_i} Initial distance: {sum_intial_distance/len(batch_sample)} per sample")

        perturbation_distance = 0
        perturbed_previous_step = init_batch_perturbed.clone().detach()


        for j in np.arange(self.num_iterations):
            current_iter = j + 1

            # Binary search to approach the boundary.
            perturbed, _ = self.binary_search_batch( batch_sample,
                                                    perturbed_previous_step,
                                                    active_dataset_indices)

            # Choose delta.
            delta = self.select_delta( batch_sample,
                                      perturbed_previous_step)

            # approximate gradient.
            gradf = self.approximate_gradient(perturbed,
                                                j,
                                                delta,
                                                active_dataset_indices)
            update = sign(gradf) if self.constraint == np.inf else gradf

            # search step size.
            if self.stepsize_search == "geometric_progression":
                # find step size.
                epsilon = self.geometric_progression_for_stepsize(batch_sample,
                                                                  update,
                                                                  perturbed,
                                                                  current_iter,
                                                                  active_dataset_indices,
                                                                  b_i)

                # Update the sample.
                updated_perturbed = self.clamping(perturbed + epsilon.view(len(epsilon),1,1,1) * update )
                self.check_in_range(updated_perturbed)

            # compute new distance.
            perturbation_distance = self.compute_distance(batch_sample, updated_perturbed)
            self.check_in_range(updated_perturbed)
            perturbed_previous_step = updated_perturbed.clone().detach()
            if self.verbose:
                logger.info(f"iteration: {current_iter}, avg. distance "
                                f"{torch_sum(perturbation_distance, dim=0)/len(batch_sample)} per sample")
                if (current_iter == self.num_iterations):
                    logger.info(f"iteration: {current_iter}, avg. distance {perturbation_distance}")

        return perturbed_previous_step, perturbation_distance


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
        images = images.to(float32)
        self.check_in_range(images)
        self.model.eval()
        with no_grad():
            prob = cat([self.model(images[i].unsqueeze(0)) for i in range(images.size(0))], dim=0)
        perturbed_label = max(prob, dim=1)[1]

        model_output_labels_tensor = self.model_output_labels[active_indices].clone().detach()

        decisions = perturbed_label != model_output_labels_tensor if self.y_target is None else perturbed_label == self.y_target
        return decisions, perturbed_label


    def get_random_noise(self: Self) -> Tuple[np.ndarray, List[int]]:
        """Generate random noise for each data point in the dataset.

        Returns
        -------
            Tuple[np.ndarray, List[int]]: A tuple containing the generated random noises and the ordered indices.

        """
        success = np.zeros(len(self.data_loader.dataset), dtype=bool)
        passed_random_noises = zeros((len(self.data_loader.dataset), self.image_shape[0],
                                      self.image_shape[1], self.image_shape[2]))
        active_indices = np.arange(self.batch_size)
        num_evals = 0
        # Find a misclassified random noise.
        while not success.all():
            active_data = []
            for index in active_indices:
                active_data.append(self.data_loader.dataset[index])
            # Flatten the list of tuples into a list of tensors
            all_tensors = [unpack_tuple[0] for unpack_tuple in active_data]

            # Stack all tensors into a single tensor
            active_data = stack(all_tensors)


            random_noises = self.clip_min + rand(active_data.shape) * (self.clip_max - self.clip_min)
            random_noises = random_noises.to(self.device)
            decision, _ = self.decision_function(random_noises, active_indices)

            for i, idx in enumerate(active_indices):
                if decision[i]:
                    success[idx] = True
                    passed_random_noises[idx] = random_noises[i]

            # Update active_indices with new data points
            active_indices = np.where(~success)[0][:self.batch_size]

            num_evals += 1
            if len(active_indices) == 0:
                logger.info("All data points in the batch have been successfully perturbed by random noise "
                                 f"after {num_evals} evaluations.")

        return passed_random_noises

    def initialize(self: Self) -> Tuple[np.ndarray, List[int]]:
        """Efficient Implementation of BlendedUniformNoiseAttack in Foolbox."""
        passed_random_noises = self.get_random_noise()
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
        active_indices = np.where(~success)[0][:self.batch_size]
        passed_init = zeros((len(self.data_loader.dataset), self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        init_orderd_indices = []
        batch_i = 0

        while not success.all():

            batch_init_ordered_indices = []
            active_data_batch = []

            for idx in active_indices:
                active_data_batch.append(self.data_loader.dataset[idx])
            all_tensors = [unpack_tuple[0] for unpack_tuple in active_data_batch]
            active_data_batch = stack(all_tensors).to(self.device)

            batch_active_indices =tensor(np.arange(len(active_indices))).to(self.device)
            batch_noise = passed_random_noises[active_indices]
            batch_mid = mid[active_indices]
            batch_low = low[active_indices]
            batch_high = high[active_indices]

            while len(batch_active_indices) > 0:
                batch_mid[batch_active_indices] = (batch_high[batch_active_indices] + batch_low[batch_active_indices]) / 2.0

                # Projections of the data: blended = (1-mid) * sample + mid * noise
                midds = batch_mid[batch_active_indices].reshape(-1, 1, 1, 1)
                sampels = active_data_batch[batch_active_indices]
                noises = batch_noise[batch_active_indices]
                blended = (1 - midds) * sampels  +  midds * noises
                blended = self.clamping(blended).to(self.device)
                original_indices = batch_active_indices + batch_i * self.batch_size
                decisions, _  = self.decision_function(blended, original_indices)

                # Update high where decisions is True
                batch_high[batch_active_indices[decisions]] = batch_mid[batch_active_indices[decisions]].clone().detach()

                # Update low where decisions is False
                batch_low[batch_active_indices[~decisions]] = batch_mid[batch_active_indices[~decisions]].clone().detach()

                # Update success where the difference between high and low is less than X
                mask_out = batch_high[batch_active_indices] - batch_low[batch_active_indices] <= 0.01

                for _ , i in enumerate(batch_active_indices):
                    if  mask_out[i]:
                        idx = i + batch_i * self.batch_size
                        success[idx] = True
                        sample = active_data_batch[i]
                        opt_noise = (1 - batch_high[i]) * sample + batch_high[i] * batch_noise[i]
                        passed_init[idx] = self.clamping(opt_noise)
                        batch_init_ordered_indices.append(idx)

                batch_active_indices = batch_active_indices[~mask_out]

            # Update active_indices with new data points
            active_indices = np.where(~success)[0][:self.batch_size]
            init_orderd_indices.extend(batch_init_ordered_indices)
            batch_i += 1

        return passed_init, init_orderd_indices


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
        if torch_any(torch_isnan(batch_pert)):
            raise ValueError("NaN values found in batch_pert")

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
            projected = (1-alphas) * original_image_batch + ( alphas)  * perturbed_images_batch
        elif self.constraint == np.inf:
            projected = clamp(
                perturbed_images_batch, original_image_batch - alphas, original_image_batch + alphas
            )
        return projected


    def geometric_progression_for_stepsize(self: Self,
                                           samples: Tensor,
                                           updates: Tensor,
                                           perturbed: Tensor,
                                           current_iteration: int,
                                           active_indices: np.ndarray, b_i:int) -> Tensor:
        """Calculates the step size for geometric progression in the HopSkipJumpDistance algorithm.

        Args:
        ----
            samples (Tensor): The input samples.
            updates (Tensor): The updates to be applied to the samples.
            perturbed (Tensor): perturbed
            current_iteration (int): The current iteration of the algorithm.
            active_indices (np.ndarray): The active indices.
            b_i (int): The value of b_i.

        Returns:
        -------
            Tensor: The calculated step size for geometric progression.

        """
        dist = self.compute_distance(samples, perturbed )
        batch_epsilon = dist / np.sqrt(current_iteration)

        success = np.zeros(len(samples), dtype=bool)
        batch_active_indices = np.arange(len(samples))
        num_evals = 0

        while not success.all():

            active_disturbed = perturbed[batch_active_indices]
            active_updates = updates[batch_active_indices]
            active_epsilon = batch_epsilon[batch_active_indices].view(len(batch_active_indices), 1, 1, 1)

            active_updateed_samples = active_disturbed + active_epsilon * active_updates
            active_updateed_samples = self.clamping(active_updateed_samples)
            decisions, _  = self.decision_function(active_updateed_samples,
                                                   active_indices[batch_active_indices])
            passed_index= []
            for i, idx in enumerate(batch_active_indices):
                if decisions[i]:
                    success[idx] = True
                    passed_index.append(idx)
                else:
                    batch_epsilon[idx] =  batch_epsilon[idx] / 2.0
                    if batch_epsilon[idx] < self.epsilon_threshold:
                        success[idx] = True
                        batch_epsilon[idx] = 0
                        passed_index.append(idx)
                        logger.info(f"Batch {b_i},data point {idx}, iter {num_evals}: epsilon is less than threshold")

            if passed_index:
                # Find the positions in batch_active_indices corresponding to the values in passed_index
                positions_to_delete = [i for i, val in enumerate(batch_active_indices) if val in passed_index]

                batch_active_indices = np.delete(batch_active_indices, positions_to_delete)
            num_evals += 1
        return batch_epsilon


    def select_delta(self: Self,
                     samples: Tensor,
                     previous_perturbed: Tensor) -> Tensor:
        """Selects the delta value based on the given distance post update and current iteration.

        Args:
        ----
            samples (Tensor): The input samples.
            previous_perturbed (Tensor): The previous perturbed samples.

        Returns:
        -------
            Tensor: The selected delta value.

        Raises:
        ------
            None

        """
        dist_post_update = self.compute_distance(samples, previous_perturbed)
        d = int(np.prod(self.image_shape))
        if self.constraint == 2:
            # Proposed by the authors of the paper (Appendix B):
             delta_batch = 10  * np.sqrt(d) * self.theta * dist_post_update
        elif self.constraint == np.inf:
            delta_batch = d * self.theta * dist_post_update
        else:
            raise ValueError("The constraint should be either 2 or np.inf")

        return delta_batch.to(self.device)


    def approximate_gradient(
        self: Self,
        perturbed_batch: Tensor,
        iteration: int,
        delta: Tensor,
        active_indices: np.ndarray) -> Tensor:
        """Approximates the gradient of a function using random vectors.

        Args:
        ----
            perturbed_batch (Tensor): The input perturbed_batch.
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
        # size of the random vectors
        num_evals = int(builtins.min(self.initial_num_evals * np.sqrt(iteration + 1),
                                        self.max_num_evals))
        approx_grad = zeros((len(perturbed_batch),) + self.image_shape).to(self.device)

        for i, perturbed in enumerate(perturbed_batch):
            # create random vectors
            noise_shape = [num_evals] + list(self.image_shape)
            if self.constraint == 2:
                randv = rand(noise_shape)
            elif self.constraint == np.inf:
                randv = -1 + rand(noise_shape) * 2
            rv = randv.to(self.device)

            update_perturbed = perturbed + delta[i] * rv
            update_perturbed = self.clamping(update_perturbed)

            if torch_any(torch_isnan(update_perturbed)):
                raise ValueError(f"NaN found in perturbed at iteration {iteration}, index {i}")


            sample_index = np.full(num_evals, active_indices[i])
            decisions, _ = self.decision_function(update_perturbed, sample_index)
            phis = 2.0 * decisions.view((decisions.shape[0],) +
                                        (1,) * len(self.image_shape)) - 1.0
            if torch_any(torch_isnan(phis)):
                raise ValueError(f"NaN found in phis at iteration {iteration}, index {i}")

            # if the mean of phis is 1, then the label changes, and if it is -1, then the label does not change.
            # in both cases, the piont is on the boundary, so equation 9 is used.
            mean_phis = mean(phis)
            if mean_phis == 1.0:
                gradient_direction = mean(rv, dim=0)
            elif mean_phis == -1.0:
                gradient_direction = -mean(rv, dim=0)
            else:
                # Equation 16 in the paper
                gradient_direction = sum( (phis - mean(phis))* rv) /(num_evals - 1)

            # Equation 12 in the paper
            approx_grad[i] = gradient_direction / norm(gradient_direction, p=2)

        return approx_grad


    def binary_search_batch(self: Self,
                            original_image_batch: Tensor, perturbed_images_batch: Tensor,
                            active_indices: np.ndarray
                            ) -> Tuple[Tensor, Tensor, Tensor]:
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
        out_images = zeros((len(perturbed_images_batch),) + self.image_shape).to(self.device)
        dists_pre_update = self.compute_distance(original_image_batch, perturbed_images_batch)
        # Choose upper thresholds in binary searches based on constraint.
        if self.constraint == np.inf:
            highs = dists_pre_update
            # Stopping criteria.
            thresholds = min(dists_pre_update * self.theta, self.theta)
        else:
            highs = ones(len(perturbed_images_batch)).to(self.device)
            thresholds = self.theta

        lows = zeros(len(perturbed_images_batch)).to(self.device)
        mids = zeros(len(perturbed_images_batch)).to(self.device)

        active_data_batch = []
        for idx in active_indices:
            active_data_batch.append(self.data_loader.dataset[idx])
        all_tensors = [unpack_tuple[0] for unpack_tuple in active_data_batch]
        active_data_batch = stack(all_tensors).to(self.device)
        batch_active_indices =np.arange(len(active_indices))

        while len(batch_active_indices) > 0:
            mids[batch_active_indices] = (highs[batch_active_indices] + lows[batch_active_indices]) / 2.0
            blended_images = self.project(original_image_batch[batch_active_indices],
                                          perturbed_images_batch[batch_active_indices],
                                          mids[batch_active_indices])
            blended_images = self.clamping(blended_images)
            decisions, _ = self.decision_function(blended_images,
                                                active_indices[batch_active_indices])
            decisions = decisions.cpu().numpy()

            # Update high where decisions is True
            highs[batch_active_indices[decisions]] = mids[batch_active_indices[decisions]].clone().detach()

            # Update low where decisions is False
            lows[batch_active_indices[~decisions]] = mids[batch_active_indices[~decisions]].clone().detach()

            # Update success where the difference between high and low is less than X
            a = (highs[batch_active_indices] - lows[batch_active_indices]) / thresholds
            mask_out = (a.cpu().numpy() <= 1)
            for _ , i in enumerate(batch_active_indices):
                if  mask_out[i]:
                    idx = active_indices[batch_active_indices[i]]
                    sample = original_image_batch[i]
                    perturbed = perturbed_images_batch[i]
                    out_images[i] = self.project(sample.unsqueeze(0), perturbed.unsqueeze(0), highs[i].unsqueeze(0))
                    out_images[i] = self.clamping(out_images[i])
                    if self.decision_function(out_images[i].unsqueeze(0), idx)[0] is False:
                        logger.info(f"Using alpha_ mid instead of high for data point {idx}")
                        out_images[i] = blended_images[i]
            batch_active_indices = batch_active_indices[~mask_out]

        return out_images, dists_pre_update


    def clamping(self: Self, data: Tensor) -> Tensor:
        """Scaling function for the input tensor.

        Args:
        ----
            data (Tensor): The input tensor

        Returns:
        -------
            Tensor: The scaled tensor.

        """
        return  clamp(data, self.clip_min, self.clip_max)

    def check_in_range(self: Self, x: Tensor) -> Tensor:
        """Check if the input tensor is within the specified range.

        Args:
        ----
            x (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The tensor with values clamped to the specified range.

        """
        same_max_min =  bool(min(x) == self.clip_max or max(x) == self.clip_min)
        assert same_max_min is False, "The input tensor is out of range"

        out_of_range = bool(min(x) < self.clip_min or max(x) > self.clip_max)
        assert out_of_range is False, "The input tensor is out of range"
