# This file is part of celevrhans and is released under MIT License.
# Copyright (c) 2019 Google Inc., OpenAI and Pennsylvania State University
# See https://github.com/cleverhans-lab/cleverhans?tab=MIT-1-ov-file#readme for details.
import numpy as np
from torch import cuda, device, max, min, clamp, ones, zeros, stack, where, randn, rand, sign, no_grad, cat, from_numpy, norm, mean 
from torch import sum as torch_sum, sqrt as torch_sqrt
import builtins

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

class HopSkipJumpDistance:
    """
    PyTorch implementation of HopSkipJumpAttack.
    HopSkipJumpAttack was originally proposed by Chen, Jordan and Wainwright.
    It is a decision-based attack that requires access to output
    labels of a model alone.
    Paper link: https://arxiv.org/abs/1904.02144
    At a high level, this attack is an iterative attack composed of three
    steps: Binary search to approach the boundary; gradient estimation;
    stepsize search. HopSkipJumpAttack requires fewer model queries than
    Boundary Attack which was based on rejective sampling.

    :param model_fn: a callable that takes an input tensor and returns the model logits.
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
    def __init__(self, 
                 model, 
                 data_loader, 
                 norm, 
                 y_target=None, 
                 image_target=None, 
                 initial_num_evals=100, 
                 max_num_evals=10000, 
                 stepsize_search="geometric_progression", 
                 num_iterations=100, 
                 gamma=1.0, 
                 constraint=2, 
                 batch_size=128, 
                 verbose=True, 
                 clip_min=-1, 
                 clip_max=1):
        
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
        self.clip_min = clip_min
        self.clip_max = clip_max

        # TODO: Chnage this
        shape = (1, 3, 32, 32)
        d = int(np.prod(shape))

        if self.constraint == 2:
            self.theta = self.gamma / (np.sqrt(d) * d)
        else:
            self.theta = self.gamma / (d * d)

        self.HopSkipJump()

    def HopSkipJump(self):

        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")

        if self.y_target is not None:
            assert self.image_target is not None, "Require a target image for targeted attack."
        if self.clip_min is not None and self.clip_max is not None:
            if self.clip_min > self.clip_max:
                raise ValueError(
                    "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                        self.clip_min, self.clip_max
                    )
                )

        # Compute the distance on one instance at a time
        adv_x = []
        perturbation_distances = []
        for i, (batch_data, batch_label) in enumerate(tqdm(self.dataloader, desc=f"Epoch")):
            if i> 10000:
                break
            else:
                print(f"Processing {i} data points")
                if self.verbose:
                    print(f"Traget image and its label: {batch_label}")
                    plt.figure(figsize=(5, 3)) 
                    plt.imshow(((batch_data+1)/2).squeeze().permute(1, 2, 0).cpu().numpy())
                    plt.show()

                if self.y_target is not None:
                    # targeted attack that requires target label and image.
                    pert, perturbation_distance = self.hsja(batch_data,
                                                            self.y_target[i],
                                                            self.image_target[i],
                                                            gpu_or_cpu)
                else:
                    if self.image_target is not None:
                        pert, perturbation_distance = self.hsja(batch_data,
                                                                None,
                                                                self.image_target[i],
                                                                gpu_or_cpu)
                    else:
                        # untargeted attack without an initialized image.
                        pert, perturbation_distance = self.hsja(batch_data,
                                                                 None,
                                                                 None,
                                                                 gpu_or_cpu)
                adv_x.append(pert)
                perturbation_distances.append(perturbation_distance)
        return cat(adv_x, 0), perturbation_distances





    def decision_function(self, images ):
                """
                Decision function output 1 on the desired side of the boundary,
                0 otherwise.
                """
                # model_fn.to(device)
                # images_permuted = images.permute(0, 3, 1, 2)
                gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
                images = clamp(images, self.clip_min, self.clip_max)
                prob = []
                for i in range(0, len(images), self.batch_size):
                    batch = images[i : i + self.batch_size]
                    batch = batch.to(gpu_or_cpu)
                    with no_grad():
                        prob_i = self.model_fn(batch)
                    prob.append(prob_i)
                prob = cat(prob, dim=0)
                perturbed_label = max(prob, dim=1)[1]
                if self.target_label is None:
                    return max(prob, dim=1)[1] != self.original_label, perturbed_label
                else:
                    return max(prob, dim=1)[1] == self.target_label, perturbed_label

    def hsja(self, sample, target_label, target_image, gpu_or_cpu):
            
            
            self.model_fn.eval()
            
            if target_label is None:
                with no_grad():
                    # sample_permuted = sample.permute(0, 3, 1, 2)
                    y_sample = self.model_fn(sample.to(gpu_or_cpu))
                    # print(y_sample)
                    _, original_label = max(y_sample, 1)

            

            # Initialize.
            if target_image is None:
                perturbed = self.initialize(self.decision_function,self.model_fn, 
                                            sample, self.shape, self.clip_min, self.clip_max)
            else:
                perturbed = target_image.to(gpu_or_cpu)

            # Project the initialization to the boundary.
            perturbed, dist_post_update, perturbed_label = self.binary_search_batch(
                sample, perturbed, self.decision_function, self.shape, self.constraint, self.theta
            )
            dist = self.compute_distance(perturbed, sample, self.constraint)

            perturbation_distance = 0
            for j in np.arange(self.num_iterations):
                current_iteration = j + 1

                # Choose delta.
                delta = self.select_delta(
                    dist_post_update,
                    current_iteration,
                    self.clip_max,
                    self.clip_min,
                    self.d,
                    self.theta,
                    self.constraint,
                )

                # Choose number of evaluations.
                num_evals = int(builtins.min(self.initial_num_evals * np.sqrt(j + 1), self.max_num_evals))

                # approximate gradient.
                gradf = self.approximate_gradient(
                    self.decision_function,
                    perturbed,
                    num_evals,
                    delta,
                    self.constraint,
                    self.shape[1:],
                    self.clip_min,
                    self.clip_max,
                )
                if self.constraint == np.inf:
                    update = sign(gradf)
                else:
                    update = gradf

                # search step size.
                if self.stepsize_search == "geometric_progression":
                    # find step size.
                    epsilon = self.geometric_progression_for_stepsize(
                        perturbed, update, dist, self.decision_function, current_iteration
                    )

                    # Update the sample.
                    perturbed = clamp(
                        perturbed + epsilon * update, self.clip_min, self.clip_max
                    )

                    # Binary search to return to the boundary.
                    perturbed, dist_post_update , perturbed_label= self.binary_search_batch(
                        sample, perturbed, self.decision_function, self.shape, self.constraint, self.theta
                    )

                elif self.stepsize_search == "grid_search":
                    # Grid search for stepsize.
                    epsilons = (
                        from_numpy(np.logspace(-4, 0, num=20, endpoint=True))
                        .to(gpu_or_cpu)
                        .float()
                        * dist
                    )
                    perturbeds = (
                        perturbed + epsilons.view((20,) + (1,) * (len(self.shape) - 1)) * update
                    )
                    perturbeds = clamp(perturbeds, self.clip_min, self.clip_max)
                    idx_perturbed,perturbed_label = self.decision_function(perturbeds)

                    if torch_sum(idx_perturbed) > 0:
                        # Select the perturbation that yields the minimum distance # after binary search.
                        perturbed, dist_post_update, perturbed_label = self.binary_search_batch(
                            sample,
                            perturbeds[idx_perturbed],
                            self.decision_function,
                            self.shape,
                            self.constraint,
                            self.theta,
                        )

                # compute new distance.
                dist = self.compute_distance(perturbed, sample, self.constraint)
                if j == self.num_iterations - 1:
                    perturbation_distance = dist
                
                if j % 20 == 0 and self.verbose:
                    print(f"Perturbed image in iter: {j} and distance: {dist} , label: {perturbed_label}")
                    plt.figure(figsize=(4, 3)) 
                    plt.imshow(((perturbed+1)/2).squeeze().permute(1, 2, 0).cpu().numpy())
                    plt.show()
                    print( "iteration: {:d}, {:s} distance {:.4E}".format(
                            j + 1, str(self.constraint), dist
                        ))

                # if verbose:
                #     print(
                #         "iteration: {:d}, {:s} distance {:.4E}".format(
                #             j + 1, str(constraint), dist
                #         )
                #     )

            return perturbed, perturbation_distance


    def compute_distance(self, x_ori, x_pert, constraint=2):
        """ Compute the distance between two images. """

        x_ori = x_ori.to("cpu")
        x_pert = x_pert.to("cpu")
        if constraint == 2:
            dist = norm(x_ori - x_pert, p=2)
        elif constraint == np.inf:
            dist = max(abs(x_ori - x_pert))
        return dist


    def project(self, original_image, perturbed_images, alphas, shape, constraint):
        """ Projection onto given l2 / linf balls in a batch. """
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        alphas = alphas.to(gpu_or_cpu)
        original_image = original_image.to(gpu_or_cpu)
        perturbed_images = perturbed_images.to(gpu_or_cpu)
        alphas = alphas.view((alphas.shape[0],) + (1,) * (len(shape) - 1))
        if constraint == 2:
            projected = (1 - alphas) * original_image + alphas * perturbed_images
        elif constraint == np.inf:
            projected = clamp(
                perturbed_images, original_image - alphas, original_image + alphas
            )
        return projected


    def geometric_progression_for_stepsize(self, 
        x, update, dist, decision_function, current_iteration
    ):
        """Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary.
        """
        epsilon = dist / np.sqrt(current_iteration)
        while True:
            updated = x + epsilon * update
            success = decision_function(updated)[0]
            if success:
                break
            else:
                epsilon = epsilon / 2.0

        return epsilon


    def select_delta( slef,
        dist_post_update, current_iteration, clip_max, clip_min, d, theta, constraint
    ):
        """
        Choose the delta at the scale of distance
        between x and perturbed sample.
        """
        if current_iteration == 1:
            delta = 0.1 * (clip_max - clip_min)
        else:
            if constraint == 2:
                delta = np.sqrt(d) * theta * dist_post_update
            elif constraint == np.inf:
                delta = d * theta * dist_post_update

        return delta


    def approximate_gradient(
            self,
            decision_function,
            sample,
            num_evals,
            delta,
            constraint,
            shape,
            clip_min,
            clip_max
             ):
            """ Gradient direction estimation """
            # Generate random vectors.
            gpu_or_cpu =device("cuda" if cuda.is_available() else "cpu")
            noise_shape = [num_evals] + list(shape)
            if constraint == 2:
                rv = randn(noise_shape)
            elif constraint == np.inf:
                rv = -1 + rand(noise_shape) * 2

            axis = tuple(range(1, 1 + len(shape)))
            rv = rv / torch_sqrt(torch_sum(rv ** 2, dim=axis, keepdim=True))
            perturbed = sample + delta * rv.to(gpu_or_cpu)
            perturbed = clamp(perturbed, clip_min, clip_max)
            rv = (perturbed - sample) / delta

            # query the model.
            decisions, _ = decision_function(perturbed)
            fval = 2.0 * decisions.view((decisions.shape[0],) + (1,) * len(shape)) - 1.0

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
            gradf = gradf / norm(gradf, p=2)

            return gradf



    def binary_search_batch(self, 
        original_image, perturbed_images, decision_function, shape, constraint, theta
    ):
        """ Binary search to approach the boundary. """
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        # Compute distance between each of perturbed image and original image.
        dists_post_update = stack(
            [
                self.compute_distance(original_image, perturbed_image, constraint)
                for perturbed_image in perturbed_images
            ]
        )

        # Choose upper thresholds in binary searchs based on constraint.
        if constraint == np.inf:
            highs = dists_post_update
            # Stopping criteria.
            thresholds = min(dists_post_update * theta, theta)
        else:
            highs = ones(len(perturbed_images)).to(gpu_or_cpu)
            thresholds = theta

        lows = zeros(len(perturbed_images)).to(gpu_or_cpu)

        while max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_images = self.project(original_image, perturbed_images, mids, shape, constraint)

            # Update highs and lows based on model decisions.
            decisions, perturbed_label = decision_function(mid_images)
            lows = where(decisions == 0, mids, lows)
            highs = where(decisions == 1, mids, highs)

        out_images = self.project(original_image, perturbed_images, highs, shape, constraint)

        # Compute distance of the output image to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = stack(
            [
                self.compute_distance(original_image, out_image, constraint)
                for out_image in out_images
            ]
        )
        _, idx = min(dists, 0)

        dist = dists_post_update[idx]
        out_image = out_images[idx].unsqueeze(0)
        return out_image, dist, perturbed_label


    def initialize(self, decision_function, model_fn, sample, shape, clip_min, clip_max):
        """
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        sample = sample.to(gpu_or_cpu)
        model_fn.eval()


        # Find a misclassified random noise.
        while True:
            random_noise = clip_min + rand(shape).to(gpu_or_cpu) * (
                clip_max - clip_min
            )
            # random_noise_permuted = random_noise.permute(0, 3, 1, 2)
            success= decision_function(random_noise)[0]
            if success:
                print(f"success in finding a misclassified random noise.{num_evals}")
                break
            num_evals += 1
            # if(num_evals%1000==0):
            #     print(f"try {num_evals} times to find a misclassified random noise.")
            message = (
                "Initialization failed! Try to use a misclassified image as `target_image`"
            )
            assert num_evals < 1e6, message

        # Binary search to minimize l2 distance to original image.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
            mid = (high + low) / 2.0
            blended = (1 - mid) * sample + mid * random_noise
            # blended_permuted = blended.permute(0, 3, 1, 2)
            success = decision_function(blended)[0]
            if success:
                high = mid
            else:
                low = mid

        initialization = (1 - high) * sample + high * random_noise
        return initialization









