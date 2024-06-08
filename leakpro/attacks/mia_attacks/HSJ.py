import os
import pickle
from logging import Logger

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F  # noqa: N812
from torch import argmax, cuda, device, load, no_grad, optim, save, tensor
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.attack_data import get_attack_data
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import HopSkipJumpDistance


import torch
import torch.nn as nn
# from cleverhans.torch.attacks import HopSkipJump, CarliniWagnerL2
from leakpro.attacks.mia_attacks.hop_skip_jump_attack import hop_skip_jump_attack


class CallableModelWrapper(torch.nn.Module):
    def __init__(self,callable_fn, output_layer):
        super(CallableModelWrapper, self).__init__()
        self.output_layer = output_layer
        self.callable_fn = callable_fn

    def forward(self, x, **kwargs):
        output = self.callable_fn(x, **kwargs)
        if self.output_layer == "probs":
            assert output.max().item() <= 1.0
            assert output.min().item() >= 0.0
        elif self.output_layer == "logits":
            assert not torch.allclose(output, F.softmax(output, dim=-1))
        return output
    
class AttackHopSkipJump(AbstractMIA):
    def __init__(self: Self,
                 population: np.ndarray,
                 audit_dataset: dict,
                 target_model: nn.Module,
                 logger: Logger,
                 configs: dict
                ) -> None:
        super().__init__(population, audit_dataset, target_model, logger)

        self.logger.info("Configuring label only attack")
        self._configure_attack(configs)
        self.signal = HopSkipJumpDistance()


    def _configure_attack(self:Self,
                          configs: dict) -> None:
        """Configure the attack using the configurations."""
        self.configs = configs
        self.target_metadata_path = configs.get("trained_model_metadata_path", "./target/model_metadata.pkl")
        with open(self.target_metadata_path, "rb") as f:
             self.target_model_metadata = pickle.load(f)
        
        target_train_indices = self.target_model_metadata["model_metadata"]["train_indices"]
        target_test_indices = self.target_model_metadata["model_metadata"]["test_indices"]
        self.target_train_dataset =  self.population.subset(target_train_indices)
        self.target_test_dataset = self.population.subset(target_test_indices)
        

        self.attack_data_fraction = configs.get("attack_data_fraction", 0.4)
        self.num_shadow_models = configs.get("num_shadow_models", 1)
        self.norm = configs.get("norm", 2)
        self.y_target = configs.get("y_target", None)
        self.image_target = configs.get("image_target", None)
        self.initial_num_evals = configs.get("initial_num_evals", 100)
        self.max_num_evals = configs.get("max_num_evals", 10000)
        self.stepsize_search = configs.get("stepsize_search", "geometric_progression")
        self.num_iterations = configs.get("num_iterations", 100)
        self.gamma = configs.get("gamma", 1.0)
        self.constraint = configs.get("constraint", 2)
        self.batch_size = configs.get("batch_size", 128)
        self.verbose = configs.get("verbose", True)
        self.clip_min = configs.get("clip_min", -1)
        self.clip_max = configs.get("clip_max", 1)



    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Label-Only Membership Inference Attacks"
        reference_str = "Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini and Nicolas Papernot\
            Label-Only Membership Inference Attacks. (2020)."
        summary_str = ""
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }
    
    def prepare_attack(self:Self) -> None:
        """Prepare the attack by loading the shadow model and target model.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """
        self.logger.info("Preparing the data for Hop Skip Jump attack")
        include_target_training_data = False
        include_target_testing_data = False

        # Get all available indices for the only shadow model dataset
        shadow_data_index = get_attack_data(
            self.population_size,
            self.train_indices,
            self.test_indices,
            include_target_training_data,
            include_target_testing_data,
            self.logger
        )

        # create auxiliary dataset
        shadow_data_size = len(shadow_data_index)
        shadow_train_data_size = int(shadow_data_size * self.attack_data_fraction)
        shadow_train_data_indices = np.random.choice(shadow_data_index, shadow_train_data_size, replace=False)
        shadow_test_data_indices = np.setdiff1d(shadow_data_index, shadow_train_data_indices)
    
        # np.random.shuffle(shadow_train_data_indices)
        self.shadow_train_dataset = self.population.subset(shadow_train_data_indices)
        self.shadow_test_dataset = self.population.subset(shadow_test_data_indices)


        # train shadow models
        self.logger.info(f"Training shadow models on {len(self.shadow_train_dataset)} points")
        ShadowModelHandler().create_shadow_models(
            self.num_shadow_models,
            self.shadow_train_dataset,
            shadow_train_data_indices,
            training_fraction = 1.0,
            retrain= True,
        )
        # load shadow models
        self.shadow_models, self.shadow_model_indices = \
            ShadowModelHandler().get_shadow_models(self.num_shadow_models)
        self.shadow_metadata = ShadowModelHandler().get_shadow_model_metadata(1)
        


    def run_attack(self:Self) -> CombinedMetricResult:
        """Run the attack and return the combined metric result.

        Returns
        -------
            CombinedMetricResult: The combined metric result containing predicted labels, true labels,
            predictions probabilities, and signal values.

        """

        target_model = self.target_model.model_obj
        shadow_model = self.shadow_models[0].model_obj


        self.logger.info("Running Hop Skip Jump distance attack")
        results = self.HopSkipJump_distance(target_model,
                                          shadow_model)
        
        
        
        HSJ_distance_shadow_in, perturbation_distances_in = self.dists(shadow_model, 
                                            self.shadow_train_dataset,
                                            attack="HSJ", 
                                            max_samples= self.max_samples,
                                            input_dim=[None, self.input_dim[0], self.input_dim[1], self.input_dim[2]],
                                            n_classes=n_classes)
        HSJ_distance_shadow_out, perturbation_distances_out = self.dists(shadow_model,
                                             self.shadow_test_dataset,
                                             attack="HSJ",
                                             max_samples=self.max_samples,
                                            input_dim=[None, self.input_dim[0], self.input_dim[1], self.input_dim[2]],
                                            n_classes=self.n_classes)
        
        # dists_target = np.concatenate([HSJ_distance_target_in, HSJ_distance_target_out], axis=0)
        dist_shadow = np.concatenate([perturbation_distances_in, perturbation_distances_out], axis=0)

    
        # create thresholds
        min_signal_val = np.min(dist_shadow)
        max_signal_val = np.max(dist_shadow)
        thresholds = np.linspace(min_signal_val, max_signal_val,1000)
        num_threshold = len(thresholds)


        # compute the signals for the in-members and out-members
        member_signals = (np.array(perturbation_distances_in).reshape(-1, 1).repeat(num_threshold, 1).T)
        non_member_signals = (np.array(perturbation_distances_out).reshape(-1, 1).repeat(num_threshold, 1).T)
        member_preds = np.less(member_signals, thresholds)
        non_member_preds = np.less(non_member_signals, thresholds)
        
        # what does the attack predict on test and train dataset
        predictions = np.concatenate([member_preds, non_member_preds], axis=1)
        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [
                np.ones(len(perturbation_distances_in)),
                np.zeros(len(perturbation_distances_out)),
            ]
        )
        signal_values = np.concatenate(
            [perturbation_distances_in, perturbation_distances_out]
        )

        # member_preds = np.greater(perturbation_distances_in, thresholds).T
        # non_member_preds = np.greater(perturbation_distances_out, thresholds).T

        # what does the attack predict on test and train dataset
        # predictions = np.concatenate([member_preds, non_member_preds])[None, :]
        # # set true labels for being in the training dataset
        # true_labels = np.concatenate(
        #     [
        #         np.ones(len(perturbation_distances_in)),
        #         np.zeros(len(perturbation_distances_out)),
        #     ]
        # )
        # signal_values = np.concatenate(
        #     [perturbation_distances_in, perturbation_distances_out]
        # )

        # compute ROC, TP, TN etc
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
        )

        # return results




    def HopSkipJump_distance(self:Self, 
                             target_model, 
                             shadow_model,
                             max_samples=100,
                             input_dim=[None, 32, 32, 3], n_classes=10):

        HSJ_distance_shadow_in, perturbation_distances_in = self.dists(shadow_model, 
                                            self.shadow_train_dataset,
                                            attack="HSJ", 
                                            max_samples=max_samples,
                                            input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                            n_classes=n_classes)
        HSJ_distance_shadow_out, perturbation_distances_out = self.dists(shadow_model,
                                             self.shadow_test_dataset,
                                             attack="HSJ",
                                             max_samples=max_samples,
                                            input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                            n_classes=n_classes)
        
        # dists_target = np.concatenate([HSJ_distance_target_in, HSJ_distance_target_out], axis=0)
        dist_shadow = np.concatenate([perturbation_distances_in, perturbation_distances_out], axis=0)

    
        # create thresholds
        min_signal_val = np.min(dist_shadow)
        max_signal_val = np.max(dist_shadow)
        thresholds = np.linspace(min_signal_val, max_signal_val,1000)
        num_threshold = len(thresholds)


        # compute the signals for the in-members and out-members
        member_signals = (np.array(perturbation_distances_in).reshape(-1, 1).repeat(num_threshold, 1).T)
        non_member_signals = (np.array(perturbation_distances_out).reshape(-1, 1).repeat(num_threshold, 1).T)
        member_preds = np.less(member_signals, thresholds)
        non_member_preds = np.less(non_member_signals, thresholds)
        
        # what does the attack predict on test and train dataset
        predictions = np.concatenate([member_preds, non_member_preds], axis=1)
        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [
                np.ones(len(perturbation_distances_in)),
                np.zeros(len(perturbation_distances_out)),
            ]
        )
        signal_values = np.concatenate(
            [perturbation_distances_in, perturbation_distances_out]
        )

        # member_preds = np.greater(perturbation_distances_in, thresholds).T
        # non_member_preds = np.greater(perturbation_distances_out, thresholds).T

        # what does the attack predict on test and train dataset
        # predictions = np.concatenate([member_preds, non_member_preds])[None, :]
        # # set true labels for being in the training dataset
        # true_labels = np.concatenate(
        #     [
        #         np.ones(len(perturbation_distances_in)),
        #         np.zeros(len(perturbation_distances_out)),
        #     ]
        # )
        # signal_values = np.concatenate(
        #     [perturbation_distances_in, perturbation_distances_out]
        # )

        # compute ROC, TP, TN etc
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
        )



    # def confidence_vector_attack(self:Self, target_model, shadow_model) -> None:

    #     target_model_labels = np.concatenate([self.target_train_dataset._labels,
    #                                          self.target_test_dataset._labels ], axis=0)
    #     shadow_model_labels = np.concatenate([self.shadow_train_dataset._labels,
    #                                           self.shadow_test_dataset._labels], axis=0)
                
    #     target_outputs_in = self.call_model(target_model, self.target_train_dataset)
    #     target_outputs_out = self.call_model(target_model,self.target_test_dataset)

    #     shadow_outputs_in = self.call_model(shadow_model, self.shadow_train_dataset)
    #     shadow_outputs_out = self.call_model(shadow_model, self.shadow_test_dataset)    

    #     target_outputs_in = self.softmax(target_outputs_in)
    #     target_outputs_out = self.softmax(target_outputs_out)
    #     shadow_outputs_in = self.softmax(shadow_outputs_in)
    #     shadow_outputs_out = self.softmax(shadow_outputs_out)

    #     target_features = np.concatenate([target_outputs_in, target_outputs_out], axis=0)
    #     shadow_features = np.concatenate([shadow_outputs_in, shadow_outputs_out], axis=0)
    #     print(f"shadow_features: {shadow_features.shape}, target_features: {target_features.shape}")

    #     target_membership_status = np.concatenate([np.ones(len(self.target_train_dataset)),
    #                             np.zeros(len(self.target_test_dataset))], axis=0)
    #     shadow_membership_status = np.concatenate([np.ones(len(self.shadow_train_dataset)),
    #                             np.zeros(len(self.shadow_test_dataset))], axis=0)

    #     # just look at confidence in predicted label
    #     conf_shadow = np.max(shadow_features, axis=-1)
    #     conf_target = np.max(target_features, axis=-1)
    #     # print("threshold on predicted label:")
    #     # acc1, prec1, _, _ = self.get_threshold(shadow_membership_status, conf_shadow,
    #     #                                        target_membership_status, conf_target)

    #     # look at confidence in true label
    #     conf_shadow= shadow_features[range(len(shadow_features)), shadow_model_labels]
    #     conf_target = target_features[range(len(target_features)), target_model_labels]
    #     # print("threshold on true label:")
    #     # acc2, prec2, _, _ = self.get_threshold(shadow_membership_status, conf_shadow,
    #     #                                   target_membership_status, conf_target)
        
    #     # NOTE: In the original code, the threshold leading to max accuracy in shadow model is used as a threshold for the target model signals 
    #     min_signal_val = np.min(np.concatenate([self.in_member_signals, self.out_member_signals]))
    #     max_signal_val = np.max(np.concatenate([self.in_member_signals, self.out_member_signals]))
    #     thresholds = np.linspace(min_signal_val, max_signal_val, 1000)

    #     member_preds = np.greater(self.in_member_signals, thresholds).T
    #     non_member_preds = np.greater(self.out_member_signals, thresholds).T

    #     # what does the attack predict on test and train dataset
    #     predictions = np.concatenate([member_preds, non_member_preds], axis=1)
    #     # set true labels for being in the training dataset
    #     true_labels = np.concatenate(
    #         [
    #             np.ones(len(self.in_member_signals)),
    #             np.zeros(len(self.out_member_signals)),
    #         ]
    #     )
    #     signal_values = np.concatenate(
    #         [self.in_member_signals, self.out_member_signals]
    #     )

    #     # compute ROC, TP, TN etc
    #     return CombinedMetricResult(
    #         predicted_labels=predictions,
    #         true_labels=true_labels,
    #         predictions_proba=None,
    #         signal_values=signal_values,
    #     )


    # def get_max_accuracy(self, y_true, probs, thresholds=None):
    #     """Return the max accuracy possible given the correct labels and guesses. Will try all thresholds unless passed.

    #     Args:
    #         y_true: True label of `in' or `out' (member or non-member, 1/0)
    #         probs: The scalar to threshold
    #         thresholds: In a blackbox setup with a shadow/source model, the threshold obtained by the source model can be passed
    #         here for attackin the target model. This threshold will then be used.

    #     Returns: max accuracy possible, accuracy at the threshold passed (if one was passed), the max precision possible,
    #     and the precision at the threshold passed.

    #     """
    #     if thresholds is None:
    #         fpr, tpr, thresholds = roc_curve(y_true, probs)

    #     accuracy_scores = []
    #     precision_scores = []
    #     for thresh in thresholds:
    #         accuracy_scores.append(accuracy_score(y_true,
    #                                             [1 if m > thresh else 0 for m in probs]))
    #         precision_scores.append(precision_score(y_true, [1 if m > thresh else 0 for m in probs]))

    #     accuracies = np.array(accuracy_scores)
    #     precisions = np.array(precision_scores)
    #     max_accuracy = accuracies.max()
    #     max_precision = precisions.max()
    #     max_accuracy_threshold = thresholds[accuracies.argmax()]
    #     max_precision_threshold = thresholds[precisions.argmax()]
    #     return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold


    # def get_threshold(self, ource_m, source_stats, target_m, target_stats):
    #     """ Train a threshold attack model and get teh accuracy on source and target models.

    #     Args:
    #         source_m: membership labels for source dataset (1 for member, 0 for non-member)
    #         source_stats: scalar values to threshold (attack features) for source dataset
    #         target_m: membership labels for target dataset (1 for member, 0 for non-member)
    #         target_stats: scalar values to threshold (attack features) for target dataset

    #     Returns: best acc from source thresh, precision @ same threshold, threshold for best acc,
    #         precision at the best threshold for precision. all tuned on source model.

    #     """
    #     # find best threshold on source data
    #     acc_source, t, prec_source, tprec = self.get_max_accuracy(source_m, source_stats)

    #     # find best accuracy on test data (just to check how much we overfit)
    #     acc_test, _, prec_test, _ = self.get_max_accuracy(target_m, target_stats)

    #     # get the test accuracy at the threshold selected on the source data
    #     acc_test_t, _, _, _ = self.get_max_accuracy(target_m, target_stats, thresholds=[t])
    #     _, _, prec_test_t, _ = self.get_max_accuracy(target_m, target_stats, thresholds=[tprec])
    #     print("acc src: {}, acc test (best thresh): {}, acc test (src thresh): {}, thresh: {}".format(acc_source, acc_test,
    #                                                                                                     acc_test_t, t))
    #     print(
    #         "prec src: {}, prec test (best thresh): {}, prec test (src thresh): {}, thresh: {}".format(prec_source, prec_test,
    #                                                                                                 prec_test_t, tprec))

    #     return acc_test_t, prec_test_t, t, tprec
    
    
    # def continuous_rand_robust(self, model, ds, max_samples=100, noise_samples=2500, stddev=0.025, input_dim=[None, 32, 32, 3],
    #                        num=[1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000, 2500]):
    #     """Calculate robustness to random noise for Adv-x MI attack on continuous-featured datasets (+ UCI adult).

    #     :param model: model to approximate distances on (attack).
    #     :param ds: PyTorch dataset should be either the training set or the test set.
    #     :param max_samples: maximum number of samples to take from the ds
    #     :param noise_samples: number of noised samples to take for each sample in the ds.
    #     :param stddev: the standard deviation to use for Gaussian noise (only for Adult, which has some continuous features)
    #     :param input_dim: dimension of inputs for the dataset.
    #     :param num: subnumber of samples to evaluate. max number is noise_samples
    #     :return: a list of lists. each sublist of the accuracy on up to $num noise_samples.
    #     """
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model.to(device)
    #     model.eval()

    #     data_loader = torch.utils.data.DataLoader(ds, batch_size=max_samples, shuffle=True)
    #     robust_accs = [[] for _ in num]
    #     all_correct = []
    #     num_samples = 0

    #     with torch.no_grad():
    #         for xbatch, ybatch in data_loader:
    #             if num_samples >= max_samples:
    #                 break

    #             xbatch, ybatch = xbatch.to(device), ybatch.to(device)
    #             labels = ybatch.argmax(dim=-1)
    #             y_pred = model(xbatch).argmax(dim=-1)
    #             correct = (y_pred == labels).cpu().numpy()
    #             all_correct.extend(correct)

    #             for i in range(len(xbatch)):
    #                 if correct[i]:
    #                     noise = stddev * torch.randn(noise_samples, *input_dim[1:], device=device)
    #                     x_noisy = torch.clamp(xbatch[i:i + 1] + noise, 0, 1)
    #                     preds = []

    #                     bsize = 50
    #                     num_batches = noise_samples // bsize
    #                     for j in range(num_batches):
    #                         batch_noisy = x_noisy[j * bsize:(j + 1) * bsize]
    #                         preds_batch = model(batch_noisy).argmax(dim=-1)
    #                         preds.extend(preds_batch.cpu().numpy())

    #                     for idx, n in enumerate(num):
    #                         if n == 0:
    #                             robust_accs[idx].append(1)
    #                         else:
    #                             robust_accs[idx].append(np.mean(preds[:n] == labels[i].cpu().numpy()))
    #                 else:
    #                     for idx in range(len(num)):
    #                         robust_accs[idx].append(0)

    #             num_samples += len(xbatch)

    #     return robust_accs


    # def binary_rand_robust(self, model, ds, p, max_samples=100, noise_samples=10000, stddev=0.025, input_dim=[None, 107],
    #                     num=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 250, 500, 1000, 2500, 5000, 10000],
    #                     dataset='adult'):
    #     """Calculate robustness to random noise for Adv-x MI attack on binary-featured datasets (+ UCI adult).

    #     :param model: model to approximate distances on (attack).
    #     :param ds: PyTorch dataset should be either the training set or the test set.
    #     :param p: probability for Bernoulli flips for binary features.
    #     :param max_samples: maximum number of samples to take from the ds.
    #     :param noise_samples: number of noised samples to take for each sample in the ds.
    #     :param stddev: the standard deviation to use for Gaussian noise (only for Adult, which has some continuous features).
    #     :param input_dim: dimension of inputs for the dataset.
    #     :param num: subnumber of samples to evaluate. max number is noise_samples.
    #     :return: a list of lists. each sublist of the accuracy on up to $num noise_samples.
    #     """
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model.to(device)
    #     model.eval()

    #     data_loader = torch.utils.data.DataLoader(ds, batch_size=max_samples, shuffle=True)
    #     robust_accs = [[] for _ in num]
    #     all_correct = []
    #     num_samples = 0

    #     with torch.no_grad():
    #         for xbatch, ybatch in data_loader:
    #             if num_samples >= max_samples:
    #                 break

    #             xbatch, ybatch = xbatch.to(device), ybatch.to(device)
    #             labels = ybatch.argmax(dim=-1)
    #             y_pred = model(xbatch).argmax(dim=-1)
    #             correct = (y_pred == labels).cpu().numpy()
    #             all_correct.extend(correct)

    #             for i in range(len(xbatch)):
    #                 if correct[i]:
    #                     if dataset == 'adult':
    #                         noise = np.random.binomial(1, p, [noise_samples, xbatch[i: i+1, 6:].shape[-1]])
    #                         x_sampled = np.tile(np.copy(xbatch[i:i+1].cpu()), (noise_samples, 1))
    #                         x_noisy = np.copy(x_sampled[:, 6:])
    #                         np.bitwise_xor(x_sampled[:, 6:], noise, out=x_noisy)
    #                         x_noisy = np.concatenate([x_sampled[:, :6] + stddev * np.random.randn(noise_samples, xbatch[i: i+1, :6].shape[-1]), x_noisy], axis=1)
    #                     else:
    #                         noise = np.random.binomial(1, p, [noise_samples, xbatch[i: i + 1].shape[-1]])
    #                         x_sampled = np.tile(np.copy(xbatch[i:i + 1].cpu()), (noise_samples, 1))
    #                         x_noisy = np.copy(x_sampled)
    #                         np.bitwise_xor(x_sampled, noise, out=x_noisy)

    #                     x_noisy = torch.tensor(x_noisy, dtype=torch.float32).to(device)
    #                     preds = []

    #                     bsize = 100
    #                     num_batches = noise_samples // bsize
    #                     for j in range(num_batches):
    #                         batch_noisy = x_noisy[j * bsize:(j + 1) * bsize]
    #                         preds_batch = model(batch_noisy).argmax(dim=-1)
    #                         preds.extend(preds_batch.cpu().numpy())

    #                     for idx, n in enumerate(num):
    #                         if n == 0:
    #                             robust_accs[idx].append(1)
    #                         else:
    #                             robust_accs[idx].append(np.mean(preds[:n] == labels[i].cpu().numpy()))
    #                 else:
    #                     for idx in range(len(num)):
    #                         robust_accs[idx].append(0)

    #             num_samples += len(xbatch)

    #     return robust_accs



    # def distance_augmentation_attack(self, model, train_set, test_set, max_samples, attack_type='d', distance_attack='CW', augment_kwarg=1, batch=100, input_dim=[None, 32, 32, 3], n_classes=10):
    #     """Combined MI attack using the distances for each augmentation.

    #     Args:
    #         model: model to approximate distances on (attack).
    #         train_set: the training set for the model.
    #         test_set: the test set for the model.
    #         max_samples: max number of samples to attack.
    #         attack_type: either 'd' or 'r' for translation and rotation attacks, respectively.
    #         augment_kwarg: the kwarg for each augmentation. If rotations, augment_kwarg defines the max rotation, with n=2r+1
    #         rotated images being used. If translations, then 4n+1 translations will be used at a max displacement of
    #         augment_kwarg.
    #         batch: batch size for querying model in the attack.

    #     Returns: 2D array where rows correspond to samples and columns correspond to the distance to boundary in an untargeted
    #     attack for that rotated/translated sample.
    #     """
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model.to(device)
    #     model.eval()

    #     if attack_type == 'r':
    #         augments = self.create_rotates(augment_kwarg)
    #     elif attack_type == 'd':
    #         augments = self.create_translates(augment_kwarg)
    #     else:
    #         raise ValueError(f"attack_type: {attack_type} is not valid.")

    #     m = np.concatenate([np.ones(max_samples), np.zeros(max_samples)], axis=0)
    #     attack_in = np.zeros((max_samples, len(augments)))
    #     attack_out = np.zeros((max_samples, len(augments)))

    #     for i, augment in enumerate(augments):
    #         train_augment = self.apply_augment(train_set, augment, attack_type)
    #         test_augment = self.apply_augment(test_set, augment, attack_type)

    #         train_ds = DataLoader(TensorDataset(torch.tensor(train_augment[0]), torch.tensor(train_augment[1])), batch_size=batch, shuffle=False)
    #         test_ds = DataLoader(TensorDataset(torch.tensor(test_augment[0]), torch.tensor(test_augment[1])), batch_size=batch, shuffle=False)

    #         attack_in[:, i] = self.dists(model, train_ds, attack=distance_attack, max_samples=max_samples, input_dim=input_dim, n_classes=n_classes, device=device)
    #         attack_out[:, i] = self.dists(model, test_ds, attack=distance_attack, max_samples=max_samples, input_dim=input_dim, n_classes=n_classes, device=device)

    #     attack_set = (np.concatenate([attack_in, attack_out], 0),
    #                 np.concatenate([train_set[1], test_set[1]], 0),
    #                 m)
    #     return attack_set




    def dists(self, model, ds, attack="HSJ", max_samples=100, input_dim=[None, 3, 32, 32], n_classes=10):
        """
        Calculate untargeted distance to decision boundary for Adv-x MI attack.
    
        :param model: model to approximate distances on (attack).
        :param ds: PyTorch dataset should be either the training set or the test set.
        :param attack: "CW" for Carlini-Wagner or "HSJ" for Hop Skip Jump
        :param max_samples: maximum number of samples to take from the ds
        :return: an array of the first samples from the ds, of len max_samples, with the untargeted distances.
        """
        
        # Move model to the device
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        # model = CallableModelWrapper(lambda x: model_(x), "logits")
        
        # data_tensor = torch.tensor(ds._data, dtype=torch.float32)
        # data_tensor = data_tensor.permute(0, 3, 1, 2)
        dataloader = DataLoader(ds, batch_size=1, shuffle=True)
        if attack == "CW":
            # attack_fn = L2CarliniWagnerAttack(model, device=device)
            print("Remeber me!")
        elif attack == "HSJ":
             attack_fn, perturbation_distance = hop_skip_jump_attack(model,
                                                dataloader,
                                                norm = 2,
                                                y_target = None,
                                                image_target = None,
                                                )
    
        else:
            raise ValueError("Unknown attack {}".format(attack))
        
        return attack_fn.cpu().numpy(), perturbation_distance

    # def create_rotates(self , r):
    #     """Creates vector of rotation degrees compatible with scipy' rotate.

    #     Args:
    #         r: param r for rotation augmentation attack. Defines max rotation by +/-r. Leads to 2*r+1 total images per sample.

    #     Returns: vector of rotation degrees compatible with scipy' rotate.

    #     """
    #     if r is None:
    #         return None
    #     if r == 1:
    #         return [0.0]
    #     # rotates = [360. / r * i for i in range(r)]
    #     rotates = np.linspace(-r, r, (r * 2 + 1))
    #     return rotates

    # def all_shifts(self, mshift):

    #     if mshift == 0:
    #       return [(0, 0, 0, 0)]
    #     all_pairs = []
    #     start = (0, mshift, 0, 0)
    #     end = (0, mshift, 0, 0)
    #     vdir = -1
    #     hdir = -1
    #     first_time = True
    #     while (start[1] != end[1] or start[2] != end[2]) or first_time:
    #         all_pairs.append(start)
    #         start = (0, start[1] + vdir, start[2] + hdir, 0)
    #         if abs(start[1]) == mshift:
    #             vdir *= -1
    #         if abs(start[2]) == mshift:
    #             hdir *= -1
    #         first_time = False
    #     all_pairs = [(0, 0, 0, 0)] + all_pairs  # add no shift
    #     return all_pairs

    # def create_translates(self, d):
    #     """Creates vector of translation displacements compatible with scipy' translate.

    #     Args:
    #         d: param d for translation augmentation attack. Defines max displacement by d. Leads to 4*d+1 total images per sample.

    #     Returns: vector of translation displacements compatible with scipy' translate.
    #     """
    #     if d is None:
    #         return None
    #     translates = self.all_shifts(d)
    #     return translates

    # def apply_augment(self, ds, augment, type_):
    #     """Applies an augmentation from create_rotates or create_translates.

    #     Args:
    #         ds: tuple of (images, labels) describing a datset. Images should be 4D of (N,H,W,C) where N is total images.
    #         augment: the augment to apply. (one element from augments returned by create_rotates/translates)
    #         type_: attack type, either 'd' or 'r'

    #     Returns:

    #     """
    #     if type_ == 'd':
    #         ds = (interpolation.shift(ds[0], augment, mode='nearest'), ds[1])
    #     else:
    #         ds = (interpolation.rotate(ds[0], augment, (1, 2), reshape=False), ds[1])
    #     return ds



    # def confidence_vector_attack(self:Self, target_model, shadow_model) -> None:

    #     target_model_labels = np.concatenate([self.target_train_dataset._labels,
    #                                          self.target_test_dataset._labels ], axis=0)
    #     shadow_model_labels = np.concatenate([self.shadow_train_dataset._labels,
    #                                           self.shadow_test_dataset._labels], axis=0)
                
    #     target_outputs_in = self.call_model(target_model, self.target_train_dataset)
    #     target_outputs_out = self.call_model(target_model,self.target_test_dataset)

    #     shadow_outputs_in = self.call_model(shadow_model, self.shadow_train_dataset)
    #     shadow_outputs_out = self.call_model(shadow_model, self.shadow_test_dataset)    

    #     target_outputs_in = self.softmax(target_outputs_in)
    #     target_outputs_out = self.softmax(target_outputs_out)
    #     shadow_outputs_in = self.softmax(shadow_outputs_in)
    #     shadow_outputs_out = self.softmax(shadow_outputs_out)

    #     target_features = np.concatenate([target_outputs_in, target_outputs_out], axis=0)
    #     shadow_features = np.concatenate([shadow_outputs_in, shadow_outputs_out], axis=0)
    #     print(f"shadow_features: {shadow_features.shape}, target_features: {target_features.shape}")

    #     target_membership_status = np.concatenate([np.ones(len(self.target_train_dataset)),
    #                             np.zeros(len(self.target_test_dataset))], axis=0)
    #     shadow_membership_status = np.concatenate([np.ones(len(self.shadow_train_dataset)),
    #                             np.zeros(len(self.shadow_test_dataset))], axis=0)

    #     # just look at confidence in predicted label
    #     conf_shadow = np.max(shadow_features, axis=-1)
    #     conf_target = np.max(target_features, axis=-1)
    #     # print("threshold on predicted label:")
    #     # acc1, prec1, _, _ = self.get_threshold(shadow_membership_status, conf_shadow,
    #     #                                        target_membership_status, conf_target)

    #     # look at confidence in true label
    #     conf_shadow= shadow_features[range(len(shadow_features)), shadow_model_labels]
    #     conf_target = target_features[range(len(target_features)), target_model_labels]
    #     # print("threshold on true label:")
    #     # acc2, prec2, _, _ = self.get_threshold(shadow_membership_status, conf_shadow,
    #     #                                   target_membership_status, conf_target)
        
    #     # NOTE: In the original code, the threshold leading to max accuracy in shadow model is used as a threshold for the target model signals 
    #     min_signal_val = np.min(np.concatenate([self.in_member_signals, self.out_member_signals]))
    #     max_signal_val = np.max(np.concatenate([self.in_member_signals, self.out_member_signals]))
    #     thresholds = np.linspace(min_signal_val, max_signal_val, 1000)

    #     member_preds = np.greater(self.in_member_signals, thresholds).T
    #     non_member_preds = np.greater(self.out_member_signals, thresholds).T

    #     # what does the attack predict on test and train dataset
    #     predictions = np.concatenate([member_preds, non_member_preds], axis=1)
    #     # set true labels for being in the training dataset
    #     true_labels = np.concatenate(
    #         [
    #             np.ones(len(self.in_member_signals)),
    #             np.zeros(len(self.out_member_signals)),
    #         ]
    #     )
    #     signal_values = np.concatenate(
    #         [self.in_member_signals, self.out_member_signals]
    #     )

    #     # compute ROC, TP, TN etc
    #     return CombinedMetricResult(
    #         predicted_labels=predictions,
    #         true_labels=true_labels,
    #         predictions_proba=None,
    #         signal_values=signal_values,
    #     )


    # def get_max_accuracy(self, y_true, probs, thresholds=None):
    #     """Return the max accuracy possible given the correct labels and guesses. Will try all thresholds unless passed.

    #     Args:
    #         y_true: True label of `in' or `out' (member or non-member, 1/0)
    #         probs: The scalar to threshold
    #         thresholds: In a blackbox setup with a shadow/source model, the threshold obtained by the source model can be passed
    #         here for attackin the target model. This threshold will then be used.

    #     Returns: max accuracy possible, accuracy at the threshold passed (if one was passed), the max precision possible,
    #     and the precision at the threshold passed.

    #     """
    #     if thresholds is None:
    #         fpr, tpr, thresholds = roc_curve(y_true, probs)

    #     accuracy_scores = []
    #     precision_scores = []
    #     for thresh in thresholds:
    #         accuracy_scores.append(accuracy_score(y_true,
    #                                             [1 if m > thresh else 0 for m in probs]))
    #         precision_scores.append(precision_score(y_true, [1 if m > thresh else 0 for m in probs]))

    #     accuracies = np.array(accuracy_scores)
    #     precisions = np.array(precision_scores)
    #     max_accuracy = accuracies.max()
    #     max_precision = precisions.max()
    #     max_accuracy_threshold = thresholds[accuracies.argmax()]
    #     max_precision_threshold = thresholds[precisions.argmax()]
    #     return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold


    # def get_threshold(self, ource_m, source_stats, target_m, target_stats):
    #     """ Train a threshold attack model and get teh accuracy on source and target models.

    #     Args:
    #         source_m: membership labels for source dataset (1 for member, 0 for non-member)
    #         source_stats: scalar values to threshold (attack features) for source dataset
    #         target_m: membership labels for target dataset (1 for member, 0 for non-member)
    #         target_stats: scalar values to threshold (attack features) for target dataset

    #     Returns: best acc from source thresh, precision @ same threshold, threshold for best acc,
    #         precision at the best threshold for precision. all tuned on source model.

    #     """
    #     # find best threshold on source data
    #     acc_source, t, prec_source, tprec = self.get_max_accuracy(source_m, source_stats)

    #     # find best accuracy on test data (just to check how much we overfit)
    #     acc_test, _, prec_test, _ = self.get_max_accuracy(target_m, target_stats)

    #     # get the test accuracy at the threshold selected on the source data
    #     acc_test_t, _, _, _ = self.get_max_accuracy(target_m, target_stats, thresholds=[t])
    #     _, _, prec_test_t, _ = self.get_max_accuracy(target_m, target_stats, thresholds=[tprec])
    #     print("acc src: {}, acc test (best thresh): {}, acc test (src thresh): {}, thresh: {}".format(acc_source, acc_test,
    #                                                                                                     acc_test_t, t))
    #     print(
    #         "prec src: {}, prec test (best thresh): {}, prec test (src thresh): {}, thresh: {}".format(prec_source, prec_test,
    #                                                                                                 prec_test_t, tprec))

    #     return acc_test_t, prec_test_t, t, tprec




    # def call_model(self:Self, 
    #        model,
    #        dataset, 
    #        batch_size=32):
    #     """
    #     Run inference on the given data using the specified model.

    #     Parameters
    #     ----------
    #     model : torch.nn.Module
    #         The PyTorch model to use for inference.
    #     dataset : list or numpy.ndarray
    #         The dataset on which to run inference. Each element should be a sample.
    #     batch_size : int
    #         The batch size to use for inference. Defaults to 32.

    #     Returns
    #     -------
    #     output : list
    #         The model's outputs for each input in the dataset.
    #     """
    #     gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    #     data_loader = DataLoader(dataset, batch_size, shuffle=True)

    #     model.eval()
    #     model.to(gpu_or_cpu)
        
    #     outputs = []

    #     # No need to calculate gradients for inference
    #     with torch.no_grad():
    #         for batch in data_loader:
    #             # Move data to the device
    #             input_tensor = batch[0].to(gpu_or_cpu)
                
    #             # Run the batch through the model
    #             output = model(input_tensor)
                
    #             # Append the output to the outputs list (moving it back to CPU and converting to numpy)
    #             outputs.append(output.cpu().numpy())
        
    #     return np.concatenate(outputs, axis=0)



    # def softmax(self: Self, input_array):
    #     """
    #     Compute the softmax function for the given array.

    #     Parameters
    #     ----------
    #     input_array : numpy.ndarray
    #         Input array.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         Softmax of the input array.
    #     """
    #     # Ensure input_array is at least 2-dimensional
    #     if input_array.ndim == 1:
    #         input_array = input_array[np.newaxis, :]

    #     # Subtract the maximum value for numerical stability
    #     max_values = np.max(input_array, axis=1, keepdims=True)
    #     exponentiated = np.exp(input_array - max_values)

    #     # Compute the denominator for normalization
    #     divisor = np.sum(exponentiated, axis=1, keepdims=True)

    #     # Compute softmax probabilities
    #     return exponentiated / divisor


    # def gap_attack(self:Self, configs: str) -> None:
    #     """Run the gap attack."""
        


    #     train_acc = self.target_model_metadata["model_metadata"]["train_acc"]
    #     test_acc = self.target_model_metadata["model_metadata"]["test_acc"]
    #     shadow_train_acc = self.target_model_metadata["model_metadata"]["train_acc"]
    #     shadow_test_acc = self.target_model_metadata["model_metadata"]["train_acc"]

    #     self.logger.info(f"Gap attack| target model: {50 + (train_acc - test_acc) * 50}, shadow model: {50 + (shadow_train_acc - shadow_test_acc) * 50}")
        
