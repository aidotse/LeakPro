# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from opacus.accountants import create_accountant
# file is from MIMIC Extract Paper 
import copy, math, os, pickle, time, pandas as pd, numpy as np, scipy.stats as ss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

from opacus import PrivacyEngine, GradSampleModule
import torch, torch.utils.data as utils, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau


from pathlib import Path


MAX_SIGMA = 1e6


def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Optional[int] = None,
    steps: Optional[int] = None,
    accountant: str = "rdp",
    epsilon_tolerance: float = 0.01,
    **kwargs,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    if (steps is None) == (epochs is None):
        raise ValueError(
            "get_noise_multiplier takes as input EITHER a number of steps or a number of epochs"
        )
    if steps is None:
        steps = int(epochs / sample_rate)

    eps_high = float("inf")
    accountant = create_accountant(mechanism=accountant)

    sigma_low, sigma_high = 0, 10
    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        accountant.history = [(sigma_high, sample_rate, steps)]
        eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        accountant.history = [(sigma, sample_rate, steps)]
        eps = accountant.get_epsilon(delta=target_delta, **kwargs)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high


def Train_Model_DPSGD(pre_model, loss_fn, pre_train_dataloader, noise_multiplier, 
                      max_grad_norm = 1, num_epochs = 300, patience = 1000, 
                      learning_rate=1e-3, batch_size=None):
    """
    Inputs:
        pre_model: a GRUD model
        loss_fn: the loss function to use
        pre_train_dataloader: training data
        noise_multiplier: the noise multiplier for dpsgd
        max_grad_norm: the max norm for gradient in dpsgd
        num_epochs: number of times over the training data
        patience: used for decreasing learning rate
        min_delta: if the loss stays within this value on the next step stop early
        batch_size: size of a batch
        
    Returns:
        best_model
        losses_train 
        losses_epochs_train
    """
    pre_opt = torch.optim.Adam(pre_model.parameters(), lr = learning_rate)

    # make private
    privacy_engine = PrivacyEngine(accountant = 'prv')
    priv_model, priv_opt, priv_train_dataloader = privacy_engine.make_private(
        module=pre_model,
        optimizer=pre_opt,
        data_loader=pre_train_dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    scheduler = ReduceLROnPlateau(priv_opt, 'min', patience=patience, verbose = True) 

#    losses_train = []
#    losses_epochs_train = []
    niter_per_epoch = 0

    # BE CAREFUL! The mean should be computed privately.
    X_mean = priv_model._module.X_mean
    for epoch in range(num_epochs):
#        losses_epoch_train = []

        for X, labels in priv_train_dataloader:
            if epoch == 0:
                niter_per_epoch += 1 # needed to compute epsilon later if we want to
            mask = X[:, np.arange(0, X.shape[1], 3), :]
            measurement = X[:, np.arange(1, X.shape[1], 3), :]
            time_ = X[:, np.arange(2, X.shape[1], 3), :]

            mask = torch.transpose(mask, 1, 2)
            measurement = torch.transpose(measurement, 1, 2)
            time_ = torch.transpose(time_, 1, 2)
#            measurement_last_obsv = measurement
            m_shape = measurement.shape[0]
            # we delete last column and prepend mean so that the last observed is used
            measurement_last_obsv = measurement[:, 0:measurement.shape[1]-1, :]
            measurement_last_obsv = torch.cat((torch.stack([X_mean[:, 0, :]]*m_shape), 
                                               measurement_last_obsv), dim = 1)

            convert_to_tensor = lambda x: torch.autograd.Variable(x)
            X, X_last_obsv, Mask, Delta, labels = map(convert_to_tensor, 
                                                 [measurement, 
                                                  measurement_last_obsv,
                                                  mask,
                                                  time_,
                                                  labels])
        
            priv_model.zero_grad()

            prediction = priv_model(X, X_last_obsv, Mask, Delta)

            loss_train = loss_fn(torch.squeeze(prediction), torch.squeeze(labels))
#            with torch.no_grad():
#                losses_train.append(loss_train.item())
#                losses_epoch_train.append(loss_train.item())

            priv_opt.zero_grad()
            loss_train.backward()
            priv_opt.step()
            scheduler.step(loss_train)

#        avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
#        losses_epochs_train.append(avg_losses_epoch_train)

    return priv_model, niter_per_epoch, privacy_engine#, [losses_train, losses_epochs_train]

def get_results_df(RESULTS_FOLDER, h_pass, run, task, verbose = False):
    task_d = {}
    i = 0
    folder = Path(RESULTS_FOLDER, f"{h_pass}{run}")
    for filename in folder.glob('*'):
        if os.path.isdir(filename):
            for subfilename in filename.glob('*'):
                if task in str(filename) and 'json' in str(subfilename) and 'results' in str(subfilename):
                    task_d[i] = unjsonify(subfilename)
                    i += 1
        if task in str(filename) and 'json' in str(filename):
            task_d[i] = unjsonify(filename)
            i += 1
    if verbose: print(f'---Processing {h_pass}{run} run for {task} ICU hyperparameter results -----')
    task_df = pd.concat([pd.json_normalize(task_d[j]) for j in range(0,i)])
    return task_df