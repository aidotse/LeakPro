"""
This file is inspired by https://github.com/MLforHealth/MIMIC_Extract 
MIT License
Copyright (c) 2019 MIT Laboratory for Computational Physiology
"""
import math
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data as utils
from sklearn.metrics import accuracy_score
from torch import Tensor, cat, cuda, device, exp, eye, from_numpy, isnan, max, nn, optim, save, sigmoid, squeeze, tanh, zeros
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def to_3D_tensor(df):
    idx = pd.IndexSlice
    np_3D = np.dstack([df.loc[idx[:, :, :, i], :].values for i in sorted(set(df.index.get_level_values("hours_in")))])
    return from_numpy(np_3D)

def prepare_dataloader(df, Ys, batch_size, shuffle=True):
    """Dfs = (df_train, df_dev, df_test).
    df_* = (subject, hadm, icustay, hours_in) X (level2, agg fn \ni {mask, mean, time})
    Ys_series = (subject, hadm, icustay) => label.
    """
    X     = from_numpy(to_3D_tensor(df).astype(np.float32))
    label = from_numpy(Ys.values.astype(np.int64))
    dataset = utils.TensorDataset(X, label)
    return utils.DataLoader(dataset, batch_size =int(batch_size) , shuffle=shuffle, drop_last = True)

class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, device, bias=True):
        """filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        """
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert in_features > 1 and out_features > 1, "Passing in nonsense sizes"

        self.filter_square_matrix = None
        self.filter_square_matrix = Variable(filter_square_matrix.to(device), requires_grad=False)

        self.weight = Parameter(Tensor(out_features, in_features)).to(device)

        if bias:
            self.bias = Parameter(Tensor(out_features)).to(device)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return F.linear(
            x,
            self.filter_square_matrix.mul(self.weight),
            self.bias
        )

    def __repr__(self):
        return self.__class__.__name__ + "(" \
            + "in_features=" + str(self.in_features) \
            + ", out_features=" + str(self.out_features) \
            + ", bias=" + str(self.bias is not None) + ")"

class GRUD(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, X_mean, batch_size = 0, output_last = False):
        """With minor modifications from https://github.com/zhiyongc/GRU-D/

        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        cell_size is the size of cell_state.
        
        Implemented based on the paper: 
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }
        
        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the historical input data
        """

        super(GRUD, self).__init__()

        # Save init params to a dictionary
        self.init_params = {
            "input_size": input_size,
            "cell_size": cell_size,
            "hidden_size": hidden_size,
            "X_mean": X_mean,
            "batch_size": batch_size,
            "output_last": output_last
        }

        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size

        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.identity = eye(input_size).to(self.device)
        self.X_mean = Variable(Tensor(X_mean).to(self.device))

        # Wz, Uz are part of the same network. the bias is bz
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size).to(self.device)

        # Wr, Ur are part of the same network. the bias is br
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size).to(self.device)

        # W, U are part of the same network. the bias is b
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size).to(self.device)

        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity, self.device)
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size).to(self.device)
        self.output_last = output_last

        self.fc = nn.Linear(self.hidden_size, 2).to(self.device)
        self.bn= nn.BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True).to(self.device)
        self.drop=nn.Dropout(p=0.7, inplace=False)


    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        """Inputs:
            x: input tensor
            x_last_obsv: input tensor with forward fill applied
            x_mean: the mean of each feature
            h: the hidden state of the network
            mask: the mask of whether or not the current value is observed
            delta: the tensor indicating the number of steps since the last time a feature was observed.
            
        Returns:
            h: the updated hidden state of the network

        """

        # Assert to check for NaNs in x_mean
        assert not isnan(x_mean).any(), "NaN values found in x_mean"

        batch_size = x.size()[0]
        feature_size = x.size()[1]
        zero_x = zeros(batch_size, feature_size).to(self.device)
        zero_h = zeros(batch_size, self.hidden_size).to(self.device)


        gamma_x_l_delta = self.gamma_x_l(delta)
        delta_x = exp(-max(zero_x, gamma_x_l_delta))

        gamma_h_l_delta = self.gamma_h_l(delta)
        delta_h = exp(-max(zero_h, gamma_h_l_delta))

        x_mean = x_mean.repeat(batch_size, 1)

        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h

        combined = cat((x, h, mask), 1)
        # Assert to check for NaNs in combined
        assert not isnan(combined).any(), "NaN values found in combined"

        z = sigmoid(self.zl(combined)) #sigmoid(W_z*x_t + U_z*h_{t-1} + V_z*m_t + bz)
        r = sigmoid(self.rl(combined)) #sigmoid(W_r*x_t + U_r*h_{t-1} + V_r*m_t + br)
        combined_new = cat((x, r*h, mask), 1)
        h_tilde = tanh(self.hl(combined_new)) #tanh(W*x_t +U(r_t*h_{t-1}) + V*m_t) + b
        h = (1 - z) * h + z * h_tilde

        return h


    def forward(self, X):
        """X: Input tensor of shape (batch_size, time_steps * 3, features)
        The tensor includes Mask, Measurement, and Delta sequentially for each time step.
        """

        # Step 1: Split the input tensor into Mask, Measurement, and Delta
        batch_size = X.size(0)
        time_steps = X.size(1) // 3  # Since every 3 consecutive steps represent Mask, Measurement, and Delta

        # Reshape X into 3 separate tensors for Mask, Measurement, and Delta
        Mask = X[:, np.arange(0, X.size(1), 3), :]       # Extract Mask
        Measurement = X[:, np.arange(1, X.size(1), 3), :]  # Extract Measurement
        Delta = X[:, np.arange(2, X.size(1), 3), :]       # Extract Delta

        # Transpose tensors to match (batch_size, time_steps, features)
        Mask = Mask.transpose(1, 2)
        Measurement = Measurement.transpose(1, 2)
        Delta = Delta.transpose(1, 2)

        # X_last_obsv is initialized to Measurement at the starting point
        X_last_obsv = Measurement

        # Step 2: Initialize hidden state
        step_size = Measurement.size(1)  # Number of time points
        Hidden_State = self.initHidden(batch_size)

        # Step 3: Iterate through time steps and update the GRU hidden state
        outputs = None
        for i in range(step_size):
            Hidden_State = self.step(
                squeeze(Measurement[:, i, :], 1),
                squeeze(X_last_obsv[:, i, :], 1),
                squeeze(self.X_mean[:, i, :], 1),
                Hidden_State,
                squeeze(Mask[:, i, :], 1),
                squeeze(Delta[:, i, :], 1),
            )
            # Collect hidden states
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = cat((Hidden_State.unsqueeze(1), outputs), 1)

        # Step 4: Predict a binary outcome using FC, BatchNorm, and Dropout layers
        return self.drop(self.bn(self.fc(Hidden_State)))

    def initHidden(self, batch_size):
        Hidden_State = Variable(zeros(batch_size, self.hidden_size)).to(self.device)
        return Hidden_State

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.is_cuda else tensor.detach().numpy()

def gru_trained_model_and_metadata(model,
                                   train_dataloader,
                                   test_dataloader,
                                   epochs,
                                   patience_early_stopping,
                                   patience_lr,
                                   min_delta,
                                   learning_rate):

    print("Model Structure: ", model)
    print("Start Training ... ")

    # Check if the input tensor is 3D
    # This check is nessary because the GRU-D model expects a 3D tensor, meaning the input data should not be flattened
    # The input tensor should have the shape (num_datapoints, num_features, num_timepoints)
    if train_dataloader.dataset.dataset.x.ndimension() != 3:
        warnings.warn("Input tensor is not 3D. There might be a mismatch between .", UserWarning)

    # Early Stopping
    min_loss_epoch_valid = float("inf")  # Initialize to infinity for comparison
    patient_epoch = 0  # Initialize patient counter

    device_name = device("cuda" if cuda.is_available() else "cpu")

    if isinstance(model, nn.Sequential):
        output_last = model[-1].output_last
        print("Output type dermined by the last layer")
    else:
        output_last = model.output_last
        print("Output type dermined by the model")

    criterion_CEL = nn.CrossEntropyLoss()
    criterion_MSE = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience = patience_lr)


    train_losses = []
    test_losses = []
    test_acces = []
    train_acces = []


    cur_time = time.time()
    pre_time = time.time()


    model.to(device_name)

    for epoch in tqdm(range(epochs), desc="Training Progress"):

        model.train()
        train_loss = 0.0

        test_dataloader_iter = iter(test_dataloader)

        for _, (X, labels) in enumerate(tqdm(train_dataloader, desc="Training Batches")):

            X = X.to(device_name)
            labels = labels.to(device_name)
            labels = labels.long()
            prediction = model(X)

            output_last = True
            if output_last:
                loss = criterion_CEL(squeeze(prediction), squeeze(labels))
            else:
                full_labels = cat((X[:,1:,:], labels), dim = 1)
                loss = criterion_MSE(prediction, full_labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Convert predictions to class indices
        binary_predictions = to_numpy(prediction).argmax(axis=1)

        # Ensure labels are integer and 1D
        binary_labels = to_numpy(labels).astype(int)
        # Compute accuracy
        train_acc = accuracy_score(binary_labels, binary_predictions)
        train_acces.append(train_acc)

        # test
        model.eval()
        try:
            X_test, labels_test = next(test_dataloader_iter)
        except StopIteration:
            valid_dataloader_iter = iter(test_dataloader)
            X_test, labels_test = next(valid_dataloader_iter)


        model.zero_grad()
        X_test = X_test.to(device_name)
        labels_test = labels_test.to(device_name)
        labels_test = labels_test.long()

        prediction_test = model(X_test)


        if output_last:
            test_loss = criterion_CEL(squeeze(prediction_test), squeeze(labels_test))
        else:
            full_labels_val = cat((X_test[:,1:,:], labels_test), dim = 1)
            test_loss = criterion_MSE(prediction_test, full_labels_val)

        test_loss = test_loss.cpu().item()
        test_losses.append(test_loss)

        # Convert predictions to class indices
        binary_predictions_test = to_numpy(prediction_test).argmax(axis=1)

        # Ensure labels are integer and 1D
        binary_labels_test = to_numpy(labels_test).astype(int)
        # Compute accuracy
        test_acc = accuracy_score(binary_labels_test, binary_predictions_test)
        test_acces.append(test_acc)

        # Early stopping
        # Assume test_loss is computed for validation set
        if test_loss < min_loss_epoch_valid - min_delta:  # Improvement condition
            min_loss_epoch_valid = test_loss
            patient_epoch = 0
            print(f"Epoch {epoch}: Validation loss improved to {test_loss:.4f}")
        else:
            patient_epoch += 1
            print(f"Epoch {epoch}: No improvement. Patience counter: {patient_epoch}/{patience_early_stopping}")

            if patient_epoch >= patience_early_stopping:
                print(f"Early stopping at epoch {epoch}. Best validation loss: {min_loss_epoch_valid:.4f}")
                break

        # Step the scheduler
        scheduler.step(test_loss)

        # Check the learning rate
        current_lr =  optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.6f}")

        # Stop if learning rate becomes too small
        if current_lr < 1e-6:
            print("Learning rate too small, stopping training.")
            break


        # Print training parameters
        cur_time = time.time()
        print("Epoch: {}, train_loss: {}, valid_loss: {}, time: {}".format( \
                    epoch, \
                    np.around(train_loss, decimals=8),\
                    np.around(test_loss, decimals=8),\
                    np.around(cur_time - pre_time, decimals=2)))
        pre_time = cur_time
    # Move the model back to the CPU
    # Ensure the target directory exists
    os.makedirs("target_GRUD", exist_ok=True)
    model.to("cpu")
    with open("target_GRUD/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

     # Create metadata and store it
    meta_data = {}
    meta_data["train_indices"] = train_dataloader.dataset.indices
    meta_data["test_indices"] = test_dataloader.dataset.indices
    meta_data["num_train"] = len(meta_data["train_indices"])

    # Write init params
    meta_data["init_params"] = {}
    for key, value in model.init_params.items():
        meta_data["init_params"][key] = value

    # read out optimizer parameters
    meta_data["optimizer"] = {}
    meta_data["optimizer"]["name"] = optimizer.__class__.__name__.lower()
    meta_data["optimizer"]["lr"] = optimizer.param_groups[0].get("lr", 0)
    meta_data["optimizer"]["weight_decay"] = optimizer.param_groups[0].get("weight_decay", 0)
    meta_data["optimizer"]["momentum"] = optimizer.param_groups[0].get("momentum", 0)
    meta_data["optimizer"]["dampening"] = optimizer.param_groups[0].get("dampening", 0)
    meta_data["optimizer"]["nesterov"] = optimizer.param_groups[0].get("nesterov", False)

    # read out criterion parameters
    meta_data["loss"] = {}
    meta_data["loss"]["name"] = criterion_CEL.__class__.__name__.lower()

    meta_data["batch_size"] = train_dataloader.batch_size
    meta_data["epochs"] = epochs
    meta_data["train_acc"] = train_acc
    meta_data["test_acc"] = test_acc
    meta_data["train_loss"] = train_loss
    meta_data["test_loss"] = test_loss
    meta_data["dataset"] = "mimiciii"
    with open("target_GRUD/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    return  train_losses, test_losses, train_acces, test_acces



