import math
import torch.nn.functional as F
from torch import (
    Tensor,
    cat,
    cuda,
    device,
    exp,
    eye,
    isnan,
    max,
    nn,
    sigmoid,
    squeeze,
    tanh,
    zeros,
)
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter


########################################
#        LR MODEL DEFINITIONS        #
########################################

class LR(nn.Module):
    def __init__(self, input_dim: int):
        """Initialize the logistic regression model with a single linear layer.

        Args:
        ----
            input_dim (int): The size of the input feature vector.

        """
        super(LR, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, 1)  # Binary classification (1 output)
        # Metadata initialization
        self.init_params = {"input_dim": self.input_dim}

    def forward(self, x):
        """Forward pass through the model."""
        return sigmoid(self.linear(x))  # Sigmoid to produce probabilities for binary classification


########################################
#        GRUD MODEL DEFINITIONS        #
########################################


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
    def __init__(self, input_size, hidden_size, X_mean, batch_size,
                 bn_flag = True, output_last = False):
        """With minor modifications from https://github.com/zhiyongc/GRU-D/

        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        
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
            "hidden_size": hidden_size,
            "X_mean": X_mean,
            "batch_size": batch_size,
            "output_last": output_last,
            "bn_flag": bn_flag,
        }

        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.bn_flag = bn_flag

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

        #TODO: this part differs from the cited code
        self.fc = nn.Linear(self.hidden_size, 1) # a probability score
        self.drop=nn.Dropout(p=0.57, inplace=False)
        if self.bn_flag:
            self.bn= nn.BatchNorm1d(self.hidden_size, eps=1e-05, momentum=0.1, affine=True)


    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        """Inputs:
            x: input tensor
            x_last_obsv: input tensor with forward fill applied
            x_mean: the mean of each feature
            h: the hidden state of the network
            mask: the mask of whether or not the current value is observed
            delta: the tensor indicating the number of steps since the last time a feature was observed.
            
        Returns
        -------
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
        if self.bn_flag:
            return self.fc(self.bn(self.drop(Hidden_State)))
        else:
            return self.fc(self.drop(Hidden_State))

    def initHidden(self, batch_size):
        Hidden_State = Variable(zeros(batch_size, self.hidden_size)).to(self.device)
        return Hidden_State
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.is_cuda else tensor.detach().numpy()

