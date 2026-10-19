import torch.nn as nn

class Bhowmick(nn.Module):
    """
    Model inspired by the architecture used in Bhowmick et al. 2020
    Note: paper studied univariate one-step predictions on EEG
    Main changes:   supporting multivariate data with arbitrary horizon
                    utilizing two dense layers instead of three
                    using LeakyReLU (instead of regular ReLU) to avoid bad local minimas
                    no lambda layer
    """
    
    def __init__(self, input_dim, horizon, conv_hidden_channels=32, kernel_size=5, lstm_hidden_size=64, num_lstm_layers=2, dense_units=64):
        super().__init__()
        self.init_params = {"input_dim": input_dim,
                            "horizon": horizon,
                            "conv_hidden_channels": conv_hidden_channels,
                            "kernel_size": kernel_size,
                            "lstm_hidden_size": lstm_hidden_size,
                            "num_lstm_layers": num_lstm_layers,
                            "dense_units": dense_units}
        
        self.input_dim = input_dim
        self.horizon = horizon
        self.conv_hidden_channels = conv_hidden_channels
        self.kernel_size = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

        self.causal_conv = nn.Conv1d(in_channels=input_dim, out_channels=conv_hidden_channels, kernel_size=kernel_size, padding=kernel_size-1)
        self.lstm = nn.LSTM(conv_hidden_channels, lstm_hidden_size, batch_first=True, num_layers=num_lstm_layers)
        self.linear1 = nn.Linear(lstm_hidden_size, dense_units)
        self.linear2 = nn.Linear(dense_units, input_dim * horizon)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)                  # Conv expects channel dim before time dim
        conv_out = self.causal_conv(x)[:,:,:-(self.kernel_size-1)]
        conv_out = self.activation(conv_out)
        conv_out = conv_out.permute(0, 2, 1)    # Permute back as expected by LSTM
        lstm_out, (h_n, c_n) = self.lstm(conv_out)  # h_n shape: (num_lstm_layers, batch_size, lstm_hidden_size) 
        h_n = h_n[-1]                               # use only hidden state of last LSTM layer
        linear1_out = self.activation(self.linear1(h_n))
        linear2_out = self.activation(self.linear2(linear1_out))
        return linear2_out.view(-1, self.horizon, self.input_dim)   # reshape to (batch_size, horizon, num_variables)