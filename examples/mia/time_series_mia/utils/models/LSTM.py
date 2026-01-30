import torch.nn as nn

class LSTM(nn.Module):
    """LSTM for multi-variate forecasting"""
    
    def __init__(self, input_dim, horizon, hidden_dim=64, num_layers=2, bidirectional = False):
        super().__init__()
        self.init_params = {"input_dim": input_dim,
                            "horizon": horizon,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "bidirectional": bidirectional}
        self.input_dim = input_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, self.hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(num_layers * self.hidden_dim * (2 if bidirectional else 1), input_dim * horizon)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x) # h_n shape: (num_layers, batch_size, hidden_size) 
        h_n = h_n.movedim(1, 0) # h_n shape: (batch_size, num_layers, hidden_size)
        h_n = h_n.flatten(start_dim = 1) # h_n shape: (batch_size, num_layers * hidden_size)
        linear_out = self.linear(h_n)
        return linear_out.view(-1, self.horizon, self.input_dim)   # reshape to (batch_size, horizon, num_variables)