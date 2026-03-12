import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    DLinear
    """
    def __init__(self, input_dim, lookback, horizon, individual=False):
        super(DLinear, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.individual = individual
        self.input_dim = input_dim
        self.init_params = {"input_dim": input_dim,
                            "lookback": lookback,
                            "horizon": horizon,
                            "individual": individual}

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.input_dim):
                self.Linear_Seasonal.append(nn.Linear(self.lookback,self.horizon))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.lookback)*torch.ones([self.horizon,self.lookback]))
                self.Linear_Trend.append(nn.Linear(self.lookback,self.horizon))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.lookback)*torch.ones([self.horizon,self.lookback]))
                self.Linear_Decoder.append(nn.Linear(self.lookback,self.horizon))
        else:
            self.Linear_Seasonal = nn.Linear(self.lookback,self.horizon)
            self.Linear_Trend = nn.Linear(self.lookback,self.horizon)
            self.Linear_Decoder = nn.Linear(self.lookback,self.horizon)
            self.Linear_Seasonal.weight = nn.Parameter((1/self.lookback)*torch.ones([self.horizon,self.lookback]))
            self.Linear_Trend.weight = nn.Parameter((1/self.lookback)*torch.ones([self.horizon,self.lookback]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.horizon],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.horizon],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.input_dim):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]