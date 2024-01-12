"""
@inproceedings{Zeng2022AreTE,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
"""


import torch
import torch.nn as nn
from models._model_basic import ModelBasic


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


class Model(ModelBasic):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        assert all([k in configs for k in ['seq_len', 'pred_len', 'in_dim', 'out_dim', 'individual']]), 'DLinear: Parameters are not completed!'
        super().__init__(configs.seq_len, configs.pred_len, configs.in_dim, configs.out_dim)
        self.configs = configs
        # Decompsition Kernel Size
        kernel_size = configs.kernel_size if configs.kernel_size else 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.in_dim

        if self.individual:
            assert configs.in_dim == configs.out_dim, 'DLinear: Individual mode requires in_dim==out_dim'
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
    
    def get_model_info(self):
        # super().get_model_info()
        print(f"\n//===== {self.configs.model_name} Model Info =====")
        print(f'Seq_len: {self.seq_len}, Pred_len: {self.pred_len}')
        print(f'In_dim: {self.in_dim}, Out_dim: {self.out_dim}')
        print(f'Individual: {self.individual}')

        print(f'\n{self.configs.model_name} Model:\n{self}')
        print(f"===== {self.configs.model_name} Model Info =====//\n")


if __name__ == '__main__':

    class AttrDict(dict):
        def __getattr__(self, name):
            if name in self:
                return self[name]
            else:
                return None
            # raise AttributeError(f"'AttrDict' object has no attribute '{name}'")

        def __setattr__(self, name, value):
            self[name] = value

    configs = AttrDict(dict({'seq_len': 10, 'pred_len': 5, 'in_dim': 1, 'out_dim': 1, 'individual': False, 'model_name': 'DLinear'}))
    model = Model(configs)
    model.get_model_info()
    x = torch.randn(32, 10, 1)   # [Batch, seq_len, in_dim]
    print(model(x).shape)       # [Batch, pred_len, out_dim]


















