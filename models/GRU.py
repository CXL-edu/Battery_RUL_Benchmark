import torch
import torch.nn as nn
from models._model_basic import ModelBasic


class Model(ModelBasic):
    def __init__(self, configs):
        assert all([k in configs for k in ['seq_len', 'pred_len', 'in_dim', 'out_dim', 'batch_size',
                                           'n_layer', 'hidden_dim', 'dropout', 'bidirect',
                                           ]]), 'GRU: Parameters are not completed!'
        assert configs.in_dim == configs.out_dim, 'Autoregression: in_dim should be equal to out_dim!'
        super().__init__(configs.seq_len, configs.pred_len, configs.in_dim, configs.out_dim)

        self.h0 = nn.Parameter(torch.randn(configs.n_layer * (2 if configs.bidirect else 1), 1, configs.hidden_dim))
        self.gru = nn.GRU(input_size=configs.in_dim, hidden_size=configs.hidden_dim, num_layers=configs.n_layer,
                          batch_first=True, bidirectional=configs.bidirect, dropout=configs.dropout, bias=True)
        self.fc = nn.Linear(configs.hidden_dim, configs.out_dim)


    def forward(self, x):
        # x: (batch, seq_len, in_dim),  output: (batch, pred_len, out_dim)
        output = []
        h0 = self.h0.repeat(1, x.size(0), 1)
        output_temp, hn = self.gru(x, h0)
        output.append(self.fc(output_temp[:, -1:, :]))
        for i in range(1, self.pred_len):
            output_temp, hn = self.gru(output[-1], hn)
            output.append(self.fc(output_temp[:, -1:, :]))
        output = torch.cat(output, dim=1)
        return output

    def get_model_info(self, mode='detail'):
        # super().get_model_info()
        print(f"\n//===== GRU Model Info =====")
        print(f'Seq_len: {self.seq_len}, Pred_len: {self.pred_len}')
        print(f'In_dim: {self.in_dim}, Out_dim: {self.out_dim}')

        print(f'\nGRU Model:\n{self}\n')

        if mode == 'detail':
            print(f'GRU Model Params:')
            for k, v in self.named_parameters():
                print(f'{k}: {v.shape}')

        print(f"===== GRU Model Info =====//\n")





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

    configs = AttrDict(dict({
        'model_name': 'GRU',
        'seq_len': 10, 'pred_len': 5, 'in_dim': 2, 'out_dim': 2, 'batch_size': 32,
        'n_layer': 2, 'hidden_dim': 30, 'dropout': 0.1, 'bidirect': False,
                             }))

    model = Model(configs)
    model.get_model_info()

    x = torch.randn(configs.batch_size, configs.seq_len, configs.in_dim)   # [Batch, seq_len, in_dim]
    out = model(x)
    print(out.shape)
