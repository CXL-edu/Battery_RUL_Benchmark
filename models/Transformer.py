import math

import torch
import torch.nn as nn
from models._model_basic import ModelBasic
from torch.nn.init import xavier_uniform_


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, device='cpu', dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term)[:, :-1]
        pe = pe.transpose(0, 1).to(device)
        self.register_buffer('pe', pe)  # 保存在buffer中的数据不会被更新，也不会被backward

    def forward(self, x: 'Tensor', pos: [int,int]) -> 'Tensor':
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        assert x.size(1) == pos[1]-pos[0], f'positional encoding size error! {x.size(1)} != {pos[1]-pos[0]}'
        x = x + self.pe[:, pos[0]:pos[1]]
        return self.dropout(x)



class Model(ModelBasic):
    def __init__(self, configs):
        assert all([k in configs for k in ['seq_len', 'pred_len', 'in_dim', 'out_dim', 'batch_size',
                                           'nhead', 'd_model', 'n_encoder_layer', 'n_decoder_layer', 'feed_hidden_dim'
                                           ]]), 'Transformer: Parameters are not completed!'
        assert configs.in_dim == configs.out_dim, 'Autoregression: in_dim should be equal to out_dim!'
        super().__init__(configs.seq_len, configs.pred_len, configs.in_dim, configs.out_dim)
        (d_model, nhead, dim_feedforward, n_encoder_layers, n_decoder_layers) = \
            (configs.d_model, configs.nhead, configs.feed_hidden_dim, configs.n_encoder_layer, configs.n_decoder_layer)

        self.configs = configs
        self.device = configs.device if 'device' in configs else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 相对位置编码和向量编码
        self.pos_embedding = PositionalEncoding(d_model, max_len=configs.seq_len+configs.pred_len+1, device=self.device)
        self.enc_x_embedding = nn.Linear(configs.in_dim, configs.d_model) if configs.in_dim != configs.d_model else None
        self.dec_x_embedding = nn.Linear(configs.out_dim, configs.d_model) if configs.out_dim != configs.d_model else None

        # Transformer架构
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True, norm_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers, encoder_norm)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True, norm_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers, decoder_norm)

        # decoder输入的第一个token
        self.dec_in_start = nn.Parameter(torch.randn(1, 1, configs.d_model)).to(self.device)  # set start token
        dec_in = torch.zeros(1, configs.pred_len-1, configs.d_model).to(self.device)
        self.dec_in = torch.cat((self.dec_in_start, dec_in), dim=1)  # [1, pred_len, d_model]

        # 输出层
        self.proj = nn.Linear(configs.d_model, configs.out_dim)

        self._reset_parameters()    # 初始化参数

    def forward(self, x, dec_in=None, epoch=None):
        # TODO: 利用epoch来控制teacher-forcing的使用真实值的概率。
        # DONE: 需要更新dec_in。dec_in的第一个token是start token，后面的token是真实值或者是预测值。
        # x: (batch, seq_len, in_dim), dec_in: (batch, pred_len+1, out_dim)  output: (batch, pred_len, out_dim)
        x = x.to(self.device)
        dec_in = dec_in.to(self.device) if dec_in is not None else None
        enc_x = self.enc_x_embedding(x) if self.enc_x_embedding is not None else x
        enc_x = self.pos_embedding(enc_x, [0, x.size(1)])
        enc_out = self.encoder(enc_x)
        if dec_in is None:
            # inference
            dec_out = torch.zeros(x.size(0), self.pred_len, self.configs.d_model).to(self.device)
            dec_in = self.dec_in.repeat(x.size(0), 1, 1).to(self.device)
            pos_embed = self.pos_embedding.pe[:, x.size(1):x.size(1) + dec_in.size(1)].repeat(x.size(0), 1, 1)
            for i in range(self.pred_len):
                dec_out_temp = self.decoder(dec_in + pos_embed, enc_out)
                dec_out[:, i] = dec_out_temp[:, i]
                if i+1 < self.pred_len:
                    dec_in[:, i+1] = dec_out[:, i]
        else:
            # Using teacher-forcing for parallel training
            dec_in = self.dec_x_embedding(dec_in[:, :-1]) if self.dec_x_embedding is not None else dec_in[:, :-1]
            dec_in = torch.cat((self.dec_in_start.repeat(x.size(0), 1, 1), dec_in), dim=1)
            dec_in = self.pos_embedding(dec_in, [enc_x.size(1), enc_x.size(1) + dec_in.size(1)])
            dec_in_mask = torch.triu(torch.full((dec_in.size(1), dec_in.size(1)), float('-inf')), diagonal=1).to(x.device)
            dec_out = self.decoder(dec_in, enc_out, tgt_mask=dec_in_mask)

        output = self.proj(dec_out)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def get_model_info(self, mode='detail'):
        print(f"\n//===== Transformer Model Info =====")
        print(f'Seq_len: {self.seq_len}, Pred_len: {self.pred_len}')
        print(f'In_dim: {self.in_dim}, Out_dim: {self.out_dim}')

        print(f'\nTransformer Model:\n{self}\n')

        if mode == 'detail':
            print(f'Transformer Model Params:')
            for k, v in self.named_parameters():
                print(f'{k}: {v.shape}')

        print(f"===== Transformer Model Info =====//\n")


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
        'model_name': 'Transformer',
        'seq_len': 10, 'pred_len': 10, 'in_dim': 1, 'out_dim': 1, 'batch_size': 32,
        'nhead': 2, 'd_model': 4, 'n_encoder_layer': 2, 'n_decoder_layer': 2, 'feed_hidden_dim': 30
    }))
    model = Model(configs).to('cuda') if torch.cuda.is_available() else Model(configs)
    # model.get_model_info()

    x = torch.randn(configs.batch_size, configs.seq_len, configs.in_dim)  # [Batch, seq_len, in_dim]
    tgt = torch.randn(configs.batch_size, configs.pred_len, configs.out_dim)  # [Batch, pred_len, out_dim]
    output = model(x, tgt)  # training
    print(output.shape)
    output = model(x)  # inference
    print(output.shape)

