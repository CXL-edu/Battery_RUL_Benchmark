from abc import abstractmethod, ABCMeta
import torch.nn as nn


class ModelBasic(nn.Module, metaclass=ABCMeta):
    """
    seq_len: The length of the input sequence.
    pred_len: The length of the output sequence.
    in_dim: The number of input feature variables.
    out_dim: The number of output feature variables.
    """
    @abstractmethod
    def __init__(self, seq_len=1, pred_len=1, in_dim=1, out_dim=1):
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.in_dim, self.out_dim = in_dim, out_dim

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_model_info(self):
        print("\n//===== Model Info =====")
        print(f'Seq_len: {self.seq_len}, Pred_len: {self.pred_len}')
        print(f'In_dim: {self.in_dim}, Out_dim: {self.out_dim}\n')
        print(f'Model:\n{self}')
        print("===== Model Info =====//\n")

