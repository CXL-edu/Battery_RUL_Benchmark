import os
import torch
from abc import abstractmethod, ABCMeta


class ExpBasic(object, metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.device = self.args.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            assert torch.cuda.is_available(), 'No GPU is available'
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def vali(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def _get_data(self):
        raise NotImplementedError


