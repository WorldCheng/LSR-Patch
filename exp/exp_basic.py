import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        use_mps = bool(getattr(self.args, 'use_mps', False))
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if use_mps and mps_available:
            device = torch.device('mps')
            print('Use MPS')
            return device
        if use_mps and not mps_available:
            print('MPS requested but not available. Falling back to CUDA/CPU.')

        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
