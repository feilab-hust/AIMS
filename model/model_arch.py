import torch.nn as nn
from model.model_used.VIT import *
from model.model_used.DiNA import *

class Net_arch(nn.Module):
    # Network architecture
    def __init__(self, hp):
        super(Net_arch, self).__init__()
        self.model = MODEL_REGISTRY.get(str(hp.model.name).lower())(**eval(f'hp.model.{hp.model.name}'))

    def forward(self, x):
        return self.model(x)

