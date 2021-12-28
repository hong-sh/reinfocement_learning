import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, model:object):
        super(model, self).__init__()

    def get_action(self):
        raise NotImplementedError

    def save_xp(self, trajectory:tuple):
        raise NotImplementedError

    def training(self):
        raise NotImplementedError