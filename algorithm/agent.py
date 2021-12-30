import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()

    def get_action(self, state:object):
        raise NotImplementedError

    def save_xp(self, trajectory:tuple):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError