
from agent.agent import Agent

class MultiEnvWrapper(Agent):
    def __init__(self):
        super(Agent, self).__init__()

    def get_actions(self, states:list):
        raise NotImplementedError

    def save_xps(self, index:int, trajectory:tuple):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError