import gymnasium as gym
import numpy as np
from model import SAC
import torch
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_dim = 21
        self.observation_dim = 67
        self.agent = SAC(state_dim=self.observation_dim, action_dim=self.action_dim,device = torch.device('cpu'))
        self.agent.load('best_model')
    def act(self, observation):
        print(observation.shape)
        return self.agent.select_action(observation, deterministic=True)
