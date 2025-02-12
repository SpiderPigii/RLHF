import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from ppo_config import Args

args = Args()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOPolicy(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.actor_logstd.data.clamp_(-5.0, 2.0)

    def get_action_and_value(self, observation_space, action=None):
        action_mean = self.actor_mean(observation_space)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        if torch.isnan(action_mean).any():
            print("NaN detected in action_mean")
        if torch.isnan(action_std).any():
            print("NaN detected in action_std")
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)