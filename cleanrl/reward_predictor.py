import sys
import time
import abc
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from dataclasses import dataclass
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from ppo_config import Args
from torch.nn import init

sys.path.append('/Users/maxi/Developer/Python/rlhf_f')

args = Args()

segment_list = []
preference_list = []

def segment_list_next(segment_list, segment):
    segment = [torch.tensor(item) if isinstance(item, np.ndarray) else item for item in segment]
    segment_list = [[torch.tensor(item) if isinstance(item, np.ndarray) else item for item in seg] for seg in segment_list]
    for i, seg in enumerate(segment_list):
        if all(torch.equal(a, b) for a, b in zip(seg, segment)):
            if i + 1 < len(segment_list):
                return segment_list[i + 1]
            else:
                print("segment vom anfang weil ende erreicht")
                return segment_list[0]
    raise ValueError("Segment not found in segment_list")

class AbstractRewardPredictor(abc.ABC):
    @abc.abstractmethod
    def predict_reward(self, state_action):
        pass
    @abc.abstractmethod
    def parameters(self):
        pass

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.Tanh())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class RewardPredictor(AbstractRewardPredictor, nn.Module):
    def __init__(self, envs):
        super().__init__()
        input_size = np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape)
        layer_sizes = [64, 64, 64]
        output_size = 1
        self.reward_predictor = FeedForwardNet(input_size, layer_sizes, output_size)

    def predict_reward(self, state_action):
        out = self.reward_predictor(state_action)
        out = torch.clamp(out, -10, 10)
        return out

    def forward(self, state_action):
        return self.predict_reward(state_action)

    def parameters(self):
        return self.reward_predictor.parameters()