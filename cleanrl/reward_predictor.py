from dataclasses import dataclass # Import dataclass decorator for automatically generating methods like __init__, __repr__ etc. in classes.
import sys # Import the sys module for system-specific parameters and functions, like modifying the Python path.
import time # Import the time module for time-related functionalities.
sys.path.append('/Users/maxi/Developer/Python/rlhf_f') # Add a path to system's path to import modules from this directory.
import abc # Import the abc module for Abstract Base Classes.
import gymnasium as gym # Import Gymnasium library, an environment library for reinforcement learning.
import numpy as np # Import NumPy library for numerical operations, especially for arrays and matrices.
import torch # Import PyTorch library, a deep learning framework.
import torch.nn as nn # Import PyTorch's neural network module for building neural networks.
import torch.optim as optim # Import PyTorch's optimization module for optimization algorithms like Adam.
import tyro # Import Tyro library for parsing command-line arguments into Python dataclasses.
from torch.distributions.normal import Normal # Import Normal distribution from PyTorch distributions (not used in this file but imported).
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter from TensorBoard for logging and visualization (not used in this file but imported).
from ppo_config import Args # Import Args dataclass from ppo_config.py, which holds hyperparameters.
from torch.nn import init # Import init module from torch.nn for weight initialization.

args = Args() # Create an instance of the Args dataclass, which will hold the configuration parameters.

segment_list = [] # Initialize an empty list to store segments of experience for reward prediction.
preference_list = [] # Initialize an empty list to store preference pairs for reward predictor training (not directly used in this file).

def segment_list_next(segment_list, segment):
    """
    Retrieves the next segment in a cyclic segment list. If the given segment is the last one, it returns the first segment.

    Args:
        segment_list (list): A list of segments.
        segment (list): The current segment to find the next one after.

    Returns:
        list: The segment immediately following the input segment in the list, or the first segment if the input is the last.

    Raises:
        ValueError: If the provided segment is not found in the segment_list.
    """
    # Ensure all elements are tensors for comparison
    segment = [torch.tensor(item) if isinstance(item, np.ndarray) else item for item in segment] # Convert numpy arrays in segment to tensors for comparison.
    segment_list = [[torch.tensor(item) if isinstance(item, np.ndarray) else item for item in seg] for seg in segment_list] # Convert numpy arrays in all segments in segment_list to tensors.

    for i, seg in enumerate(segment_list): # Iterate through the segment list with index.
        if all(torch.equal(a, b) for a, b in zip(seg, segment)): # Check if the current segment in the list is equal to the input segment.
            if i + 1 < len(segment_list): # Check if there is a next segment in the list.
                return segment_list[i + 1] # Return the next segment.
            else: # If the current segment is the last one in the list.
                print("segment vom anfang weil ende erreicht") # Print a message indicating cyclic behavior.
                return segment_list[0] # Return the first segment, making it cyclic.
    raise ValueError("Segment not found in segment_list") # Raise ValueError if the provided segment is not found in the list.

class AbstractRewardPredictor(abc.ABC):
    """Abstract base class for reward predictors."""

    @abc.abstractmethod
    def predict_reward(self, state_action):
        """Predict the reward for a given state-action pair (or sequence)."""
        pass # Abstract method, must be implemented by subclasses.

    @abc.abstractmethod
    def parameters(self):
        """Return an iterator over the predictor's parameters."""
        pass # Abstract method, must be implemented by subclasses.


class FeedForwardNet(nn.Module):
    """A feedforward neural network."""
    def __init__(self, input_size, layer_sizes, output_size):
        """
        Initializes the FeedForwardNet.

        Args:
            input_size (int): Size of the input layer.
            layer_sizes (list of int): Sizes of the hidden layers.
            output_size (int): Size of the output layer.
        """
        super().__init__() # Call the constructor of the parent class (nn.Module).

        layers = [] # Initialize an empty list to store layers.
        prev_size = input_size # Set the previous layer size to the input size.
        for size in layer_sizes: # Iterate through the specified hidden layer sizes.
            layers.append(nn.Linear(prev_size, size)) # Add a linear layer from prev_size to current size.
            layers.append(nn.Tanh()) # Add a Tanh activation function after each linear layer.
            prev_size = size # Update previous size to the current layer size for the next iteration.
        layers.append(nn.Linear(prev_size, output_size)) # Add the final linear layer to the output size.

        self.net = nn.Sequential(*layers) # Create a sequential model from the layers.
        self._initialize_weights() # Initialize the weights of the network.

    def _initialize_weights(self):
        """Initializes weights of linear layers using Xavier uniform initialization."""
        for m in self.net.modules(): # Iterate through modules in the network.
            if isinstance(m, nn.Linear): # Check if the module is a linear layer.
                init.xavier_uniform_(m.weight) # Initialize weights with Xavier uniform initialization.
                if m.bias is not None: # Check if the layer has a bias.
                    init.constant_(m.bias, 0) # Initialize biases to zero.

    def forward(self, x):
        """
        Forward pass through the feedforward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x) # Pass the input through the sequential network and return the output.


class RewardPredictor(AbstractRewardPredictor, nn.Module):
    """Reward predictor network."""
    def __init__(self, envs):
        """
        Initializes the RewardPredictor.

        Args:
            envs (gym.vector.SyncVectorEnv or gym.vector.AsyncVectorEnv): Vectorized environment to infer observation and action spaces.
        """
        super().__init__() # Call the constructor of the parent class (nn.Module).
        input_size = np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape) # Calculate input size by summing observation and action space dimensions.
        layer_sizes = [64,64,64] # Define hidden layer sizes for the feedforward network.
        output_size = 1 # Define output size as 1 (single reward prediction value).

        self.reward_predictor = FeedForwardNet(input_size, layer_sizes, output_size) # Instantiate FeedForwardNet as the reward predictor network.

    def predict_reward(self, state_action):
        """
        Predicts the reward for a given state-action input.

        Args:
            state_action (torch.Tensor): Concatenated state and action tensor.

        Returns:
            torch.Tensor: Predicted reward value (clamped to [-10, 10]).
        """
        out = self.reward_predictor(state_action) # Pass the state-action input through the feedforward reward predictor network.
        out = torch.clamp(out, -10, 10) # Clamp the output reward prediction to the range [-10, 10].
        return out # Return the clamped reward prediction.

    def forward(self, state_action):
        """
        Forward pass through the RewardPredictor (calls predict_reward).

        Args:
            state_action (torch.Tensor): Concatenated state and action tensor.

        Returns:
            torch.Tensor: Predicted reward value.
        """
        return self.predict_reward(state_action) # Call the predict_reward method for forward pass.

    def parameters(self):
        """
        Returns an iterator over the reward predictor network's parameters.

        Returns:
            iterator: Iterator over the parameters.
        """
        return self.reward_predictor.parameters() # Return the parameters of the underlying feedforward reward predictor network.