from collections import deque
import random
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """
    Replay buffer to store agent transitions in a reinforcement learning environment
    and sample minibatches for training.

    Attributes:
        buf (deque): Deque with a maximum length `capacity` to store transitions.
    """

    def __init__(self, capacity):
        """
        Initializes the replay buffer with a maximum capacity.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        """
        Adds a transition to the buffer.

        Args:
            s (np.ndarray): Current state.
            a (int): Action taken in the current state.
            r (float): Reward received after taking the action.
            s2 (np.ndarray): Next state after taking the action.
            d (float): Done flag indicating if the episode ended (0.0 or 1.0).
        """
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size):
        """
        Samples a random minibatch of transitions and converts them into PyTorch tensors.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Tensors (s, a, r, s2, d) with shapes:
                - s: (batch_size, 3, G, G) or the state shape
                - a: (batch_size,)
                - r: (batch_size,)
                - s2: (batch_size, 3, G, G)
                - d: (batch_size,)
        """
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        s  = torch.tensor(np.array(s),  dtype=torch.float32, device=device)
        a  = torch.tensor(a,            dtype=torch.long,    device=device)
        r  = torch.tensor(r,            dtype=torch.float32, device=device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=device)
        d  = torch.tensor(d,            dtype=torch.float32, device=device)
        return s, a, r, s2, d

    def __len__(self):
        """
        Returns the current number of transitions stored in the buffer.

        Returns:
            int: Current buffer length.
        """
        return len(self.buf)
