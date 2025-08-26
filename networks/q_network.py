import torch.nn as nn


class QNet(nn.Module):
    """
    Convolutional Q-network for GridWorld.

    Args:
        grid_size (int): Size of the GridWorld (grid_size x grid_size).
        n_actions (int): Number of possible actions.
    """

    def __init__(self, grid_size, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flat = 64 * grid_size * grid_size
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, n_actions)
        )

    def forward(self, x):
        """
        Forward pass of the Q-network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, grid_size, grid_size).

        Returns:
            torch.Tensor: Q-values for each action, shape (batch, n_actions).
        """
        return self.head(self.conv(x))
