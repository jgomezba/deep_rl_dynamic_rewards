import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, grid_size, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        flat = 64 * grid_size * grid_size
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    def forward(self, x):
        return self.head(self.conv(x))