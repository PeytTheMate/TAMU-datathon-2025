# training/model.py
import torch
import torch.nn as nn
from gymnasium import spaces

class TronTransformer(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__()
        _ = observation_space

        self.conv = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pool
        )
        self.flatten = nn.Flatten()
        self.features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.conv(observations)
        x = self.flatten(x)
        x = x.view(x.size(0), -1)
        return x