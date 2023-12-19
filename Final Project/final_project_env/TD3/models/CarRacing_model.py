import numpy as np
import torch
import torch.nn as nn

class ActorNetSimple(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, N_frame: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_frame, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh()
        )

    def forward(self, state, brake_rate=0.015):
        state = state.float() / 255.0
        # print(state.shape)
        h = self.conv(state)
        # print(h.shape)
        h = torch.flatten(h, start_dim=1)
        # print(h.shape)
        h = self.linear(h)

        h_clone = h.clone()
        # h_clone = (h_clone / 4) + 0.75
        # map to valid action space: {steer:[-1, 1], motor:[-1, 1]}
        h_clone[:, 0] = (h_clone[:, 0])
        h_clone[:, 1] = (h_clone[:, 1])
        
        return h_clone
    
class CriticNetSimple(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, N_frame: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_frame, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.action_linear = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LayerNorm(256),
            nn.ELU()
        )

        self.state_linear = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ELU(),
        )

        self.concat_linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        # extract the state features
        # state = state.float() / 255.0
        state_h = self.conv(state)
        state_h = torch.flatten(state_h, start_dim=1)

        state_h = self.state_linear(state_h)
        # action features
        action_h = self.action_linear(action)

        # concat
        h = self.concat_linear(torch.concat((state_h, action_h), dim=1))
        # h = (h / 4) + 0.75
        
        return h
