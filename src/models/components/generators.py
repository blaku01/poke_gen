import torch
import torch.nn as nn


class NNGen(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh(),  # Outputs in the range [-1, 1]
        )
        self.input_shape = z_dim

    def forward(self, noise):
        return self.generator(noise)
