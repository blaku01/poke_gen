import torch
import torch.nn as nn


class NNDisc(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Outputs a probability (0 to 1)
        )

    def forward(self, image):
        return self.discriminator(image)
