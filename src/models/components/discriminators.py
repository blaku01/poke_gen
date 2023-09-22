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


class DCDisc(nn.Module):
    def __init__(self, img_dim, features_d, use_instance_norm=True):
        super().__init__()
        self.use_instance_norm = use_instance_norm
        self.disc = nn.Sequential(
            nn.Conv2d(img_dim, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        ]
        if self.use_instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.disc(x)
