import torch
import torch.nn as nn


class NNGen(nn.Module):
    def __init__(
        self,
        z_dim,
        img_dim,
        lin1_size: int = 128,
        lin2_size: int = 256,
        lin3_size: int = 512,
        relu_slope: float = 0.2,
    ):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim, lin1_size),
            nn.LeakyReLU(relu_slope),
            nn.Linear(lin1_size, lin2_size),
            nn.LeakyReLU(relu_slope),
            nn.Linear(lin2_size, lin3_size),
            nn.LeakyReLU(relu_slope),
            nn.Linear(lin3_size, img_dim),
            nn.Tanh(),  # Outputs in the range [-1, 1]
        )
        self.input_shape = z_dim

    def forward(self, noise):
        return self.generator(noise)


class DCGen(nn.Module):
    def __init__(
        self, z_dim, channels_img, features_g, use_instance_norm=True, use_batch_norm=False
    ):
        super().__init__()
        self.use_instance_norm = use_instance_norm
        self.use_batch_norm = use_batch_norm
        self.net = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.input_shape = z_dim
        self.apply(self._initialize_weights)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
        ]
        if self.use_instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.constant_(m.bias, 0)
