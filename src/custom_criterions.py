import torch
import torch.nn as nn


class NegativeMeanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_output, label_tensor):
        sign_tensor = label_tensor.clone().reshape(-1)  # might need to detach
        sign_tensor[sign_tensor == 0] = -1
        disc_output = disc_output.reshape(-1)
        return -torch.mean(disc_output.mul(sign_tensor))
