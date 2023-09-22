import math
import random

import torch


class ImageAugmentor:
    def __init__(self, p=0.5):
        self.p = p

    def apply_transform_if_lte_p(self, transform_fn):
        if random.random() < self.p:
            return transform_fn
        return lambda x: x

    def flip_x(self, x):
        x[0, :, :] *= -1
        return x

    def rotate_90(self, x):
        num_rotations = random.randint(0, 3)
        return torch.rot90(x, num_rotations, (1, 2))

    def integer_translation(self, x):
        w, h = x.size(-1), x.size(-2)
        tx = round(random.uniform(-0.125, 0.125) * w)
        ty = round(random.uniform(-0.125, 0.125) * h)
        return torch.roll(x, shifts=(tx, ty), dims=(2, 1))

    def isotropic_scaling(self, x):
        scale = torch.exp(torch.normal(0, (0.2 * torch.log(2)) ** 2))
        scaling_matrix = torch.tensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        return torch.nn.functional.affine_grid(
            scaling_matrix[:2].unsqueeze(0), x.unsqueeze(0)
        ).squeeze()

    def pre_rotation(self, x):
        theta = random.uniform(-math.pi, math.pi)
        pre_rotation_matrix = torch.tensor(
            [
                [math.cos(-theta), -math.sin(-theta), 0],
                [math.sin(-theta), math.cos(-theta), 0],
                [0, 0, 1],
            ]
        )
        return torch.nn.functional.affine_grid(
            pre_rotation_matrix[:2].unsqueeze(0), x.unsqueeze(0)
        ).squeeze()

    def anisotropic_scaling(self, x):
        scale_x = torch.exp(torch.normal(0, (0.2 * torch.log(2)) ** 2))
        scale_y = torch.exp(torch.normal(0, (0.2 * torch.log(2)) ** 2))
        scaling_matrix = torch.tensor([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
        return torch.nn.functional.affine_grid(
            scaling_matrix[:2].unsqueeze(0), x.unsqueeze(0)
        ).squeeze()

    def post_rotation(self, x):
        theta = random.uniform(-math.pi, math.pi)
        post_rotation_matrix = torch.tensor(
            [
                [math.cos(-theta), -math.sin(-theta), 0],
                [math.sin(-theta), math.cos(-theta), 0],
                [0, 0, 1],
            ]
        )
        return torch.nn.functional.affine_grid(
            post_rotation_matrix[:2].unsqueeze(0), x.unsqueeze(0)
        ).squeeze()

    def fractional_translation(self, x):
        w, h = x.size(-1), x.size(-2)
        tx = round(random.normal(0, 0.125) * w)
        ty = round(random.normal(0, 0.125) * h)
        return torch.roll(x, shifts=(tx, ty), dims=(2, 1))

    def __call__(
        self,
        x,
        flip=True,
        rotate=True,
        translate=True,
        scale=True,
        pre_rotation=True,
        anisotropic_scaling=True,
        post_rotation=True,
        fractional_translation=True,
    ):
        if flip:
            x = self.apply_transform_if_lte_p(self.flip_x)(x)
        if rotate:
            x = self.apply_transform_if_lte_p(self.rotate_90)(x)
        if translate:
            x = self.apply_transform_if_lte_p(self.integer_translation)(x)
        if scale:
            x = self.apply_transform_if_lte_p(self.isotropic_scaling)(x)
        if pre_rotation:
            x = self.apply_transform_if_lte_p(self.pre_rotation)(x)
        if anisotropic_scaling:
            x = self.apply_transform_if_lte_p(self.anisotropic_scaling)(x)
        if post_rotation:
            x = self.apply_transform_if_lte_p(self.post_rotation)(x)
        if fractional_translation:
            x = self.apply_transform_if_lte_p(self.fractional_translation)(x)

        return x


# Example usage
if __name__ == "__main__":
    augmentor = ImageAugmentor(p=0.5)
    input_image = torch.randn(3, 64, 64)  # Replace with your image data
    augmented_image = augmentor.augment_image(input_image)
    print(augmented_image.shape)
