import torch
import deepinv as dinv


class TomographyProblem:
    def __init__(self, img_size, angles=100, device="cpu"):
        """
        Tomography (Radon transform) physics operator.
        """
        self.img_size = img_size
        self.angles = angles
        self.device = device

        self.physics = dinv.physics.Tomography(
            img_width=img_size[1],
            angles=angles,
            circle=False,
            device=device,
        )
        self.scaling = torch.pi / (2 * angles)

    def A(self, x):
        return self.physics.A(x)

    def A_dagger(self, x):
        return self.physics.A_dagger(x)