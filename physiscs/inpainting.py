import torch
import deepinv as dinv


class InpaintingProblem:
    def __init__(self, img_size, mask=None, device="cpu"):
        """
        Inpainting physics operator using deepinv.
        """
        self.img_size = img_size
        self.device = device

        if mask is None:
            mask = torch.ones(1, img_size[1], img_size[2])
            mask[:, img_size[1] // 4: 3 * img_size[1] // 4,
                 img_size[2] // 4: 3 * img_size[2] // 4] = 0

        self.physics = dinv.physics.Inpainting(tensor_size=img_size, mask=mask, device=device)

    def A(self, x):
        return self.physics.A(x)

    def A_dagger(self, x):
        return self.physics.A_dagger(x)