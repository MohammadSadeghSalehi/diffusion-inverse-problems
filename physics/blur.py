import torch
import deepinv as dinv


class GaussianBlurProblem:
    def __init__(self, sigma=1.0, device="cpu"):
        """
        Gaussian blur physics operator.
        """
        self.kernel = dinv.physics.blur.gaussian_blur(sigma=(sigma, sigma), angle=0.0)
        self.physics = dinv.physics.Blur(self.kernel, device=device, padding="replicate")

    def A(self, x):
        return self.physics.A(x)

    def A_dagger(self, x):
        return self.physics.A_dagger(x)


class MotionBlurProblem:
    def __init__(self, sigma=(2, 0.1), angle=0.0, device="cpu"):
        """
        Motion blur physics operator (approximated as anisotropic Gaussian).
        """
        self.kernel = dinv.physics.blur.gaussian_blur(sigma=sigma, angle=angle)
        self.physics = dinv.physics.Blur(self.kernel, device=device, padding="replicate")

    def A(self, x):
        return self.physics.A(x)

    def A_dagger(self, x):
        return self.physics.A_dagger(x)