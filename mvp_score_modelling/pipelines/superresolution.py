import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import CustomConditionalScoreVePipeline, VeTweedie

class _SuperResolutionPipeline(CustomConditionalScoreVePipeline):
    @staticmethod
    def upscale(img, kernel_size):
        upscaled = torch.repeat_interleave(img, kernel_size, dim=-2)
        upscaled = torch.repeat_interleave(upscaled, kernel_size, dim=-1)
        return upscaled

    def initialise_inference(self, y, generator, batch_size: int, n: int):
        self.downsampled, self.kernel_size = y
        self.pool = nn.AvgPool2d(
            kernel_size=self.kernel_size,
            stride=self.kernel_size
        )
        return super().initialise_inference(y, generator, batch_size, n)

class SuperResolutionProjectionPipeline(_SuperResolutionPipeline):
    def initialise_inference(self, y, generator, batch_size: int, n: int):
        res = super().initialise_inference(y, generator, batch_size, n)
        self.y = _SuperResolutionPipeline.upscale(self.downsampled, self.kernel_size)
        return res
    def constraint_projection(self, y, x_t, sigma_t):
        diff = x_t - _SuperResolutionPipeline.upscale(self.pool(x_t), self.kernel_size)
        return self.y + diff

class PrYtGuidedSuperResolutionPipeline(_SuperResolutionPipeline):
    def calculate_score(self, y, x_t, sigma_t):
        s_x =  super().calculate_score(y, x_t, sigma_t)
        with torch.enable_grad():
            x_t.requires_grad = True
            dist = Normal(self.downsampled, sigma_t / self.kernel_size)
            dist.log_prob(self.pool(x_t)).sum().backward()
        return s_x + x_t.grad

class PseudoinverseGuidedSuperResolutionPipeline(_SuperResolutionPipeline):
    def initialise_inference(self, y, generator, batch_size: int, n: int):
        self.tweedie = VeTweedie(self.unet)
        return super().initialise_inference(y, generator, batch_size, n)
    def calculate_score(self, y, x_t, sigma_t):
        s_x = super().calculate_score(y, x_t, sigma_t)
        with torch.enable_grad():
            x_t.requires_grad = True
            x_hat = self.tweedie(x_t, sigma_t)
            r = sigma_t
            sigma_0 = r / self.kernel_size
            dist = Normal(self.pool(x_hat), sigma_0)
            dist.log_prob(self.downsampled).sum().backward()
        return s_x + x_t.grad
