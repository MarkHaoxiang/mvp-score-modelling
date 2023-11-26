import torch
from torch.distributions import Normal
from torchvision.transforms import GaussianBlur
from .utils import CustomConditionalScoreVePipeline

class _DeblurringPipeline(CustomConditionalScoreVePipeline):
    def initialise_inference(self, y, generator, batch_size: int, n: int):
        self.blurred = y[0]
        self.gaussian_blur = GaussianBlur(y[1])
        return super().initialise_inference(y, generator, batch_size, n)

class PrYtDeblurringPipeline(_DeblurringPipeline):
    def calculate_score(self, y, x_t, sigma_t):
        s_x = super().calculate_score(y, x_t, sigma_t)
        with torch.enable_grad():
            x_t.requires_grad = True
            blurred_sample = self.gaussian_blur(x_t)
            dist = Normal(self.blurred, sigma_t)
            dist.log_prob(blurred_sample).sum().backward()
        return s_x + x_t.grad