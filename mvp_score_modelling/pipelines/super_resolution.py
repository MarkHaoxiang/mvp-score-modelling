from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.distributions import Normal
from PIL.Image import Image
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
        x_hat = self.tweedie(x_t, sigma_t, score=s_x)
        v = self.upscale(self.downsampled, self.kernel_size)\
              - self.upscale(self.pool(x_hat), self.kernel_size)
        v = torch.flatten((v / sigma_t ** 2), 1)
        grad = torch.autograd.functional.vjp(
            lambda x: torch.flatten(self.tweedie(x, sigma_t), start_dim=1),
            inputs=x_t,
            v=v
        )[1]
        return s_x + grad

class ManifoldConstrainedGradientSuperResolutionPipeline(SuperResolutionProjectionPipeline):

    @torch.no_grad()
    def __call__(
        self,
        y = None,
        batch_size: int = 1,
        num_inference_steps: int = 2000,
        generator = None,
        output_type: Optional[str] = "pil",
        initial_sample: Optional[Tuple] = None,
        alpha = 1
    ) -> Union[torch.Tensor, Image]:
        # Initialise
        tweedie = VeTweedie(self.unet)
        sample, start = self.initialise_inference(y, generator, batch_size, num_inference_steps), 0
        if not initial_sample is None:
            sample, start = initial_sample[0], num_inference_steps - initial_sample[1]

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps[start:])):
            std = self.scheduler.sigmas[i+start]
            sigma_t = std * torch.ones(batch_size, device=self.device)

            # Constraint Projection (prepend for assumption on sigma)
            sample = self.constraint_projection(y, sample, sigma_t)
            x_t = sample

            # correction step
            for _ in range(self.scheduler.config.correct_steps):
                score = self.calculate_score(y, sample, sigma_t)
                sample = self.scheduler.step_correct(score, sample, generator=generator).prev_sample

            # prediction step
            sample = self.constraint_projection(y, sample, sigma_t)
            score = self.calculate_score(y, sample, sigma_t)
            output = self.scheduler.step_pred(score, t, sample, generator=generator)
            sample = output.prev_sample

            # MCG
            with torch.enable_grad():
                x_t.requires_grad = True
                x_hat = tweedie(x_t, sigma_t)
                y_hat = self.pool(x_hat)
                dist = (y_hat - self.downsampled).pow(2)
                dist_norm = torch.sqrt(dist.sum([-1,-2,-3]))
                a = alpha / dist_norm
                dist.sum().backward()
                a.reshape((*a.shape, 1, 1, 1))
                sample = sample - a * x_t.grad

        return self.organise_output(output, output_type)
