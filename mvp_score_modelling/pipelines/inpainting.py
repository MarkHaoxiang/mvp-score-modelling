from typing import Optional, Union, Tuple
import torch
from torch.distributions import Normal
from PIL.Image import Image
from .utils import CustomConditionalScoreVePipeline, VeTweedie

class MaskGenerator:
    def __init__(self, shape, device='cpu'):
        self.shape = shape
        if len(self.shape) == 3:
            shape = (1, *shape)
        self.n = shape[-1] // 2
        self.device = device

    def generate_box_mask(self, size: int = 40) -> torch.Tensor:
        mask = torch.ones(self.shape)
        mask[:, :, self.n - size: self.n + size, self.n-size : self.n+size] = 0
        return mask.to(device=self.device)

    def generate_random_mask(self, p: float = 0.9) -> torch.Tensor:
        mask = torch.rand(self.shape)
        mask = torch.where(mask > p, torch.ones(self.shape), torch.zeros(self.shape))
        return mask.to(device=self.device)

class _InpaintingPipeline(CustomConditionalScoreVePipeline):
    def initialise_inference(self, y, generator, batch_size: int, n: int):
        self.reference_image, self.mask = y
        self.masked_reference_image = self.reference_image * self.mask
        return super().initialise_inference(y, generator, batch_size, n)

class InpaintingProjectionPipeline(_InpaintingPipeline):
    """ Solves the conditional inpainting problem
    """

    def constraint_projection(self, y, x_t, sigma_t):
        """ Replaces the unmasked areas of backwards sample x_t with generated y_t
        """
        y_t = (self.reference_image + torch.randn_like(x_t) * sigma_t) * self.mask
        x_t = x_t * (1. - self.mask) + y_t
        return x_t

class PseudoinverseGuidedInpaintingPipeline(_InpaintingPipeline):
    def initialise_inference(self, y, generator, batch_size: int, n: int):
        self.tweedie = VeTweedie(self.unet)
        return super().initialise_inference(y, generator, batch_size, n)

    def calculate_score(self, y, x_t, sigma_t):
        s_x = super().calculate_score(y, x_t, sigma_t)
        x_hat = self.tweedie(x_t, sigma_t, score=s_x)
        v = self.masked_reference_image - self.mask * x_hat
        v = torch.flatten((v / sigma_t**2), 1)
        grad = torch.autograd.functional.vjp(
            lambda x: torch.flatten(self.tweedie(x, sigma_t), start_dim=1),
            inputs=x_t,
            v=v
        )[1]
        return s_x + grad

class PrYtGuidedInpaintingPipeline(_InpaintingPipeline):
    def calculate_score(self, y, x_t, sigma_t):
        s_x = super().calculate_score(y, x_t, sigma_t)
        with torch.enable_grad():
            x_t.requires_grad = True
            diff = self.masked_reference_image - self.mask * x_t
            dist = Normal(0, sigma_t)
            dist.log_prob(diff).sum().backward()
        return s_x + x_t.grad

class ManifoldConstrainedGradientInpaintingPipeline(InpaintingProjectionPipeline):

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
                y_hat = self.mask * x_hat
                dist = (y_hat - self.masked_reference_image).pow(2)
                dist_norm = torch.sqrt(dist.sum([-1,-2,-3]))
                a = alpha / dist_norm
                dist.sum().backward()
                a.reshape((*a.shape, 1, 1, 1))
                sample = sample - a * x_t.grad

        return self.organise_output(output, output_type)

