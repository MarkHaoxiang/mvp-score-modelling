from typing import Optional, Union, Tuple
import torch
from torch.distributions import Normal
from PIL.Image import Image
from .utils import CustomConditionalScoreVePipeline, VeTweedie


def greyscale(img: torch.Tensor) -> torch.Tensor:
    return torch.mean(img, dim=-3, keepdims=True).repeat(1,3,1,1)

class _ColorisationPipeline(CustomConditionalScoreVePipeline):
    M = torch.tensor(
        [[5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
         [5.7735026e-01, 4.0824834e-01,  7.0710671e-01],
         [5.7735026e-01, 4.0824822e-01, -7.0710683e-01]])
    invM = torch.inverse(M)

    @staticmethod
    def decouple(img):
        return torch.einsum(
            'bihw,ij->bjhw',
            img,
            _ColorisationPipeline.M.to(img.device)
        )

    @staticmethod
    def couple(inputs):
        return torch.einsum(
            'bihw,ij->bjhw',
            inputs,
            _ColorisationPipeline.invM.to(inputs.device)
        )

class ColorisationProjectionPipeline(_ColorisationPipeline):
    def initialise_inference(self, y, generator, batch_size: int, n: int):
        self.grey = y
        self.mask = torch.cat(
            [
                torch.ones_like(self.grey[:, :1, ...]),
                torch.zeros_like(self.grey[:, 1:, ...])
            ],
             dim=1
        )
        self.decoupled_greyscale = _ColorisationPipeline.decouple(self.grey)

        return super().initialise_inference(y, generator, batch_size, n)

    def constraint_projection(self, y, x_t, sigma_t):
        y_t = self.decoupled_greyscale + torch.randn_like(x_t) * sigma_t
        x_t = self.couple(
            self.decouple(x_t) * (1. - self.mask) + y_t * self.mask
        )
        return x_t

class PrYtGuidedColorisationPipeline(_ColorisationPipeline):
    def calculate_score(self, y, x_t, sigma_t):
        s_x =  super().calculate_score(y, x_t, sigma_t)
        with torch.enable_grad():
            x_t.requires_grad = True
            diff = greyscale(x_t)[:, 0] - y[0][0].expand(3, 256, 256)
            dist = Normal(0, sigma_t / 3. ** 0.5)
            dist.log_prob(diff).sum().backward()
        return s_x + x_t.grad
    
class PseudoinverseGuidedColorisationPipeline(_ColorisationPipeline):
    def initialise_inference(self, y, generator, batch_size: int, n: int):
        self.grey = y
        self.tweedie = VeTweedie(self.unet)
        return super().initialise_inference(y, generator, batch_size, n)

    def calculate_score(self, y, x_t, sigma_t):
        s_x = super().calculate_score(y, x_t, sigma_t)
        x_hat = self.tweedie(x_t, sigma_t, score=s_x)
        v = self.grey - greyscale(x_hat)
        v = torch.flatten((v / sigma_t ** 2), 1)
        grad = torch.autograd.functional.vjp(
            lambda x: torch.flatten(self.tweedie(x, sigma_t), start_dim=1),
            inputs=x_t,
            v=v
        )[1]
        return s_x + grad

class ManifoldConstrainedGradientColorisationPipeline(ColorisationProjectionPipeline):

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
                y_hat = greyscale(x_hat) 
                dist = (y_hat - y).pow(2)
                dist_norm = torch.sqrt(dist.sum([-1,-2,-3]))
                a = alpha / dist_norm
                dist.sum().backward()
                a.reshape((*a.shape, 1, 1, 1))
                sample = sample - a * x_t.grad

        return self.organise_output(output, output_type)
