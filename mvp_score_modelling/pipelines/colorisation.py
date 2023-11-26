import torch
from torch.distributions import Normal
from .utils import CustomConditionalScoreVePipeline

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
