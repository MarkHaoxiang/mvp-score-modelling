from typing import Union, List, Optional, Tuple

from diffusers import ScoreSdeVePipeline
from diffusers.utils.torch_utils import randn_tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image

from mvp_score_modelling.utils import tensor_to_PIL

class VeTweedie(nn.Module):
    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, x_t, sigma, score=None):
        """ Calculates the VE expectation of x_0 from Tweedie's formula

        Args:
            x_t (_type_): (b, c, h, w)
            sigma (_type_): (b)
            score (_type_, optional): Defaults to passing through unet.

        Returns:
            Tensor: the expectation of x_0
        """
        if len(x_t.shape) == 3:
            x_t = x_t.unsqueeze(0)
        if score is None:
            score = self.unet(x_t, sigma).sample
        if isinstance(sigma, torch.Tensor) and len(sigma.shape) == 1:
            sigma = sigma[:, None, None, None]
        return x_t + score * sigma**2

class CustomConditionalScoreVePipeline(ScoreSdeVePipeline):

    def initialise_inference(self, y, generator, batch_size: int, n: int) -> torch.Tensor:
        img_size = self.unet.config.sample_size
        sample = randn_tensor(
            shape=(batch_size, 3, img_size, img_size),
            generator=generator
        ).to(self.device) * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(n)
        self.scheduler.set_sigmas(n)
        return sample

    def organise_output(self, output, output_type: Optional[str]):
        sample = output.prev_sample_mean.clamp(0,1)
        if output_type == "pil":
            return [tensor_to_PIL(img) for img in sample]
        return sample

    def calculate_score(self, y, x_t, sigma_t):
        return self.unet(x_t, sigma_t).sample
    
    def constraint_projection(self, y, x_t, sigma_t):
        return x_t
    

    @torch.no_grad()
    def __call__(
        self,
        y = None,
        batch_size: int = 1,
        num_inference_steps: int = 2000,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        initial_sample: Optional[Tuple] = None
    ) -> Union[torch.Tensor, Image]:
        # Initialise
        sample, start = self.initialise_inference(y, generator, batch_size, num_inference_steps), 0
        if not initial_sample is None:
            sample, start = initial_sample[0], num_inference_steps - initial_sample[1]

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps[start:])):
            std = self.scheduler.sigmas[i+start]
            sigma_t = std * torch.ones(batch_size, device=self.device)

            # correction step
            for _ in range(self.scheduler.config.correct_steps):
                sample = self.constraint_projection(y, sample, sigma_t)
                score = self.calculate_score(y, sample, sigma_t)
                sample = self.scheduler.step_correct(score, sample, generator=generator).prev_sample

            # prediction step
            sample = self.constraint_projection(y, sample, sigma_t)
            score = self.calculate_score(y, sample, sigma_t)
            output = self.scheduler.step_pred(score, t, sample, generator=generator)
            sample = output.prev_sample

        return self.organise_output(output, output_type)

def apply_custom_blur(images, kernel, padding_mode='reflect'):
    """
    Apply custom blur to a batch of images using a specified kernel.

    :param images: Tensor of shape [n, c, h, w]
    :param kernel: 2D tensor representing the blurring kernel
    :param padding_mode: Mode used for padding (default is 'reflect')
    :return: Blurred images
    """
    kernel_size = kernel.size(0)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(images.shape[1], 1, 1, 1)
    kernel = kernel.to(images.device).type(images.dtype)

    padding = kernel_size // 2
    images_padded = F.pad(images, (padding, padding, padding, padding), mode=padding_mode)

    blurred_images = F.conv2d(images_padded, kernel, groups=images.shape[1])
    return blurred_images

def kron(a, b):
    return torch.einsum('ij,kl->ikjl', a, b).reshape(a.size(0) * b.size(0), a.size(1) * b.size(1))

def pseudo_inverse(U, S, V, P2=None):
    # Threshold for considering singular values as zero
    threshold = 1e-6

    if P2 != None:
        S = P2 @ S @ P2.T
        U = U @ P2.T
        V = (P2 @ V.mH).mH

    if (len(S.shape) > 1):
        S = torch.diag(S)

    S_inv = torch.diag_embed(1 / S)

    H_plus = V @ S_inv @ U.mH
    return H_plus