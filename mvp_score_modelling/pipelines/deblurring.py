import torch
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F
from torchvision.transforms import functional as VF
from .utils import CustomConditionalScoreVePipeline, VeTweedie, apply_custom_blur, kron, pseudo_inverse

class _DeblurringPipeline(CustomConditionalScoreVePipeline):
    def initialise_inference(self, y, generator, batch_size: int, n: int, padding_mode='constant'):
        self.blurred = y[0]
        self.kernel = y[1] # can be any kernel, uniform, gaussian, etc.
        self.tweedie = VeTweedie(self.unet)
        self.padding_mode = padding_mode
        return super().initialise_inference(y, generator, batch_size, n)

class PrYtDeblurringPipeline(_DeblurringPipeline):
    def calculate_score(self, y, x_t, sigma_t):
        s_x = super().calculate_score(y, x_t, sigma_t)
        with torch.enable_grad():
            x_t.requires_grad = True
            blurred_sample = apply_custom_blur(x_t, self.kernel, padding_mode=self.padding_mode)
            dist = Normal(self.blurred, sigma_t)
            dist.log_prob(blurred_sample).sum().backward()
        return s_x + x_t.grad

class ReconstrutionGuidanceDeblurringPipeline(_DeblurringPipeline):
    """
    A simplified pseudoinverse guided deblurring pipeline 
    """
    def calculate_score(self, y, x_t, sigma_t):
        s_x = super().calculate_score(y, x_t, sigma_t)
        with torch.enable_grad():
            x_t.requires_grad = True
            x_hat = self.tweedie(x_t, sigma_t, score=s_x)
            mean = apply_custom_blur(x_hat, self.kernel, padding_mode=self.padding_mode)
            dist = torch.distributions.Normal(mean, sigma_t[:,None,None,None])
            dist.log_prob(y).sum().backward()
        return s_x + x_t.grad

class DirectPseudoInverseDeblurringPipeline():
    """
    Calculate the pseudo-inverse of the blurred image and apply it directly to the blurred image
    """
    def __init__(self, kernel, image_size, padding_mode='constant'):
        self.kernel = kernel
        self.padding_mode = padding_mode
        self.kernel_size = kernel.shape[0]
        self.padding = self.kernel_size // 2
        self.image_size = image_size
        self.padded_size = image_size + self.padding * 2

    def get_pseudo_inverse(self):
        # Ar and Ac
        Ar = np.zeros((self.padded_size, self.padded_size))
        Ac = np.zeros((self.padded_size, self.padded_size))

        for i in range(self.padded_size):
            start = max(0, i-self.padding)
            end = min(self.padded_size-1,i+self.padding) + 1
            k_start = start-i+self.padding
            Ar[i, start:end] = self.kernel[k_start:k_start+end-start]
            Ac[start:end, i] = self.kernel[k_start:k_start+end-start]

        # SVD of Ar and Ac
        Ar = torch.tensor(Ar)
        Ur, Sr, Vr = torch.svd(Ar)
        Sr = torch.diag_embed(Sr)  

        Ac = torch.tensor(Ac)
        Uc, Sc, Vc = torch.svd(Ac)
        Sc = torch.diag_embed(Sc) 

        # SVD of H
        Uh = kron(Ur, Uc).float()
        Sh = kron(Sr, Sc).float()
        Vh = kron(Vr, Vc).float()

        # Sorting permuation of sigma
        diagonal = torch.diag(Sh)
        sorted_diagonal, indices = torch.sort(diagonal, descending=True)
        P2 = torch.eye(Sh.size(0))[indices]

        # Pseudo-inverse of H
        H_pinv = pseudo_inverse(Uh, Sh, Vh, P2)
        return H_pinv

    def __call__(self, image):
        image = VF.resize(image, (self.image_size, self.image_size), antialias=True)
        image = F.pad(image, (self.padding, self.padding, self.padding, self.padding), mode=self.mode, value=0)

        blurred = apply_custom_blur(image.unsqueeze(0), self.uni_kernel, padding_mode=self.mode)

        H_pinv = self.get_pseudo_inverse()
        deblur = H_pinv.cpu().float() @ blurred.view(3,-1,1)
        deblur = deblur.view(3,self.padded_size,-1)
        return deblur