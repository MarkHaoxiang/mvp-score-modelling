import torch
import torch.nn as nn
from .utils import CustomConditionalScoreVePipeline, VeTweedie


class _ClassConditionalClassifierPipeline(CustomConditionalScoreVePipeline):
    softmax = nn.Softmax()
    def initialise_inference(self, y, generator, batch_size: int, n: int):
        self.classifier, self.target = y
        return super().initialise_inference(y, generator, batch_size, n)
    
class ClassConditionalClassifierPipeline(_ClassConditionalClassifierPipeline):
    def calculate_score(self, y, x_t, sigma_t):
        s_x =  super().calculate_score(y, x_t, sigma_t)
        with torch.enable_grad():
            x_t.requires_grad = True
            class_probabilities = self.softmax(self.classifier(x_t))
            target = class_probabilities[:, self.target]
            torch.log(target).sum().backward()
        return s_x + x_t.grad

class NoiselessConditionalClassifierPipeline(_ClassConditionalClassifierPipeline):
    def initialise_inference(self, y, generator, batch_size: int, n: int):
        self.tweedie = VeTweedie(self.unet)
        return super().initialise_inference(y, generator, batch_size, n)

    def calculate_score(self, y, x_t, sigma_t):
        s_x =  super().calculate_score(y, x_t, sigma_t)
        with torch.enable_grad():
            x_t.requires_grad = True
            x_hat = self.tweedie(x_t, sigma_t)
            class_probabilities = self.softmax(self.classifier(x_hat))
            target = class_probabilities[:, self.target]
            torch.log(target).sum().backward()
        return s_x + torch.nan_to_num(x_t.grad)
