import torch
from torchvision.transforms import Resize, ToPILImage, CenterCrop

crop = CenterCrop(256)
resize = Resize(256)
softmax = torch.nn.Softmax()
tensor_to_PIL = ToPILImage()


def plt_img(ax, img):
    ax.imshow(img.detach().cpu().permute(-2,-1,-3))

def process_from_raw(img: torch.Tensor) -> torch.Tensor:
    """ Transforms an image from a huggingface tensor dataset
    to the format expected by the pipeline

    Args:
        img (Tensor): _description_

    Returns:
        Tensor: _description_
    """
    return resize(img.to(torch.float32).transpose(-1,-3).transpose(-1,-2) / 256)

def compute_accuracy(dataloader, model, loss_fn, process):
    # Adapted from the MVP exercises
    num_batches, size, test_loss, correct = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            X, y = data['image'], data['label']
            size += X.shape[0]
            num_batches += 1
            y_p = model(process(X))
            test_loss += loss_fn(y_p, y).item()
            y_p = softmax(y_p)
            correct += torch.all(((y_p > 0.5) == y), dim=1).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss