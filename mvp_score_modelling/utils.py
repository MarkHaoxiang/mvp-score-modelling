import torch
from torchvision.transforms import Resize, ToPILImage, CenterCrop
from torchvision.datasets import ImageFolder

crop = CenterCrop(256)
resize = Resize(256)
softmax = torch.nn.Softmax()
_tensor_to_PIL = ToPILImage()

def tensor_to_PIL(img: torch.Tensor):
    shape = img.shape
    if len(shape) == 3:
        return _tensor_to_PIL(img)
    elif shape[0] == 1:
        return _tensor_to_PIL(img.squeeze())
    else:
        return [_tensor_to_PIL(im) for im in img]

def plt_img(ax, img):
    ax.imshow(img.squeeze().detach().cpu().permute(-2,-1,-3))

def process_from_raw(img: torch.Tensor) -> torch.Tensor:
    """ Transforms an image from a huggingface tensor dataset
    to the format expected by the pipeline

    Args:
        img (Tensor): _description_

    Returns:
        Tensor: _description_
    """
    return resize(img.to(torch.float32).transpose(-1,-3).transpose(-1,-2) / 256)

def compute_accuracy(dataloader, model, loss_fn, process, device=None):
    # Adapted from the MVP exercises
    num_batches, size, test_loss, correct = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            X, y = data['image'], data['label']
            if device is not None:
                X, y = X.to(device), y.to(device)
            size += X.shape[0]
            num_batches += 1
            sigmas = torch.zeros(X.shape[0]).to(X.device)
            y_p = model(process(X), sigmas)
            test_loss += loss_fn(y_p, y).item()
            y_p = softmax(y_p)
            correct += ((y_p.argmax(1)) == y).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss

def load_images(image_folder, batch_size, transforms):
    dataset = ImageFolder(root=image_folder, transform=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader