import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torchvision.datasets import ImageFolder
import argparse

def load_images(image_folder, batch_size, transforms):
    dataset = ImageFolder(root=image_folder, transform=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def calculate_fid(real_images_loader, generated_images_loader, fid):
    for real_batch, _ in real_images_loader:
        fid.update(real_batch, real=True)

    for gen_batch, _ in generated_images_loader:
        fid.update(gen_batch, real=False)

    fid_score = fid.compute()
    return fid_score

def main(args):
    transforms = Compose([Resize((args.image_size, args.image_size)), ToTensor()])
                    #   Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    real_images_loader = load_images(args.real_path, args.batch_size, transforms)
    generated_images_loader = load_images(args.generated_path, args.batch_size, transforms)

    torch.manual_seed(0)
    fid = FrechetInceptionDistance(feature=args.feature, normalize=True).set_dtype(torch.float64)
    fid_score = calculate_fid(real_images_loader, generated_images_loader, fid)

    print(f"Real images: {len(real_images_loader.dataset)}, Generated images: {len(generated_images_loader.dataset)}")
    print(f"Batch size: {args.batch_size}, Image size: {args.image_size}, Feature size: {args.feature}")
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID score between real and generated images")
    parser.add_argument('--real_path', type=str, default="output/images/real",
                         required=True, help='Path to directory with real images')
    parser.add_argument('--generated_path', type=str, default="output/images/generated",
                        required=True, help='Path to directory with generated images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images')
    parser.add_argument('--image_size', type=int, default=299, help='Size of the images (299 for InceptionV3)')
    parser.add_argument('--feature', type=int, default=64, help='Feature size for FID')

    args = parser.parse_args()
    main(args)

# Use case:
"""sh
python mvp_score_modelling/evaluation/fid.py \
    --real_path output/images/real \
    --generated_path output/images/generated \
    --batch_size 32 --image_size 299 --feature 64 
"""

"""windows
python mvp_score_modelling/evaluation/fid.py ^
    --real_path output/images/real ^
    --generated_path output/images/generated ^
    --batch_size 32 --image_size 299 --feature 64 
"""