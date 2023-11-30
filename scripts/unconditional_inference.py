import argparse
import os
import shutil

import torch
from diffusers import ScoreSdeVePipeline

from mvp_score_modelling.utils import process_from_raw 
from mvp_score_modelling.pipelines.utils import CustomConditionalScoreVePipeline

def main(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
        print(f"Deleted {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)

    print(f"Generating {args.num_batch} batches of {args.batch_size} images, in total {args.num_batch * args.batch_size} images")

    PRETRAINED = "google/ncsnpp-celebahq-256"
    unconditional_pipeline: ScoreSdeVePipeline = ScoreSdeVePipeline.from_pretrained(PRETRAINED).to(device=DEVICE)

    pipeline = CustomConditionalScoreVePipeline(
        unconditional_pipeline.unet,
        unconditional_pipeline.scheduler
    ).to(DEVICE)

    for b in range(args.num_batch):
        # interleave targets
        target = b % 2
        images = pipeline(
            num_inference_steps=args.inference_steps,
            y=None,
            batch_size=args.batch_size
        )
        for i, image in enumerate(images):
            filepath = os.path.join(args.save_path, f'image_b{b}_t{target}_{i}.png')
            image.save(filepath)
        print(f"Inference {b} completed, saved {len(images)} images to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image generation pipeline with unconditional model.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save generated images.")
    parser.add_argument("--num_batch", type=int, required=True, help="Number of batches to generate.")
    parser.add_argument("--batch_size", type=int, required=True, help="Size of each batch.")
    parser.add_argument("--inference_steps", type=int, required=True, help="Number of inference steps.")

    args = parser.parse_args()
    print(args)
    main(args)