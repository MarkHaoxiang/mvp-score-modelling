import argparse
import os
import shutil

import torch
import torch.nn as nn
from diffusers import ScoreSdeVeScheduler, ScoreSdeVePipeline

from mvp_score_modelling.nn import ClassificationNet, ClassificationNetCNN
from mvp_score_modelling.utils import process_from_raw 
from mvp_score_modelling.pipelines.class_conditional import ClassConditionalClassifierPipeline

def main(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')

    model = torch.load(args.model_path, map_location=DEVICE)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
        print(f"Deleted {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)

    print(f"Generating {args.num_batch} batches of {args.batch_size} images, in total {args.num_batch * args.batch_size} images")

    PRETRAINED = "google/ncsnpp-celebahq-256"
    unconditional_pipeline: ScoreSdeVePipeline = ScoreSdeVePipeline.from_pretrained(PRETRAINED).to(device=DEVICE)

    pipeline = ClassConditionalClassifierPipeline(
        unconditional_pipeline.unet,
        unconditional_pipeline.scheduler
    ).to(DEVICE)

    for b in range(args.num_batch):
        # interleave targets
        target = b % 2
        images = pipeline(
            num_inference_steps=args.inference_steps,
            y=(model, target),
            batch_size=args.batch_size,
            temperature=args.temperature
        )
        for i, image in enumerate(images):
            filepath = os.path.join(args.save_path, f'image_b{b}_t{target}_{i}.png')
            image.save(filepath)
        print(f"Inference {b} completed, saved {len(images)} images to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image generation pipeline with class conditional models.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save generated images.")
    parser.add_argument("--num_batch", type=int, required=True, help="Number of batches to generate.")
    parser.add_argument("--batch_size", type=int, required=True, help="Size of each batch.")
    parser.add_argument("--inference_steps", type=int, required=True, help="Number of inference steps.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for classifier.")

    args = parser.parse_args()
    print(args)
    main(args)

# use case:
"""sh
python scripts/class_conditional_inference.py \
    --model_path checkpoints/test \
    --save_path output/class_conditional_inference/test/all \
    --num_batch 1 \
    --batch_size 1 \
    --inference_steps 800 \
    --temperature 1.0
"""

"""windows
python scripts/class_conditional_inference.py ^
    --model_path checkpoints/test ^
    --save_path output/class_conditional_inference/test/all ^
    --num_batch 1 ^
    --batch_size 1 ^
    --inference_steps 800 ^
    --temperature 1.0
"""