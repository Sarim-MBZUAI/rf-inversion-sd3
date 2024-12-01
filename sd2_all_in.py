import torch
import argparse
import os
from PIL import Image
from diffusers import StableDiffusionPipeline, FlowMatchEulerDiscreteScheduler
from torchvision import transforms
from tqdm import tqdm
import sys
from pathlib import Path


from sd2_rf_inversion import encode_imgs, interpolated_inversion, decode_imgs

def process_directory(
    input_dir,
    output_dir,
    pipeline,
    dtype=torch.float16,
    gamma=0.5,
    num_steps=28,
    seed=42
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files from input directory
    image_files = list(Path(input_dir).glob('*.png'))
    
    # Set up image transformation
    transform = transforms.Compose([
        transforms.Resize(768, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(768),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load and preprocess image
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0).to(pipeline.device).to(dtype)
            
            # Encode image to latents
            img_latent = encode_imgs(img, pipeline, dtype)
            
            # Perform inversion
            inversed_latent = interpolated_inversion(
                pipeline,
                img_latent,
                gamma=gamma,
                prompt="",
                DTYPE=dtype,
                num_steps=num_steps,
                seed=seed
            )
            
            # Decode latents back to image
            out = decode_imgs(inversed_latent, pipeline)[0]
            
            # Save with the same filename in output directory
            output_path = os.path.join(output_dir, img_path.name)
            out.save(output_path)
            
            print(f"Processed: {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Batch process images for inversion')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save inverted images')
    parser.add_argument('--model_path', type=str, default='/shared/shashmi/stable-diffusion-2', help='Path to the pretrained model')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter for interpolated_inversion')
    parser.add_argument('--num_steps', type=int, default=28, help='Number of steps for timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'], help='Data type for computations')
    
    args = parser.parse_args()
    
    # Set up dtype
    if args.dtype == 'bfloat16':
        DTYPE = torch.bfloat16
    elif args.dtype == 'float16':
        DTYPE = torch.float16
    else:
        DTYPE = torch.float32
    
    # Initialize the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=DTYPE,
        local_files_only=True
    ).to("cuda")
    
    # Set scheduler
    pipe.scheduler = FlowMatchEulerDiscreteScheduler()
    
    # Process all images
    process_directory(
        args.input_dir,
        args.output_dir,
        pipe,
        dtype=DTYPE,
        gamma=args.gamma,
        num_steps=args.num_steps,
        seed=args.seed
    )

if __name__ == "__main__":
    main()