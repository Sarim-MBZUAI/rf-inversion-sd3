import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np
from DDIM_inversion import ddim_inversion
from null_text_inversion import NullInversion, load_1024
from diffusers import StableDiffusionXLPipeline, DDIMScheduler 


def setup_pipeline():
    model_id = "/shared/shashmi/stable-diffusion-xl-base-1.0"
    
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False
    )
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,  # Make sure this is set
        use_safetensors=True
    ).to("cuda")
    
    # Ensure all model components are in float16
    pipe.to(dtype=torch.float16)
    
    return pipe




def process_image(image_path, prompt, num_steps=50, guidance_scale=7.5):
    # Load and prepare the pipeline
    pipe = setup_pipeline()
    
    # Initialize the NullInversion object
    null_inverter = NullInversion(
        model=pipe,
        prompt=prompt,
        ddim_steps=num_steps,
        guidance_scale=guidance_scale
    )
    
    # Perform inversion
    image_gt, ddim_latents, uncond_embeddings = null_inverter.invert(
        image_path=image_path,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
        verbose=True
    )
    
    return image_gt, ddim_latents, uncond_embeddings

def generate_variations(pipe, ddim_latents, uncond_embeddings, prompt, num_variations=4):
    # Generate variations using the inverted latents
    results = []
    
    for i in range(num_variations):
        # Use the latents and embeddings to condition the generation
        with torch.no_grad():
            variation = pipe(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                latents=ddim_latents[-1],
                negative_prompt_embeds=uncond_embeddings[-1]
            ).images[0]
            
        results.append(variation)
    
    return results

def save_results(original, variations, output_dir="outputs"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original
    Image.fromarray(original).save(f"{output_dir}/original.png")
    
    # Save variations
    for i, var in enumerate(variations):
        var.save(f"{output_dir}/variation_{i}.png")

def main():
    # Example usage
    image_path = "/ephemeral/shashmi/rf-inversion-sd3/images/cat.png"
    prompt = "a professional photograph of a cat in a garden"
    
    # Setup
    pipe = setup_pipeline()
    
    # Process the image
    print("Processing image...")
    image_gt, ddim_latents, uncond_embeddings = process_image(
        image_path=image_path,
        prompt=prompt
    )
    
    # Generate variations
    print("Generating variations...")
    variations = generate_variations(
        pipe=pipe,
        ddim_latents=ddim_latents,
        uncond_embeddings=uncond_embeddings,
        prompt=prompt
    )
    
    # Save results
    print("Saving results...")
    save_results(image_gt, variations)
    
    print("Done! Check the outputs directory for results.")

if __name__ == "__main__":
    main()