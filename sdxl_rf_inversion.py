import torch
import argparse
import os
from PIL import Image
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler
from torchvision import transforms
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
import numpy as np
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@torch.inference_mode()
def decode_imgs(latents, pipeline):
    # Scale and decode with VAE
    latents = 1 / pipeline.vae.config.scaling_factor * latents
    imgs = pipeline.vae.decode(latents)[0]
    
    # Move to CPU and convert to float32 for numeric stability
    imgs = imgs.cpu().float()
    
    # Normalize from [-1, 1] to [0, 1] with proper clamping
    imgs = ((imgs + 1) / 2).clamp(0, 1)
    
    # Scale to [0, 255]
    imgs = (imgs * 255).round()
    
    # Convert to numpy and ensure values are in valid range
    imgs = imgs.numpy()
    imgs = np.maximum(np.minimum(imgs, 255), 0)
    
    # Convert to uint8
    imgs = imgs.astype(np.uint8)
    
    # Rearrange dimensions
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    
    # Convert to PIL Image (return single image since we're processing one at a time)
    return Image.fromarray(imgs[0])

@torch.inference_mode()
def encode_imgs(imgs, pipeline, DTYPE):
    latents = pipeline.vae.encode(imgs).latent_dist.sample()
    latents = pipeline.vae.config.scaling_factor * latents
    latents = latents.to(dtype=DTYPE)
    return latents

def generate_eta_values(timesteps, start_step, end_step, eta, eta_trend):
    assert start_step < end_step and start_step >= 0 and end_step <= len(timesteps), "Invalid start_step and end_step"
    eta_values = [0.0] * len(timesteps)
    
    if eta_trend == 'constant':
        for i in range(start_step, end_step):
            eta_values[i] = eta
    elif eta_trend == 'linear_increase':
        total_time = timesteps[start_step] - timesteps[end_step - 1]
        for i in range(start_step, end_step):
            eta_values[i] = eta * (timesteps[start_step] - timesteps[i]) / total_time
    elif eta_trend == 'linear_decrease':
        total_time = timesteps[start_step] - timesteps[end_step - 1]
        for i in range(start_step, end_step):
            eta_values[i] = eta * (timesteps[i] - timesteps[end_step - 1]) / total_time
    return eta_values

@torch.inference_mode()
def interpolated_denoise(
    pipeline, 
    img_latents,
    eta_base,
    eta_trend,
    start_step,
    end_step,
    inversed_latents,
    use_inversed_latents=True,
    guidance_scale=7.5,
    prompt='photo of a tiger',
    negative_prompt="",
    DTYPE=torch.float16,
    num_steps=28,
    seed=42
):
    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_steps, pipeline.device)
    
    # Text conditioning
    do_classifier_free_guidance = guidance_scale > 1.0
    
    if prompt == "":
        prompt_embeds = negative_prompt_embeds = torch.zeros(
            (1, 77, 2048), device=pipeline.device, dtype=DTYPE
        )
        pooled_prompt_embeds = negative_pooled_prompt_embeds = torch.zeros(
            (1, 1280), device=pipeline.device, dtype=DTYPE
        )
    else:
        prompt_embeds = pipeline._encode_prompt(
            prompt=prompt,
            device=pipeline.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_embeds

    if use_inversed_latents:
        latents = inversed_latents
    else:
        set_seed(seed)
        latents = torch.randn_like(img_latents)
    
    target_img = img_latents.clone().to(torch.float32)
    eta_values = generate_eta_values(timesteps, start_step, end_step, eta_base, eta_trend)

    with pipeline.progress_bar(total=num_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # Expand the latents for classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
                
            timestep = t.expand(latent_model_input.shape[0])

            # Add time ids for SDXL - ensure they're on the correct device
            add_time_ids = pipeline._get_add_time_ids(
                original_size=(1024, 1024),
                crops_coords_top_left=(0, 0),
                target_size=(1024, 1024),
                dtype=latents.dtype,
                text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
            ).to(pipeline.device)  # Add this .to(pipeline.device)

            if do_classifier_free_guidance:
                add_time_ids = add_time_ids.repeat(2, 1)

            # Predict the noise residual
            pred_velocity = pipeline.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]) if do_classifier_free_guidance else prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]) if do_classifier_free_guidance else pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                },
                return_dict=False,
            )[0]

            # Handle classifier free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = pred_velocity.chunk(2)
                pred_velocity = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Prevents precision issues
            latents = latents.to(torch.float32)
            pred_velocity = pred_velocity.to(torch.float32)

            # Target image velocity
            t_curr = t / pipeline.scheduler.config.num_train_timesteps
            target_velocity = -(target_img - latents) / t_curr

            # interpolated velocity
            eta = eta_values[i]
            interpolate_velocity = pred_velocity + eta * (target_velocity - pred_velocity)

            # denoising
            latents = pipeline.scheduler.step(interpolate_velocity, t, latents, return_dict=False)[0]
            
            latents = latents.to(DTYPE)
            progress_bar.update()
    
    return latents

@torch.inference_mode()
def interpolated_inversion(
    pipeline,
    latents,
    gamma,
    DTYPE,
    prompt="",
    num_steps=28,
    seed=42
):
    pipeline.scheduler.set_timesteps(num_steps, device=pipeline.device)

    if not hasattr(pipeline.scheduler, "sigmas"):
        raise Exception("Cannot find sigmas variable in scheduler. Please use FlowMatchEulerDiscreteScheduler for RF Inversion")
    
    timesteps = pipeline.scheduler.sigmas
    timesteps = torch.flip(timesteps, dims=[0])

    # For inversion, generate empty embeddings - make sure they're on the correct device
    empty_embeds = torch.zeros((1, 77, 2048), device=pipeline.device, dtype=DTYPE)
    empty_pooled_embeds = torch.zeros((1, 1280), device=pipeline.device, dtype=DTYPE)

    set_seed(seed)
    target_noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32)

    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((latents.shape[0],), t_curr * 1000, dtype=latents.dtype, device=latents.device)

            # Add time ids for SDXL - ensure they're on the correct device
            add_time_ids = pipeline._get_add_time_ids(
                original_size=(1024, 1024),
                crops_coords_top_left=(0, 0),
                target_size=(1024, 1024),
                dtype=latents.dtype,
                text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
            ).to(pipeline.device)  # Add this .to(pipeline.device)

            # Null-text velocity
            pred_velocity = pipeline.unet(
                latents,
                t_vec,
                encoder_hidden_states=empty_embeds,
                added_cond_kwargs={
                    "text_embeds": empty_pooled_embeds,
                    "time_ids": add_time_ids,
                },
                return_dict=False,
            )[0]

            # Prevents precision issues
            latents = latents.to(torch.float32)
            pred_velocity = pred_velocity.to(torch.float32)

            # Target noise velocity
            target_noise_velocity = (target_noise - latents) / (1.0 - t_curr)
            
            # interpolated velocity
            interpolated_velocity = gamma * target_noise_velocity + (1 - gamma) * pred_velocity
            
            # one step Euler
            latents = latents + (t_prev - t_curr) * interpolated_velocity
            
            latents = latents.to(DTYPE)
            progress_bar.update()
            
    return latents

def main():
    parser = argparse.ArgumentParser(description='Test interpolated_denoise with SDXL.')
    parser.add_argument('--model_path', type=str, default='/shared/shashmi/stable-diffusion-xl-base-1.0', help='Path to the pretrained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output images')
    parser.add_argument('--eta_base', type=float, default=0.95, help='Eta parameter for interpolated_denoise')
    parser.add_argument('--eta_trend', type=str, default='constant', choices=['constant', 'linear_increase', 'linear_decrease'], help='Eta trend for interpolated_denoise')
    parser.add_argument('--start_step', type=int, default=0, help='Start step for eta values')
    parser.add_argument('--end_step', type=int, default=9, help='End step for eta values')
    parser.add_argument('--no_inversion', action='store_true', help='Skip the inversion progress')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for interpolated_denoise')    
    parser.add_argument('--num_steps', type=int, default=28, help='Number of steps for timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter for interpolated_inversion')
    parser.add_argument('--prompt', type=str, default='', help='Prompt for generation')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative prompt for generation')
    parser.add_argument('--source_prompt', type=str, default='', help='Prompt of the source image')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'], help='Data type for computations')
    
    args = parser.parse_args()

    if args.dtype == 'bfloat16':
        DTYPE = torch.bfloat16
    elif args.dtype == 'float16':
        DTYPE = torch.float16
    elif args.dtype == 'float32':
        DTYPE = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    # Initialize SDXL pipeline
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16" if args.dtype == "float16" else None,
        local_files_only=True
    ).to("cuda")
    
    # Set scheduler
    pipe.scheduler = FlowMatchEulerDiscreteScheduler()

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess the image
    img = Image.open(args.image_path)
    transform = transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),  # SDXL uses 1024x1024
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    img = transform(img).unsqueeze(0).to(pipe.device).to(DTYPE)

    # Encode image to latents
    img_latent = encode_imgs(img, pipe, DTYPE)

    if not args.no_inversion:
        inversed_latent = interpolated_inversion(
            pipe, 
            img_latent,
            gamma=args.gamma,
            prompt=args.source_prompt,
            DTYPE=DTYPE,
            num_steps=args.num_steps,
            seed=args.seed
        )    
    else:
        inversed_latent = None

    # Denoise
    img_latents = interpolated_denoise(
        pipe, 
        img_latent,
        eta_base=args.eta_base,
        eta_trend=args.eta_trend,
        start_step=args.start_step,
        end_step=args.end_step,
        inversed_latents=inversed_latent,
        use_inversed_latents=not args.no_inversion,
        guidance_scale=args.guidance_scale,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        DTYPE=DTYPE,
        seed=args.seed
    )

    # Decode latents to images
    out = decode_imgs(img_latents, pipe)

    # Save output image
    output_filename = f"eta{args.eta_base}_{args.eta_trend}_start{args.start_step}_end{args.end_step}_inversed{not args.no_inversion}_guidance{args.guidance_scale}_seed{args.seed}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    out.save(output_path)
    print(f"Saved output image to {output_path}")

if __name__ == "__main__":
    main()