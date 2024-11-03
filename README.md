# rf-inversion-sd3
[Unofficial] [RF Inversion](https://rf-inversion.github.io/) implemented for [Stable Diffusion 3](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3) (Also compatible with 3.5)

| Source Image (Dog) | **Prompt:** Photograph of a cat on the grass |
| ---- | ----- |
| ![Dog](images/dog.jpg)   | ![Cat](images/cat.png) |


## Getting started 

```
python sd3_rf_inversion.py
```

This code have a few option that you can tuning around 

```
options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the pretrained model
  --image_path IMAGE_PATH
                        Path to the input image
  --output_dir OUTPUT_DIR
                        Directory to save output images
  --eta_base ETA_BASE   Eta parameter for interpolated_denoise
  --eta_trend {constant,linear_increase,linear_decrease}
                        Eta trend for interpolated_denoise
  --start_step START_STEP
                        Start step for eta values, 0-based indexing, closed interval
  --end_step END_STEP   End step for eta values, 0-based indexing, open interval
  --no_inversion        Skip the inversion progress. Useful for comparing between with and without inversion
  --guidance_scale GUIDANCE_SCALE
                        Guidance scale for interpolated_denoise
  --num_steps NUM_STEPS
                        Number of steps for timesteps
  --seed SEED           seed for generation
  --gamma GAMMA         Gamma parameter for interpolated_inversion
  --prompt PROMPT       Prompt text for generation
  --dtype {float16,bfloat16,float32}
                        Data type for computations
```

## Requirements 
This repository need diffusers 0.31.0 or newer


## Acknowledgements 
Some portion of code taken from [rf-inversion-diffuser](https://github.com/DarkMnDragon/rf-inversion-diffuser) which is design to run on Flux
