# RF Inversion for Stable Diffusion Models

This repository implements RF (Reverse Flow) inversion techniques for various versions of Stable Diffusion models, including SD2, SD3, SD3.5, and SDXL. The implementation enables better control over image generation through interpolated denoising and inversion processes.

## Features

- Support for multiple Stable Diffusion versions:
  - Stable Diffusion 2.0
  - Stable Diffusion 3.0
  - Stable Diffusion 3.5
  - Stable Diffusion XL
- Interpolated denoising with configurable parameters
- Flexible eta scheduling (constant, linear increase, linear decrease)
- Support for different precision types (float16, bfloat16, float32)
- Customizable guidance scale and number of steps
- Source prompt conditioning for better inversion

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd rf-inversion-sd3

# Install required dependencies
pip install torch diffusers transformers pillow
```

## Usage

### Basic Example

```bash
python sd3_rf_inversion.py \
    --image_path "path/to/your/image.png" \
    --model_path "path/to/sd3/model" \
    --prompt "your generation prompt" \
    --output_dir "output"
```

### Advanced Parameters

```bash
python sd3_rf_inversion.py \
    --image_path "path/to/your/image.png" \
    --model_path "path/to/sd3/model" \
    --prompt "your generation prompt" \
    --eta_base 0.95 \
    --eta_trend "linear_decrease" \
    --start_step 0 \
    --end_step 9 \
    --guidance_scale 3.5 \
    --num_steps 28 \
    --seed 42 \
    --gamma 0.5 \
    --dtype "float16"
```

## Parameters

- `--model_path`: Path to the pretrained Stable Diffusion model
- `--image_path`: Path to the input image
- `--output_dir`: Directory to save output images (default: 'output')
- `--eta_base`: Base eta parameter for interpolated denoising (default: 0.95)
- `--eta_trend`: Eta scheduling trend ['constant', 'linear_increase', 'linear_decrease']
- `--start_step`: Starting step for eta values (default: 0)
- `--end_step`: Ending step for eta values (default: 9)
- `--no_inversion`: Flag to skip the inversion process
- `--guidance_scale`: Guidance scale for interpolated denoising (default: 3.5)
- `--num_steps`: Number of denoising steps (default: 28)
- `--seed`: Random seed for reproducibility (default: 42)
- `--gamma`: Gamma parameter for interpolated inversion (default: 0.5)
- `--prompt`: Generation prompt
- `--source_prompt`: Prompt describing the source image
- `--dtype`: Computation precision ['float16', 'bfloat16', 'float32']

## Directory Structure

```
├── LICENSE
├── README.md
├── sd2_rf_inversion.py    # RF inversion for SD2
├── sd3_rf_inversion.py    # RF inversion for SD3
├── sd3_5_rf_inversion.py  # RF inversion for SD3.5
├── sdxl_rf_inversion.py   # RF inversion for SDXL
├── images/               # Input images directory
├── output/              # Generated images directory
└── generated_images/    # Additional output directory
```

