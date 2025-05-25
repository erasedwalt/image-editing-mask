# Image Editing Mask

This repository implements various techniques for generating binary masks to identify objects targeted for editing, using only a target diffusion model ([SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) in our case).

We examine three types of techniques (the only known ones to our knowledge):
- Noise-based technique (see for example [Diffedit](https://arxiv.org/abs/2210.11427))
- Attention-based technique (see for example [Prompt-to-prompt](https://arxiv.org/abs/2208.01626))
- Our proposed technique

## Usage

### Installation

Use this code as a Python module by installing it via:
```bash
pip install .
```

After installation, import the module using `image_editing_mask`.

This implementation includes a modified version of [Turbo-Edit](https://arxiv.org/abs/2408.00735) that supports binary masks for object editing. For reference, please see the original repository: https://github.com/GiilDe/turbo-edit.

### Image Editing

The EditingPipeline class enables image editing by combining Turbo-Edit with one of three mask generation techniques:
1. `ours`
1. `attn_based`
1. `noise_based`

The first two methods share a binarization threshold range of 0 to 1, while the noise_based method uses a range of 0 to ~10 (consistent with Diffedit's implementation [here](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/stable_diffusion_diffedit/pipeline_stable_diffusion_diffedit.py#L1046)). Adjusting `num_repeat`, `strength`, `seed`, and `sample_mode` may improve results.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Required to set device (turbo-edit limitation)

from PIL import Image
from image_editing_mask.editing import EditingPipeline

cache_dir = "/HF/cache/dir" # or None

pipeline = EditingPipeline(cache_dir)
image = Image.open("/path/to/image.png")

result = pipeline.edit(
    image,
    source_prompt = "a face",
    target_prompt = "a face with glasses",
    mask_obtaining = "ours", # Alternatives: "attn_based", "noise_based"
    binarization_threshold = 0.5,
    num_repeat = 1, # Higher values produce more stable masks
    strength = 0.5, # Options: 0.25, 0.5, 0.75
    sample_mode = "argmax", # Alternative: "sample"
    seed = 8128,
)
```

To generate only an object mask:

```python
import random

from diffusers import AutoPipelineForImage2Image
from image_editing_mask.ours import get_map_ours
from image_editing_mask.attn_based import get_map_attn_based
from image_editing_mask.noise_based import get_map_noise_based
from image_editing_mask.patcher import NoiseAttentionPatcher


pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir="/hf/cache/dir",
).to("device")

patcher = NoiseAttentionPatcher()
patcher.patch_attention(pipe.unet, patch_cross=True, patch_self=False)
patcher.patch_call(pipe)

image = Image.open("/path/to/image.png")

func = random.choice([get_map_ours, get_map_attn_based, get_map_noise_based])
mask = func(
    pipe,
    patcher,
    image,
    "a face",
    "a face with glasses",
    num_repeat=1,
    strength=0.5,
    sample_mode="argmax"
)
```

### Scripts for Pie-Bench

Run these scripts to evaluate all three techniques on the [Pie-Bench](https://arxiv.org/abs/2310.01506) dataset for both segmentation and editing tasks (with Turbo-Edit):

```bash
# Parse dataset
python scripts/parse_pie.py --ds-path /path/to/pie/bench

# Generate masks using all methods
python scripts/generate_masks_pie.py --cache-dir /hf/cache/dir --device cuda:1

# Edit images using generated masks
CUDA_VISIBLE_DEVICES=1 python scripts/edit_pie.py --mode one --cache-dir /hf/cache/dir

# Calculate editing metrics
python scripts/calculate_editing_metrics.py --device cuda:1

# Calculate segmentation metrics
python scripts/calculate_segmentation_metrics.py 
```

The editing metrics script uses the original implementation from Pie-Bench authors. Reference their [paper](https://arxiv.org/abs/2310.01506) and [code](https://github.com/cure-lab/PnPInversion) for details.

## Binary Segmentation Quality

The table below shows Dice metric scores for different techniques on the Pie-Bench dataset, with the best performance highlighted in bold.

<img src="assets/segmentation_table.png" alt="drawing" width="500"/>

Example segmentation results for different prompts:

<img src="assets/segmentation_examples.png" alt="drawing" width="800"/>


## Image Editing Quality

The graph below compares CLIP Similarity (within target mask) and LPIPS scores (outside target mask) across techniques when combined with Turbo-Edit with varying binarization thresholds. Better performance appears in the upper-left quadrant.

<img src="assets/editing_graph.png" alt="drawing" width="500"/>

Example editing results from Pie-Bench:

<img src="assets/editing_examples.png" alt="drawing" width="1000"/>

Comparison table of our method (with Turbo-Edit) against other fast editing methods ([InfEdit](https://arxiv.org/abs/2312.04965) and [Adobe's Turbo-Edit](https://arxiv.org/abs/2408.08332)):

<img src="assets/editing_table.png" alt="drawing" width="1000"/>
