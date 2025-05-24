import argparse
import os

import pandas as pd
import torch
from diffusers import AutoPipelineForImage2Image
from image_editing_mask.attn_based import get_map_attn_based
from image_editing_mask.noise_based import get_map_noise_based
from image_editing_mask.ours import get_map_ours
from image_editing_mask.patcher import NoiseAttentionPatcher
from PIL import Image
from tqdm.auto import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate masks for Pie using different techniques."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        required=False,
        help="HuggingFace cache dir."
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="A device where to do processing."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=False,
        default="./generated_masks",
        help="A directory where to save feature maps."
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if not os.path.exists("pie_dataset.csv"):
        raise RuntimeError("Please run `process_pie.py` script firstly.")

    df = pd.read_csv("pie_dataset.csv", dtype={"idx": str})
    df.original_prompt = df.original_prompt.str.replace("[", "").str.replace("]", "")
    df.edit_prompt = df.edit_prompt.str.replace("[", "").str.replace("]", "")

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=args.cache_dir,
    ).to(args.device)

    patcher = NoiseAttentionPatcher()
    patcher.patch_attention(pipe.unet, patch_cross=True, patch_self=False)
    patcher.patch_call(pipe)

    for mask_obtaining_func in [get_map_attn_based, get_map_noise_based, get_map_ours]:
        os.makedirs(os.path.join(args.save_dir, mask_obtaining_func.__name__), exist_ok=True)

        for row in tqdm(df.itertuples(), total=df.shape[0], desc=f"Generate masks using {mask_obtaining_func.__name__}"):
            image = Image.open(row.path)
            fm = mask_obtaining_func(
                pipe,
                patcher,
                image,
                row.original_prompt,
                row.edit_prompt,
                num_repeat=1,
                strength=0.5,
                sample_mode="argmax"
            )
            torch.save(fm, os.path.join(args.save_dir, mask_obtaining_func.__name__, row.idx + ".data"))


if __name__ == "__main__":
    main()
