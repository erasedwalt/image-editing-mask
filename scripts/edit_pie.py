import argparse
import os

import numpy as np
import pandas as pd
import torch
from image_editing_mask.common_utils import binarize
from image_editing_mask.noise_based import binarize_noise_based
from image_editing_mask.turbo_edit.main import load_pipe, run
from PIL import Image
from tqdm.auto import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit images from Pie using different techniques. "
            "Please set device using `CUDA_VISIBLE_DEVICES=num python scripts/edit_pie.py`. "
            "This is because of turbo-edit's original limitations."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["full", "one"],
        help="Whether to run the full threshold search or just use one selected threshold for all methods."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        required=False,
        help="HuggingFace cache dir."
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        required=False,
        default="./generated_masks",
        help="A directory where masks have been saved."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=False,
        default="./edited_pics",
        help="A directory where to save edited images."
    )
    return parser.parse_args()


BINARIZE_MAP = {
    "get_map_attn_based": binarize,
    "get_map_noise_based": binarize_noise_based,
    "get_map_ours": binarize,
}

T_RANGES_FULL = {
    "get_map_attn_based": np.linspace(0, 1, 21),
    "get_map_noise_based": np.linspace(0, 5, 21),
    "get_map_ours": np.linspace(0, 1, 21),
}

T_RANGES_ONE = {
    "get_map_attn_based": [0.6],
    "get_map_noise_based": [3.],
    "get_map_ours": [0.7],
}

def main():
    args = _parse_args()

    if not os.path.exists("pie_dataset.csv"):
        raise RuntimeError("Please run `process_pie.py` script firstly.")

    df = pd.read_csv("pie_dataset.csv", dtype={"idx": str})
    df.original_prompt = df.original_prompt.str.replace("[", "").str.replace("]", "")
    df.edit_prompt = df.edit_prompt.str.replace("[", "").str.replace("]", "")

    pipeline = load_pipe(fp16=True, cache_dir=args.cache_dir)
    pipeline.set_progress_bar_config(disable=True)

    if args.mode == "full":
        trange = T_RANGES_FULL
    else:
        trange = T_RANGES_ONE

    for method, binarization_func in BINARIZE_MAP.items():
        for row in tqdm(df.itertuples(), total=df.shape[0], desc=f"Edit images using {method}'s masks"):
            image = Image.open(row.path)
            fm = torch.load(os.path.join(args.masks_dir, method, row.idx + ".data"), map_location=pipeline.device)

            for t in tqdm(trange[method], leave=False, desc="Iterate over the thresholds..."):
                os.makedirs(os.path.join(args.save_dir, method, f"{t:.3f}"), exist_ok=True)

                mask = binarization_func(fm, t)

                mask = torch.nn.functional.interpolate(
                    input=mask[None, None].float(),
                    size=(64, 64),
                    mode="nearest-exact"
                )[0, 0].byte()

                res = run(
                    image=image,
                    src_prompt=row.original_prompt,
                    tgt_prompt=row.edit_prompt,
                    seed=8128,
                    w1=1.5,
                    num_timesteps=4,
                    pipeline=pipeline,
                    mask=mask
                )
                res.save(os.path.join(args.save_dir, method, f"{t:.3f}", row.idx + ".png"))


if __name__ == "__main__":
    main()
