from collections import defaultdict
import argparse
import json
import os
from pathlib import Path

import torch
from image_editing_mask.common_utils import binarize
from image_editing_mask.noise_based import binarize_noise_based
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate segmentation metrics"
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        required=False,
        default="./generated_masks",
        help="A directory where generated masks have been saved."
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


def segmentation_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.flatten()
    target = target.flatten()
    npred = 1 - pred
    ntarget = 1 - target
    tp = torch.sum(pred * target)
    fp = torch.sum(pred * ntarget)
    fn = torch.sum(npred * target)
    result = {
        "dice": ((2 * tp + 1e-7) / (2 * tp + fn + fp + 1e-7)).item(),
        "precision": ((tp + 1e-7) / (tp + fp + 1e-7)).item(),
        "recall": ((tp + 1e-7) / (tp + fn + 1e-7)).item(),
        "iou": ((tp + 1e-7) / (tp + fn + fp + 1e-7)).item(),
    }
    return result


def mask_decode(encoded_mask, image_shape=[512,512]):
    length=image_shape[0] * image_shape[1]
    mask_array=np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def main():
    args = _parse_args()

    if not os.path.exists("pie_dataset.csv"):
        raise RuntimeError("Please run `process_pie.py` script firstly.")

    with open("mapping_file_pie.json") as fp:
        mapping = json.load(fp)

    df = pd.read_csv("pie_dataset.csv", dtype={"idx": str})
    df.original_prompt = df.original_prompt.str.replace("[", "").str.replace("]", "")
    df.edit_prompt = df.edit_prompt.str.replace("[", "").str.replace("]", "")

    metrics = {}
    masks_path = Path(args.masks_dir)
    for method_path in masks_path.glob("*"):
        metrics[method_path.name] = defaultdict(list)
        for row in tqdm(df.itertuples(), total=df.shape[0], desc=f"Calculate metrics for {method_path.name}"):
            fm = torch.load(method_path / (row.idx + ".data"), map_location="cpu")
            tgt_mask = torch.from_numpy(mask_decode(mapping[row.idx]["mask"])).byte()

            for t in T_RANGES_FULL[method_path.name]:
                mask = BINARIZE_MAP[method_path.name](fm, t)

                metrics[method_path.name][f"{t:.3f}"].append({
                    **row._asdict(),
                    **segmentation_metrics(mask, tgt_mask),
                    **{k + "_inv": v for k, v in segmentation_metrics(1 - mask, 1 - tgt_mask).items()},
                })

    with open("total_segmentation_metrics.json", "w") as fp:
        json.dump(metrics, fp)

    print("Total metrics table was saved into `total_segmentation_metrics.json`.")


if __name__ == "__main__":
    main()
