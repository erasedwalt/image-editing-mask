import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from image_editing_mask.metrics_calculator import MetricsCalculator
from PIL import Image
from tqdm.auto import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate editing metrics."
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda:0",
        help="A device for calculations."
    )
    parser.add_argument(
        "--edits-dir",
        type=str,
        required=False,
        default="./edited_pics",
        help="A directory where edited images have been saved."
    )
    return parser.parse_args()


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

    calculator = MetricsCalculator(args.device)

    metrics = {}
    edits_path = Path(args.edits_dir)
    for method_path in edits_path.glob("*"):
        for t_path in method_path.glob("*"):
            metrics[f"{method_path.name}~{t_path.name}"] = []
            for row in tqdm(df.itertuples(), total=df.shape[0], desc=f"Calculate metrics for {method_path.name} with t={t_path.name}"):
                src_image = Image.open(row.path)
                tgt_image = Image.open(t_path / (row.idx + ".png"))
                mask = mask_decode(mapping[row.idx]["mask"])[..., None].repeat([3], axis=-1)

                metrics[f"{method_path.name}~{t_path.name}"].append({
                    "clip_whole": calculator.calculate_clip_similarity(tgt_image, row.edit_prompt),
                    "clip_edited": calculator.calculate_clip_similarity(tgt_image, row.edit_prompt, mask) if mask.sum() > 0 else None,
                    "lpips": calculator.calculate_lpips(src_image, tgt_image, 1 - mask, 1 - mask) if (1 - mask).sum() > 0 else None,
                    "psnr": calculator.calculate_psnr(src_image, tgt_image, 1 - mask, 1 - mask) if (1 - mask).sum() > 0 else None,
                    "ssim": calculator.calculate_ssim(src_image, tgt_image, 1 - mask, 1 - mask) if (1 - mask).sum() > 0 else None,
                    "mse": calculator.calculate_mse(src_image, tgt_image, 1 - mask, 1 - mask) if (1 - mask).sum() > 0 else None,
                    **row._asdict(),
                })

    with open("total_editing_metrics.json", "w") as fp:
        json.dump(metrics, fp)

    print("Total metrics table was saved into `total_editing_metrics.json`.")


if __name__ == "__main__":
    main()
