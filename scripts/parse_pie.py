import argparse
import json
import os
import shutil

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse the Pie-Bench dataset, save metadata into a CSV file and copy mapping file into current dir."
    )
    parser.add_argument(
        "--ds-path",
        type=str,
        required=True,
        help="A path to the Pie-Bench dataset."
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    with open(os.path.join(args.ds_path, "mapping_file.json")) as fp:
        mapping = json.load(fp)

    ds = []
    for key in mapping:
        item = mapping[key]
        row = {
            "idx": key,
            "path": os.path.join(args.ds_path, "annotation_images", item["image_path"]),
            "original_prompt": item["original_prompt"],
            "edit_prompt": item["editing_prompt"],
            "edit_type": item["image_path"].split("/")[0],
            "edit_instruction": item["editing_instruction"],
        }
        ds.append(row)

    df = pd.DataFrame(ds)
    df.to_csv("pie_dataset.csv", index=False)
    shutil.copyfile(os.path.join(args.ds_path, "mapping_file.json"), "mapping_file_pie.json")

    print("Pie-Bench dataset was parsed. A CSV file is saved into `pie_dataset.csv`. The mapping file is copied.")


if __name__ == "__main__":
    main()
