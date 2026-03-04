import os
import glob
import shutil
import zipfile
import torch
import argparse
import cv2 as cv
from torchvision.utils import save_image
from .base_dataset import BaseDataset, Patchify


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def sort_by_last_number(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    last_num = stem.split("_")[-1]
    return int(last_num)


def patches_exist(processed_dir, split):
    a_dir = os.path.join(processed_dir, split, "A")
    return os.path.exists(a_dir) and len(os.listdir(a_dir)) > 0


# -------------------------------------------------
# Data Preparation
# -------------------------------------------------

def prepare_whu(zip_path, raw_dir, processed_dir,
                patch_size=(256, 256),
                verbose=False):

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    zip_name = os.path.basename(zip_path)
    unzip_dir = os.path.join(raw_dir, zip_name[:-4])

    if not os.path.exists(unzip_dir):
        print(f"Unzipping {zip_path} → {unzip_dir}")
        shutil.copy(zip_path, raw_dir)
        with zipfile.ZipFile(os.path.join(raw_dir, zip_name), 'r') as z:
            z.extractall(unzip_dir)
    else:
        print("Unzip folder exists. Skipping unzip.")

    patcher = Patchify(*patch_size)
    for split in ["train", "test"]:
        if patches_exist(processed_dir, split):
            print(f"{split} already patchified. Skipping.")
            continue

        print(f"Patchifying {split}...")

        for sub in ["A", "B", "label"]:
            os.makedirs(os.path.join(processed_dir, split, sub),
                        exist_ok=True)

        base_path = os.path.join(
            unzip_dir,
            os.path.splitext(os.path.basename(zip_path))[0],
            "Building change detection dataset_add",
            "1. The two-period image data"
        )
        image_dict = {
            "A": os.path.join(base_path,
                              f"2012/whole_image/{split}/image/2012_{split}.tif"),
            "B": os.path.join(base_path,
                              f"2016/whole_image/{split}/image/2016_{split}.tif"),
            "label": os.path.join(base_path,
                                  f"change_label/{split}/change_label.tif")
        }

        for sub, image_path in image_dict.items():
            if not os.path.exists(image_path):
                raise RuntimeError(f"Missing file: {image_path}")
            img = torch.from_numpy(cv.imread(image_path)).permute(2, 0, 1)
            patches = patcher(img)
            for i, patch in enumerate(patches):
                save_dest = os.path.join(
                    processed_dir,
                    split,
                    sub,
                    f"{sub}_{i}.png"
                )
                if verbose:
                    print(f"Saving {save_dest}")

                save_image(patch.float() / 255.0, save_dest)

    print("WHU preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=256)

    args = parser.parse_args()
    prepare_whu(
        zip_path=args.zip_path,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        patch_size=(args.patch_size, args.patch_size)
    )
