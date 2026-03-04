import os
import glob
import zipfile
import torch
import argparse
import cv2 as cv
from torchvision.utils import save_image
from ChangeMambaVision.datasets.base_dataset import Patchify

def unzip_raw(raw_dir, unzip_dir):
    zip_files = glob.glob(os.path.join(raw_dir, "*.zip"))
    if not zip_files: 
        raise RuntimeError(f"Zip files not found in {raw_dir}")

    for zfile in zip_files:
        split_name = os.path.splitext(os.path.basename(zfile))[0]
        split_dest = os.path.join(unzip_dir, split_name)
        if os.path.exists(split_dest):
            print(f"{split_name} already unzipped. Skipping.")
            continue
        print(f"Unzipping {zfile} → {split_dest}")
        os.makedirs(split_dest, exist_ok=True)

        with zipfile.ZipFile(zfile, 'r') as z:
            z.extractall(split_dest)

def patchify_dataset(unzip_dir, processed_dir, patch_size):
    patcher = Patchify(*patch_size)
    for split in ["train", "val", "test"]:
        split_src = os.path.join(unzip_dir, split)
        split_dest = os.path.join(processed_dir, split)
        if os.path.exists(os.path.join(split_dest, "A")) and \
           len(os.listdir(os.path.join(split_dest, "A"))) > 0:
            print(f"{split} split already patchified. Skipping.")
            continue
        print(f"Patchifying {split}")
        for sub in ["A", "B", "label"]:
            os.makedirs(os.path.join(split_dest, sub), exist_ok=True)

            for img_path in glob.glob(
                os.path.join(split_src, sub, "*.png")
            ):
                img = torch.from_numpy(cv.imread(img_path)).permute(2, 0, 1)
                patches = patcher(img)

                for i, patch in enumerate(patches):
                    save_path = os.path.join(
                        split_dest,
                        sub,
                        f"{os.path.splitext(os.path.basename(img_path))[0]}_{i}.png"
                    )
                    save_image(patch.float()/255.0, save_path)

def prepare_levir(raw_dir, processed_dir, patch_size=(256, 256)):
    unzip_dir = os.path.join(processed_dir, "unzipped")

    unzip_raw(raw_dir, unzip_dir)
    patchify_dataset(unzip_dir, processed_dir, patch_size)

    print(f"LEVIR-CD data preparation complete, patched dataset is at {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=256)

    args = parser.parse_args()
    prepare_levir(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        patch_size=(args.patch_size, args.patch_size)
    )
