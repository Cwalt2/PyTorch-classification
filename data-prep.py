import os, shutil
import numpy as np
import kagglehub
from PIL import Image
from glob import glob
from pathlib import Path

# Download latest version from kaggle
dataset_path_str = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")

base_path = Path(dataset_path_str)

data_dir = base_path / "PetImages"

print(f"Correct data directory: {data_dir}")
output_dir = "data"  # where train/val folders will go

# Make dirs
for split in ["train", "val"]:
    for cls in ["cats", "dogs"]:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

split_ratio = 0.8

for cls in ["Cat", "Dog"]:
    images = os.listdir(os.path.join(data_dir, cls))
    images = [f for f in images if f.endswith(".jpg")]  # only jpg
    np.random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_imgs, val_imgs = images[:split_idx], images[split_idx:]

    # Move images
    for img in train_imgs:
        shutil.copy(
            os.path.join(data_dir, cls, img),
            os.path.join(output_dir, "train", cls.lower()+"s", img)
        )
    for img in val_imgs:
        shutil.copy(
            os.path.join(data_dir, cls, img),
            os.path.join(output_dir, "val", cls.lower()+"s", img)
        )

# clean data
print(f"Scanning for corrupted images in: {output_dir}")

# Get a list of all .jpg files recursively
image_files = glob(os.path.join(output_dir, '**', '*.jpg'), recursive=True)
removed_count = 0

for file_path in image_files:
    try:
        with Image.open(file_path) as img:
            img.verify()
    except (IOError, SyntaxError) as e:
        print(f"DELETING corrupted file: {file_path}")
        os.remove(file_path)
        removed_count += 1

print(f"\nScan complete. Removed {removed_count} corrupted images.")
