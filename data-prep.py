import os, shutil
import numpy as np
import kagglehub

# Download latest version
path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")

data_dir = path
output_dir = "data"  # where your train/val folders will go

# Make dirs
for split in ["train", "val"]:
    for cls in ["cats", "dogs"]:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Split ratio
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
