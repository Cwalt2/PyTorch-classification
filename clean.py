import os
from PIL import Image
from glob import glob

# --- CONFIGURATION ---
# Set this to the base directory containing your 'train' and 'test' folders
data_dir = "data" 
# ---------------------

print(f"Scanning for corrupted images in: {data_dir}")

# Get a list of all .jpg files recursively
image_files = glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)
removed_count = 0

for file_path in image_files:
    try:
        # Try to open the image file
        with Image.open(file_path) as img:
            img.verify() # Verify that it is, in fact, an image
    except (IOError, SyntaxError) as e:
        # If it fails, print the path and delete the file
        print(f"DELETING corrupted file: {file_path}")
        os.remove(file_path)
        removed_count += 1

print(f"\nScan complete. Removed {removed_count} corrupted images.")
