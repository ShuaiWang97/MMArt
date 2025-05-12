import os
import pandas as pd
import shutil

# CONFIGURE THESE PATHS
CSV_PATH = "wikiart_balanced_200.csv"
SRC_IMAGE_ROOT = "../data/wikiart/Images"  # Change to your actual source images root
DST_IMAGE_ROOT = "copied_images"        # Change to your desired destination

# Read CSV
df = pd.read_csv(CSV_PATH)

for idx, row in df.iterrows():
    rel_path = row['relative_path']
    src_path = os.path.join(SRC_IMAGE_ROOT, rel_path)
    dst_path = os.path.join(DST_IMAGE_ROOT, rel_path)

    # Make sure destination directory exists
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src_path} -> {dst_path}")
    else:
        print(f"Source not found: {src_path}")

print("Done copying images.")