import pandas as pd

# File paths (update Artemis2 path as needed)
WIKIART_CSV = "wikiart_balanced_200.csv"
ARTEMIS2_CSV = "artemis-v2/dataset/combined/train/artemis_preprocessed.csv"  # <-- Change this to your Artemis2 CSV path
OUTPUT_CSV = "output/artemis2_subset_200.csv"

# Load the datasets
wikiart_df = pd.read_csv(WIKIART_CSV)
artemis2_df = pd.read_csv(ARTEMIS2_CSV)

# Use the 'image' column for matching
wikiart_images = set(wikiart_df['image'].unique())

# Print some debug information
print("Number of unique paintings in Artemis2:", len(artemis2_df['painting'].unique()))
print("Number of images in WikiArt sample set:", len(wikiart_images))


# Remove file extensions from WikiArt images for matching 
wikiart_images_clean = set(img.replace('.jpg', '').replace('.png', '') for img in wikiart_images)

# Filter Artemis2 for matching images (need to handle extensions in the filter)
subset_df = artemis2_df[artemis2_df['painting'].isin(wikiart_images_clean)].drop(columns=['tokens', 'tokens_encoded'])
print("Number of matches found:", len(subset_df))

import pdb; pdb.set_trace()
# Save the subset
subset_df.to_csv(OUTPUT_CSV, index=False)
print(f"Subset saved to {OUTPUT_CSV} with {len(subset_df)} rows.")