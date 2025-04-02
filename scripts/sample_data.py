import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../data/wikiart/wikiart_full.csv')

# Get the number of unique styles
num_styles = len(df['style_classification'].unique())
print(f"Number of styles: {num_styles}, {df['style_classification'].unique()}")
# Calculate how many samples we want per style to get 200 total
samples_per_style = 200 // num_styles

# Make sure we'll get exactly 200 samples
remaining_samples = 200 - (samples_per_style * num_styles)

# Sample from each style
balanced_df = pd.DataFrame()
for idx, style in enumerate(df['style_classification'].unique()):
    style_df = df[df['style_classification'] == style]
    # Add one extra sample to some styles if we need to reach 200 exactly
    current_samples = samples_per_style + (1 if idx < remaining_samples else 0)
    # Make sure we don't try to sample more than available
    current_samples = min(current_samples, len(style_df))
    sampled_style_df = style_df.sample(n=current_samples, random_state=42)
    balanced_df = pd.concat([balanced_df, sampled_style_df])

# Shuffle the final dataset
balanced_df = balanced_df.sample(frac=1, random_state=42)

# Save the balanced dataset
balanced_df.to_csv('wikiart_balanced_200.csv', index=False)

print(f"Number of styles: {num_styles}")
print(f"Base samples per style: {samples_per_style}")
print(f"Extra samples distributed: {remaining_samples}")
print(f"Total samples: {len(balanced_df)}")
print("\nSamples per style:")
print(balanced_df['style_classification'].value_counts())