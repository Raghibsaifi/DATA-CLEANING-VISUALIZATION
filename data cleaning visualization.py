
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Load the original dataset
input_file = Path("raw_dataset.csv")

if not input_file.exists():
    raise FileNotFoundError(
        "Please upload your dataset as raw_dataset.csv"
    )

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Check data quality
print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Fill missing numeric values using column medians
numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Fill missing text values using the most frequent value
text_cols = df.select_dtypes(
    include=["object", "string"]
).columns

for col in text_cols:
    if df[col].isnull().any():
        mode_values = df[col].mode()
        if not mode_values.empty:
            df[col] = df[col].fillna(mode_values.iloc[0])

# Save cleaned data
df.to_csv("cleaned_dataset.csv", index=False)

# Create a missing-values chart
missing = df.isnull().sum()
missing = missing[missing > 0]

plt.figure(figsize=(10, 5))

if not missing.empty:
    missing.plot(kind="bar")
    plt.ylabel("Count")
else:
    plt.text(
        0.5, 0.5,
        "No missing values remain",
        ha="center", va="center"
    )

plt.title("Missing Values After Cleaning")
plt.tight_layout()
plt.savefig("missing_values.png")
plt.close()

# Create distributions for numeric columns
if len(numeric_cols) > 0:
    df[numeric_cols].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Numeric Data Distributions")
    plt.tight_layout()
    plt.savefig("numeric_distributions.png")
    plt.close()

print("\nCleaned dataset shape:", df.shape)
print("Cleaning workflow finished.")
