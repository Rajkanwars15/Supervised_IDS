"""
Preprocessing script for Kyoto 2006 binary classification dataset

This script:
- Combines all daily TXT files from Kyoto folder
- Converts labels to binary (0=normal, 1 or -1=attack)
- Handles categorical features
- Performs data validation and cleaning
- Splits into train/test sets
- Saves cleaned data to data/kyoto-silver/

Requirements:
    pip install pandas numpy scikit-learn tqdm
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set paths
data_dir = Path(__file__).parent.parent.parent.parent / "data" / "Kyoto"
output_dir = Path(__file__).parent.parent.parent.parent / "data" / "kyoto-silver"
output_dir.mkdir(parents=True, exist_ok=True)

output_train = output_dir / "Kyoto_Train_Binary.csv"
output_test = output_dir / "Kyoto_Test_Binary.csv"

print("="*80)
print("Kyoto 2006 Binary Classification - Data Preprocessing")
print("="*80)

# Find all TXT files
txt_files = sorted(glob.glob(str(data_dir / "*.txt")))
print(f"\n1. Found {len(txt_files)} TXT files to process")

# Load and combine all files
print(f"\n2. Loading and combining TXT files...")
all_dataframes = []

for txt_file in tqdm(txt_files, desc="Loading files"):
    try:
        df = pd.read_csv(txt_file, sep='\t', header=None, low_memory=False)
        all_dataframes.append(df)
    except Exception as e:
        print(f"  Warning: Error loading {txt_file}: {e}")
        continue

print(f"  Combined {len(all_dataframes)} files")

# Combine all dataframes
print(f"\n3. Combining dataframes...")
combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f"  Combined shape: {combined_df.shape}")

# Kyoto 2006 has 24 columns
# Column 17 (index 17) is the label: 0=normal, 1=attack, -1=attack
print(f"\n4. Processing labels...")
print(f"   Total columns: {combined_df.shape[1]}")

# Extract labels (column 17, index 17)
labels = combined_df.iloc[:, 17].copy()

print(f"   Label distribution (before binary conversion):")
label_counts = labels.value_counts()
print(f"   {label_counts.to_dict()}")

# Convert to binary classification
# 0 = normal, 1 or -1 = attack
print(f"\n5. Converting to binary classification...")
combined_df['is_attack'] = (labels != 0).astype(int)

print(f"   Binary target distribution:")
print(f"     {combined_df['is_attack'].value_counts().to_dict()}")

# Drop the original label column (column 17)
combined_df = combined_df.drop(columns=[17])

# Identify feature columns (all columns except is_attack)
feature_cols = [col for col in combined_df.columns if col != 'is_attack']
print(f"\n6. Feature columns identified: {len(feature_cols)} features")

# Convert categorical columns to numeric
print(f"\n7. Converting features to numeric...")
usable_features = []
label_encoders = {}

for col in tqdm(feature_cols, desc="Processing features"):
    # Check if column is already numeric
    if combined_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
        usable_features.append(col)
    else:
        # Try to convert to numeric
        try:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            if combined_df[col].notna().sum() > len(combined_df) * 0.5:  # If >50% valid
                usable_features.append(col)
        except:
            # If conversion fails, try label encoding for categorical
            try:
                le = LabelEncoder()
                combined_df[col] = le.fit_transform(combined_df[col].astype(str))
                label_encoders[col] = le
                usable_features.append(col)
            except:
                pass

print(f"   Usable numeric features: {len(usable_features)}")
feature_cols = usable_features

# Check for missing values
print(f"\n8. Checking for missing values...")
missing = combined_df[feature_cols].isnull().sum().sum()
print(f"   Missing values: {missing}")

if missing > 0:
    print("   Filling missing values with median...")
    for col in tqdm(feature_cols, desc="Filling missing values"):
        if combined_df[col].dtype in [np.float64, np.int64]:
            median_val = combined_df[col].median()
            if pd.isna(median_val):
                median_val = 0
            combined_df[col].fillna(median_val, inplace=True)
        else:
            combined_df[col].fillna(0, inplace=True)

# Check for infinite values
print(f"\n9. Checking for infinite values...")
numeric_cols = combined_df[feature_cols].select_dtypes(include=[np.number]).columns
inf_count = np.isinf(combined_df[numeric_cols]).sum().sum()
print(f"   Infinite values: {inf_count}")

if inf_count > 0:
    print("   Replacing infinite values...")
    combined_df[numeric_cols] = combined_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        median_val = combined_df[col].median()
        if pd.isna(median_val):
            median_val = 0
        combined_df[col].fillna(median_val, inplace=True)

# Final feature cleaning
print(f"\n10. Final feature cleaning...")
final_features = []
for col in feature_cols:
    if combined_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
        final_features.append(col)
    else:
        try:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            if combined_df[col].notna().sum() > len(combined_df) * 0.5:
                final_features.append(col)
        except:
            pass

print(f"   Final usable features: {len(final_features)}")
feature_cols = final_features

# Final fill for any remaining NaN
for col in feature_cols:
    if combined_df[col].isnull().sum() > 0:
        median_val = combined_df[col].median()
        if pd.isna(median_val):
            median_val = 0
        combined_df[col].fillna(median_val, inplace=True)

# Split into train/test
print(f"\n11. Splitting into train/test sets (80/20)...")
X = combined_df[feature_cols]
y = combined_df['is_attack']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print(f"   Training set: {len(train_df):,} samples")
print(f"   Test set: {len(test_df):,} samples")
print(f"   Training class distribution: {y_train.value_counts().to_dict()}")
print(f"   Test class distribution: {y_test.value_counts().to_dict()}")

# Final validation
print(f"\n12. Final validation...")
print(f"   Train shape: {train_df.shape}")
print(f"   Test shape: {test_df.shape}")
print(f"   Train missing: {train_df.isnull().sum().sum()}")
print(f"   Test missing: {test_df.isnull().sum().sum()}")

# Save preprocessed data
print(f"\n13. Saving preprocessed data...")
print(f"   Train output: {output_train}")
print(f"   Test output: {output_test}")

train_df.to_csv(output_train, index=False)
test_df.to_csv(output_test, index=False)

print(f"\n✓ Preprocessing complete!")
print(f"   Preprocessed data saved to: {output_dir}")
print(f"   Train samples: {len(train_df):,}")
print(f"   Test samples: {len(test_df):,}")
print(f"   Features: {len(feature_cols)}")
