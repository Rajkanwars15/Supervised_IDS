"""
Preprocessing script for UNSW-NB15 binary classification dataset

This script:
- Loads the train/test CSV files
- Validates data quality
- Performs any necessary preprocessing
- Saves cleaned data to data/unsw-nb15-silver/

Requirements:
    pip install pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Set paths
data_dir = Path(__file__).parent.parent.parent.parent / "data" / "UNSW-NB15" / "Training and Testing Sets"
input_train = data_dir / "UNSW_NB15_training-set.csv"
input_test = data_dir / "UNSW_NB15_testing-set.csv"
output_dir = Path(__file__).parent.parent.parent.parent / "data" / "unsw-nb15-silver"
output_dir.mkdir(parents=True, exist_ok=True)

output_train = output_dir / "UNSW-NB15_Train_Binary.csv"
output_test = output_dir / "UNSW-NB15_Test_Binary.csv"

print("="*80)
print("UNSW-NB15 Binary Classification - Data Preprocessing")
print("="*80)

# Load data
print(f"\n1. Loading data...")
print(f"   Train file: {input_train}")
print(f"   Test file: {input_test}")

train_df = pd.read_csv(input_train)
test_df = pd.read_csv(input_test)

print(f"   Train shape: {train_df.shape}")
print(f"   Test shape: {test_df.shape}")

# Validate target column
print(f"\n2. Validating target column...")
if 'label' not in train_df.columns or 'label' not in test_df.columns:
    print("ERROR: 'label' column not found!")
    sys.exit(1)

print(f"   Train target distribution:")
print(f"     {train_df['label'].value_counts().to_dict()}")
print(f"   Test target distribution:")
print(f"     {test_df['label'].value_counts().to_dict()}")

# Rename label to is_attack for consistency
train_df = train_df.rename(columns={'label': 'is_attack'})
test_df = test_df.rename(columns={'label': 'is_attack'})

# Ensure target is binary (0/1)
print(f"\n3. Ensuring binary target...")
train_df['is_attack'] = train_df['is_attack'].astype(int)
test_df['is_attack'] = test_df['is_attack'].astype(int)

print(f"   Train target unique values: {sorted(train_df['is_attack'].unique())}")
print(f"   Test target unique values: {sorted(test_df['is_attack'].unique())}")

# Identify feature columns (exclude target and non-numeric columns)
print(f"\n4. Identifying feature columns...")
exclude_cols = ['id', 'is_attack', 'attack_cat']  # Exclude ID, target, and attack category
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

# Convert categorical columns to numeric if possible
print(f"\n5. Converting features to numeric...")
usable_features = []
for col in feature_cols:
    if train_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
        usable_features.append(col)
    else:
        # Try to convert to numeric
        try:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
            if train_df[col].notna().sum() > len(train_df) * 0.5:  # If >50% valid
                usable_features.append(col)
        except:
            pass

print(f"   Usable numeric features: {len(usable_features)}")
feature_cols = usable_features

# Check for missing values
print(f"\n6. Checking for missing values...")
train_missing = train_df[feature_cols].isnull().sum().sum()
test_missing = test_df[feature_cols].isnull().sum().sum()
print(f"   Train missing values: {train_missing}")
print(f"   Test missing values: {test_missing}")

if train_missing > 0 or test_missing > 0:
    print("   WARNING: Missing values found. Filling with median...")
    # Fill missing values with median for numeric columns
    for col in tqdm(feature_cols, desc="Filling missing values"):
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)
        test_df[col].fillna(median_val, inplace=True)

# Check for infinite values
print(f"\n7. Checking for infinite values...")
train_inf = np.isinf(train_df[feature_cols]).sum().sum()
test_inf = np.isinf(test_df[feature_cols]).sum().sum()
print(f"   Train infinite values: {train_inf}")
print(f"   Test infinite values: {test_inf}")

if train_inf > 0 or test_inf > 0:
    print("   WARNING: Infinite values found. Replacing with NaN then median...")
    train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)
        test_df[col].fillna(median_val, inplace=True)

# Final validation
print(f"\n8. Final validation...")
print(f"   Train shape: {train_df.shape}")
print(f"   Test shape: {test_df.shape}")
print(f"   Train missing: {train_df[feature_cols].isnull().sum().sum()}")
print(f"   Test missing: {test_df[feature_cols].isnull().sum().sum()}")
print(f"   Train target distribution:")
print(f"     {train_df['is_attack'].value_counts().to_dict()}")
print(f"   Test target distribution:")
print(f"     {test_df['is_attack'].value_counts().to_dict()}")

# Save preprocessed data (only keep feature columns and target)
output_train_df = train_df[feature_cols + ['is_attack']].copy()
output_test_df = test_df[feature_cols + ['is_attack']].copy()

print(f"\n9. Saving preprocessed data...")
print(f"   Train output: {output_train}")
print(f"   Test output: {output_test}")

output_train_df.to_csv(output_train, index=False)
output_test_df.to_csv(output_test, index=False)

print(f"\n✓ Preprocessing complete!")
print(f"   Preprocessed data saved to: {output_dir}")
print(f"   Train samples: {len(output_train_df):,}")
print(f"   Test samples: {len(output_test_df):,}")
print(f"   Features: {len(feature_cols)}")
