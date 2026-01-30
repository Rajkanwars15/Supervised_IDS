"""
Preprocessing script for Farm-Flow binary classification dataset

This script:
- Loads the binary train/test CSV files
- Validates data quality
- Performs any necessary preprocessing
- Saves cleaned data to data/farm-flow-silver/

Requirements:
    pip install pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Set paths
data_dir = Path(__file__).parent.parent.parent.parent / "data"
input_train = data_dir / "farm-flow" / "Datasets" / "Farm-Flow_Train_Binary.csv"
input_test = data_dir / "farm-flow" / "Datasets" / "Farm-Flow_Test_Binary.csv"
output_dir = data_dir / "farm-flow-silver"
output_dir.mkdir(parents=True, exist_ok=True)

output_train = output_dir / "Farm-Flow_Train_Binary.csv"
output_test = output_dir / "Farm-Flow_Test_Binary.csv"

print("="*80)
print("Farm-Flow Binary Classification - Data Preprocessing")
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
if 'is_attack' not in train_df.columns or 'is_attack' not in test_df.columns:
    print("ERROR: 'is_attack' column not found!")
    sys.exit(1)

print(f"   Train target distribution:")
print(f"     {train_df['is_attack'].value_counts().to_dict()}")
print(f"   Test target distribution:")
print(f"     {test_df['is_attack'].value_counts().to_dict()}")

# Check for missing values
print(f"\n3. Checking for missing values...")
train_missing = train_df.isnull().sum().sum()
test_missing = test_df.isnull().sum().sum()
print(f"   Train missing values: {train_missing}")
print(f"   Test missing values: {test_missing}")

if train_missing > 0 or test_missing > 0:
    print("   WARNING: Missing values found. Filling with median...")
    # Fill missing values with median for numeric columns
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'is_attack':
            median_val = train_df[col].median()
            train_df[col].fillna(median_val, inplace=True)
            test_df[col].fillna(median_val, inplace=True)

# Ensure target is binary (0/1)
print(f"\n4. Ensuring binary target...")
train_df['is_attack'] = train_df['is_attack'].astype(int)
test_df['is_attack'] = test_df['is_attack'].astype(int)

print(f"   Train target unique values: {sorted(train_df['is_attack'].unique())}")
print(f"   Test target unique values: {sorted(test_df['is_attack'].unique())}")

# Identify feature columns (exclude target)
feature_cols = [col for col in train_df.columns if col != 'is_attack']
print(f"\n5. Feature columns identified: {len(feature_cols)} features")

# Check for infinite values
print(f"\n6. Checking for infinite values...")
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
print(f"\n7. Final validation...")
print(f"   Train shape: {train_df.shape}")
print(f"   Test shape: {test_df.shape}")
print(f"   Train missing: {train_df.isnull().sum().sum()}")
print(f"   Test missing: {test_df.isnull().sum().sum()}")
print(f"   Train target distribution:")
print(f"     {train_df['is_attack'].value_counts().to_dict()}")
print(f"   Test target distribution:")
print(f"     {test_df['is_attack'].value_counts().to_dict()}")

# Save preprocessed data
print(f"\n8. Saving preprocessed data...")
print(f"   Train output: {output_train}")
print(f"   Test output: {output_test}")

train_df.to_csv(output_train, index=False)
test_df.to_csv(output_test, index=False)

print(f"\n✓ Preprocessing complete!")
print(f"   Preprocessed data saved to: {output_dir}")
print(f"   Train samples: {len(train_df):,}")
print(f"   Test samples: {len(test_df):,}")
print(f"   Features: {len(feature_cols)}")
