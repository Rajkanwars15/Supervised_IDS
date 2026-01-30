"""
Preprocessing script for NSL-KDD binary classification dataset

This script:
- Loads the train/test TXT files (KDDTrain+.txt and KDDTest+.txt)
- Converts attack-type labels to binary (normal=0, attacks=1)
- Handles categorical features
- Performs data validation and cleaning
- Saves cleaned data to data/nsl-kdd-silver/

Requirements:
    pip install pandas numpy scikit-learn tqdm
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Set paths
data_dir = Path(__file__).parent.parent.parent.parent / "data" / "NSL-KDD-Dataset"
input_train = data_dir / "KDDTrain+.txt"
input_test = data_dir / "KDDTest+.txt"
output_dir = Path(__file__).parent.parent.parent.parent / "data" / "nsl-kdd-silver"
output_dir.mkdir(parents=True, exist_ok=True)

output_train = output_dir / "NSL-KDD_Train_Binary.csv"
output_test = output_dir / "NSL-KDD_Test_Binary.csv"

print("="*80)
print("NSL-KDD Binary Classification - Data Preprocessing")
print("="*80)

# Load data (no header in TXT files)
print(f"\n1. Loading data...")
print(f"   Train file: {input_train}")
print(f"   Test file: {input_test}")

print(f"   Loading training data (this may take a while)...")
train_df = pd.read_csv(input_train, header=None, low_memory=False)
print(f"   Training data shape: {train_df.shape}")

print(f"   Loading test data...")
test_df = pd.read_csv(input_test, header=None, low_memory=False)
print(f"   Test data shape: {test_df.shape}")

# NSL-KDD has 43 columns: 41 features + 1 attack-type label + 1 difficulty level
# The last column (index 42) is difficulty level
# The second-to-last column (index 41) is the attack-type label
print(f"\n2. Processing labels...")
print(f"   Total columns: {train_df.shape[1]}")

# Extract labels (second-to-last column, index 41)
train_labels = train_df.iloc[:, 41].copy()
test_labels = test_df.iloc[:, 41].copy()

print(f"   Training label distribution (before binary conversion):")
print(f"     {train_labels.value_counts().to_dict()}")
print(f"   Test label distribution (before binary conversion):")
print(f"     {test_labels.value_counts().to_dict()}")

# Convert to binary: normal=0, everything else=1
print(f"\n3. Converting to binary classification...")
train_df['is_attack'] = (train_labels != 'normal').astype(int)
test_df['is_attack'] = (test_labels != 'normal').astype(int)

print(f"   Binary target distribution (training):")
print(f"     {train_df['is_attack'].value_counts().to_dict()}")
print(f"   Binary target distribution (test):")
print(f"     {test_df['is_attack'].value_counts().to_dict()}")

# Drop the label columns (index 41 and 42)
print(f"\n4. Dropping label columns...")
train_df = train_df.drop(columns=[41, 42])
test_df = test_df.drop(columns=[41, 42])

print(f"   Training shape after dropping labels: {train_df.shape}")
print(f"   Test shape after dropping labels: {test_df.shape}")

# Identify feature columns (all columns except is_attack)
feature_cols = [col for col in train_df.columns if col != 'is_attack']
print(f"\n5. Feature columns identified: {len(feature_cols)} features")

# Convert categorical columns to numeric
print(f"\n6. Converting features to numeric...")
usable_features = []

for col in tqdm(feature_cols, desc="Processing features"):
    # Check if column is already numeric
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
            # If conversion fails, try label encoding for categorical
            try:
                le = LabelEncoder()
                # Fit on combined train+test to handle all categories
                combined = pd.concat([train_df[col], test_df[col]], axis=0)
                le.fit(combined.astype(str))
                train_df[col] = le.transform(train_df[col].astype(str))
                test_df[col] = le.transform(test_df[col].astype(str))
                usable_features.append(col)
            except:
                pass

print(f"   Usable numeric features: {len(usable_features)}")
feature_cols = usable_features

# Check for missing values
print(f"\n7. Checking for missing values...")
train_missing = train_df[feature_cols].isnull().sum().sum()
test_missing = test_df[feature_cols].isnull().sum().sum()
print(f"   Train missing values: {train_missing}")
print(f"   Test missing values: {test_missing}")

if train_missing > 0 or test_missing > 0:
    print("   WARNING: Missing values found. Filling with median...")
    for col in tqdm(feature_cols, desc="Filling missing values"):
        median_val = train_df[col].median()
        if pd.isna(median_val):
            median_val = 0  # Fallback to 0 if all NaN
        train_df[col].fillna(median_val, inplace=True)
        test_df[col].fillna(median_val, inplace=True)

# Check for infinite values
print(f"\n8. Checking for infinite values...")
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
        if pd.isna(median_val):
            median_val = 0
        train_df[col].fillna(median_val, inplace=True)
        test_df[col].fillna(median_val, inplace=True)

# Final validation
print(f"\n9. Final validation...")
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

print(f"\n10. Saving preprocessed data...")
print(f"   Train output: {output_train}")
print(f"   Test output: {output_test}")

output_train_df.to_csv(output_train, index=False)
output_test_df.to_csv(output_test, index=False)

print(f"\n✓ Preprocessing complete!")
print(f"   Preprocessed data saved to: {output_dir}")
print(f"   Train samples: {len(output_train_df):,}")
print(f"   Test samples: {len(output_test_df):,}")
print(f"   Features: {len(feature_cols)}")
