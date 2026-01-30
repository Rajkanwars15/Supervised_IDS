"""
Preprocessing script for CIC IDS 2017 binary classification dataset

This script:
- Combines all CSV files from MachineLearningCVE folder
- Cleans column names (removes leading spaces)
- Converts labels to binary (BENIGN=0, attacks=1)
- Splits into train/test sets
- Performs data validation and cleaning
- Saves cleaned data to data/cic-ids2017-silver/

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

# Set paths
data_dir = Path(__file__).parent.parent.parent.parent / "data" / "CIC IDS2017" / "MachineLearningCVE"
output_dir = Path(__file__).parent.parent.parent.parent / "data" / "cic-ids2017-silver"
output_dir.mkdir(parents=True, exist_ok=True)

output_train = output_dir / "CIC-IDS2017_Train_Binary.csv"
output_test = output_dir / "CIC-IDS2017_Test_Binary.csv"

print("="*80)
print("CIC IDS 2017 Binary Classification - Data Preprocessing")
print("="*80)

# Find all CSV files
csv_files = glob.glob(str(data_dir / "*.csv"))
print(f"\n1. Found {len(csv_files)} CSV files to process")

# Load and combine all files
print(f"\n2. Loading and combining CSV files...")
all_dataframes = []
all_labels = set()

for csv_file in tqdm(csv_files, desc="Loading files"):
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        all_dataframes.append(df)
        if 'Label' in df.columns:
            all_labels.update(df['Label'].unique())
    except Exception as e:
        print(f"  Warning: Error loading {csv_file}: {e}")
        continue

print(f"  Combined {len(all_dataframes)} files")
print(f"  Unique labels found: {sorted(all_labels)}")

# Combine all dataframes
print(f"\n3. Combining dataframes...")
combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f"  Combined shape: {combined_df.shape}")

# Validate target column
print(f"\n4. Validating target column...")
if 'Label' not in combined_df.columns:
    print("ERROR: 'Label' column not found!")
    sys.exit(1)

print(f"   Label distribution (before binary conversion):")
label_counts = combined_df['Label'].value_counts()
print(f"   {label_counts.to_dict()}")

# Convert to binary classification
# BENIGN = 0, everything else = 1
print(f"\n5. Converting to binary classification...")
combined_df['is_attack'] = (combined_df['Label'] != 'BENIGN').astype(int)
combined_df = combined_df.drop('Label', axis=1)

print(f"   Binary target distribution:")
print(f"     {combined_df['is_attack'].value_counts().to_dict()}")

# Identify feature columns (exclude target)
feature_cols = [col for col in combined_df.columns if col != 'is_attack']
print(f"\n6. Feature columns identified: {len(feature_cols)} features")

# Check for missing values
print(f"\n7. Checking for missing values...")
missing = combined_df[feature_cols].isnull().sum().sum()
print(f"   Missing values: {missing}")

if missing > 0:
    print("   Filling missing values with median...")
    for col in tqdm(feature_cols, desc="Filling missing values"):
        if combined_df[col].dtype in [np.float64, np.int64]:
            median_val = combined_df[col].median()
            combined_df[col].fillna(median_val, inplace=True)
        else:
            # For non-numeric, fill with mode or drop
            combined_df[col].fillna(combined_df[col].mode()[0] if len(combined_df[col].mode()) > 0 else 0, inplace=True)

# Check for infinite values
print(f"\n8. Checking for infinite values...")
numeric_cols = combined_df[feature_cols].select_dtypes(include=[np.number]).columns
inf_count = np.isinf(combined_df[numeric_cols]).sum().sum()
print(f"   Infinite values: {inf_count}")

if inf_count > 0:
    print("   Replacing infinite values...")
    combined_df[numeric_cols] = combined_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        median_val = combined_df[col].median()
        combined_df[col].fillna(median_val, inplace=True)

# Remove any remaining non-numeric columns that can't be used
print(f"\n9. Cleaning feature columns...")
usable_features = []
for col in feature_cols:
    if combined_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
        usable_features.append(col)
    else:
        # Try to convert to numeric
        try:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            if combined_df[col].notna().sum() > len(combined_df) * 0.5:  # If >50% valid
                usable_features.append(col)
        except:
            pass

print(f"   Usable numeric features: {len(usable_features)}")
feature_cols = usable_features

# Final fill for any remaining NaN
for col in feature_cols:
    if combined_df[col].isnull().sum() > 0:
        median_val = combined_df[col].median()
        combined_df[col].fillna(median_val, inplace=True)

# Split into train/test
print(f"\n10. Splitting into train/test sets (80/20)...")
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
print(f"\n11. Final validation...")
print(f"   Train shape: {train_df.shape}")
print(f"   Test shape: {test_df.shape}")
print(f"   Train missing: {train_df.isnull().sum().sum()}")
print(f"   Test missing: {test_df.isnull().sum().sum()}")

# Save preprocessed data
print(f"\n12. Saving preprocessed data...")
print(f"   Train output: {output_train}")
print(f"   Test output: {output_test}")

train_df.to_csv(output_train, index=False)
test_df.to_csv(output_test, index=False)

print(f"\n✓ Preprocessing complete!")
print(f"   Preprocessed data saved to: {output_dir}")
print(f"   Train samples: {len(train_df):,}")
print(f"   Test samples: {len(test_df):,}")
print(f"   Features: {len(feature_cols)}")
