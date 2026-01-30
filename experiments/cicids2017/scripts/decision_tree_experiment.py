"""
Decision Tree Experiment for CIC IDS 2017 Binary Classification Dataset

This script trains a decision tree classifier on the CIC IDS 2017 dataset and logs:
- All splits, thresholds, and feature selections during tree building
- Comprehensive ML metrics (accuracy, precision, recall, F1, confusion matrix, ROC-AUC)
- Experiment logs and results saved separately

Requirements:
    pip install pandas scikit-learn numpy tqdm
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json
from tqdm import tqdm

# Check for required packages
try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        average_precision_score
    )
except ImportError:
    print("Error: scikit-learn is not installed. Please install it with: pip install scikit-learn")
    sys.exit(1)

# Set paths
data_dir = Path(__file__).parent.parent.parent.parent / "data" / "cic-ids2017-silver"
train_path = data_dir / "CIC-IDS2017_Train_Binary.csv"
test_path = data_dir / "CIC-IDS2017_Test_Binary.csv"

# Check if silver data exists
if not train_path.exists() or not test_path.exists():
    print("ERROR: Preprocessed data not found!")
    print(f"Please run preprocessing script first:")
    print(f"  python experiments/cicids2017/scripts/preprocess_cicids2017.py")
    sys.exit(1)

experiment_dir = Path(__file__).parent.parent / "decision_tree_experiment"
experiment_dir.mkdir(parents=True, exist_ok=True)
logs_dir = experiment_dir / "logs"
results_dir = experiment_dir / "results"
logs_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Create timestamp for this experiment run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_log_file = logs_dir / f"experiment_log_{timestamp}.txt"
results_file = results_dir / f"results_{timestamp}.json"
tree_text_file = logs_dir / f"tree_structure_{timestamp}.txt"

# Initialize logging
log_buffer = []

def log(message, print_to_console=True):
    """Log message to both console and log file"""
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp_str}] {message}"
    log_buffer.append(log_entry)
    if print_to_console:
        print(log_entry)

# Read data
log(f"Reading data...")
log(f"  Train file: {train_path}")
log(f"  Test file: {test_path}")

log(f"Loading training data (this may take a while for large files)...")
train_df = pd.read_csv(train_path)
log(f"  Training data shape: {train_df.shape}")

log(f"Loading test data...")
test_df = pd.read_csv(test_path)
log(f"  Test data shape: {test_df.shape}")

# Prepare features and target
if 'is_attack' not in train_df.columns:
    log("ERROR: 'is_attack' column not found in training data!")
    sys.exit(1)

feature_cols = [col for col in train_df.columns if col != 'is_attack']
log(f"Found {len(feature_cols)} feature columns")

X_train = train_df[feature_cols].copy()
y_train = train_df['is_attack'].copy().astype(int)
X_test = test_df[feature_cols].copy()
y_test = test_df['is_attack'].copy().astype(int)

# Handle missing values
log(f"\nHandling missing values...")
missing_train = X_train.isnull().sum().sum()
missing_test = X_test.isnull().sum().sum()
log(f"  Training missing values: {missing_train}")
log(f"  Test missing values: {missing_test}")

if missing_train > 0 or missing_test > 0:
    log("  Filling missing values with median...")
    for col in tqdm(feature_cols, desc="Filling missing"):
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

# Check for infinite values
log(f"\nChecking for infinite values...")
train_inf = np.isinf(X_train[feature_cols]).sum().sum()
test_inf = np.isinf(X_test[feature_cols]).sum().sum()
log(f"  Training infinite values: {train_inf}")
log(f"  Test infinite values: {test_inf}")

if train_inf > 0 or test_inf > 0:
    log("  Replacing infinite values...")
    X_train[feature_cols] = X_train[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test[feature_cols] = X_test[feature_cols].replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

log(f"\nData Summary:")
log(f"  Training set: {len(X_train):,} samples")
log(f"  Test set: {len(X_test):,} samples")
log(f"  Training class distribution: {y_train.value_counts().to_dict()}")
log(f"  Test class distribution: {y_test.value_counts().to_dict()}")

# Train Decision Tree with detailed logging
log(f"\n{'='*80}")
log("Training Decision Tree Classifier...")
log(f"{'='*80}")

# Use parameters that allow the tree to grow and converge
tree_params = {
    'criterion': 'gini',
    'splitter': 'best',  # Try all splits to find best
    'max_depth': None,  # No depth limit - let it converge
    'min_samples_split': 2,  # Minimum samples to split
    'min_samples_leaf': 1,  # Minimum samples in leaf
    'min_weight_fraction_leaf': 0.0,
    'max_features': None,  # Consider all features
    'random_state': 42,
    'max_leaf_nodes': None,  # No limit on leaf nodes
    'min_impurity_decrease': 0.0,  # No minimum impurity decrease
    'class_weight': None,
    'ccp_alpha': 0.0  # No cost complexity pruning
}

log(f"Tree Parameters:")
for key, value in tree_params.items():
    log(f"  {key}: {value}")

# Train the model
log(f"\nTraining tree (this may take a while for full convergence)...")
clf = DecisionTreeClassifier(**tree_params)
clf.fit(X_train, y_train)

log(f"✓ Tree training completed!")
log(f"  Tree depth: {clf.tree_.max_depth}")
log(f"  Number of nodes: {clf.tree_.node_count}")
log(f"  Number of leaves: {clf.tree_.n_leaves}")

# Extract and log tree structure
log(f"\n{'='*80}")
log("Extracting tree structure and splits...")
log(f"{'='*80}")

# Get tree structure using sklearn's export_text
tree_rules = export_text(clf, feature_names=feature_cols, max_depth=100)
log(f"\nTree Structure (first 100 lines):")
log(tree_rules[:5000] if len(tree_rules) > 5000 else tree_rules)

# Save full tree structure to file
with open(tree_text_file, 'w') as f:
    f.write(f"Decision Tree Structure - CIC IDS 2017 Experiment {timestamp}\n")
    f.write("="*80 + "\n\n")
    f.write(tree_rules)

log(f"\nFull tree structure saved to: {tree_text_file}")

# Log detailed split information
log(f"\n{'='*80}")
log("Detailed Split Information:")
log(f"{'='*80}")

def log_tree_splits(tree, feature_names, node_id=0, depth=0, prefix=""):
    """Recursively log all splits in the tree"""
    if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf node
        samples = tree.n_node_samples[node_id]
        values = tree.value[node_id][0]
        class_pred = np.argmax(values)
        log(f"{prefix}LEAF: samples={samples}, class={class_pred}, values={values}")
        return
    
    # Internal node - log split
    feature_idx = tree.feature[node_id]
    threshold = tree.threshold[node_id]
    feature_name = feature_names[feature_idx]
    samples = tree.n_node_samples[node_id]
    impurity = tree.impurity[node_id]
    
    log(f"{prefix}SPLIT: feature='{feature_name}', threshold={threshold:.6f}, "
        f"samples={samples}, impurity={impurity:.6f}")
    
    # Recurse to children (limit depth for logging to avoid overwhelming output)
    if depth < 10:  # Only log first 10 levels in detail
        log_tree_splits(tree, feature_names, tree.children_left[node_id], depth+1, prefix + "  ")
        log_tree_splits(tree, feature_names, tree.children_right[node_id], depth+1, prefix + "  ")

log_tree_splits(clf.tree_, feature_cols)
log(f"\n(Note: Only first 10 levels shown in detail. Full tree structure saved to file.)")

# Feature importance
log(f"\n{'='*80}")
log("Feature Importance (Top 20):")
log(f"{'='*80}")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(20).iterrows():
    log(f"  {row['feature']}: {row['importance']:.6f}")

# Make predictions
log(f"\n{'='*80}")
log("Making predictions...")
log(f"{'='*80}")

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
y_train_proba = clf.predict_proba(X_train)[:, 1]
y_test_proba = clf.predict_proba(X_test)[:, 1]

# Calculate metrics
log(f"\n{'='*80}")
log("Calculating Metrics...")
log(f"{'='*80}")

# Training metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, zero_division=0)
train_recall = recall_score(y_train, y_train_pred, zero_division=0)
train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
train_roc_auc = roc_auc_score(y_train, y_train_proba) if len(np.unique(y_train)) > 1 else 0.0
train_avg_precision = average_precision_score(y_train, y_train_proba) if len(np.unique(y_train)) > 1 else 0.0
train_cm = confusion_matrix(y_train, y_train_pred)

log(f"\nTraining Set Metrics:")
log(f"  Accuracy:  {train_accuracy:.6f}")
log(f"  Precision: {train_precision:.6f}")
log(f"  Recall:    {train_recall:.6f}")
log(f"  F1 Score:  {train_f1:.6f}")
log(f"  ROC-AUC:   {train_roc_auc:.6f}")
log(f"  Avg Precision: {train_avg_precision:.6f}")
log(f"  Confusion Matrix:")
log(f"    {train_cm}")

# Test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_roc_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.0
test_avg_precision = average_precision_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.0
test_cm = confusion_matrix(y_test, y_test_pred)

log(f"\nTest Set Metrics:")
log(f"  Accuracy:  {test_accuracy:.6f}")
log(f"  Precision: {test_precision:.6f}")
log(f"  Recall:    {test_recall:.6f}")
log(f"  F1 Score:  {test_f1:.6f}")
log(f"  ROC-AUC:   {test_roc_auc:.6f}")
log(f"  Avg Precision: {test_avg_precision:.6f}")
log(f"  Confusion Matrix:")
log(f"    {test_cm}")

# Classification report
log(f"\nClassification Report (Test Set):")
log(classification_report(y_test, y_test_pred, target_names=['Benign', 'Attack']))

# Save results to JSON
results = {
    'experiment_timestamp': timestamp,
    'dataset': 'CIC IDS 2017 Binary Classification',
    'tree_parameters': tree_params,
    'tree_properties': {
        'max_depth': int(clf.tree_.max_depth),
        'n_nodes': int(clf.tree_.node_count),
        'n_leaves': int(clf.tree_.n_leaves)
    },
    'feature_importance': feature_importance.to_dict('records'),
    'training_metrics': {
        'accuracy': float(train_accuracy),
        'precision': float(train_precision),
        'recall': float(train_recall),
        'f1_score': float(train_f1),
        'roc_auc': float(train_roc_auc),
        'average_precision': float(train_avg_precision),
        'confusion_matrix': train_cm.tolist()
    },
    'test_metrics': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1),
        'roc_auc': float(test_roc_auc),
        'average_precision': float(test_avg_precision),
        'confusion_matrix': test_cm.tolist()
    },
    'data_info': {
        'n_train_samples': int(len(X_train)),
        'n_test_samples': int(len(X_test)),
        'n_features': int(len(feature_cols)),
        'train_class_distribution': y_train.value_counts().to_dict(),
        'test_class_distribution': y_test.value_counts().to_dict()
    }
}

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

log(f"\n✓ Results saved to: {results_file}")

# Save experiment log
with open(experiment_log_file, 'w') as f:
    f.write("\n".join(log_buffer))

log(f"✓ Experiment log saved to: {experiment_log_file}")

log(f"\n{'='*80}")
log("Experiment completed successfully!")
log(f"{'='*80}")
log(f"Summary:")
log(f"  Tree Depth: {clf.tree_.max_depth}")
log(f"  Tree Nodes: {clf.tree_.node_count}")
log(f"  Test Accuracy: {test_accuracy:.4f}")
log(f"  Test F1 Score: {test_f1:.4f}")
log(f"  Test ROC-AUC: {test_roc_auc:.4f}")
