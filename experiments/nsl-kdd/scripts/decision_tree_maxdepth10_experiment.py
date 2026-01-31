"""
Decision Tree Experiment for NSL-KDD with max_depth=10

This script trains a decision tree with max_depth=10 to compare
with the unlimited depth version.

Requirements:
    pip install pandas scikit-learn numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)

# Set paths
data_dir = Path(__file__).parent.parent.parent.parent / "data" / "nsl-kdd-silver"
train_path = data_dir / "NSL-KDD_Train_Binary.csv"
test_path = data_dir / "NSL-KDD_Test_Binary.csv"

if not train_path.exists() or not test_path.exists():
    print("ERROR: Preprocessed data not found!")
    sys.exit(1)

experiment_dir = Path(__file__).parent.parent / "decision_tree_maxdepth10_experiment"
experiment_dir.mkdir(parents=True, exist_ok=True)
results_dir = experiment_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = results_dir / f"results_{timestamp}.json"

print("="*80)
print("Decision Tree Experiment - NSL-KDD (max_depth=10)")
print("="*80)

# Load data
print("\nLoading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

feature_cols = [col for col in train_df.columns if col != 'is_attack']
X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
y_train = train_df['is_attack'].astype(int)
X_test = test_df[feature_cols].fillna(test_df[feature_cols].median())
y_test = test_df['is_attack'].astype(int)

# Handle infinite values
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

print(f"  Training set: {len(X_train):,} samples")
print(f"  Test set: {len(X_test):,} samples")

# Train tree with max_depth=10
print("\nTraining decision tree with max_depth=10...")
tree_params = {
    'criterion': 'gini',
    'splitter': 'best',
    'max_depth': 10,  # Limited depth
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'max_features': None,
    'class_weight': None
}

clf = DecisionTreeClassifier(**tree_params)
clf.fit(X_train, y_train)

print(f"✓ Tree training completed!")
print(f"  Tree depth: {clf.tree_.max_depth}")
print(f"  Number of nodes: {clf.tree_.node_count}")
print(f"  Number of leaves: {clf.tree_.n_leaves}")

# Make predictions
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
y_train_proba = clf.predict_proba(X_train)[:, 1]
y_test_proba = clf.predict_proba(X_test)[:, 1]

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, zero_division=0)
train_recall = recall_score(y_train, y_train_pred, zero_division=0)
train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
train_roc_auc = roc_auc_score(y_train, y_train_proba) if len(np.unique(y_train)) > 1 else 0.0
train_cm = confusion_matrix(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_roc_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.0
test_cm = confusion_matrix(y_test, y_test_pred)

print("\n" + "="*80)
print("Results")
print("="*80)
print(f"\nTest Set Metrics:")
print(f"  Accuracy:  {test_accuracy:.6f}")
print(f"  Precision: {test_precision:.6f}")
print(f"  Recall:    {test_recall:.6f}")
print(f"  F1 Score:  {test_f1:.6f}")
print(f"  ROC-AUC:   {test_roc_auc:.6f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

# Save results
results = {
    'timestamp': timestamp,
    'dataset': 'NSL-KDD',
    'max_depth': 10,
    'tree_properties': {
        'max_depth': int(clf.tree_.max_depth),
        'n_nodes': int(clf.tree_.node_count),
        'n_leaves': int(clf.tree_.n_leaves)
    },
    'test_metrics': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1),
        'roc_auc': float(test_roc_auc),
        'confusion_matrix': test_cm.tolist()
    },
    'train_metrics': {
        'accuracy': float(train_accuracy),
        'precision': float(train_precision),
        'recall': float(train_recall),
        'f1_score': float(train_f1),
        'roc_auc': float(train_roc_auc),
        'confusion_matrix': train_cm.tolist()
    },
    'top_10_features': feature_importance.head(10).to_dict('records')
}

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {results_file}")
print("\n" + "="*80)
print("Experiment Complete!")
print("="*80)
