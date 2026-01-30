"""
Decision Stump (1-Rule) Experiment for NSL-KDD

This experiment:
1. Trains a decision stump using only the top feature
2. Tests performance with noise added to the dominant feature
3. Performs cross-validation to check generalization

Requirements:
    pip install pandas scikit-learn numpy matplotlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set paths
data_dir = Path(__file__).parent.parent.parent.parent / "data" / "nsl-kdd-silver"
train_path = data_dir / "NSL-KDD_Train_Binary.csv"
test_path = data_dir / "NSL-KDD_Test_Binary.csv"

if not train_path.exists() or not test_path.exists():
    print("ERROR: Preprocessed data not found!")
    print(f"Please run preprocessing script first")
    sys.exit(1)

experiment_dir = Path(__file__).parent.parent / "decision_stump_experiment"
experiment_dir.mkdir(parents=True, exist_ok=True)
results_dir = experiment_dir / "results"
figs_dir = experiment_dir / "figs"
results_dir.mkdir(parents=True, exist_ok=True)
figs_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = results_dir / f"results_{timestamp}.json"

print("="*80)
print("Decision Stump Experiment - NSL-KDD")
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

# Train full tree to get feature importance
print("\nTraining full decision tree to identify top feature...")
full_tree = DecisionTreeClassifier(random_state=42, max_depth=None)
full_tree.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': full_tree.feature_importances_
}).sort_values('importance', ascending=False)

top_feature = feature_importance.iloc[0]['feature']
top_importance = feature_importance.iloc[0]['importance']

print(f"\nTop feature: {top_feature} (importance: {top_importance:.6f})")

# Experiment 1: Decision Stump with top feature
print("\n" + "="*80)
print("Experiment 1: Decision Stump (max_depth=1) with Top Feature")
print("="*80)

X_train_stump = X_train[[top_feature]]
X_test_stump = X_test[[top_feature]]

stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train_stump, y_train)

y_train_pred = stump.predict(X_train_stump)
y_test_pred = stump.predict(X_test_stump)
y_test_proba = stump.predict_proba(X_test_stump)[:, 1]

stump_train_acc = accuracy_score(y_train, y_train_pred)
stump_test_acc = accuracy_score(y_test, y_test_pred)
stump_test_prec = precision_score(y_test, y_test_pred, zero_division=0)
stump_test_rec = recall_score(y_test, y_test_pred, zero_division=0)
stump_test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
stump_test_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.0

print(f"\nDecision Stump Performance:")
print(f"  Training Accuracy: {stump_train_acc:.6f}")
print(f"  Test Accuracy: {stump_test_acc:.6f}")
print(f"  Test Precision: {stump_test_prec:.6f}")
print(f"  Test Recall: {stump_test_rec:.6f}")
print(f"  Test F1: {stump_test_f1:.6f}")
print(f"  Test ROC-AUC: {stump_test_auc:.6f}")

# Get the threshold
threshold = stump.tree_.threshold[0]
print(f"\n  Threshold: {threshold:.6f}")
print(f"  Rule: if {top_feature} <= {threshold:.6f} then Benign, else Attack")

# Experiment 2: Add noise to dominant feature
print("\n" + "="*80)
print("Experiment 2: Performance with Noise Added to Dominant Feature")
print("="*80)

noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0]
noise_results = []

for noise_level in noise_levels:
    X_test_noisy = X_test_stump.copy()
    # Add Gaussian noise scaled by feature std
    feature_std = X_test_noisy[top_feature].std()
    noise = np.random.RandomState(42).normal(0, noise_level * feature_std, size=len(X_test_noisy))
    X_test_noisy[top_feature] = X_test_noisy[top_feature] + noise
    
    y_pred_noisy = stump.predict(X_test_noisy)
    acc_noisy = accuracy_score(y_test, y_pred_noisy)
    prec_noisy = precision_score(y_test, y_pred_noisy, zero_division=0)
    rec_noisy = recall_score(y_test, y_pred_noisy, zero_division=0)
    f1_noisy = f1_score(y_test, y_pred_noisy, zero_division=0)
    
    noise_results.append({
        'noise_level': noise_level,
        'accuracy': acc_noisy,
        'precision': prec_noisy,
        'recall': rec_noisy,
        'f1': f1_noisy
    })
    
    print(f"  Noise level {noise_level:.2f}: Accuracy = {acc_noisy:.4f}, F1 = {f1_noisy:.4f}")

# Plot noise impact
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
noise_df = pd.DataFrame(noise_results)

axes[0, 0].plot(noise_df['noise_level'], noise_df['accuracy'], marker='o')
axes[0, 0].set_xlabel('Noise Level (std)')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy vs Noise Level')
axes[0, 0].grid(True)

axes[0, 1].plot(noise_df['noise_level'], noise_df['precision'], marker='o', color='orange')
axes[0, 1].set_xlabel('Noise Level (std)')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision vs Noise Level')
axes[0, 1].grid(True)

axes[1, 0].plot(noise_df['noise_level'], noise_df['recall'], marker='o', color='green')
axes[1, 0].set_xlabel('Noise Level (std)')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].set_title('Recall vs Noise Level')
axes[1, 0].grid(True)

axes[1, 1].plot(noise_df['noise_level'], noise_df['f1'], marker='o', color='red')
axes[1, 1].set_xlabel('Noise Level (std)')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('F1 Score vs Noise Level')
axes[1, 1].grid(True)

plt.tight_layout()
noise_plot_file = figs_dir / f"noise_impact_{timestamp}.png"
plt.savefig(noise_plot_file, dpi=300, bbox_inches='tight')
print(f"\nNoise impact plot saved to: {noise_plot_file}")
plt.close()

# Experiment 3: Cross-validation
print("\n" + "="*80)
print("Experiment 3: Cross-Validation (5-fold)")
print("="*80)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_train_fold = X_train.iloc[train_idx][[top_feature]]
    X_val_fold = X_train.iloc[val_idx][[top_feature]]
    y_train_fold = y_train.iloc[train_idx]
    y_val_fold = y_train.iloc[val_idx]
    
    stump_fold = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump_fold.fit(X_train_fold, y_train_fold)
    
    y_pred_fold = stump_fold.predict(X_val_fold)
    acc_fold = accuracy_score(y_val_fold, y_pred_fold)
    f1_fold = f1_score(y_val_fold, y_pred_fold, zero_division=0)
    
    cv_scores.append({'fold': fold, 'accuracy': acc_fold, 'f1': f1_fold})
    print(f"  Fold {fold}: Accuracy = {acc_fold:.4f}, F1 = {f1_fold:.4f}")

cv_mean_acc = np.mean([s['accuracy'] for s in cv_scores])
cv_std_acc = np.std([s['accuracy'] for s in cv_scores])
cv_mean_f1 = np.mean([s['f1'] for s in cv_scores])
cv_std_f1 = np.std([s['f1'] for s in cv_scores])

print(f"\n  CV Mean Accuracy: {cv_mean_acc:.4f} (+/- {cv_std_acc:.4f})")
print(f"  CV Mean F1: {cv_mean_f1:.4f} (+/- {cv_std_f1:.4f})")

# Save results
results = {
    'timestamp': timestamp,
    'dataset': 'NSL-KDD',
    'top_feature': top_feature,
    'top_feature_importance': float(top_importance),
    'stump_threshold': float(threshold),
    'stump_performance': {
        'train_accuracy': float(stump_train_acc),
        'test_accuracy': float(stump_test_acc),
        'test_precision': float(stump_test_prec),
        'test_recall': float(stump_test_rec),
        'test_f1': float(stump_test_f1),
        'test_roc_auc': float(stump_test_auc)
    },
    'noise_experiment': noise_results,
    'cross_validation': {
        'fold_scores': cv_scores,
        'mean_accuracy': float(cv_mean_acc),
        'std_accuracy': float(cv_std_acc),
        'mean_f1': float(cv_mean_f1),
        'std_f1': float(cv_std_f1)
    }
}

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {results_file}")
print("\n" + "="*80)
print("Experiment Complete!")
print("="*80)
