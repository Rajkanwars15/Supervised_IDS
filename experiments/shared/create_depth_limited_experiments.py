"""
Automatically create max_depth=10 experiments for datasets with trees > 10 depth

This script:
1. Checks all decision tree experiment results
2. Identifies trees with depth > 10
3. Creates max_depth=10 experiment scripts if they don't exist
4. Runs the experiments

Requirements:
    pip install pandas scikit-learn numpy
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

def find_latest_results(experiment_dir):
    """Find the latest results JSON file"""
    results_dir = experiment_dir / "results"
    if not results_dir.exists():
        return None
    
    result_files = sorted(results_dir.glob("results_*.json"),
                         key=lambda p: p.stat().st_mtime,
                         reverse=True)
    
    if not result_files:
        return None
    
    try:
        with open(result_files[0], 'r') as f:
            return json.load(f)
    except:
        return None

def create_maxdepth10_script(dataset_name, dataset_path, base_path):
    """Create a max_depth=10 experiment script for a dataset"""
    
    script_template = '''"""
Decision Tree Experiment for {dataset_name} with max_depth=10

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
data_dir = Path(__file__).parent.parent.parent.parent / "data" / "{data_path}"
train_path = data_dir / "{train_file}"
test_path = data_dir / "{test_file}"

if not train_path.exists() or not test_path.exists():
    print("ERROR: Preprocessed data not found!")
    sys.exit(1)

experiment_dir = Path(__file__).parent.parent / "decision_tree_maxdepth10_experiment"
experiment_dir.mkdir(parents=True, exist_ok=True)
results_dir = experiment_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = results_dir / f"results_{{timestamp}}.json"

print("="*80)
print("Decision Tree Experiment - {dataset_name} (max_depth=10)")
print("="*80)

# Load data
print("\\nLoading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

feature_cols = [col for col in train_df.columns if col != '{target_col}']
X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
y_train = train_df['{target_col}'].astype(int)
X_test = test_df[feature_cols].fillna(test_df[feature_cols].median())
y_test = test_df['{target_col}'].astype(int)

# Handle infinite values
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

print(f"  Training set: {{len(X_train):,}} samples")
print(f"  Test set: {{len(X_test):,}} samples")

# Train tree with max_depth=10
print("\\nTraining decision tree with max_depth=10...")
tree_params = {{
    'criterion': 'gini',
    'splitter': 'best',
    'max_depth': 10,  # Limited depth
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'max_features': None,
    'class_weight': None
}}

clf = DecisionTreeClassifier(**tree_params)
clf.fit(X_train, y_train)

print(f"✓ Tree training completed!")
print(f"  Tree depth: {{clf.tree_.max_depth}}")
print(f"  Number of nodes: {{clf.tree_.node_count}}")
print(f"  Number of leaves: {{clf.tree_.n_leaves}}")

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

print("\\n" + "="*80)
print("Results")
print("="*80)
print(f"\\nTest Set Metrics:")
print(f"  Accuracy:  {{test_accuracy:.6f}}")
print(f"  Precision: {{test_precision:.6f}}")
print(f"  Recall:    {{test_recall:.6f}}")
print(f"  F1 Score:  {{test_f1:.6f}}")
print(f"  ROC-AUC:   {{test_roc_auc:.6f}}")

# Feature importance
feature_importance = pd.DataFrame({{
    'feature': feature_cols,
    'importance': clf.feature_importances_
}}).sort_values('importance', ascending=False)

# Save results
results = {{
    'timestamp': timestamp,
    'dataset': '{dataset_name}',
    'max_depth': 10,
    'tree_properties': {{
        'max_depth': int(clf.tree_.max_depth),
        'n_nodes': int(clf.tree_.node_count),
        'n_leaves': int(clf.tree_.n_leaves)
    }},
    'test_metrics': {{
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1),
        'roc_auc': float(test_roc_auc),
        'confusion_matrix': test_cm.tolist()
    }},
    'train_metrics': {{
        'accuracy': float(train_accuracy),
        'precision': float(train_precision),
        'recall': float(train_recall),
        'f1_score': float(train_f1),
        'roc_auc': float(train_roc_auc),
        'confusion_matrix': train_cm.tolist()
    }},
    'top_10_features': feature_importance.head(10).to_dict('records')
}}

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\\n✓ Results saved to: {{results_file}}")
print("\\n" + "="*80)
print("Experiment Complete!")
print("="*80)
'''
    
    # Dataset-specific configurations
    dataset_configs = {
        'SensorNetGuard': {
            'data_path': '',
            'train_file': 'sensornetguard_data.csv',
            'test_file': None,
            'target_col': 'Is_Malicious',
            'needs_split': True
        },
        'Farm-Flow': {
            'data_path': 'farm-flow-silver',
            'train_file': 'Farm-Flow_Train_Binary.csv',
            'test_file': 'Farm-Flow_Test_Binary.csv',
            'target_col': 'is_attack',
            'needs_split': False
        },
        'CIC IDS 2017': {
            'data_path': 'cic-ids2017-silver',
            'train_file': 'CIC-IDS2017_Train_Binary.csv',
            'test_file': 'CIC-IDS2017_Test_Binary.csv',
            'target_col': 'is_attack',
            'needs_split': False
        },
        'UNSW-NB15': {
            'data_path': 'unsw-nb15-silver',
            'train_file': 'UNSW-NB15_Train_Binary.csv',
            'test_file': 'UNSW-NB15_Test_Binary.csv',
            'target_col': 'is_attack',
            'needs_split': False
        },
        'NSL-KDD': {
            'data_path': 'nsl-kdd-silver',
            'train_file': 'NSL-KDD_Train_Binary.csv',
            'test_file': 'NSL-KDD_Test_Binary.csv',
            'target_col': 'is_attack',
            'needs_split': False
        }
    }
    
    config = dataset_configs.get(dataset_name)
    if not config:
        print(f"Unknown dataset: {dataset_name}")
        return None
    
    # For SensorNetGuard, handle data splitting differently
    if config['needs_split']:
        # Modify the load data section for SensorNetGuard
        load_section = '''# Load data
print("\\nLoading data...")
df = pd.read_csv(train_path)
exclude_cols = ['Node_ID', 'Timestamp', 'IP_Address', '{target_col}']
feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
X = df[feature_cols].fillna(df[feature_cols].median())
y = df['{target_col}'].astype(int)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)'''
        
        # Replace the load section in template
        old_load = '# Load data\nprint("\\nLoading data...")\ntrain_df = pd.read_csv(train_path)\ntest_df = pd.read_csv(test_path)\n\nfeature_cols = [col for col in train_df.columns if col != \'{target_col}\']\nX_train = train_df[feature_cols].fillna(train_df[feature_cols].median())\ny_train = train_df[\'{target_col}\'].astype(int)\nX_test = test_df[feature_cols].fillna(test_df[feature_cols].median())\ny_test = test_df[\'{target_col}\'].astype(int)'
        
        script_content = script_template.replace(old_load, load_section).format(
            dataset_name=dataset_name,
            data_path=config['data_path'],
            train_file=config['train_file'],
            test_file=config['test_file'] or config['train_file'],
            target_col=config['target_col']
        )
    else:
        script_content = script_template.format(
            dataset_name=dataset_name,
            data_path=config['data_path'],
            train_file=config['train_file'],
            test_file=config['test_file'] or config['train_file'],
            target_col=config['target_col']
        )
    
    script_path = dataset_path / "scripts" / "decision_tree_maxdepth10_experiment.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent
    
    datasets_to_check = [
        ('SensorNetGuard', base_path / "experiments" / "sensornetguard"),
        ('Farm-Flow', base_path / "experiments" / "farmflow"),
        ('CIC IDS 2017', base_path / "experiments" / "cicids2017"),
        ('UNSW-NB15', base_path / "experiments" / "unsw-nb15"),
        ('NSL-KDD', base_path / "experiments" / "nsl-kdd")
    ]
    
    print("="*80)
    print("Checking Tree Depths and Creating max_depth=10 Experiments")
    print("="*80)
    
    for dataset_name, exp_path in datasets_to_check:
        decision_tree_dir = exp_path / "decision_tree_experiment"
        maxdepth10_dir = exp_path / "decision_tree_maxdepth10_experiment"
        
        # Check if max_depth=10 experiment already exists
        if (maxdepth10_dir / "results").exists() and list((maxdepth10_dir / "results").glob("results_*.json")):
            print(f"\n✓ {dataset_name}: max_depth=10 experiment already exists")
            continue
        
        # Check original tree depth
        results = find_latest_results(decision_tree_dir)
        if not results:
            print(f"\n⚠ {dataset_name}: No decision tree results found, skipping")
            continue
        
        tree_depth = results.get('tree_properties', {}).get('max_depth', 0)
        print(f"\n{dataset_name}: Tree depth = {tree_depth}")
        
        if tree_depth > 10:
            print(f"  → Creating max_depth=10 experiment...")
            script_path = create_maxdepth10_script(dataset_name, exp_path, base_path)
            if script_path:
                print(f"  ✓ Created script: {script_path}")
                print(f"  → Running experiment...")
                try:
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        capture_output=True,
                        text=True,
                        timeout=3600  # 1 hour timeout
                    )
                    if result.returncode == 0:
                        print(f"  ✓ Experiment completed successfully")
                    else:
                        print(f"  ✗ Experiment failed: {result.stderr[:200]}")
                except Exception as e:
                    print(f"  ✗ Error running experiment: {e}")
            else:
                print(f"  ✗ Failed to create script")
        else:
            print(f"  → Tree depth ≤ 10, skipping max_depth=10 experiment")
    
    print("\n" + "="*80)
    print("Depth-Limited Experiment Creation Complete!")
    print("="*80)
