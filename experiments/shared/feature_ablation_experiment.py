"""
Feature Ablation Experiment

This script performs feature ablation by:
1. Loading a trained decision tree model
2. Dropping features starting from lowest importance
3. Retraining and evaluating at each step
4. Plotting performance vs number of features

Requirements:
    pip install pandas scikit-learn numpy matplotlib joblib
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_ablation_experiment(dataset_name, model_file_path, train_path, test_path, output_dir, target_col=None, train_df=None, test_df=None, feature_cols=None):
    """Run feature ablation experiment for a dataset"""
    
    print("="*80)
    print(f"Feature Ablation Experiment - {dataset_name}")
    print("="*80)
    
    # Load the trained model to get feature importance
    print("\nLoading trained model...")
    clf_original = joblib.load(model_file_path)
    
    # Load data
    print("Loading data...")
    if train_df is None:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        feature_cols = [col for col in train_df.columns if col != 'is_attack' and col != 'Is_Malicious']
        target_col = 'is_attack' if 'is_attack' in train_df.columns else 'Is_Malicious'
    
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
    y_train = train_df[target_col].astype(int)
    X_test = test_df[feature_cols].fillna(test_df[feature_cols].median())
    y_test = test_df[target_col].astype(int)
    
    # Handle infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    # Get feature importance from original model
    if hasattr(clf_original, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': clf_original.feature_importances_
        }).sort_values('importance', ascending=True)  # Sort ascending (lowest first)
    else:
        # If model doesn't have feature_importances, use equal importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': [1.0/len(feature_cols)] * len(feature_cols)
        }).sort_values('importance', ascending=True)
    
    print(f"\nStarting with {len(feature_cols)} features")
    print("Dropping features from lowest to highest importance...")
    
    # Ablation: drop features one by one from lowest importance
    ablation_results = []
    current_features = feature_cols.copy()
    
    # Start with all features, then drop one at a time
    num_features_to_test = min(20, len(feature_cols))  # Test up to 20 different feature counts
    step_size = max(1, len(feature_cols) // num_features_to_test)
    
    for num_features in range(len(feature_cols), 0, -step_size):
        if num_features < 1:
            break
            
        # Select top N features (highest importance)
        features_to_keep = feature_importance.tail(num_features)['feature'].tolist()
        
        X_train_subset = X_train[features_to_keep]
        X_test_subset = X_test[features_to_keep]
        
        # Train new model with subset of features
        clf = DecisionTreeClassifier(
            criterion='gini',
            splitter='best',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        clf.fit(X_train_subset, y_train)
        
        # Evaluate
        y_test_pred = clf.predict(X_test_subset)
        y_test_proba = clf.predict_proba(X_test_subset)[:, 1]
        
        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.0
        
        ablation_results.append({
            'num_features': int(num_features),  # Convert to Python int
            'features': features_to_keep,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'roc_auc': float(auc)
        })
        
        print(f"  {num_features:3d} features: Accuracy={acc:.4f}, F1={f1:.4f}, ROC-AUC={auc:.4f}")
    
    # Create plots
    ablation_df = pd.DataFrame(ablation_results)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Accuracy
    axes[0, 0].plot(ablation_df['num_features'], ablation_df['accuracy'], marker='o')
    axes[0, 0].set_xlabel('Number of Features')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Number of Features')
    axes[0, 0].grid(True)
    axes[0, 0].invert_xaxis()
    
    # Precision
    axes[0, 1].plot(ablation_df['num_features'], ablation_df['precision'], marker='o', color='orange')
    axes[0, 1].set_xlabel('Number of Features')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Number of Features')
    axes[0, 1].grid(True)
    axes[0, 1].invert_xaxis()
    
    # Recall
    axes[0, 2].plot(ablation_df['num_features'], ablation_df['recall'], marker='o', color='green')
    axes[0, 2].set_xlabel('Number of Features')
    axes[0, 2].set_ylabel('Recall')
    axes[0, 2].set_title('Recall vs Number of Features')
    axes[0, 2].grid(True)
    axes[0, 2].invert_xaxis()
    
    # F1 Score
    axes[1, 0].plot(ablation_df['num_features'], ablation_df['f1'], marker='o', color='red')
    axes[1, 0].set_xlabel('Number of Features')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score vs Number of Features')
    axes[1, 0].grid(True)
    axes[1, 0].invert_xaxis()
    
    # ROC-AUC
    axes[1, 1].plot(ablation_df['num_features'], ablation_df['roc_auc'], marker='o', color='purple')
    axes[1, 1].set_xlabel('Number of Features')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_title('ROC-AUC vs Number of Features')
    axes[1, 1].grid(True)
    axes[1, 1].invert_xaxis()
    
    # Combined metrics
    axes[1, 2].plot(ablation_df['num_features'], ablation_df['accuracy'], marker='o', label='Accuracy')
    axes[1, 2].plot(ablation_df['num_features'], ablation_df['precision'], marker='s', label='Precision')
    axes[1, 2].plot(ablation_df['num_features'], ablation_df['recall'], marker='^', label='Recall')
    axes[1, 2].plot(ablation_df['num_features'], ablation_df['f1'], marker='d', label='F1')
    axes[1, 2].set_xlabel('Number of Features')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('All Metrics vs Number of Features')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].invert_xaxis()
    
    plt.tight_layout()
    
    plot_file = output_dir / f"feature_ablation_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_file}")
    plt.close()
    
    # Save results (convert numpy types to Python native types for JSON)
    results_file = output_dir / f"feature_ablation_{dataset_name.lower().replace(' ', '_')}.json"
    
    # Convert ablation_results to JSON-serializable format
    serializable_results = []
    for result in ablation_results:
        serializable_results.append({
            'num_features': int(result['num_features']),
            'features': result['features'],  # List of strings, already serializable
            'accuracy': float(result['accuracy']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1': float(result['f1']),
            'roc_auc': float(result['roc_auc'])
        })
    
    results = {
        'dataset': dataset_name,
        'ablation_results': serializable_results,
        'summary': {
            'max_features': int(len(feature_cols)),
            'min_features_tested': int(ablation_df['num_features'].min()),
            'best_accuracy': float(ablation_df['accuracy'].max()),
            'best_f1': float(ablation_df['f1'].max()),
            'best_roc_auc': float(ablation_df['roc_auc'].max())
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    return ablation_df

def find_latest_model(model_dir):
    """Find the latest model file in a directory"""
    if not model_dir.exists():
        return None
    
    model_files = sorted(model_dir.glob("decision_tree_model_*.joblib"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True)
    
    return model_files[0] if model_files else None

if __name__ == "__main__":
    # Calculate base path: shared -> experiments -> Supervised_IDS
    # __file__ is at: experiments/shared/feature_ablation_experiment.py
    # parent = experiments/shared/
    # parent.parent = experiments/
    # parent.parent.parent = Supervised_IDS/ (project root)
    base_path = Path(__file__).parent.parent.parent
    
    # Run for each dataset
    datasets = [
        {
            'name': 'SensorNetGuard',
            'model_dir': base_path / "experiments" / "sensornetguard" / "decision_tree_experiment" / "models",
            'train': base_path / "data" / "sensornetguard_data.csv",
            'test': None,  # Will split from train
            'target_col': 'Is_Malicious'
        },
        {
            'name': 'Farm-Flow',
            'model_dir': base_path / "experiments" / "farmflow" / "decision_tree_experiment" / "models",
            'train': base_path / "data" / "farm-flow-silver" / "Farm-Flow_Train_Binary.csv",
            'test': base_path / "data" / "farm-flow-silver" / "Farm-Flow_Test_Binary.csv",
            'target_col': 'is_attack'
        },
        {
            'name': 'CIC IDS 2017',
            'model_dir': base_path / "experiments" / "cicids2017" / "decision_tree_experiment" / "models",
            'train': base_path / "data" / "cic-ids2017-silver" / "CIC-IDS2017_Train_Binary.csv",
            'test': base_path / "data" / "cic-ids2017-silver" / "CIC-IDS2017_Test_Binary.csv",
            'target_col': 'is_attack'
        },
        {
            'name': 'UNSW-NB15',
            'model_dir': base_path / "experiments" / "unsw-nb15" / "decision_tree_experiment" / "models",
            'train': base_path / "data" / "unsw-nb15-silver" / "UNSW-NB15_Train_Binary.csv",
            'test': base_path / "data" / "unsw-nb15-silver" / "UNSW-NB15_Test_Binary.csv",
            'target_col': 'is_attack'
        },
        {
            'name': 'NSL-KDD',
            'model_dir': base_path / "experiments" / "nsl-kdd" / "decision_tree_experiment" / "models",
            'train': base_path / "data" / "nsl-kdd-silver" / "NSL-KDD_Train_Binary.csv",
            'test': base_path / "data" / "nsl-kdd-silver" / "NSL-KDD_Test_Binary.csv",
            'target_col': 'is_attack'
        }
    ]
    
    output_dir = base_path / "experiments" / "shared" / "feature_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets:
        # Find latest model
        model_file = find_latest_model(dataset['model_dir'])
        if model_file is None:
            print(f"\n⚠ Skipping {dataset['name']}: No model file found in {dataset['model_dir']}")
            continue
        
        print(f"\n✓ Found model for {dataset['name']}: {model_file.name}")
        
        if dataset['test'] is None:
            # For SensorNetGuard, split the data
            print(f"\n{dataset['name']}: Splitting data for train/test...")
            if not dataset['train'].exists():
                print(f"⚠ Skipping {dataset['name']}: Training data file not found: {dataset['train']}")
                continue
            
            df = pd.read_csv(dataset['train'])
            exclude_cols = ['Node_ID', 'Timestamp', 'IP_Address', dataset['target_col']]
            feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
            X = df[feature_cols].fillna(df[feature_cols].median())
            y = df[dataset['target_col']].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Create temporary dataframes
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
            # Update dataset info
            dataset['train_df'] = train_df
            dataset['test_df'] = test_df
            dataset['feature_cols'] = feature_cols
        else:
            # Check if train and test files exist
            if not dataset['train'].exists() or not dataset['test'].exists():
                print(f"\n⚠ Skipping {dataset['name']}: Data files not found")
                print(f"  Train: {dataset['train'].exists()}")
                print(f"  Test: {dataset['test'].exists()}")
                continue
        
        try:
            if dataset['test'] is None:
                # SensorNetGuard with split data
                run_ablation_experiment(
                    dataset['name'],
                    model_file,
                    None,  # train_path
                    None,  # test_path
                    output_dir,
                    target_col=dataset['target_col'],
                    train_df=dataset['train_df'],
                    test_df=dataset['test_df'],
                    feature_cols=dataset['feature_cols']
                )
            else:
                run_ablation_experiment(
                    dataset['name'],
                    model_file,
                    dataset['train'],
                    dataset['test'],
                    output_dir,
                    target_col=dataset['target_col']
                )
        except Exception as e:
            print(f"\n✗ Error processing {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("Feature Ablation Experiments Complete!")
    print("="*80)
