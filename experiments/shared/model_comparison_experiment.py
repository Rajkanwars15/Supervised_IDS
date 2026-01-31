"""
Comprehensive Model Comparison Experiment

This script compares multiple models across all datasets:
- Baseline models: Decision Stump (1-rule), Shallow Trees (max_depth=3, 5)
- Tree-based: Random Forest, XGBoost
- Neural: Multi-Layer Perceptron (MLP)
- Linear: Logistic Regression, Linear SVM

Also performs temporal/source-split experiments where possible.

Requirements:
    pip install pandas scikit-learn numpy xgboost matplotlib joblib
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train a model and return evaluation metrics"""
    try:
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Get probabilities if available
        try:
            y_test_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.0
        except:
            auc = 0.0
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        return {
            'model_name': model_name,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'test_f1': float(test_f1),
            'test_roc_auc': float(auc),
            'status': 'success'
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'status': 'error',
            'error': str(e)
        }

def run_model_comparison(dataset_name, train_path, test_path, output_dir, target_col=None, 
                         train_df=None, test_df=None, feature_cols=None, temporal_split=False):
    """Run comprehensive model comparison for a dataset"""
    
    print("="*80)
    print(f"Model Comparison Experiment - {dataset_name}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    if train_df is None:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        feature_cols = [col for col in train_df.columns if col not in ['is_attack', 'Is_Malicious']]
        target_col = 'is_attack' if 'is_attack' in train_df.columns else 'Is_Malicious'
    
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
    y_train = train_df[target_col].astype(int)
    X_test = test_df[feature_cols].fillna(test_df[feature_cols].median())
    y_test = test_df[target_col].astype(int)
    
    # Handle infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training class distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test class distribution: {y_test.value_counts().to_dict()}")
    
    # Prepare models
    models = []
    
    # Baseline models
    print("\n" + "="*80)
    print("Training Baseline Models")
    print("="*80)
    
    models.append(('Decision Stump (max_depth=1)', DecisionTreeClassifier(max_depth=1, random_state=42)))
    models.append(('Shallow Tree (max_depth=3)', DecisionTreeClassifier(max_depth=3, random_state=42)))
    models.append(('Shallow Tree (max_depth=5)', DecisionTreeClassifier(max_depth=5, random_state=42)))
    
    # Linear models
    print("\n" + "="*80)
    print("Training Linear Models")
    print("="*80)
    
    models.append(('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)))
    models.append(('Linear SVM', LinearSVC(max_iter=1000, random_state=42, dual=False)))
    
    # Tree-based ensemble models
    print("\n" + "="*80)
    print("Training Ensemble Models")
    print("="*80)
    
    models.append(('Random Forest (n_estimators=100)', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)))
    models.append(('Random Forest (n_estimators=500)', RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)))
    
    if XGBOOST_AVAILABLE:
        models.append(('XGBoost', xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')))
    
    # Neural network
    print("\n" + "="*80)
    print("Training Neural Network")
    print("="*80)
    
    models.append(('MLP (hidden_layers=100)', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)))
    models.append(('MLP (hidden_layers=100,50)', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)))
    
    # Train and evaluate all models
    results = []
    for model_name, model in models:
        print(f"\nTraining {model_name}...")
        result = train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"  ✓ Test Accuracy: {result['test_accuracy']:.4f}, F1: {result['test_f1']:.4f}, ROC-AUC: {result['test_roc_auc']:.4f}")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown error')}")
    
    # Create comparison plot
    print("\n" + "="*80)
    print("Creating Comparison Plots")
    print("="*80)
    
    results_df = pd.DataFrame([r for r in results if r['status'] == 'success'])
    
    if len(results_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy comparison
        axes[0, 0].barh(results_df['model_name'], results_df['test_accuracy'])
        axes[0, 0].set_xlabel('Test Accuracy')
        axes[0, 0].set_title('Model Comparison - Test Accuracy')
        axes[0, 0].grid(True, axis='x')
        
        # F1 Score comparison
        axes[0, 1].barh(results_df['model_name'], results_df['test_f1'], color='orange')
        axes[0, 1].set_xlabel('Test F1 Score')
        axes[0, 1].set_title('Model Comparison - Test F1 Score')
        axes[0, 1].grid(True, axis='x')
        
        # ROC-AUC comparison
        axes[1, 0].barh(results_df['model_name'], results_df['test_roc_auc'], color='green')
        axes[1, 0].set_xlabel('Test ROC-AUC')
        axes[1, 0].set_title('Model Comparison - Test ROC-AUC')
        axes[1, 0].grid(True, axis='x')
        
        # Combined metrics
        x_pos = np.arange(len(results_df))
        width = 0.25
        axes[1, 1].bar(x_pos - width, results_df['test_accuracy'], width, label='Accuracy', alpha=0.8)
        axes[1, 1].bar(x_pos, results_df['test_f1'], width, label='F1 Score', alpha=0.8)
        axes[1, 1].bar(x_pos + width, results_df['test_roc_auc'], width, label='ROC-AUC', alpha=0.8)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Combined Metrics Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(results_df['model_name'], rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        plot_file = output_dir / f"model_comparison_{dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_file}")
        plt.close()
    
    # Save results
    results_file = output_dir / f"model_comparison_{dataset_name.lower().replace(' ', '_')}.json"
    output_data = {
        'dataset': dataset_name,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'temporal_split': temporal_split,
        'data_info': {
            'n_train_samples': int(len(X_train)),
            'n_test_samples': int(len(X_test)),
            'n_features': int(len(feature_cols)),
            'train_class_distribution': y_train.value_counts().to_dict(),
            'test_class_distribution': y_test.value_counts().to_dict()
        },
        'model_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent
    
    datasets = [
        {
            'name': 'SensorNetGuard',
            'train': base_path / "data" / "sensornetguard_data.csv",
            'test': None,  # Will split from train
            'target_col': 'Is_Malicious'
        },
        {
            'name': 'Farm-Flow',
            'train': base_path / "data" / "farm-flow-silver" / "Farm-Flow_Train_Binary.csv",
            'test': base_path / "data" / "farm-flow-silver" / "Farm-Flow_Test_Binary.csv",
            'target_col': 'is_attack'
        },
        {
            'name': 'CIC IDS 2017',
            'train': base_path / "data" / "cic-ids2017-silver" / "CIC-IDS2017_Train_Binary.csv",
            'test': base_path / "data" / "cic-ids2017-silver" / "CIC-IDS2017_Test_Binary.csv",
            'target_col': 'is_attack'
        },
        {
            'name': 'UNSW-NB15',
            'train': base_path / "data" / "unsw-nb15-silver" / "UNSW-NB15_Train_Binary.csv",
            'test': base_path / "data" / "unsw-nb15-silver" / "UNSW-NB15_Test_Binary.csv",
            'target_col': 'is_attack'
        },
        {
            'name': 'NSL-KDD',
            'train': base_path / "data" / "nsl-kdd-silver" / "NSL-KDD_Train_Binary.csv",
            'test': base_path / "data" / "nsl-kdd-silver" / "NSL-KDD_Test_Binary.csv",
            'target_col': 'is_attack'
        }
    ]
    
    output_dir = base_path / "experiments" / "shared" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets:
        print("\n" + "="*80)
        print(f"Processing {dataset['name']}")
        print("="*80)
        
        if dataset['test'] is None:
            # For SensorNetGuard, split the data
            if not dataset['train'].exists():
                print(f"⚠ Skipping {dataset['name']}: Training data file not found")
                continue
            
            df = pd.read_csv(dataset['train'])
            exclude_cols = ['Node_ID', 'Timestamp', 'IP_Address', dataset['target_col']]
            feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
            X = df[feature_cols].fillna(df[feature_cols].median())
            y = df[dataset['target_col']].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
            run_model_comparison(
                dataset['name'],
                None,
                None,
                output_dir,
                target_col=dataset['target_col'],
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols
            )
        else:
            if not dataset['train'].exists() or not dataset['test'].exists():
                print(f"⚠ Skipping {dataset['name']}: Data files not found")
                continue
            
            run_model_comparison(
                dataset['name'],
                dataset['train'],
                dataset['test'],
                output_dir,
                target_col=dataset['target_col']
            )
    
    print("\n" + "="*80)
    print("Model Comparison Experiments Complete!")
    print("="*80)
