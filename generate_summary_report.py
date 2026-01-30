"""
Generate a comprehensive summary report from all experiment results

This script:
- Reads all result JSON files from each experiment
- Extracts key metrics
- Generates a comparison report
- Saves to markdown and JSON formats

Usage:
    python generate_summary_report.py
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def load_latest_results(experiment_dir):
    """Load the most recent results JSON from an experiment directory"""
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
    except Exception as e:
        print(f"Error loading {result_files[0]}: {e}")
        return None

def generate_markdown_report(results_data):
    """Generate a markdown summary report"""
    report = []
    report.append("# Decision Tree Experiments - Summary Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("="*80 + "\n")
    
    # Dataset information
    report.append("## Dataset Information\n")
    datasets = {
        'SensorNetGuard': {
            'path': 'experiments/sensornetguard',
            'name': 'SensorNetGuard IDS'
        },
        'Farm-Flow': {
            'path': 'experiments/farmflow',
            'name': 'Farm-Flow Binary Classification'
        },
        'CIC IDS 2017': {
            'path': 'experiments/cicids2017',
            'name': 'CIC IDS 2017 Binary Classification'
        }
    }
    
    for dataset_name, info in datasets.items():
        if dataset_name in results_data and results_data[dataset_name]:
            data = results_data[dataset_name]
            data_info = data.get('data_info', {})
            report.append(f"### {info['name']}\n")
            report.append(f"- **Training samples**: {data_info.get('n_train_samples', 'N/A'):,}")
            report.append(f"- **Test samples**: {data_info.get('n_test_samples', 'N/A'):,}")
            report.append(f"- **Features**: {data_info.get('n_features', 'N/A')}")
            report.append(f"- **Training class distribution**: {data_info.get('train_class_distribution', {})}")
            report.append(f"- **Test class distribution**: {data_info.get('test_class_distribution', {})}")
            report.append("")
    
    # Tree properties
    report.append("## Tree Properties\n")
    report.append("| Dataset | Max Depth | Nodes | Leaves |")
    report.append("|---------|-----------|-------|--------|")
    
    for dataset_name in datasets.keys():
        if dataset_name in results_data and results_data[dataset_name]:
            props = results_data[dataset_name].get('tree_properties', {})
            report.append(f"| {dataset_name} | {props.get('max_depth', 'N/A')} | "
                         f"{props.get('n_nodes', 'N/A'):,} | {props.get('n_leaves', 'N/A'):,} |")
    report.append("")
    
    # Test metrics comparison
    report.append("## Test Set Performance Metrics\n")
    report.append("| Dataset | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Avg Precision |")
    report.append("|---------|----------|-----------|--------|----------|---------|---------------|")
    
    for dataset_name in datasets.keys():
        if dataset_name in results_data and results_data[dataset_name]:
            metrics = results_data[dataset_name].get('test_metrics', {})
            report.append(f"| {dataset_name} | "
                         f"{metrics.get('accuracy', 0):.4f} | "
                         f"{metrics.get('precision', 0):.4f} | "
                         f"{metrics.get('recall', 0):.4f} | "
                         f"{metrics.get('f1_score', 0):.4f} | "
                         f"{metrics.get('roc_auc', 0):.4f} | "
                         f"{metrics.get('average_precision', 0):.4f} |")
    report.append("")
    
    # Top features
    report.append("## Top 10 Most Important Features (by Dataset)\n")
    
    for dataset_name in datasets.keys():
        if dataset_name in results_data and results_data[dataset_name]:
            features = results_data[dataset_name].get('feature_importance', [])
            if features:
                report.append(f"### {dataset_name}\n")
                top_features = sorted(features, key=lambda x: x.get('importance', 0), reverse=True)[:10]
                for i, feat in enumerate(top_features, 1):
                    report.append(f"{i}. **{feat.get('feature', 'N/A')}**: {feat.get('importance', 0):.6f}")
                report.append("")
    
    # Confusion matrices
    report.append("## Confusion Matrices (Test Set)\n")
    
    for dataset_name in datasets.keys():
        if dataset_name in results_data and results_data[dataset_name]:
            cm = results_data[dataset_name].get('test_metrics', {}).get('confusion_matrix', [])
            if cm and len(cm) == 2 and len(cm[0]) == 2:
                report.append(f"### {dataset_name}\n")
                report.append("```")
                report.append(f"                Predicted")
                report.append(f"              Benign  Attack")
                report.append(f"Actual Benign   {cm[0][0]:5d}   {cm[0][1]:5d}")
                report.append(f"       Attack   {cm[1][0]:5d}   {cm[1][1]:5d}")
                report.append("```")
                report.append("")
    
    # Model and visualization paths
    report.append("## Generated Artifacts\n")
    
    for dataset_name in datasets.keys():
        if dataset_name in results_data and results_data[dataset_name]:
            data = results_data[dataset_name]
            report.append(f"### {dataset_name}\n")
            if 'model_file' in data:
                report.append(f"- **Model**: `{data['model_file']}`")
            if 'visualization_files' in data:
                viz = data['visualization_files']
                report.append(f"- **Visualizations**:")
                for fmt, path in viz.items():
                    report.append(f"  - {fmt.upper()}: `{path}`")
            report.append("")
    
    return "\n".join(report)

def main():
    base_path = Path(__file__).parent
    
    experiments = {
        'SensorNetGuard': base_path / "experiments" / "sensornetguard" / "decision_tree_experiment",
        'Farm-Flow': base_path / "experiments" / "farmflow" / "decision_tree_experiment",
        'CIC IDS 2017': base_path / "experiments" / "cicids2017" / "decision_tree_experiment"
    }
    
    print("="*80)
    print("Generating Summary Report")
    print("="*80)
    
    results_data = {}
    
    for name, exp_dir in experiments.items():
        print(f"\nLoading results for {name}...")
        results = load_latest_results(exp_dir)
        if results:
            results_data[name] = results
            print(f"  ✓ Loaded results from {name}")
        else:
            print(f"  ✗ No results found for {name}")
    
    if not results_data:
        print("\nNo results found. Please run experiments first.")
        return
    
    # Generate markdown report
    markdown_report = generate_markdown_report(results_data)
    
    # Save markdown report
    report_file = base_path / "EXPERIMENT_SUMMARY_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(markdown_report)
    print(f"\n✓ Markdown report saved to: {report_file}")
    
    # Save JSON summary
    summary_json = {
        'generated_at': datetime.now().isoformat(),
        'datasets': list(results_data.keys()),
        'results': results_data
    }
    
    summary_file = base_path / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"✓ JSON summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("Summary Report Generation Complete")
    print("="*80)

if __name__ == "__main__":
    main()
