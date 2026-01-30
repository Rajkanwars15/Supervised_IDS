"""
Generate a comprehensive summary report from all experiment results

This script:
- Reads all result JSON files from each experiment type
- Extracts key metrics
- Generates a comparison report with embedded plots
- Saves to markdown and JSON formats
- Uses relative paths for GitHub compatibility

Usage:
    python generate_summary_report.py
"""

import json
from pathlib import Path
from datetime import datetime
import glob

def load_latest_results(experiment_dir, pattern="results_*.json"):
    """Load the most recent results JSON from an experiment directory"""
    results_dir = experiment_dir / "results"
    if not results_dir.exists():
        return None
    
    result_files = sorted(results_dir.glob(pattern), 
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

def find_latest_plot(plot_dir, pattern):
    """Find the latest plot file matching pattern"""
    if not plot_dir.exists():
        return None
    
    plot_files = sorted(plot_dir.glob(pattern),
                       key=lambda p: p.stat().st_mtime,
                       reverse=True)
    
    return plot_files[0] if plot_files else None

def get_relative_path(file_path, base_path):
    """Get relative path from base_path for GitHub compatibility"""
    try:
        rel_path = file_path.relative_to(base_path)
        return str(rel_path).replace('\\', '/')  # Use forward slashes for GitHub
    except:
        return str(file_path)

def generate_markdown_report(results_data, base_path):
    """Generate a markdown summary report with embedded plots"""
    report = []
    report.append("# Decision Tree Experiments - Comprehensive Summary Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("="*80 + "\n")
    
    # Main Decision Tree Experiments
    report.append("## 1. Main Decision Tree Experiments\n")
    
    datasets = {
        'SensorNetGuard': {
            'path': 'experiments/sensornetguard',
            'name': 'SensorNetGuard IDS',
            'exp_dir': 'decision_tree_experiment'
        },
        'Farm-Flow': {
            'path': 'experiments/farmflow',
            'name': 'Farm-Flow Binary Classification',
            'exp_dir': 'decision_tree_experiment'
        },
        'CIC IDS 2017': {
            'path': 'experiments/cicids2017',
            'name': 'CIC IDS 2017 Binary Classification',
            'exp_dir': 'decision_tree_experiment'
        }
    }
    
    # Dataset information
    report.append("### Dataset Information\n")
    for dataset_name, info in datasets.items():
        key = dataset_name
        if key in results_data.get('decision_trees', {}) and results_data['decision_trees'][key]:
            data = results_data['decision_trees'][key]
            data_info = data.get('data_info', {})
            report.append(f"#### {info['name']}\n")
            report.append(f"- **Training samples**: {data_info.get('n_train_samples', 'N/A'):,}")
            report.append(f"- **Test samples**: {data_info.get('n_test_samples', 'N/A'):,}")
            report.append(f"- **Features**: {data_info.get('n_features', 'N/A')}")
            report.append(f"- **Training class distribution**: {data_info.get('train_class_distribution', {})}")
            report.append(f"- **Test class distribution**: {data_info.get('test_class_distribution', {})}")
            report.append("")
    
    # Tree properties
    report.append("### Tree Properties\n")
    report.append("| Dataset | Max Depth | Nodes | Leaves |")
    report.append("|---------|-----------|-------|--------|")
    
    for dataset_name in datasets.keys():
        key = dataset_name
        if key in results_data.get('decision_trees', {}) and results_data['decision_trees'][key]:
            props = results_data['decision_trees'][key].get('tree_properties', {})
            report.append(f"| {dataset_name} | {props.get('max_depth', 'N/A')} | "
                         f"{props.get('n_nodes', 'N/A'):,} | {props.get('n_leaves', 'N/A'):,} |")
    report.append("")
    
    # Test metrics comparison
    report.append("### Test Set Performance Metrics\n")
    report.append("| Dataset | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Avg Precision |")
    report.append("|---------|----------|-----------|--------|----------|---------|---------------|")
    
    for dataset_name in datasets.keys():
        key = dataset_name
        if key in results_data.get('decision_trees', {}) and results_data['decision_trees'][key]:
            metrics = results_data['decision_trees'][key].get('test_metrics', {})
            report.append(f"| {dataset_name} | "
                         f"{metrics.get('accuracy', 0):.4f} | "
                         f"{metrics.get('precision', 0):.4f} | "
                         f"{metrics.get('recall', 0):.4f} | "
                         f"{metrics.get('f1_score', 0):.4f} | "
                         f"{metrics.get('roc_auc', 0):.4f} | "
                         f"{metrics.get('average_precision', 0):.4f} |")
    report.append("")
    
    # Tree visualizations
    report.append("### Decision Tree Visualizations\n")
    for dataset_name, info in datasets.items():
        exp_dir = base_path / info['path'] / info['exp_dir']
        viz_dir = exp_dir / "visualizations"
        
        # Find tree visualization
        tree_viz = find_latest_plot(viz_dir, "tree_visualization_*.png")
        if tree_viz:
            rel_path = get_relative_path(tree_viz, base_path)
            report.append(f"#### {dataset_name}\n")
            report.append(f"![{dataset_name} Decision Tree]({rel_path})\n")
            report.append("")
    
    # Decision Stump Experiments
    report.append("## 2. Decision Stump Experiments (1-Rule Models)\n")
    report.append("These experiments test model robustness by using only the top feature.\n\n")
    
    stump_datasets = ['SensorNetGuard', 'Farm-Flow']
    for dataset_name in stump_datasets:
        key = dataset_name
        if key in results_data.get('decision_stumps', {}):
            stump_data = results_data['decision_stumps'][key]
            report.append(f"### {dataset_name} Decision Stump\n")
            
            if 'top_feature' in stump_data:
                report.append(f"- **Top Feature**: {stump_data['top_feature']}")
                report.append(f"- **Feature Importance**: {stump_data.get('top_feature_importance', 0):.6f}")
                report.append(f"- **Threshold**: {stump_data.get('stump_threshold', 0):.6f}")
                report.append("")
            
            if 'stump_performance' in stump_data:
                perf = stump_data['stump_performance']
                report.append("**Performance:**\n")
                report.append(f"- Test Accuracy: {perf.get('test_accuracy', 0):.4f}")
                report.append(f"- Test F1: {perf.get('test_f1', 0):.4f}")
                report.append(f"- Test ROC-AUC: {perf.get('test_roc_auc', 0):.4f}")
                report.append("")
            
            # Noise impact plot
            if dataset_name == 'SensorNetGuard':
                stump_exp_dir = base_path / "experiments" / "sensornetguard" / "decision_stump_experiment"
            else:
                stump_exp_dir = base_path / "experiments" / "farmflow" / "decision_stump_experiment"
            
            noise_plot = find_latest_plot(stump_exp_dir / "figs", "noise_impact_*.png")
            if noise_plot:
                rel_path = get_relative_path(noise_plot, base_path)
                report.append(f"**Noise Impact Analysis:**\n")
                report.append(f"![{dataset_name} Noise Impact]({rel_path})\n")
                report.append("")
            
            # Cross-validation results
            if 'cross_validation' in stump_data:
                cv = stump_data['cross_validation']
                report.append(f"**Cross-Validation (5-fold):**\n")
                report.append(f"- Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} (±{cv.get('std_accuracy', 0):.4f})")
                report.append(f"- Mean F1: {cv.get('mean_f1', 0):.4f} (±{cv.get('std_f1', 0):.4f})")
                report.append("")
    
    # CIC IDS 2017 max_depth=10 Experiment
    report.append("## 3. CIC IDS 2017 Depth-Limited Experiment\n")
    if 'CIC IDS 2017' in results_data.get('maxdepth10', {}):
        maxdepth_data = results_data['maxdepth10']['CIC IDS 2017']
        report.append("Comparison of unlimited depth vs max_depth=10:\n\n")
        
        # Get original results for comparison
        original = results_data.get('decision_trees', {}).get('CIC IDS 2017', {})
        
        report.append("| Metric | Unlimited Depth | max_depth=10 | Difference |")
        report.append("|--------|-----------------|--------------|------------|")
        
        if original and 'test_metrics' in original and 'test_metrics' in maxdepth_data:
            orig_metrics = original['test_metrics']
            maxd_metrics = maxdepth_data['test_metrics']
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                orig_val = orig_metrics.get(metric, 0)
                maxd_val = maxd_metrics.get(metric, 0)
                diff = orig_val - maxd_val
                report.append(f"| {metric.capitalize()} | {orig_val:.4f} | {maxd_val:.4f} | {diff:+.4f} |")
        
        report.append("")
    
    # Feature Ablation Experiments
    report.append("## 4. Feature Ablation Experiments\n")
    report.append("These experiments show how performance changes as features are removed, starting from lowest importance.\n\n")
    
    ablation_dir = base_path / "experiments" / "shared" / "feature_ablation"
    if ablation_dir.exists():
        ablation_plots = list(ablation_dir.glob("feature_ablation_*.png"))
        for plot_file in sorted(ablation_plots):
            dataset_name = plot_file.stem.replace('feature_ablation_', '').replace('_', ' ').title()
            rel_path = get_relative_path(plot_file, base_path)
            report.append(f"### {dataset_name}\n")
            report.append(f"![{dataset_name} Feature Ablation]({rel_path})\n")
            report.append("")
    
    # Top features
    report.append("## 5. Top 10 Most Important Features (by Dataset)\n")
    
    for dataset_name in datasets.keys():
        key = dataset_name
        if key in results_data.get('decision_trees', {}) and results_data['decision_trees'][key]:
            features = results_data['decision_trees'][key].get('feature_importance', [])
            if features:
                report.append(f"### {dataset_name}\n")
                top_features = sorted(features, key=lambda x: x.get('importance', 0), reverse=True)[:10]
                for i, feat in enumerate(top_features, 1):
                    report.append(f"{i}. **{feat.get('feature', 'N/A')}**: {feat.get('importance', 0):.6f}")
                report.append("")
    
    # Confusion matrices
    report.append("## 6. Confusion Matrices (Test Set)\n")
    
    for dataset_name in datasets.keys():
        key = dataset_name
        if key in results_data.get('decision_trees', {}) and results_data['decision_trees'][key]:
            cm = results_data['decision_trees'][key].get('test_metrics', {}).get('confusion_matrix', [])
            if cm and len(cm) == 2 and len(cm[0]) == 2:
                report.append(f"### {dataset_name}\n")
                report.append("```")
                report.append(f"                Predicted")
                report.append(f"              Benign  Attack")
                report.append(f"Actual Benign   {cm[0][0]:5d}   {cm[0][1]:5d}")
                report.append(f"       Attack   {cm[1][0]:5d}   {cm[1][1]:5d}")
                report.append("```")
                report.append("")
    
    return "\n".join(report)

def main():
    base_path = Path(__file__).parent
    
    print("="*80)
    print("Generating Comprehensive Summary Report")
    print("="*80)
    
    results_data = {
        'decision_trees': {},
        'decision_stumps': {},
        'maxdepth10': {},
        'feature_ablation': {}
    }
    
    # Load main decision tree experiments
    print("\nLoading main decision tree experiments...")
    main_experiments = {
        'SensorNetGuard': base_path / "experiments" / "sensornetguard" / "decision_tree_experiment",
        'Farm-Flow': base_path / "experiments" / "farmflow" / "decision_tree_experiment",
        'CIC IDS 2017': base_path / "experiments" / "cicids2017" / "decision_tree_experiment"
    }
    
    for name, exp_dir in main_experiments.items():
        results = load_latest_results(exp_dir)
        if results:
            results_data['decision_trees'][name] = results
            print(f"  ✓ Loaded {name} decision tree results")
        else:
            print(f"  ✗ No results found for {name}")
    
    # Load decision stump experiments
    print("\nLoading decision stump experiments...")
    stump_experiments = {
        'SensorNetGuard': base_path / "experiments" / "sensornetguard" / "decision_stump_experiment",
        'Farm-Flow': base_path / "experiments" / "farmflow" / "decision_stump_experiment"
    }
    
    for name, exp_dir in stump_experiments.items():
        results = load_latest_results(exp_dir)
        if results:
            results_data['decision_stumps'][name] = results
            print(f"  ✓ Loaded {name} decision stump results")
        else:
            print(f"  ✗ No results found for {name}")
    
    # Load max_depth=10 experiment
    print("\nLoading CIC IDS 2017 max_depth=10 experiment...")
    maxdepth_dir = base_path / "experiments" / "cicids2017" / "decision_tree_maxdepth10_experiment"
    maxdepth_results = load_latest_results(maxdepth_dir)
    if maxdepth_results:
        results_data['maxdepth10']['CIC IDS 2017'] = maxdepth_results
        print(f"  ✓ Loaded CIC IDS 2017 max_depth=10 results")
    else:
        print(f"  ✗ No results found for CIC IDS 2017 max_depth=10")
    
    # Check for feature ablation results
    print("\nChecking for feature ablation results...")
    ablation_dir = base_path / "experiments" / "shared" / "feature_ablation"
    if ablation_dir.exists():
        ablation_files = list(ablation_dir.glob("feature_ablation_*.json"))
        for ablation_file in ablation_files:
            try:
                with open(ablation_file, 'r') as f:
                    ablation_data = json.load(f)
                    dataset_name = ablation_file.stem.replace('feature_ablation_', '')
                    results_data['feature_ablation'][dataset_name] = ablation_data
                    print(f"  ✓ Loaded feature ablation results for {dataset_name}")
            except Exception as e:
                print(f"  ✗ Error loading {ablation_file}: {e}")
    
    if not any(results_data.values()):
        print("\nNo results found. Please run experiments first.")
        return
    
    # Generate markdown report
    markdown_report = generate_markdown_report(results_data, base_path)
    
    # Save markdown report
    report_file = base_path / "EXPERIMENT_SUMMARY_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(markdown_report)
    print(f"\n✓ Markdown report saved to: {report_file}")
    
    # Save JSON summary
    summary_json = {
        'generated_at': datetime.now().isoformat(),
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
