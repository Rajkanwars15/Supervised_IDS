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
        },
        'UNSW-NB15': {
            'path': 'experiments/unsw-nb15',
            'name': 'UNSW-NB15 Binary Classification',
            'exp_dir': 'decision_tree_experiment'
        },
        'NSL-KDD': {
            'path': 'experiments/nsl-kdd',
            'name': 'NSL-KDD Binary Classification',
            'exp_dir': 'decision_tree_experiment'
        },
        'CIC IOV 2024': {
            'path': 'experiments/cic-iov-2024',
            'name': 'CIC IOV 2024 Binary Classification',
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
                report.append(f"- Test Precision: {perf.get('test_precision', 0):.4f}")
                report.append(f"- Test Recall: {perf.get('test_recall', 0):.4f}")
                report.append(f"- Test F1: {perf.get('test_f1', 0):.4f}")
                report.append(f"- Test ROC-AUC: {perf.get('test_roc_auc', 0):.4f}")
                report.append("")
            
            # Noise impact plot and table
            if dataset_name == 'SensorNetGuard':
                stump_exp_dir = base_path / "experiments" / "sensornetguard" / "decision_stump_experiment"
            else:
                stump_exp_dir = base_path / "experiments" / "farmflow" / "decision_stump_experiment"
            
            noise_plot = find_latest_plot(stump_exp_dir / "figs", "noise_impact_*.png")
            if noise_plot:
                rel_path = get_relative_path(noise_plot, base_path)
                report.append(f"**Noise Impact Analysis (Perturbation Study):**\n")
                report.append(f"![{dataset_name} Noise Impact]({rel_path})\n")
                report.append("")
            
            # Noise experiment results table
            if 'noise_experiment' in stump_data and stump_data['noise_experiment']:
                report.append("**Performance vs Noise Level:**\n")
                report.append("| Noise Level (std) | Accuracy | Precision | Recall | F1 Score |")
                report.append("|-------------------|----------|-----------|--------|----------|")
                for noise_result in stump_data['noise_experiment']:
                    report.append(f"| {noise_result.get('noise_level', 0):.2f} | "
                                 f"{noise_result.get('accuracy', 0):.4f} | "
                                 f"{noise_result.get('precision', 0):.4f} | "
                                 f"{noise_result.get('recall', 0):.4f} | "
                                 f"{noise_result.get('f1', 0):.4f} |")
                report.append("")
            
            # Cross-validation results
            if 'cross_validation' in stump_data:
                cv = stump_data['cross_validation']
                report.append(f"**Cross-Validation (5-fold):**\n")
                report.append(f"- Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} (±{cv.get('std_accuracy', 0):.4f})")
                report.append(f"- Mean F1: {cv.get('mean_f1', 0):.4f} (±{cv.get('std_f1', 0):.4f})")
                report.append("")
        else:
            # Show that experiment hasn't been run
            report.append(f"### {dataset_name} Decision Stump\n")
            report.append("*Experiment not yet run. Results will appear here after execution.*\n")
            report.append("")
    
    # Depth-Limited Experiments (max_depth=10)
    report.append("## 3. Depth-Limited Experiments (max_depth=10)\n")
    report.append("Comparison of unlimited depth vs max_depth=10 for datasets with trees > 10 depth.\n\n")
    
    maxdepth_datasets = results_data.get('maxdepth10', {})
    if maxdepth_datasets:
        for dataset_name in sorted(maxdepth_datasets.keys()):
            maxdepth_data = maxdepth_datasets[dataset_name]
            report.append(f"### {dataset_name}\n")
            report.append("Comparison of unlimited depth vs max_depth=10:\n\n")
            
            # Get original results for comparison
            original = results_data.get('decision_trees', {}).get(dataset_name, {})
            
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
    else:
        report.append("*No depth-limited experiments found. These are created automatically for trees with depth > 10.*\n\n")
    
    # Feature Ablation Experiments
    report.append("## 4. Feature Ablation Experiments\n")
    report.append("These experiments show how performance changes as features are removed, starting from lowest importance.\n\n")
    
    ablation_dir = base_path / "experiments" / "shared" / "feature_ablation"
    ablation_found = False
    
    if ablation_dir.exists():
        ablation_plots = list(ablation_dir.glob("feature_ablation_*.png"))
        ablation_jsons = list(ablation_dir.glob("feature_ablation_*.json"))
        
        if ablation_plots or ablation_jsons:
            ablation_found = True
            # Match plots with JSON files
            for json_file in sorted(ablation_jsons):
                dataset_name_raw = json_file.stem.replace('feature_ablation_', '')
                # Handle special cases for dataset name formatting
                    dataset_name_map = {
                        'nsl-kdd': 'NSL-KDD',
                        'cic_ids_2017': 'CIC IDS 2017',
                        'unsw-nb15': 'UNSW-NB15',
                        'farm-flow': 'Farm-Flow',
                        'sensornetguard': 'SensorNetGuard',
                        'cic_iov_2024': 'CIC IOV 2024',
                        'cic-iov-2024': 'CIC IOV 2024'
                    }
                dataset_name = dataset_name_map.get(dataset_name_raw, dataset_name_raw.replace('_', ' ').title())
                report.append(f"### {dataset_name}\n")
                
                # Load JSON for summary stats and detailed table
                try:
                    with open(json_file, 'r') as f:
                        ablation_data = json.load(f)
                        summary = ablation_data.get('summary', {})
                        ablation_results = ablation_data.get('ablation_results', [])
                        
                        if summary:
                            report.append(f"- **Max Features Tested**: {summary.get('max_features', 'N/A')}")
                            report.append(f"- **Min Features Tested**: {summary.get('min_features_tested', 'N/A')}")
                            report.append(f"- **Best Accuracy**: {summary.get('best_accuracy', 0):.4f}")
                            report.append(f"- **Best F1 Score**: {summary.get('best_f1', 0):.4f}")
                            report.append(f"- **Best ROC-AUC**: {summary.get('best_roc_auc', 0):.4f}")
                            report.append("")
                        
                        # Create detailed table showing performance at different feature counts
                        if ablation_results and len(ablation_results) > 0:
                            report.append("**Performance vs Number of Features:**\n")
                            report.append("| Features | Accuracy | Precision | Recall | F1 Score | ROC-AUC |")
                            report.append("|----------|----------|-----------|--------|----------|---------|")
                            # Show first, last, and every Nth result to keep table manageable
                            step = max(1, len(ablation_results) // 15)  # Show ~15 rows max
                            for i, result in enumerate(ablation_results):
                                if i == 0 or i == len(ablation_results) - 1 or i % step == 0:
                                    report.append(f"| {result.get('num_features', 0)} | "
                                                 f"{result.get('accuracy', 0):.4f} | "
                                                 f"{result.get('precision', 0):.4f} | "
                                                 f"{result.get('recall', 0):.4f} | "
                                                 f"{result.get('f1', 0):.4f} | "
                                                 f"{result.get('roc_auc', 0):.4f} |")
                            report.append("")
                except Exception as e:
                    print(f"Error loading ablation JSON {json_file}: {e}")
                
                # Find matching plot
                plot_file = ablation_dir / f"feature_ablation_{json_file.stem.replace('feature_ablation_', '')}.png"
                if plot_file.exists():
                    rel_path = get_relative_path(plot_file, base_path)
                    report.append(f"![{dataset_name} Feature Ablation]({rel_path})\n")
                    report.append("")
                else:
                    # Try to find any plot with similar name
                    matching_plots = [p for p in ablation_plots if json_file.stem.replace('feature_ablation_', '') in p.stem]
                    if matching_plots:
                        rel_path = get_relative_path(matching_plots[0], base_path)
                        report.append(f"![{dataset_name} Feature Ablation]({rel_path})\n")
                        report.append("")
            
            # Also show any plots without JSON
            for plot_file in sorted(ablation_plots):
                dataset_name = plot_file.stem.replace('feature_ablation_', '').replace('_', ' ').title()
                # Check if we already added this
                json_exists = any(json_file.stem.replace('feature_ablation_', '') in plot_file.stem 
                                for json_file in ablation_jsons)
                if not json_exists:
                    rel_path = get_relative_path(plot_file, base_path)
                    report.append(f"### {dataset_name}\n")
                    report.append(f"![{dataset_name} Feature Ablation]({rel_path})\n")
                    report.append("")
    
    if not ablation_found:
        report.append("*Feature ablation experiments not yet run. Results will appear here after execution.*\n")
        report.append("")
        report.append("To run: `python experiments/shared/feature_ablation_experiment.py`\n")
        report.append("")
    
    # Model Comparison Experiments
    report.append("## 5. Comprehensive Model Comparison\n")
    report.append("Comparison of baseline models (decision stumps, shallow trees) against ensemble methods (Random Forest, XGBoost) and neural networks (MLP), plus linear models (Logistic Regression, Linear SVM).\n\n")
    
    model_comp_data = results_data.get('model_comparison', {})
    if model_comp_data:
        for dataset_name in sorted(model_comp_data.keys()):
            comp_data = model_comp_data[dataset_name]
            report.append(f"### {dataset_name}\n")
            
            model_results = comp_data.get('model_results', [])
            successful_results = [r for r in model_results if r.get('status') == 'success']
            
            if successful_results:
                report.append("| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |")
                report.append("|-------|---------------|-----------------|-------------|---------|--------------|")
                
                for result in sorted(successful_results, key=lambda x: x.get('test_accuracy', 0), reverse=True):
                    report.append(f"| {result.get('model_name', 'N/A')} | "
                                 f"{result.get('test_accuracy', 0):.4f} | "
                                 f"{result.get('test_precision', 0):.4f} | "
                                 f"{result.get('test_recall', 0):.4f} | "
                                 f"{result.get('test_f1', 0):.4f} | "
                                 f"{result.get('test_roc_auc', 0):.4f} |")
                report.append("")
                
                # Find and add plot
                model_comp_dir = base_path / "experiments" / "shared" / "model_comparison"
                plot_file = model_comp_dir / f"model_comparison_{dataset_name.lower().replace(' ', '_')}.png"
                if plot_file.exists():
                    rel_path = get_relative_path(plot_file, base_path)
                    report.append(f"![{dataset_name} Model Comparison]({rel_path})\n")
                    report.append("")
            else:
                report.append("*No successful model results found.*\n\n")
    else:
        report.append("*Model comparison experiments not yet run. Results will appear here after execution.*\n")
        report.append("")
        report.append("To run: `python experiments/shared/model_comparison_experiment.py`\n")
        report.append("")
    
    # Top features
    report.append("## 6. Top 10 Most Important Features (by Dataset)\n")
    
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
    report.append("## 7. Confusion Matrices (Test Set)\n")
    
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
        'feature_ablation': {},
        'model_comparison': {}
    }
    
    # Load main decision tree experiments
    print("\nLoading main decision tree experiments...")
    main_experiments = {
        'SensorNetGuard': base_path / "experiments" / "sensornetguard" / "decision_tree_experiment",
        'Farm-Flow': base_path / "experiments" / "farmflow" / "decision_tree_experiment",
        'CIC IDS 2017': base_path / "experiments" / "cicids2017" / "decision_tree_experiment",
        'UNSW-NB15': base_path / "experiments" / "unsw-nb15" / "decision_tree_experiment",
        'NSL-KDD': base_path / "experiments" / "nsl-kdd" / "decision_tree_experiment",
        'CIC IOV 2024': base_path / "experiments" / "cic-iov-2024" / "decision_tree_experiment"
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
        'Farm-Flow': base_path / "experiments" / "farmflow" / "decision_stump_experiment",
        'UNSW-NB15': base_path / "experiments" / "unsw-nb15" / "decision_stump_experiment",
        'NSL-KDD': base_path / "experiments" / "nsl-kdd" / "decision_stump_experiment",
        'CIC IOV 2024': base_path / "experiments" / "cic-iov-2024" / "decision_stump_experiment"
    }
    
    for name, exp_dir in stump_experiments.items():
        results = load_latest_results(exp_dir)
        if results:
            results_data['decision_stumps'][name] = results
            print(f"  ✓ Loaded {name} decision stump results")
        else:
            print(f"  ✗ No results found for {name}")
    
    # Load max_depth=10 experiments for all datasets
    print("\nLoading max_depth=10 experiments...")
    maxdepth_experiments = {
        'SensorNetGuard': base_path / "experiments" / "sensornetguard" / "decision_tree_maxdepth10_experiment",
        'Farm-Flow': base_path / "experiments" / "farmflow" / "decision_tree_maxdepth10_experiment",
        'CIC IDS 2017': base_path / "experiments" / "cicids2017" / "decision_tree_maxdepth10_experiment",
        'UNSW-NB15': base_path / "experiments" / "unsw-nb15" / "decision_tree_maxdepth10_experiment",
        'NSL-KDD': base_path / "experiments" / "nsl-kdd" / "decision_tree_maxdepth10_experiment",
        'CIC IOV 2024': base_path / "experiments" / "cic-iov-2024" / "decision_tree_maxdepth10_experiment"
    }
    
    for name, exp_dir in maxdepth_experiments.items():
        maxdepth_results = load_latest_results(exp_dir)
        if maxdepth_results:
            results_data['maxdepth10'][name] = maxdepth_results
            print(f"  ✓ Loaded {name} max_depth=10 results")
        else:
            print(f"  ✗ No results found for {name} max_depth=10")
    
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
    
    # Load model comparison results
    print("\nChecking for model comparison results...")
    model_comp_dir = base_path / "experiments" / "shared" / "model_comparison"
    if model_comp_dir.exists():
        model_comp_files = list(model_comp_dir.glob("model_comparison_*.json"))
        for comp_file in model_comp_files:
            try:
                with open(comp_file, 'r') as f:
                    comp_data = json.load(f)
                    dataset_name = comp_data.get('dataset', comp_file.stem.replace('model_comparison_', ''))
                    results_data['model_comparison'][dataset_name] = comp_data
                    print(f"  ✓ Loaded model comparison results for {dataset_name}")
            except Exception as e:
                print(f"  ✗ Error loading {comp_file}: {e}")
    
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
