"""
Master script to run all decision tree experiments

This script orchestrates:
1. Preprocessing (if needed) for Farm-Flow and CIC IDS 2017
2. Decision tree experiments for all datasets:
   - SensorNetGuard
   - Farm-Flow
   - CIC IDS 2017

Requirements:
    pip install -r requirements.txt
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def run_script(script_path, description):
    """Run a Python script and return success status"""
    print(f"\n{Colors.OKBLUE}{'─'*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}Running: {description}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'─'*80}{Colors.ENDC}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        print_success(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print_error(f"Script not found: {script_path}")
        return False
    except Exception as e:
        print_error(f"Error running {description}: {str(e)}")
        return False

def check_preprocessing_needed():
    """Check if preprocessing is needed for datasets"""
    base_path = Path(__file__).parent
    
    # Check Farm-Flow
    farmflow_silver = base_path / "data" / "farm-flow-silver" / "Farm-Flow_Train_Binary.csv"
    farmflow_needs_preprocessing = not farmflow_silver.exists()
    
    # Check CIC IDS 2017
    cicids_silver = base_path / "data" / "cic-ids2017-silver" / "CIC-IDS2017_Train_Binary.csv"
    cicids_needs_preprocessing = not cicids_silver.exists()
    
    return farmflow_needs_preprocessing, cicids_needs_preprocessing

def main():
    """Main function to run all experiments"""
    start_time = datetime.now()
    base_path = Path(__file__).parent
    
    print_header("Supervised IDS - All Experiments Runner")
    
    print_info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Working directory: {base_path}")
    
    # Track results
    results = {
        'start_time': start_time.isoformat(),
        'preprocessing': {},
        'experiments': {},
        'summary': {}
    }
    
    # Check preprocessing needs
    print_header("Checking Preprocessing Requirements")
    farmflow_needs, cicids_needs = check_preprocessing_needed()
    
    if farmflow_needs:
        print_warning("Farm-Flow preprocessing needed")
    else:
        print_success("Farm-Flow data already preprocessed")
    
    if cicids_needs:
        print_warning("CIC IDS 2017 preprocessing needed")
    else:
        print_success("CIC IDS 2017 data already preprocessed")
    
    # Run preprocessing if needed
    if farmflow_needs or cicids_needs:
        print_header("Preprocessing Datasets")
        
        if farmflow_needs:
            preprocess_farmflow = base_path / "experiments" / "farmflow" / "scripts" / "preprocess_farmflow.py"
            success = run_script(preprocess_farmflow, "Farm-Flow Preprocessing")
            results['preprocessing']['farmflow'] = success
            if not success:
                print_warning("Farm-Flow preprocessing failed. Experiment may fail.")
        
        if cicids_needs:
            preprocess_cicids = base_path / "experiments" / "cicids2017" / "scripts" / "preprocess_cicids2017.py"
            success = run_script(preprocess_cicids, "CIC IDS 2017 Preprocessing")
            results['preprocessing']['cicids2017'] = success
            if not success:
                print_warning("CIC IDS 2017 preprocessing failed. Experiment may fail.")
    else:
        print_info("All datasets already preprocessed. Skipping preprocessing step.")
        results['preprocessing']['farmflow'] = True
        results['preprocessing']['cicids2017'] = True
    
    # Run experiments
    print_header("Running Decision Tree Experiments")
    
    experiments = [
        {
            'name': 'SensorNetGuard',
            'script': base_path / "experiments" / "sensornetguard" / "scripts" / "decision_tree_experiment.py",
            'description': 'SensorNetGuard Decision Tree Experiment'
        },
        {
            'name': 'Farm-Flow',
            'script': base_path / "experiments" / "farmflow" / "scripts" / "decision_tree_experiment.py",
            'description': 'Farm-Flow Decision Tree Experiment'
        },
        {
            'name': 'CIC IDS 2017',
            'script': base_path / "experiments" / "cicids2017" / "scripts" / "decision_tree_experiment.py",
            'description': 'CIC IDS 2017 Decision Tree Experiment'
        }
    ]
    
    for exp in tqdm(experiments, desc="Running experiments", unit="experiment"):
        success = run_script(exp['script'], exp['description'])
        results['experiments'][exp['name']] = success
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("Experiment Summary")
    
    print(f"\n{Colors.BOLD}Preprocessing Results:{Colors.ENDC}")
    for dataset, success in results['preprocessing'].items():
        status = f"{Colors.OKGREEN}✓ Success{Colors.ENDC}" if success else f"{Colors.FAIL}✗ Failed{Colors.ENDC}"
        print(f"  {dataset}: {status}")
    
    print(f"\n{Colors.BOLD}Experiment Results:{Colors.ENDC}")
    for exp_name, success in results['experiments'].items():
        status = f"{Colors.OKGREEN}✓ Success{Colors.ENDC}" if success else f"{Colors.FAIL}✗ Failed{Colors.ENDC}"
        print(f"  {exp_name}: {status}")
    
    # Calculate statistics
    total_experiments = len(results['experiments'])
    successful_experiments = sum(1 for s in results['experiments'].values() if s)
    failed_experiments = total_experiments - successful_experiments
    
    results['summary'] = {
        'total_experiments': total_experiments,
        'successful': successful_experiments,
        'failed': failed_experiments,
        'duration_seconds': duration.total_seconds(),
        'duration_formatted': str(duration),
        'end_time': end_time.isoformat()
    }
    
    print(f"\n{Colors.BOLD}Overall Statistics:{Colors.ENDC}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Successful: {Colors.OKGREEN}{successful_experiments}{Colors.ENDC}")
    print(f"  Failed: {Colors.FAIL}{failed_experiments}{Colors.ENDC}")
    print(f"  Duration: {duration}")
    print(f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results
    results_file = base_path / "experiment_results_summary.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print_success(f"Results summary saved to: {results_file}")
    
    # Final status
    print_header("All Experiments Complete")
    
    if failed_experiments == 0:
        print_success("All experiments completed successfully!")
        return 0
    else:
        print_warning(f"{failed_experiments} experiment(s) failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠ Experiments interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
