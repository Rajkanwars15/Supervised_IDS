"""
Cleanup script to remove old experiment results

This script removes:
- Old experiment logs
- Old result JSON files
- Old model artifacts
- Old visualizations

Keeps only the most recent run for each experiment.

Usage:
    python cleanup_results.py [--keep-all] [--dry-run]
"""

import argparse
from pathlib import Path
from datetime import datetime
import shutil

def cleanup_experiment_dir(exp_dir, keep_all=False, dry_run=False):
    """Cleanup a single experiment directory"""
    if not exp_dir.exists():
        return 0
    
    removed = 0
    
    # Directories to clean
    dirs_to_clean = ['logs', 'results', 'models', 'visualizations']
    
    for subdir_name in dirs_to_clean:
        subdir = exp_dir / subdir_name
        if not subdir.exists():
            continue
        
        if keep_all:
            # Keep all files, just remove if dry_run
            if dry_run:
                files = list(subdir.glob('*'))
                print(f"  Would keep {len(files)} files in {subdir_name}/")
            continue
        
        # Get all files sorted by modification time (newest first)
        files = sorted(subdir.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if len(files) > 1:
            # Keep the newest, remove the rest
            newest = files[0]
            to_remove = files[1:]
            
            if dry_run:
                print(f"  Would remove {len(to_remove)} old files from {subdir_name}/")
                print(f"  Would keep: {newest.name}")
            else:
                for file in to_remove:
                    try:
                        if file.is_file():
                            file.unlink()
                            removed += 1
                        elif file.is_dir():
                            shutil.rmtree(file)
                            removed += 1
                    except Exception as e:
                        print(f"    Warning: Could not remove {file}: {e}")
        elif len(files) == 1:
            if dry_run:
                print(f"  Would keep only file in {subdir_name}/: {files[0].name}")
    
    return removed

def main():
    parser = argparse.ArgumentParser(description='Cleanup old experiment results')
    parser.add_argument('--keep-all', action='store_true', 
                       help='Keep all files (no cleanup)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be removed without actually removing')
    args = parser.parse_args()
    
    base_path = Path(__file__).parent
    experiment_dirs = [
        base_path / "experiments" / "sensornetguard" / "decision_tree_experiment",
        base_path / "experiments" / "farmflow" / "decision_tree_experiment",
        base_path / "experiments" / "cicids2017" / "decision_tree_experiment"
    ]
    
    print("="*80)
    print("Experiment Results Cleanup")
    print("="*80)
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be deleted]")
    
    if args.keep_all:
        print("\n[KEEP ALL MODE - No files will be deleted]")
    
    total_removed = 0
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.parent.name
        print(f"\nProcessing: {exp_name}")
        print("-" * 80)
        
        removed = cleanup_experiment_dir(exp_dir, args.keep_all, args.dry_run)
        total_removed += removed
        
        if not args.dry_run and not args.keep_all:
            print(f"  Removed {removed} old files")
    
    print("\n" + "="*80)
    if args.dry_run:
        print(f"Dry run complete. Would remove {total_removed} files total.")
    elif args.keep_all:
        print("Keep-all mode: No files removed.")
    else:
        print(f"Cleanup complete. Removed {total_removed} old files total.")
    print("="*80)

if __name__ == "__main__":
    main()
