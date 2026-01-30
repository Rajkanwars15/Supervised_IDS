# Supervised IDS - SensorNetGuard Data Analysis

This project analyzes the SensorNetGuard dataset for intrusion detection system (IDS) research.

## Setup

### 1. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Data

The dataset is located in `data/sensornetguard_data.csv` and contains network sensor data with a binary classification target `Is_Malicious` (0 for benign, 1 for malicious).

## Scripts

### Distribution Plots

Two types of distribution plots are available:

#### 1. Overlay Distribution Plots

Generate overlay histograms showing both classes on the same plot:

```bash
python experiments/sensornetguard/overlay_plots/generate_overlay_distributions.py
```

- Overlays benign (green) and malicious (red) distributions
- Uses probability density normalization
- Saves plots in `experiments/sensornetguard/overlay_plots/figs/`

#### 2. Bin-Based Distribution Plots

Generate histograms where each bin is colored by majority class:

```bash
python experiments/sensornetguard/bin_plots/generate_bin_distributions.py
```

- Green bins: >50% benign data points
- Red bins: ≤50% benign (≥50% malicious) data points
- Shows which value ranges are predominantly benign vs malicious
- Saves plots in `experiments/sensornetguard/bin_plots/figs/`

Both scripts:
- Use 100 bins for detailed distributions
- Save plots as both PNG and SVG
- Include progress tracking with tqdm

### Decision Tree Experiment

Train a decision tree classifier with comprehensive logging:

```bash
python experiments/sensornetguard/scripts/decision_tree_experiment.py
```

This script will:
- Train a decision tree with no depth limit (allows full convergence)
- Log all splits, thresholds, and feature selections
- Calculate comprehensive ML metrics:
  - Accuracy, Precision, Recall, F1 Score
  - ROC-AUC, Average Precision
  - Confusion Matrix
  - Classification Report
- Save experiment logs to `experiments/sensornetguard/decision_tree_experiment/logs/`
- Save results (JSON format) to `experiments/sensornetguard/decision_tree_experiment/results/`
- Extract and log full tree structure

## Project Structure

```
Supervised_IDS/
├── data/
│   └── sensornetguard_data.csv
├── experiments/
│   └── sensornetguard/
│       ├── overlay_plots/          # Overlay distribution plots
│       │   ├── figs/               # Generated overlay plots
│       │   └── generate_overlay_distributions.py
│       ├── bin_plots/              # Bin-based distribution plots
│       │   ├── figs/               # Generated bin plots
│       │   └── generate_bin_distributions.py
│       ├── decision_tree_experiment/  # Decision tree experiment
│       │   ├── logs/               # Experiment logs
│       │   └── results/            # Results (JSON)
│       └── scripts/                 # Analysis scripts
│           └── decision_tree_experiment.py
├── .venv/                          # Virtual environment
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Requirements

- Python 3.8+
- pandas
- plotly
- kaleido (for image export)
- numpy
- scikit-learn
- tqdm