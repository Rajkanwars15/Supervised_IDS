"""
Script to generate distribution plots for all numeric variables
colored by majority class in each bin:
- Green: bin has >50% benign (Is_Malicious=0) data points
- Red: bin has ≤50% benign (i.e., ≥50% malicious) data points

Requirements:
    pip install pandas plotly kaleido tqdm numpy
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# Check for required packages
try:
    import plotly
except ImportError:
    print("Error: plotly is not installed. Please install it with: pip install plotly")
    sys.exit(1)

try:
    import kaleido
except ImportError:
    print("Warning: kaleido is not installed. PNG export may not work.")
    print("Please install it with: pip install kaleido")
    print("SVG export should still work.")

# Set paths
data_path = Path(__file__).parent.parent.parent.parent / "data" / "sensornetguard_data.csv"
figs_dir = Path(__file__).parent.parent / "figs"
figs_dir.mkdir(parents=True, exist_ok=True)

# Read data
print(f"Reading data from {data_path}...")
df = pd.read_csv(data_path)

# Identify numeric columns (exclude Node_ID, Timestamp, IP_Address, and Is_Malicious)
exclude_cols = ['Node_ID', 'Timestamp', 'IP_Address', 'Is_Malicious']
numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

print(f"Found {len(numeric_cols)} numeric columns to plot")
print(f"Columns: {numeric_cols}")

# Set number of bins (quite a lot as requested)
n_bins = 100

# Generate plots for each numeric variable with progress tracking
print(f"\nGenerating distribution plots...")
for col in tqdm(numeric_cols, desc="Creating plots", unit="plot"):
    # Get data (remove NaN values)
    data = df[[col, 'Is_Malicious']].dropna()
    
    # Create bins
    min_val = data[col].min()
    max_val = data[col].max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Assign each data point to a bin
    data['bin'] = pd.cut(data[col], bins=bin_edges, include_lowest=True, labels=False)
    
    # Count data points in each bin by class
    bin_counts = []
    bin_colors = []
    
    for bin_idx in range(n_bins):
        bin_data = data[data['bin'] == bin_idx]
        if len(bin_data) == 0:
            bin_counts.append(0)
            bin_colors.append('gray')  # Empty bins in gray
        else:
            benign_count = len(bin_data[bin_data['Is_Malicious'] == 0])
            total_count = len(bin_data)
            benign_ratio = benign_count / total_count
            
            bin_counts.append(total_count)
            
            # Color based on majority: >50% benign = green, otherwise red
            if benign_ratio > 0.5:
                bin_colors.append('green')
            else:
                bin_colors.append('red')
    
    # Create histogram with custom colors
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=bin_counts,
        width=bin_width * 0.9,  # Slight gap between bars
        marker_color=bin_colors,
        opacity=0.7,
        name='Distribution',
        hovertemplate='<b>%{x:.2f}</b><br>Count: %{y}<br>Bin: %{customdata}<extra></extra>',
        customdata=[f"Bin {i+1}" for i in range(n_bins)]
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Distribution of {col} by Is_Malicious (Green: >50% Benign, Red: ≤50% Benign)',
        xaxis_title=col,
        yaxis_title='Count',
        legend=dict(
            x=0.7,
            y=0.95,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        template='plotly_white',
        width=1200,
        height=600
    )
    
    # Save as PNG
    try:
        png_path = figs_dir / f"{col}_distribution.png"
        fig.write_image(str(png_path), width=1200, height=600, scale=2)
    except Exception as e:
        tqdm.write(f"  Warning: Could not save PNG for {col}: {e}")
    
    # Save as SVG
    try:
        svg_path = figs_dir / f"{col}_distribution.svg"
        fig.write_image(str(svg_path), width=1200, height=600)
    except Exception as e:
        tqdm.write(f"  Warning: Could not save SVG for {col}: {e}")

print(f"\n✓ All plots generated and saved to {figs_dir}")
