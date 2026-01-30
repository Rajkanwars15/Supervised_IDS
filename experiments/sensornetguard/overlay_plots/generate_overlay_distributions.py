"""
Script to generate overlay distribution plots for all numeric variables
colored by Is_Malicious (green for 0, red for 1) - OVERLAY VERSION

Requirements:
    pip install pandas plotly kaleido tqdm
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
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
figs_dir = Path(__file__).parent / "figs"
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
print(f"\nGenerating overlay distribution plots...")
for col in tqdm(numeric_cols, desc="Creating plots", unit="plot"):
    # Separate data by Is_Malicious
    data_benign = df[df['Is_Malicious'] == 0][col].dropna()
    data_malicious = df[df['Is_Malicious'] == 1][col].dropna()
    
    # Create histogram traces
    fig = go.Figure()
    
    # Benign (green) - Is_Malicious == 0
    fig.add_trace(go.Histogram(
        x=data_benign,
        name='Benign (Is_Malicious=0)',
        nbinsx=n_bins,
        marker_color='green',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # Malicious (red) - Is_Malicious == 1
    fig.add_trace(go.Histogram(
        x=data_malicious,
        name='Malicious (Is_Malicious=1)',
        nbinsx=n_bins,
        marker_color='red',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Distribution of {col} by Is_Malicious (Overlay)',
        xaxis_title=col,
        yaxis_title='Probability Density',
        barmode='overlay',
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
        png_path = figs_dir / f"{col}_overlay_distribution.png"
        fig.write_image(str(png_path), width=1200, height=600, scale=2)
    except Exception as e:
        tqdm.write(f"  Warning: Could not save PNG for {col}: {e}")
    
    # Save as SVG
    try:
        svg_path = figs_dir / f"{col}_overlay_distribution.svg"
        fig.write_image(str(svg_path), width=1200, height=600)
    except Exception as e:
        tqdm.write(f"  Warning: Could not save SVG for {col}: {e}")

print(f"\n✓ All overlay plots generated and saved to {figs_dir}")
