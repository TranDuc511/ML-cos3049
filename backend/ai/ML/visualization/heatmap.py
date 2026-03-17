import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Load the dataset directly
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_2', 'data_encoded.json')
df = pd.read_json(DATA_PATH)

# 2. Isolate numerical columns and filter out 'ID' columns
numerical_cols = df.select_dtypes(include='number')
cols_to_keep = [c for c in numerical_cols.columns if 'id' not in c.lower()]
df_numeric = numerical_cols[cols_to_keep]

# 3. Compute the correlation matrix
correlation_matrix = df_numeric.corr()

# 4. Set up the canvas and draw the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,          # Show the correlation number
    fmt=".2f",           # Round to 2 decimal places
    cmap="coolwarm",     # Blue = negative, Red = positive
    center=0,            # Center the color scale at 0
    linewidths=0.5,      # Add gridlines for readability
    square=True          # Force cells to be square
)

# 5. Format titles/labels, save, and display
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right') # Tilt x-axis labels so they don't overlap
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), 'heatmap_output.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Heatmap saved to: {output_path}")