import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt

# 1. Load the raw, human-readable data directly
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_2', 'data.json')
df = pd.read_json(DATA_PATH)

# 2. Set up a 1x3 grid for our charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Boxplot Analysis - Data Distribution & Outliers", fontsize=16, fontweight='bold')

# 3. Plot 1: Transaction Amount by Type
sns.boxplot(data=df, x='Transaction Detail', y='Transaction amount', hue='Transaction Detail', ax=axes[0], palette='Set2', legend=False)
axes[0].set_title("Amount by Transaction Type")
axes[0].tick_params(axis='x', rotation=40)

# 4. Plot 2: Transaction Amount by Gender
sns.boxplot(data=df, x='Gender', y='Transaction amount', hue='Gender', ax=axes[1], palette='pastel', legend=False)
axes[1].set_title("Amount by Gender")

# 5. Plot 3: Account Balance by Working Status
sns.boxplot(data=df, x='Working Status', y='Account balance', hue='Working Status', ax=axes[2], palette='muted', legend=False)
axes[2].set_title("Balance by Working Status")
axes[2].tick_params(axis='x', rotation=40)

# 6. Save the image and display it
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'boxplot_output.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Boxplot saved to: {output_path}")