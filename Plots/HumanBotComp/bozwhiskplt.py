import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Load your CSV file
df = pd.read_csv('MoDataV1_5-4_tiles_MoDataV1.csv_sums_CSV.csv')

# Create bins of size 5 for 'AverageSum'
bin_edges = np.arange(df['AverageSum'].min(), df['AverageSum'].max() + 6, 5)
df['EggBin'] = pd.cut(df['AverageSum'], bins=bin_edges, right=False)

# Prepare data for plotting
grouped = df.groupby('EggBin')
human_counts_by_bin = [group[['PrimarySum', 'SecondarySum']].values.flatten() for _, group in grouped]
bot_counts_by_bin = [group['BotSum'].values for _, group in grouped]
correct_counts_by_bin = [group['AverageSum'].values for _, group in grouped]

bin_labels = [str((float(str(interval).split(',')[0][1:]) + 
              float(str(interval).split(',')[1][:-1]))//2) for interval in 
              grouped.groups.keys()]
positions = np.arange(len(bin_labels))

plt.figure(figsize=(12, 7))
plt.rc('font', size=20)

# Boxplot for human counts per bin
plt.boxplot(human_counts_by_bin, positions=positions - 0.2, widths=0.3, patch_artist=True,
            boxprops=dict(facecolor='deepskyblue', color='deepskyblue'),
            medianprops=dict(color='orange'), labels=bin_labels)

# Boxplot for bot counts per bin
plt.boxplot(bot_counts_by_bin, positions=positions + 0.2, widths=0.3, patch_artist=True,
            boxprops=dict(facecolor='lightcoral', color='lightcoral'),
            medianprops=dict(color='red'))

plt.xlabel('Correct Egg Count Bins')
plt.ylabel('Egg Count')
plt.title('Comparison of Human Counts and Model Predictions by Correct(Average) Egg Count Bins')
plt.xticks(positions, bin_labels, rotation=45)

legend_handles = [
    Patch(facecolor='deepskyblue', edgecolor='deepskyblue', label='Human Count'),
    Patch(facecolor='lightcoral', edgecolor='lightcoral', label='Model Prediction')
]
plt.legend(handles=legend_handles, loc='upper left')

plt.tight_layout()
plt.savefig("bw5_54.png")
plt.show()
