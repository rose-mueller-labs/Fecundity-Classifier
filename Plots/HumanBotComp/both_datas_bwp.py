import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Load both CSV files
df1 = pd.read_csv('MoDataV1_5-1_tiles_MoDataV1.csv_sums_CSV.csv')
df2 = pd.read_csv('MoDataV1_5-4_tiles_MoDataV1.csv_sums_CSV.csv')

# Define bin edges using the combined min/max so bins are consistent
all_avgs = pd.concat([df1['AverageSum'], df2['AverageSum']])
bin_edges = np.arange(all_avgs.min(), all_avgs.max() + 6, 5)

# Bin and group both DataFrames
df1['EggBin'] = pd.cut(df1['AverageSum'], bins=bin_edges, right=False)
df2['EggBin'] = pd.cut(df2['AverageSum'], bins=bin_edges, right=False)

grouped1 = df1.groupby('EggBin')
grouped2 = df2.groupby('EggBin')

# Prepare data for plotting
human_counts_by_bin_1 = [group[['PrimarySum', 'SecondarySum']].values.flatten() for _, group in grouped1]
bot_counts_by_bin_1 = [group['BotSum'].values for _, group in grouped1]

human_counts_by_bin_2 = [group[['PrimarySum', 'SecondarySum']].values.flatten() for _, group in grouped2]
bot_counts_by_bin_2 = [group['BotSum'].values for _, group in grouped2]

bin_labels = [str(interval) for interval in grouped1.groups.keys()]
positions = np.arange(len(bin_labels))

plt.figure(figsize=(14, 7))

# Boxplots for first CSV (shifted a bit left)
plt.boxplot(human_counts_by_bin_1, positions=positions - 0.3, widths=0.2, patch_artist=True,
            boxprops=dict(facecolor='deepskyblue', color='deepskyblue'),
            medianprops=dict(color='orange'))
plt.boxplot(bot_counts_by_bin_1, positions=positions - 0.1, widths=0.2, patch_artist=True,
            boxprops=dict(facecolor='lightcoral', color='lightcoral'),
            medianprops=dict(color='red'))

# Boxplots for second CSV (shifted a bit right)
plt.boxplot(human_counts_by_bin_2, positions=positions + 0.1, widths=0.2, patch_artist=True,
            boxprops=dict(facecolor='dodgerblue', color='dodgerblue'),
            medianprops=dict(color='gold'))
plt.boxplot(bot_counts_by_bin_2, positions=positions + 0.3, widths=0.2, patch_artist=True,
            boxprops=dict(facecolor='salmon', color='salmon'),
            medianprops=dict(color='darkred'))

plt.xlabel('Correct Egg Count Bins')
plt.ylabel('Egg Count')
plt.title('Comparison of Human Counts and Bot Predictions by Correct Egg Count Bins')
plt.xticks(positions, bin_labels, rotation=45)

legend_handles = [
    Patch(facecolor='deepskyblue', edgecolor='deepskyblue', label='Human Count CSV1'),
    Patch(facecolor='lightcoral', edgecolor='lightcoral', label='Bot Prediction CSV1'),
    Patch(facecolor='dodgerblue', edgecolor='dodgerblue', label='Human Count CSV2'),
    Patch(facecolor='salmon', edgecolor='salmon', label='Bot Prediction CSV2')
]
plt.legend(handles=legend_handles, loc='upper left')

plt.tight_layout()
plt.savefig("bw5_51_54.png")
plt.show()