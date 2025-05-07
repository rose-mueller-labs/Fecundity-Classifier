import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv('MoDataV1_5-4_tiles_MoDataV1.csv_sums_CSV.csv')

# Define bins for AverageSum
bin_edges = np.arange(df['AverageSum'].min(), df['AverageSum'].max() + 6, 5)
df['EggBin'] = pd.cut(df['AverageSum'], bins=bin_edges, right=False)

# Group by bins
grouped = df.groupby('EggBin')

# Prepare data for plotting
human_diffs_by_bin = [group[['PrimaryDiff', 'SecondaryDiff']].abs().values.flatten() for _, group in grouped]
bot_diffs_by_bin = [group['BotDiff'].abs().values for _, group in grouped]

bin_labels = [str(interval) for interval in grouped.groups.keys()]
positions = np.arange(len(bin_labels))

plt.figure(figsize=(14, 7))

# Boxplots for human diffs (shifted left)
plt.boxplot(human_diffs_by_bin, positions=positions - 0.15, widths=0.25, patch_artist=True,
            boxprops=dict(facecolor='deepskyblue', color='deepskyblue'),
            medianprops=dict(color='orange'), labels=bin_labels)

# Boxplots for bot diffs (shifted right)
plt.boxplot(bot_diffs_by_bin, positions=positions + 0.15, widths=0.25, patch_artist=True,
            boxprops=dict(facecolor='lightcoral', color='lightcoral'),
            medianprops=dict(color='red'))

plt.xlabel('Average Egg Count Bins')
plt.ylabel('Absolute Difference')
plt.title('Absolute Differences by Bin: Human (Primary & Secondary) vs Bot')
plt.xticks(positions, bin_labels, rotation=45)
plt.legend(['Human (Primary & Secondary)', 'Bot'], loc='upper left')

plt.tight_layout()
plt.savefig('abs_diff_bp1_54.png', dpi=200)
plt.show()