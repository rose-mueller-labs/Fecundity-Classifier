import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = "MoDataV1_5-4_tiles_MoDataV1.csv_sums_CSV.csv"

df = pd.read_csv(data)

# bin avg sum in intervals of size 5
bin_size = 5
min_bin = np.floor(df['AverageSum'].min() / bin_size) * bin_size
max_bin = np.ceil(df['AverageSum'].max() / bin_size) * bin_size + bin_size
bin_edges = np.arange(min_bin, max_bin, bin_size)
df['AvgSum_bin'] = pd.cut(df['AverageSum'], bins=bin_edges, right=False)

# fcn to compute 95% CI !!
def ci95(series):
    n = series.count()
    if n < 2:
        return 0
    return 1.96 * series.std(ddof=1) / np.sqrt(n)

# group by bins and calculate mean and CI for each group (which is what we graph)
grouped = df.groupby('AvgSum_bin').agg(
    PrimarySum_mean=('PrimarySum', 'mean'),
    PrimarySum_ci=('PrimarySum', ci95),
    SecondarySum_mean=('SecondarySum', 'mean'),
    SecondarySum_ci=('SecondarySum', ci95),
    BotSum_mean=('BotSum', 'mean'),
    BotSum_ci=('BotSum', ci95),
    bin_center=('AverageSum', 'mean')  # use the mean of AverageSum in each bin as x
).reset_index()

# plot
plt.figure(figsize=(8, 5))
plt.rc('font', size=14)

plt.errorbar(grouped['bin_center'], grouped['PrimarySum_mean'], yerr=grouped['PrimarySum_ci'],
             fmt='o-', label='Human 1 Count')
plt.errorbar(grouped['bin_center'], grouped['SecondarySum_mean'], yerr=grouped['SecondarySum_ci'],
             fmt='s-', label='Human 2 Count')
plt.errorbar(grouped['bin_center'], grouped['BotSum_mean'], yerr=grouped['BotSum_ci'],
             fmt='^-', label='Model Predicted Count')

plt.xlabel("Humans' Average Count")
plt.ylabel('Egg Count')
plt.title('Human and Model Counts with 95% Confidence Interval')
plt.legend()
plt.tight_layout()
plt.savefig("ci3.png")