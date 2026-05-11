import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

# data = "Alex_4-30_5-1_CC_A_v0.0_sums_COMPLETE_CD.csv"
data = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_lithium_5-4_results/Alex_5-1_5-2S_v0.0_sums__lith54_CSV.csv"
df = pd.read_csv(data)

#       CD_RootImage  BotSum  HumanSum
# 5918  3 4 D4 19.jpg      74     296.0
# 6808  3 2 C4 37.jpg      55     465.0

# bin avg sum in intervals of size 5
bin_size = 5
min_bin = np.floor(df['HumanSum'].min() / bin_size) * bin_size
max_bin = np.ceil(df['HumanSum'].max() / bin_size) * bin_size + bin_size
bin_edges = np.arange(min_bin, max_bin, bin_size)
df['HumanSum_bin'] = pd.cut(df['HumanSum'], bins=bin_edges, right=False)

# fcn to compute 95% CI !!
def ci95(series):
    n = series.count()
    if n < 2:
        return 0
    return 1.96 * series.std(ddof=1) / np.sqrt(n)

# group by bins and calculate mean and CI for each group
grouped = df.groupby('HumanSum_bin').agg(
    HumanSum_mean=('HumanSum', 'mean'),
    HumanSum_ci=('HumanSum', ci95),
    BotSum_mean=('BotSum', 'mean'),
    BotSum_ci=('BotSum', ci95),
    bin_center=('HumanSum', 'mean')
).reset_index()

# === STATISTICAL SIGNIFICANCE TESTING ===
# for each bin, compare BotSum vs average of human counts
print("Statistical Significance Analysis")
print("=" * 80)
print(f"{'Bin':<15} {'Bot Mean':<12} {'Human Mean':<12} {'t-stat':<10} {'p-value':<10} {'Sig?':<8}")
print("-" * 80)

significant_bins = []

for idx, row in grouped.iterrows():
    bin_label = row['HumanSum_bin']
   
    # get data for this bin
    bin_data = df[df['HumanSum_bin'] == bin_label]
   
    if len(bin_data) < 2:
        print(f"{str(bin_label):<15} {'N/A (insufficient data)'}")
        continue
   
    bot_values = bin_data['BotSum'].values
    # avg of the two human counts for each observation
    human_avg = bin_data['HumanSum']
   
    # paired t-test (since bot and human counts are for the same tiles/full caps)
    t_stat, p_value = stats.ttest_rel(bot_values, human_avg)
   
    is_significant = p_value < 0.05
    if is_significant:
        significant_bins.append(str(bin_label))
   
    sig_marker = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
   
    print(f"{str(bin_label):<15} {row['BotSum_mean']:>10.2f}  {row['HumanSum_mean']/2:>10.2f}  "
          f"{t_stat:>9.3f}  {p_value:>9.4f}  {sig_marker:<8}")

print("=" * 80)
print(f"\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001")
print(f"\nBins with significant differences (p < 0.05): {len(significant_bins)}")
if significant_bins:
    print(f"Bins: {', '.join(significant_bins)}")

# plot with significance markers
plt.figure(figsize=(12, 5))
plt.rc('font', size=14)

plt.errorbar(grouped['bin_center'], grouped['HumanSum_mean'], yerr=grouped['HumanSum_ci'],
             fmt='-', linewidth=1.5, capsize=7, capthick=1.5, elinewidth=1,
             color='fuchsia', label='Human Count')
plt.errorbar(grouped['bin_center'], grouped['BotSum_mean'], yerr=grouped['BotSum_ci'],
             fmt='-', linewidth=1.5, capsize=7, capthick=1.5, elinewidth=1,
             color='limegreen', label='Model Predicted Count')
r2sco = r2_score(df['HumanSum'], df['BotSum'])
plt.text(1, 55, f"R² value = {round(r2sco, 2)}")
plt.xlabel("Humans' Average Count")
plt.ylabel('Egg Count')
plt.title('Human and Model Counts with 95% Confidence Interval on 5-4')
plt.legend()
plt.tight_layout()
plt.savefig("ci4.png", dpi=150)
plt.show()