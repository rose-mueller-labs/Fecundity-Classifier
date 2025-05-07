import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

df = pd.read_csv('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/Plots/HumanBotComp/MoDataV1_5-4_tiles_MoDataV1.csv_sums_CSV.csv')

# Prepare data for plotting
human_counts = df[['PrimaryCount', 'SecondaryCount']].values.T  # shape: (2, num_images)
bot_counts = df['BotCount'].values
# If you have a separate 'CorrectCount' column, use it here
correct_counts = df[['PrimaryCount', 'SecondaryCount']].mean(axis=1).values

image_indices = np.arange(1, len(df) + 1)

plt.figure(figsize=(10, 6))

# Boxplot for human counts
plt.boxplot(human_counts, positions=image_indices, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor='deepskyblue', color='deepskyblue'),
            medianprops=dict(color='orange'))

# Plot bot predictions
plt.scatter(image_indices, bot_counts, color='red', label='Bot Prediction', zorder=5)

# Plot correct counts
plt.scatter(image_indices, correct_counts, color='green', marker='x', s=100, label='Correct Count', zorder=5)

plt.xlabel('Image Index')
plt.ylabel('Egg Count')
plt.title('Comparison of Human Counts, Bot Predictions, and Correct Counts')
plt.legend()
plt.tight_layout()
plt.show()