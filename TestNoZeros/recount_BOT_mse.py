import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import curve_fit

df = pd.read_csv('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/no_0_training_vs_CD_no_0_testing.csv')

def get_class_counts():
    counts = dict()
    for index, row in df.iterrows():
        label = int(row['Bot'])
        if label not in counts:
            counts[label]=1
        else:
            counts[label]+=1
    
    return counts

total_mse = mean_squared_error(df['Human'], df['Bot'])

print(f'TOTAL MSE: {total_mse}')
print(f'TOTAL RMSE: {np.sqrt(total_mse)}')

mse_by_counts = df.groupby('Bot').apply(lambda x: np.mean((x['Human']-x['Bot'])**2))

print("MSE FOR EACH COUNT: ")
print(mse_by_counts)
# # img_counts = get_class_counts()
# print("img counts", img_counts)

## graphing mse & class

# plt.figure(figsize=(10,6))
# mse_by_counts.plot(kind='bar')
# plt.title('Error in Egg Counts per Class')
# plt.xlabel('Class/Correct Egg Count')
# plt.ylabel('Error in Prediction (Mean Squared Error)')
# plt.ylim(0, 100)
# plt.xticks(rotation=0)
# plt.axhline(y=total_mse, color='red', linestyle='--', label='Overall Error (MSE)')
# plt.legend()
# plt.tight_layout()
# plt.plot()
# plt.savefig("DUAL_MODEL_all_and_zon_vs_CD_testing")