'''Testing between two different datasets Lithium Experiment vs CD Experiment'''

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import curve_fit

df = pd.read_csv('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingBetweenPeople/5-4_angela_julie_comparison_clusters.csv')

def get_class_counts():
    counts = dict()
    for index, row in df.iterrows():
        label = int(row['AlexCount'])
        if label not in counts:
            counts[label]=1
        else:
            counts[label]+=1
    
    return counts

total_mse = mean_squared_error(df['AlexCount'], df['MarvinCount'])
total_r2 = r2_score(df['AlexCount'], df['MarvinCount'])

mse_by_counts = df.groupby('AlexCount').apply(lambda x: np.mean((x['AlexCount']-x['MarvinCount'])**2))
mse_by_counts2 = df.groupby('MarvinCount').apply(lambda x: np.mean((x['AlexCount']-x['MarvinCount'])**2))
r2_score_by_counts = df.groupby('AlexCount').apply(lambda x: r2_score(x['AlexCount'], x['MarvinCount']))

print(f'TOTAL MSE: {total_mse}\n')
print(f'TOTAL RMSE: {np.sqrt(total_mse)}\n')
print(f'TOTAL R2SCORE: {total_r2}\n')
print("MSE FOR EACH COUNT ALEX: \n")
print(mse_by_counts)
print('\n')
print("MSE FOR EACH COUNT MARVIN: \n")
print(mse_by_counts2)
print('\n')
print("R2 SCORE FOR EACH COUNT: \n")
print(r2_score_by_counts)
print('\n')
img_counts = get_class_counts()
print("img counts", img_counts)
print('\n')

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
# plt.savefig("angela_julie_MSE_and_metrics")