import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import curve_fit

# ROOT_DIR = "/home/drosophila-lab/Documents/Fecundity/AlexanderDataClasses"
df = pd.read_csv('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/Tests3and5/SJ2_lithium_training_vs_JIMMY_lithium_testing.csv')

def get_class_counts():
    counts = dict()
    for index, row in df.iterrows():
        label = int(row['Human'])
        if label not in counts:
            counts[label]=1
        else:
            counts[label]+=1
    
    return counts

total_mse = mean_squared_error(df['Human'], df['Bot'])
total_r2 = r2_score(df['Human'], df['Bot'])

mse_by_counts = df.groupby('Human').apply(lambda x: np.mean((x['Human']-x['Bot'])**2))

r2_score_by_counts = df.groupby('Human').apply(lambda x: r2_score(x['Human'], x['Bot']))

print(f'TOTAL MSE: {total_mse}\n')
print(f'TOTAL RMSE: {np.sqrt(total_mse)}\n')
print(f'TOTAL R2SCORE: {total_r2}\n')
print("MSE FOR EACH COUNT: \n")
print(mse_by_counts)
print('\n')
print("R2 SCORE FOR EACH COUNT: \n")
print(r2_score_by_counts)
print('\n')
img_counts = get_class_counts()
print("img counts", img_counts)
print('\n')

## graphing mse & class

plt.figure(figsize=(10,6))
mse_by_counts.plot(kind='bar')
plt.title('Error in Egg Counts per Class')
plt.xlabel('Class/Correct Egg Count')
plt.ylabel('Error in Prediction (Mean Squared Error)')
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.axhline(y=total_mse, color='red', linestyle='--', label='Overall Error (MSE)')
plt.legend()
plt.tight_layout()
plt.plot()
plt.savefig("unknownlithiumtest")

# CLASS_MSES = {
# 0 :     0.038362,
# 1  :    0.391850,
# 2   :   0.922414,
# 3    :  2.892308,
# 4     : 3.071429,
# 5      :5.722222,
# 6   :   9.500000,
# 7   :  11.666667,
# 8   :   4.000000,
# 9   :  14.333333,
# 10  :  16.000000,

## graphing mse & img count per class
# img_counts = get_class_counts()

# def get_key(k):
#     return list(CLASS_MSES.keys())[list(CLASS_MSES.values()).index(k)]

# print("img counts", img_counts)
# # plt.scatter(list(CLASS_MSES.values()), list(img_counts.values()))
# plt.scatter(sorted(list(CLASS_MSES.values())), [img_counts[int(get_key(k))] for k in sorted(list(CLASS_MSES.values()))][4:])
# plt.yticks(np.arange(0, 10000, 1000)) # Add ticks every 0.5 units
# plt.title('MSE vs. Image Count')
# plt.xlabel('MSE')
# plt.ylabel('Image Count')

# CLASS_MSES = 

# best fit line
# x_data = sorted(list(CLASS_MSES.values())[4:])
# y_data = [img_counts[get_key(k)] for k in sorted(list(CLASS_MSES.values()))][4:]

# def exponential_func(x, a, b):
#     return a * np.exp(b * x)

# popt, pcov = curve_fit(exponential_func, x_data, y_data, p0=[1, 1]) 

# x_fit = np.linspace(min(x_data), max(x_data), 100)
# y_fit = exponential_func(x_fit, *popt)

# plt.plot(x_fit, y_fit, 'r-', label=f'fit: y={popt[0]:.2f}*exp({popt[1]:.2f}x)')
# print(f'BEST FIT EQUATION: y={popt[0]:.2f}*exp({popt[1]:.2f}x)')
# print("MSES SORTED (x)", sorted(list(CLASS_MSES.values())))
# print("IMAGE COUNT BASED ON SORTED MSES", [img_counts[get_key(k)] for k in sorted(list(CLASS_MSES.values()))])
# plt.plot()
# plt.show()