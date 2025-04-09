import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import curve_fit

# ROOT_DIR = "/home/drosophila-lab/Documents/Fecundity/AlexanderDataClasses"
df = pd.read_csv('testing_on_winter.csv')
CLASS_MSES = dict({'0': 0.031185, '1': 0.483871, '2': 1.223881, 
                   '3': 2.750000, '4': 2.590909, '5': 4.250000, '6': 10.750000, 
                   '7': 11.666667, '8': 1.000000, '9': 64.000000, 
                   '12': 25.000000})

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

print(f'TOTAL MSE: {total_mse}')
print(f'TOTAL RMSE: {np.sqrt(total_mse)}')

mse_by_counts = df.groupby('Human').apply(lambda x: np.mean((x['Human']-x['Bot'])**2))

print("MSE FOR EACH COUNT: ")
print(mse_by_counts)
img_counts = get_class_counts()
print("img counts", img_counts)

## graphing mse & class

plt.figure(figsize=(10,6))
mse_by_counts.plot(kind='bar')
plt.title('Error in Egg Counts per Class')
plt.xlabel('Class/Correct Egg Count')
plt.ylabel('Error in Prediction (Mean Squared Error)')
plt.ylim(0, 12)
plt.xticks(rotation=0)
plt.axhline(y=total_mse, color='red', linestyle='--', label='Overall Error (MSE)')
plt.legend()
plt.tight_layout()
plt.plot()
plt.savefig("graph")


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

# # best fit line
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