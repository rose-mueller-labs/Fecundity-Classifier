import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import curve_fit

ROOT_DIR = "/home/drosophila-lab/Documents/Fecundity/AlexanderDataClasses"
CLASS_MSES = dict({'0': 0.042881, '1': 0.282035, '2': 0.477807, '3': 1.037415, '4': 1.229814, '5': 1.911765, '6': 3.290323, '7': 5.000000, '8': 7.666667, '9': 13.200000})

def get_class_counts():
    counts = dict()
    for label in os.listdir("/home/drosophila-lab/Documents/Fecundity/AlexanderDataClasses"):
        counts[label]=0
        for img in os.listdir(f"{ROOT_DIR}/{label}"):
            counts[label]+=1
    
    return counts

df = pd.read_csv('alex.csv')

total_mse = mean_squared_error(df['Actual'], df['Predicted'])

print(f'TOTAL MSE: {total_mse}')
print(f'TOTAL RMSE: {np.sqrt(total_mse)}')

mse_by_counts = df.groupby('Actual').apply(lambda x: np.mean((x['Predicted']-x['Actual'])**2))

print("MSE FOR EACH COUNT: ")
print(mse_by_counts)

## graphing mse & class

plt.figure(figsize=(10,6))
mse_by_counts.plot(kind='bar')
plt.title('Error in Egg Counts per Class')
plt.xlabel('Class/Correct Egg Count')
plt.ylabel('Error in Prediction (Mean Squared Error)')
plt.xticks(rotation=0)
plt.axhline(y=total_mse, color='red', linestyle='--', label='Overall Error (MSE)')
plt.legend()
plt.tight_layout()
plt.plot()
plt.show()


## graphing mse & img count per class
img_counts = get_class_counts()

def get_key(k):
    return list(CLASS_MSES.keys())[list(CLASS_MSES.values()).index(k)]

print("img counts", img_counts)
# plt.scatter(list(CLASS_MSES.values()), list(img_counts.values()))
plt.scatter(sorted(list(CLASS_MSES.values()))[4:], [img_counts[get_key(k)] for k in sorted(list(CLASS_MSES.values()))][4:])
plt.yticks(np.arange(0, 10000, 1000)) # Add ticks every 0.5 units
plt.title('MSE vs. Image Count')
plt.xlabel('MSE')
plt.ylabel('Image Count')

# best fit line
x_data = sorted(list(CLASS_MSES.values())[4:])
y_data = [img_counts[get_key(k)] for k in sorted(list(CLASS_MSES.values()))][4:]

def exponential_func(x, a, b):
    return a * np.exp(b * x)

popt, pcov = curve_fit(exponential_func, x_data, y_data, p0=[1, 1]) 

x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = exponential_func(x_fit, *popt)

plt.plot(x_fit, y_fit, 'r-', label=f'fit: y={popt[0]:.2f}*exp({popt[1]:.2f}x)')
print(f'BEST FIT EQUATION: y={popt[0]:.2f}*exp({popt[1]:.2f}x)')
print("MSES SORTED (x)", sorted(list(CLASS_MSES.values())))
print("IMAGE COUNT BASED ON SORTED MSES", [img_counts[get_key(k)] for k in sorted(list(CLASS_MSES.values()))])
plt.plot()
plt.show()