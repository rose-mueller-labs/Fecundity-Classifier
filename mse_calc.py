import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('alex.csv')

total_mse = mean_squared_error(df['Actual'], df['Predicted'])

print(f'TOTAL MSE: {total_mse}')
print(f'TOTAL RMSE: {np.sqrt(total_mse)}')

mse_by_counts = df.groupby('Actual').apply(lambda x: np.mean((x['Predicted']-x['Actual'])**2))

print("MSE FR EACH COUNT: ")
print(mse_by_counts)

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