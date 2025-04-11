## alex vs. difference barplot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("5-1_marvin_alex_comparison.csv")
alex_counts = np.array(df['AlexCount'])

plt.bar(df['AlexCount'], (df['Difference']**2), width=0.8, bottom=None, align='center', data=None)
print(alex_counts)

plt.plot()
plt.show()