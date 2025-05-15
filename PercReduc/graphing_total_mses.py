import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

data_dir = 'AlexResults/mses' 

# Prepare storage for all data
# all_data = {}
x = []
y = []

# Loop through all txt files
for fname in sorted(os.listdir(data_dir)):
    if fname.endswith('.txt'):
        samples = int(fname.split('_')[0].split('n')[1])
        with open(os.path.join(data_dir, fname)) as f:
            lines = f.readlines()
        # Find the start of the HumanSum section
        start_idx = None
        for i, line in enumerate(lines):
            # print(line)
            if 'TOTAL MSE:' in line:
                print(line)
                mse = float(line.split('TOTAL MSE:')[-1].strip())
                print(mse)
                x.append(samples)
                y.append(mse)

with open('sample_mses_alex.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['X', 'Y'])
    for i in range(len(x)):
        writer.writerow([x[i], y[i]])
fig, ax = plt.subplots(figsize=(16, 8))
plt.scatter(x, y)
X = np.array(x)
Y = np.array(y)
a, b = np.polyfit(X, Y, 1)
plt.plot(X, a*X + b, color = 'olive')
plt.tight_layout()
plt.savefig("mses_ov_samps3_alex")
