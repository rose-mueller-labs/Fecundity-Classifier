import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/Plots/HumanBotComp/MoDataV1_5-4_tiles_MoDataV1.csv_sums_CSV.csv")

sum_v_diff = dict()

sum_v_diff_cnts = dict()

for index,  row in df.iterrows():
    if int(row['AverageSum']) not in sum_v_diff:
        sum_v_diff[int(row['AverageSum'])] = row['BotDiff']
        sum_v_diff_cnts[int(row['AverageSum'])] = 1
    else:
        sum_v_diff[int(row['AverageSum'])]+=row['BotDiff']
        sum_v_diff_cnts[int(row['AverageSum'])] += 1

for key, value in sum_v_diff.items():
    sum_v_diff[key] = value/sum_v_diff_cnts[key]

plt.scatter(list(sum_v_diff.keys()), list(sum_v_diff.values()))

coeffs = np.polyfit(list(sum_v_diff.keys()), list(sum_v_diff.values()), 1)
slope = coeffs[0]
cept = coeffs[1]

x_line = np.linspace(min(list(sum_v_diff.keys())), max(list(sum_v_diff.keys())), 100)
y_line = slope * x_line + cept
print("slope: ", slope)
print("cept:", cept)
plt.ylim(0, 10)
plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.plot(x_line, y_line, color="red")
plt.xlabel("Average Count", fontsize=16)
plt.ylabel("Absolute Difference", fontsize=16)
plt.title("Model Prediction")
plt.savefig("bot1.png")