import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/Plots/HumanBotComp/MoDataV1_5-4_tiles_MoDataV1.csv_sums_CSV.csv")

sum_v_diff = dict()

sum_v_diff_cnts = dict()

for index,  row in df.iterrows():
    if int(row['SecondarySum']) not in sum_v_diff:
        sum_v_diff[int(row['SecondarySum'])] = row['PrimaryDiff']
        sum_v_diff_cnts[int(row['SecondarySum'])] = 1
    else:
        sum_v_diff[int(row['SecondarySum'])]+=row['PrimaryDiff']
        sum_v_diff_cnts[int(row['SecondarySum'])] += 1

for key, value in sum_v_diff.items():
    sum_v_diff[key] = value/sum_v_diff_cnts[key]

plt.scatter(list(sum_v_diff.keys()), list(sum_v_diff.values()))

coeffs = np.polyfit(list(sum_v_diff.keys()), list(sum_v_diff.values()), 1)
slope = coeffs[0]
cept = coeffs[1]

x_line = np.linspace(min(list(sum_v_diff.keys())), max(list(sum_v_diff.keys())), 100)
y_line = slope * x_line + cept
print("slope: ", slope)
plt.plot(x_line, y_line, color="red")
plt.xlabel("Secondary Sum")
plt.ylabel("Human Absolute Difference from Primary Sum")
plt.title("Secondary Sum v. Primary Human Absolute Difference")
plt.savefig("HumanFCorrect***")