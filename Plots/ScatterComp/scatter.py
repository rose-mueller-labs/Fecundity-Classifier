import matplotlib.pyplot as plt
import pandas as pd

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
plt.xlabel("Correct (Average) Sum")
plt.ylabel("Bot Absolute Difference from Sum")
plt.savefig("Correct**")