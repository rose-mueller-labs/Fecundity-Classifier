import numpy as np
from scipy import stats
import pandas as pd

def calculate_p_value(x1, y1, x2, y2):

    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)

    t_stat, p_value = stats.ttest_ind_from_stats(
        mean1=slope1, std1=std_err1, nobs1=len(x1),
        mean2=slope2, std2=std_err2, nobs2=len(x2),
        equal_var=False
    )

    return p_value

df = pd.read_csv("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/Plots/HumanBotComp/MoDataV1_5-4_tiles_MoDataV1.csv_sums_CSV.csv")
x_bot = list(df['AverageSum'])
y_bot = list(df['BotDiff'])
x_hum = list(df['AverageSum'])
y_hum = list(df['PrimaryDiff'])

p_value = calculate_p_value(x_bot, y_bot, x_hum, y_hum)
print("P-value:", p_value)