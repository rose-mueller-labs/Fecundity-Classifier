import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_row', None)



#df="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_lithium_5-4_results/Alex_5-1_5-2O_v0.0_sums__lith54_CSV.csv"
df="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_lithium_5-4_results/Alex_5-1_5-2O_v0.0_tile_counts_lith.csv"
#df="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_CD_results/Alex_5-1_5-2O_v0.0_sums_CD_CSV.csv"
data = pd.read_csv(df)

print(sum(data['Bot'] > data['Human']))
print(sum(data['Bot'] < data['Human']))
print(sum(data['Bot'] == data['Human']))
print(min(data['Bot']-data['Human']))
print(max(data['Bot']-data['Human']))

def MSEs_metrics_and_graph(caps_csvs, name):
    df = pd.read_csv(caps_csvs)

    mse_by_counts = df.groupby('Human').apply(lambda x: np.mean((x['Human']-x['Bot'])**2))
    std_dev_by_counts = df.groupby('Human').apply(lambda x: np.std(x['Bot']))
    var_by_counts = df.groupby('Human').apply(lambda x: np.var(x['Bot']))
    mean_by_counts = df.groupby('Human').apply(lambda x: np.mean(x['Bot']))

    print(mean_by_counts.index)

    plt.scatter(data['Human'], data['Bot'])
    plt.xlabel("Human Counts")
    plt.ylabel("Bot Counts")
    plt.savefig(f'{name}_CD.png')

    print(np.corrcoef(data['Bot'],data['Human']))

    with open(f"{name}_metrics_54_tiles.txt", "w") as file:
        print("MSE FOR EACH COUNT: \n", file=file)
        print(mse_by_counts, file=file)
        print('\n', file=file)
        print("STD FOR EACH COUNT: \n", file=file)
        print(std_dev_by_counts, file=file)
        print('\n', file=file)
        print("VAR FOR EACH COUNT: \n", file=file)
        print(var_by_counts, file=file)
        print('\n', file=file)
        print("MEAN FOR EACH COUNT: \n", file=file)
        print(mean_by_counts, file=file)
        print('\n', file=file)

MSEs_metrics_and_graph(df, 'example_5152O_54_tiles')