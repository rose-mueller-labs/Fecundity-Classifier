
import pandas as pd
import os
import shutil

df_names = pd.read_csv("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/FullCapsSet/sj2_training_indivs.csv")
clean = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/FullCapsSplit"
dest = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/FullCapsSplitCounted"

for cls in os.listdir(clean):
    for file in os.listdir(f'{clean}/{cls}'):
        # value = int(file in df_names.loc[df_names['ImageName'], 'Sum'].values)
        mask = df_names['ImageName'].str.contains(file, na=False)
        try:
            value = int(df_names.loc[mask, 'Sum'])
            print(df_names.loc[mask, 'Sum'])
        except TypeError:
            continue
        try:
                shutil.move(f'{clean}/{cls}/{file}', f'{dest}/{value}/{file}')
        except FileNotFoundError:
                path = f'{dest}/{value}'
                os.mkdir(path)
                shutil.move(f'{clean}/{cls}/{file}', f'{dest}/{value}/{file}')