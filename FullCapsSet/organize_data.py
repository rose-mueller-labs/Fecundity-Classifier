DATA_PATH = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/UsableData/FullCaps"
DATA_PATH_DEST = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/FullCaps"

import os
import shutil
import pandas as pd

df_counts = pd.read_csv("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/FullCapsSet/sj2_training_sums.csv")

def orgdat(datapath):
    for file in os.listdir(datapath):
        file_size_kb = os.stat(f'{datapath}/{file}').st_size / 1024
        if file_size_kb <= 2.00:
             continue
        try:
            egg_count = int(df_counts.loc[df_counts['RootImage'] == file[:-4], 'Sum'].values)
        except TypeError:
             continue
        try:
                shutil.move(f'{datapath}/{file}', f'{DATA_PATH_DEST}/{egg_count}/{file}')
        except FileNotFoundError:
                path = f'{DATA_PATH_DEST}/{egg_count}'
                os.mkdir(path)
                shutil.move(f'{datapath}/{file}', f'{DATA_PATH_DEST}/{egg_count}/{file}')

for fld in os.listdir(DATA_PATH):
    for folder in os.listdir(f'{DATA_PATH}/{fld}'):
        orgdat(f'{DATA_PATH}/{fld}/{folder}')

#orgdat("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/DATA/FullCaps")