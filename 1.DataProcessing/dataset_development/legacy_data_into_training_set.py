ROOT_DEST_PATH = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/1.DataProcessing/DATASETS"
ROOT_INPUT_PATH = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA"

multiple_sets = bool(int(input("Are there multiple sets in this dataset? If yes, type 1. Else, type 0: ")))
# (source, dest)
import os
import shutil

if multiple_sets:
    for folder in os.listdir(ROOT_INPUT_PATH):
        for file in os.listdir(f'{ROOT_DEST_PATH}/{folder}'):
            if 'eggs' not in file or 'unsure' in file:
                continue
            egg_count = file.split('eggs')[1].split('count')[0]
            try:
                shutil.move(f'{ROOT_INPUT_PATH}/{folder}/{file}', f'{ROOT_DEST_PATH}/{egg_count}/{file}')
            except FileNotFoundError:
                os.mkdir(f"{ROOT_DEST_PATH}/{egg_count}")
else:
    for file in os.listdir(f'{ROOT_INPUT_PATH}'):
        if 'eggs' not in file or 'unsure' in file:
            continue
        egg_count = file.split('eggs')[1].split('count')[0]
        try:
            shutil.move(f'{ROOT_INPUT_PATH}/{file}', f'{ROOT_INPUT_PATH}/{egg_count}/{file}')
        except FileNotFoundError:
                os.mkdir(f"{ROOT_DEST_PATH}/{egg_count}")