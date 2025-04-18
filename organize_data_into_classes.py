DATA_PATH = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/Winter 2017 2 21 C pops cap-sliced"
DATA_PATH_DEST = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingClasses"

import os
import shutil

for file in os.listdir(DATA_PATH):
    print(file)
    if 'eggs' not in file or 'unsure' in file:
        continue
    egg_count = file.split('eggs')[1].split('count')[0]
    try:
        if int(egg_count) > 10:
            print(egg_count)
        try:
            shutil.move(f'{DATA_PATH}/{file}', f'{DATA_PATH_DEST}/{egg_count}/{file}')
        except FileNotFoundError:
            path = f'{DATA_PATH_DEST}/{egg_count}'
            os.mkdir(path)
            shutil.move(f'{DATA_PATH}/{file}', f'{DATA_PATH_DEST}/{egg_count}/{file}')
    except ValueError:
        pass