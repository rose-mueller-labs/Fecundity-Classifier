DATA_PATH = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/UsableData"
DATA_PATH_DEST = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/SilkyJohnson3"

import os
import shutil

def orgdat(datapath):
    for file in os.listdir(datapath):
        print(file)
        if 'eggs' not in file or 'unsure' in file:
            continue
        egg_count = file.split('eggs')[1].split('count')[0]
        if egg_count == 0:
            continue
        try:
            if int(egg_count) > 10:
                print(egg_count)
            try:
                shutil.move(f'{datapath}/{file}', f'{DATA_PATH_DEST}/{egg_count}/{file}')
            except FileNotFoundError:
                path = f'{DATA_PATH_DEST}/{egg_count}'
                os.mkdir(path)
                shutil.move(f'{datapath}/{file}', f'{DATA_PATH_DEST}/{egg_count}/{file}')
        except ValueError:
            pass

for folder in os.listdir(DATA_PATH):
    orgdat(f'{DATA_PATH}/{folder}')

# orgdat("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/UsableData/Winter 2017 2 21 C pops cap-sliced")