DATA_PATH = "/home/drosophila-lab/Documents/Fecundity/04-30-cap-800x800-sliced-Alexander"
DATA_PATH_DEST = "/home/drosophila-lab/Documents/Fecundity/AlexanderDataClasses"

import os
import shutil

for file in os.listdir(DATA_PATH):
    print(file)
    egg_count = file[4]
    print(egg_count)
    shutil.move(f'{DATA_PATH}/{file}', f'{DATA_PATH_DEST}/{egg_count}/{file}')