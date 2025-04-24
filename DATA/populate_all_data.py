DEST="/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/DATA/AllLithiumData"
PATH="/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/UsableData"

import os
import shutil

for folder in os.listdir(PATH):
    for file in os.listdir(f'{PATH}/{folder}'):
        if 'eggs' not in file or 'unsure' in file:
            continue
        egg_count = file.split('eggs')[1].split('count')[0]
        shutil.move(f'{PATH}/{folder}/{file}', f'{DEST}/{file}')