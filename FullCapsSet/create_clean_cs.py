dpt = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/FullCaps copy"

DATA_PATH_DEST = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/FullCapsClean"

import os
import shutil

def orgdat(dp):
    for cls  in os.listdir(dp):
        datapath = f'{dp}/{cls}/clean'
        for file in os.listdir(f'{datapath}'):
            try:
                    shutil.move(f'{datapath}/{file}', f'{DATA_PATH_DEST}/{cls}/{file}')
            except FileNotFoundError:
                    path = f'{DATA_PATH_DEST}/{cls}'
                    os.mkdir(path)
                    shutil.move(f'{datapath}/{file}', f'{DATA_PATH_DEST}/{cls}/{file}')

orgdat(dpt)