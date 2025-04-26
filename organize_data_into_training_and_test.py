DATA_PATH = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/SilkyJohnson2Training"
DATA_PATH_DEST = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestingSets/SilkyJohnson2Testing"

import os
import shutil
import math 

counts = dict()
test_counts = dict()

def get_counts(datapath):
    for cls in os.listdir(datapath):
        for file in os.listdir(f'{datapath}/{cls}'):
            if cls in counts:
                counts[cls]+=1
            else:
                counts[cls]=1

def get_proportions():
    for key, value in counts.items():
        test_counts[key] = math.floor(0.20 * value)
    print(test_counts)


def orgdat(datapath):
    get_counts(datapath)
    get_proportions()

    for cls in os.listdir(datapath):
        testing_cnt = 0
        for file in os.listdir(f'{datapath}/{cls}'):
            if testing_cnt == int(test_counts[cls]):
                print(testing_cnt)
                break
            else:
                print(file)
                try:
                    shutil.move(f'{datapath}/{cls}/{file}', f'{DATA_PATH_DEST}/{cls}/{file}')
                except FileNotFoundError:
                    path = f'{DATA_PATH_DEST}/{cls}'
                    os.mkdir(path)
                    shutil.move(f'{datapath}/{cls}/{file}', f'{DATA_PATH_DEST}/{cls}/{file}')
                testing_cnt+=1

# for folder in os.listdir(DATA_PATH):
#     orgdat(f'{DATA_PATH}/{folder}')

orgdat("//home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/SilkyJohnson2Training")