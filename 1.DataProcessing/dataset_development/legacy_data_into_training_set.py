import os
import shutil
from PIL import Image 


ROOT_DEST_PATH = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/1.DataProcessing/DATASETS"

DS_PATHS = {'4-30': '/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/04-30-cap-800x800-sliced-Alexander',
            '5-1': '/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/5-1-cap-800x800-sliced-Alexander',
            '5-2S': '/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/5-2-cap-800x800-sliced-Alex-supplemented',
            '5-2O': '/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/5-2-cap-800x800-sliced-Alex-supl-no-5-2-zeros'
            }

BW_THRESH = 150 ## change to grayscale

multiple_sets = bool(int(input("Are there multiple sets in this dataset? If yes, type 1. Else, type 0: ")))
# (source, dest)

set_names = input("List the names of the subset(s) delimited with a space. Valid names: 4-30, 5-1, 5-2S, 5-2O. ").split(' ')
valid = bool(int(input(f"You inputted: {set_names}. Is this correct? Type 1 if yes. Type 0 if not: ")))
if not valid:
    set_names = input("List the names of the subset(s) delimited with a space. Valid names: 4-30, 5-1, 5-2S, 5-2O.   ").split(' ')

bw = bool(int(input('Regular: 0. Black and White: 1. Type: ')))
if bw:
    valid = bool(int(input('You selected black and white. Is this true (1 if yes 0 if no): ')))
    if valid:
        destination_path = f'{ROOT_DEST_PATH}/BW_{'_'.join(set_names)}'
        os.mkdir(destination_path)
    else:
        bw = False
        destination_path = f'{ROOT_DEST_PATH}/{'_'.join(set_names)}'
        os.mkdir(destination_path)
else:
    destination_path = f'{ROOT_DEST_PATH}/{'_'.join(set_names)}'
    os.mkdir(destination_path)

print(f"Your training set will be located at: {destination_path}")
print("Beginning transfer")
for set in set_names:
    for file in os.listdir(DS_PATHS[set]):
        egg_cnt = file.split('eggs')[1].split('count')[0]
        # print(egg_cnt)
        try:
            # print(f'{DS_PATHS[set]}/{file}', f'{destination_path}/{egg_cnt}/')
            shutil.copy(f'{DS_PATHS[set]}/{file}', f'{destination_path}/{egg_cnt}/')
            if bw:
                img = Image.open(f'{destination_path}/{egg_cnt}/{file}') # open colour image
                thresh = BW_THRESH
                fn = lambda x : 255 if x > thresh else 0
                r = img.convert('L').point(fn, mode='1')
                r.save(f'{destination_path}/{egg_cnt}/{file}')
        except FileNotFoundError:
            # print(f"{destination_path}/{egg_cnt}")
            os.mkdir(f"{destination_path}/{egg_cnt}")
            shutil.copy(f'{DS_PATHS[set]}/{file}', f'{destination_path}/{egg_cnt}/')
            if bw:
                img = Image.open(f'{destination_path}/{egg_cnt}/{file}') # open colour image
                thresh = BW_THRESH
                fn = lambda x : 255 if x > thresh else 0
                r = img.convert('L').point(fn, mode='1')
                r.save(f'{destination_path}/{egg_cnt}/{file}')