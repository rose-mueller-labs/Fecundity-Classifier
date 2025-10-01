import os
import shutil

ROOT_DEST_PATH = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/1.DataProcessing/DATASETS"

DS_PATHS = {'4-30': '/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/04-30-cap-800x800-sliced-Alexander',
            '5-1': '/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/5-1-cap-800x800-sliced-Alexander',
            '5-2': '/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/5-2-cap-800x800-sliced-Alex'
            }

multiple_sets = bool(int(input("Are there multiple sets in this dataset? If yes, type 1. Else, type 0: ")))
# (source, dest)

set_names = input("List the names of the subset(s) delimited with a space. Valid names: 4-30, 5-1, 5-2. ").split(' ')
valid = bool(input(f"You inputted: {set_names}. Is this correct? Type 1 if yes. Type 0 if not: "))
if not valid:
    set_names = input("List the names of the subset(s) delimited with a space. Valid names: 4-30, 5-1, 5-2.   ").split(' ')

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
        except FileNotFoundError:
            # print(f"{destination_path}/{egg_cnt}")
            os.mkdir(f"{destination_path}/{egg_cnt}")