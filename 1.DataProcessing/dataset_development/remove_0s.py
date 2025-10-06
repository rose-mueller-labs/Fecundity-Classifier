import os
import shutil

DS_PATH = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/5-2-cap-800x800-sliced-Alex-supl-no-5-2-zeros"

max_0s = 1801
curr_0s = 0

for file in os.listdir(DS_PATH):
    if 'unsure' in file:
        continue
    egg_cnt = int(file.split('eggs')[1].split('count')[0])
    if egg_cnt != 0:
        continue
    if egg_cnt == 0:
        curr_0s += 1
    if curr_0s >= max_0s and egg_cnt == 0:
        os.remove(f"{DS_PATH}/{file}")