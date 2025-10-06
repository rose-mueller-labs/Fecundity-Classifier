import os
import shutil

SET_PATH = input("What is the dataset path: ")

freq = dict()

for file in os.listdir(SET_PATH):
    egg_cnt = file.split('eggs')[1].split('count')[0]
    freq[egg_cnt] = freq.get(egg_cnt, 0) + 1

for key, value in freq.items():
    print(f'{key} Egg Tiles\t: {value} tiles')