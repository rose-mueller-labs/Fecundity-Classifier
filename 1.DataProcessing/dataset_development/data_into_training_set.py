import os
import shutil
import pandas as pd

ALL_TILES="/home/drosophila-lab/Documents/All Lithium Caps-sliced"
CSV_PATH = ""
DATA_DEST_ROOT_PATH="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA"
while True:
    whose_csvs = input("Whose CSVs? Jacob or Alex: ")

    if whose_csvs == "Jacob":
        CSV_PATH="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/jacob_data_csvs"
        break
    elif whose_csvs == "Alex":
        CSV_PATH="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/alex_data_csvs"
        break

which_csvs = input().split(', ')
dataset_prefix = '_'.join(input("What are the days: ").split(', '))
dataset_name = f'CC_{dataset_prefix}_{'J' if whose_csvs == "Jacob" else 'A'}'
os.mkdir(f"{DATA_DEST_ROOT_PATH}/{dataset_name}")
# csv headers: Image,Part,Count,Whitespace
# format of all tiles:
# AO5 Lithium 5-17 13 pt33.jpg
# {image} pt{part}.jpg
# normalize to: eggs1countACO1 Control 04-30 1 pt13.jpg
# eggs{Count}{Image} pt{Part}.jpg
# send off to legacy script to create paritioned set

for csv in os.listdir(CSV_PATH):
    if csv not in which_csvs:
        continue
    df_csv = pd.read_csv(f"{CSV_PATH}/{csv}")
    df_csv['Filename'] = f"eggs{df_csv["Count"]}count{df_csv["Image"]} pt{df_csv["Part"]}.jpg"
    for filename in df_csv["Filename"]:
        shutil.copy(f"{ALL_TILES}/{filename}", f"{DATA_DEST_ROOT_PATH}/{dataset_name}")

