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

all = False
which_csvs = input("If you want all csvs write A: ").split(', ')
# 'A' = all of them
if 'A' in which_csvs:
    print("You selected: ")
    print(os.listdir(CSV_PATH))
    all = True
if not all:
    dataset_prefix = '_'.join(input("What are the days?: ").split(', '))
else:
    dataset_prefix = "5-2_5-4_5-6_5-8_5-10_5-12_5-14_5-16_5-18"
dataset_name = f'CC_{dataset_prefix}_{'J' if whose_csvs == "Jacob" else 'A'}'
try:
    os.mkdir(f"{DATA_DEST_ROOT_PATH}/{dataset_name}")
except FileExistsError:
    if len(os.listdir(f"{DATA_DEST_ROOT_PATH}/{dataset_name}")) == 0:
        pass
    else:
        raise FileExistsError("There are contents to this directory")
# csv headers: Image,Part,Count,Whitespace
# format of all tiles:
# AO5 Lithium 5-17 13 pt33.jpg
# {image} pt{part}.jpg
# normalize to: eggs1countACO1 Control 04-30 1 pt13.jpg
# eggs{Count}{Image} pt{Part}.jpg
# send off to legacy script to create paritioned set

# print(os.listdir(ALL_TILES))[:10]

for csv in os.listdir(CSV_PATH):
    if not all and csv not in which_csvs:
        continue
    df_csv = pd.read_csv(f"{CSV_PATH}/{csv}")
    for index, row in df_csv.iterrows():
        filename = f"eggs{row["Count"]}count{row["Image"]} pt{row["Part"]}.jpg"
        src_name = f"{row["Image"]} pt{row["Part"]}.jpg"
        shutil.copy(f"{ALL_TILES}/{src_name}", f"{DATA_DEST_ROOT_PATH}/{dataset_name}/{filename}")

