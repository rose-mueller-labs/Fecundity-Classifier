import os
import shutil
import pandas as pd

ALL_TILES="/home/drosophila-lab/Documents/All CD Caps"
CSV_PATH = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/CD_Image_to_Count.csv"
DATA_DEST_ROOT_PATH="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA"


dataset_name = f'ALL_CD_CAPS'
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
        filename = f"eggs{row["Count"]}count{row["ImageName"]}.jpg"
        src_name = f"{row["ImageName"]}.jpg"
        shutil.copy(f"{ALL_TILES}/{src_name}", f"{DATA_DEST_ROOT_PATH}/{dataset_name}/{filename}")

