import os
import numpy as np
import time
import csv
import pandas as pd

# create a csv file with the predicted eggs and the actual
def create_csv_data_file(csv_name, tiles_csv):
    tiles_df = pd.read_csv(tiles_csv)
    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName','x', 'y', 'Part', 'EggCount', 'FileName'])
        for i, row in tiles_df.iterrows():
            part = row['Part']
            x, y = get_x_y_coordinate_from_part(part)
            actual_total = row['Sum']
            img = f"eggs{actual_total}count{row['CD_RootImage']} pt{row['Part']}.jpg"
            writer.writerow([img, x, y, row['Part'], row['Bot'], row['CD_RootImage']])

def get_actual_total(csv_path):
    # get all unique names => get the ones with the same names => get the actual counts => sum
    df = pd.read_csv(csv_path)
    root_image_names = np.array(df['RootImage'].unique())
    # print(root_image_names)
    actual_counts = dict()
    expected_counts = dict()
    for cap_name in root_image_names:
        actual_counts[cap_name] = 0
        expected_counts[cap_name] = 0

    for index, row in df.iterrows():
        actual_counts[row['RootImage']] += row['Actual']
        expected_counts[row['RootImage']] += row['Expected']

def get_x_y_coordinate_from_part(part):
    # part_number = int(part.split('pt')[1])
    part_number = int(part)

    if part_number % 10 == 0:
        return (int(part_number/10)-1, 10-1)
    x = int(np.floor(part_number / 10))
    y = (part_number % 10)-1

    return (x, y)


def main(csv_name, tiles_df):
    # create_csv_data_file('alex2.csv', "/home/drosophila-lab/Documents/Fecundity/AlexanderDataClasses")
    # get_actual_total('alex2.csv', 'alex2_sums.csv')
    create_csv_data_file(csv_name, tiles_df)
                         # "/home/drosophila-lab/Documents/Fecundity/Lithium-Caps-Organization/Alex_4-30_5-1_CC_A_v0.0_tile_counts_CD_Complete.csv