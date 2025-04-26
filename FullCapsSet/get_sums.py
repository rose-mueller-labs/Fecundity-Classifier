import os
import numpy as np
import time
import csv
import pandas as pd

# create a csv file with the predicted eggs and the actual
def create_csv_data_file(csv_name, data_path):
    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Sum'])

    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Sum'])
        for label in os.listdir(data_path):
            for img in os.listdir(f"{data_path}/{label}"):
                root_image_a = img.split('count')[1]
                root_image_b = root_image_a.split('pt')
                root_image = root_image_b[0]
                root_image = root_image.strip()
                writer.writerow([img, root_image, label])

def get_actual_total(csv_path, actual_csv_name):
    # get all unique names => get the ones with the same names => get the actual counts => sum
    df = pd.read_csv(csv_path)
    root_image_names = np.array(df['RootImage'].unique())
    # print(root_image_names)
    actual_counts = dict()
    for cap_name in root_image_names:
        actual_counts[cap_name] = 0

    for index, row in df.iterrows():
        actual_counts[row['RootImage']] += row['Sum']
    
    with open(actual_csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RootImage', 'Sum'])
        for root_img, actual in actual_counts.items():
            actual = actual_counts[root_img]
            writer.writerow([root_img, actual])



if __name__ == '__main__':
    create_csv_data_file('sj2_training_indivs.csv', "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/SilkyJohnson2")
    get_actual_total('sj2_training_indivs.csv', 'sj2_training_sums.csv')