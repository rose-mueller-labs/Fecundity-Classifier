import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import time
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit

IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42
MODEL = tf.keras.models.load_model("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_mo_data_v1.h5")

## CSV creation

# get human avg count for caps and the abs difference for 5-1, 5-2, 5-4
# get the bots' full cap count for that too
# get its abs difference as well 

def predict_egg_count_default(image_path, name, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])

    return egg_count

MARVIN = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/DATA/5-1-cap-800x800-sliced-Marvin"
ALEX = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/DATA/5-1-cap-800x800-sliced-Alexander"

def get_raw(file):
    raw_file_name = file.split('eggs')[1]
    raw_file_name_2 = raw_file_name.split('count')[1]
    count = raw_file_name.split('count')[0]
    return count, raw_file_name_2

all_files = os.listdir(ALEX)

def get_alex_file(file, all_files):
    for named in all_files:
        if file == get_raw(named)[1]:
            if 'unsure' in named:
                return None, None
            alexcount, rawfile = get_raw(named)
            return alexcount, named

def get_tile_preds_data_file(name, model):
    csv_name = f"5-1_tiles_{name}.csv"
    count_same = 0
    total = 0
    count_diff = 0
    with open(csv_name, "w", newline='') as cmp_file:
        writer = csv.writer(cmp_file)
        writer.writerow(['RootImage', 'FileName', 'PrimaryFileName', 'SecondaryFileName', 'PrimaryCount', 'SecondaryCount', 'BotCount', 'Mean', 'PrimaryDiff', 'SecondaryDiff', 'BotDiff'])
        for file in os.listdir(MARVIN):
            if 'eggs0' in file or 'unsure' in file:
                continue
            path_name = f"{MARVIN}/{file}"
            
            try:
                count, raw_file_name = get_raw(file)
            except IndexError:
                continue
            total += 1
            bot_pred = predict_egg_count_default(path_name, "MoDataV1", model)
            root_image_a = file.split('count')[1]
            root_image_b = root_image_a.split('pt')
            root_image = root_image_b[0]
            root_image = root_image.strip()
            if f'{file}' in all_files:
                count_same+=1
                alex_count, alex_filename = get_alex_file(raw_file_name, all_files)
                count = int(count)
                mean = ((int(alex_count)+int(count))/2)
                writer.writerow([root_image, raw_file_name, alex_filename, file, alex_count, count, bot_pred, mean, abs(int(alex_count)-mean), abs(int(count)-mean), abs(bot_pred-mean)])
            else:
                count_diff += 1
                alex_count, alex_filename = get_alex_file(raw_file_name, all_files)
                if (alex_count == None):
                    continue
                count = int(count)
                alex_count = int(alex_count)
                mean = ((int(alex_count)+int(count))/2)
                writer.writerow([root_image, raw_file_name, alex_filename, file, alex_count, count, bot_pred, mean, abs(alex_count-mean), abs(count-mean), abs(bot_pred-mean)])
                # print(alex_filename)
                # print(alex_count)
                # print(count)
                # print(file)
                # print('-----')
    return csv_name

def get_actual_total(csv_path, name):
    actual_csv_name = f'{name}_{csv_path}_sums_CSV.csv'
    # get all unique names => get the ones with the same names => get the actual counts => sum
    df = pd.read_csv(csv_path)
    root_image_names = np.array(df['RootImage'].unique())
    # print(root_image_names)
    actual_counts_prim = dict()
    actual_counts_sec = dict()
    actual_counts_bot = dict()
    for cap_name in root_image_names:
        actual_counts_prim[cap_name] = 0
        actual_counts_sec[cap_name] = 0
        actual_counts_bot[cap_name] = 0

    for index, row in df.iterrows():
        actual_counts_prim[row['RootImage']] += row['PrimaryCount']
        actual_counts_sec[row['RootImage']] += row['SecondaryCount']
        actual_counts_bot[row['RootImage']] += row['BotCount']
    
    with open(actual_csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RootImage', 'PrimarySum', 'SecondarySum', 'BotSum', 'AverageSum', 'PrimaryDiff', 'SecondaryDiff', 'BotDiff'])
        for root_img, prim_cnt in actual_counts_prim.items():
            sec_cnt = int(actual_counts_sec[root_img])
            bot_cnt = int(actual_counts_bot[root_img])
            avg_cnt = (int(prim_cnt)+int(sec_cnt))/2
            writer.writerow([root_img, prim_cnt, sec_cnt, bot_cnt, avg_cnt, abs(int(prim_cnt)-avg_cnt), abs(sec_cnt-avg_cnt), abs(bot_cnt-avg_cnt)])
    return actual_csv_name

if __name__ == '__main__':
    name = 'MoDataV1'
    print(f'Getting tiles')
    tiles_csv_name = get_tile_preds_data_file(name, MODEL)
    print(f'Getting sums')
    sums_csv_name = get_actual_total(tiles_csv_name, name)