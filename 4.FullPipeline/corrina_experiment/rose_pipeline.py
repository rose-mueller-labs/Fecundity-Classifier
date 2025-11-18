'''
Full pipeline to output full cap counts from grid image.
'''

from preprocess import main
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
from datetime import datetime
from image_shredder import main
import os
import os.path
import cv2

# First, split the gridded image into the squares

# Put set name here
set_name="SecondBigCap"

if not os.path.isdir(f"/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/{set_name}-sliced"):
    main(f"/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/{set_name}")


TESTING_SET=f"/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/{set_name}-sliced"


def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42
BASE_DIR="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/1.DataProcessing/model_architecture/models"

TOP_MODEL_NAMES_AND_PATHS = {
    'Alex_FecundityModelMoDataV1': (f'{BASE_DIR}/fecundity_model_mo_data_v1.h5', None),
    'Alex_5-1_5-2S_v0.0':(f'{BASE_DIR}/alex_5-1_5-2S_v0.0.h5',None),
    'Alex_4-30_5-1_CC_A_v0.0': (f'{BASE_DIR}/alex_4-30_5-1_CC_A_v0.0.h5', None)
}

def predict_egg_count_default(image_path, name, model, model2=None):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])

    return egg_count

# make a csv contianing tile predictions from split images
def get_tile_preds(name, model, model2):
    mod1 = tf.keras.models.load_model(model)
    mod2 = None
    if model2 != None:
        mod2 = tf.keras.models.load_model(model2)
    
    csv_name = f'{set_name}_{name}_tiles.csv'

    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'Part', 'Bot'])
        for img in os.listdir(f"{TESTING_SET}"):
            root_image = img.split("JPG")[0].split(".")[0]
            predicted_eggs = int(predict_egg_count_default(f"{TESTING_SET}/{img}", name, mod1, mod2))
            part = img.split("pt")[-1].split('.')[0]
            writer.writerow([root_image, part, predicted_eggs])
    return csv_name

# loop through all tiles of images 
def get_sums(csv_path, name):
    actual_csv_name = f'{set_name}_{name}_sums.csv'

    df = pd.read_csv(csv_path)
    root_image_names = np.array(df['ImageName'].unique())

    actual_counts = dict()
    for cap_name in root_image_names:
        actual_counts[cap_name] = 0

    for index, row in df.iterrows():
        # print(row['Bot'])
        actual_counts[row['ImageName']] += row['Bot']
    
    with open(actual_csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'BotSum'])
        for root_img, actual in actual_counts.items():
            writer.writerow([root_img, actual])
    return actual_csv_name

if __name__ == '__main__':
    for name, paths in TOP_MODEL_NAMES_AND_PATHS.items():
        print(f'Getting tiles for {name}')
        print_time()
        tiles_csv_name = get_tile_preds(name, paths[0], paths[1])
        print(f'Getting sums for {name}')
        print_time()
        sums_csv_name = get_sums(tiles_csv_name, name)