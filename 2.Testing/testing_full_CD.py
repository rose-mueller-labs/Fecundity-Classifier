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

CD_IMAGES=""
SPLIT_CD_IMAGES=""

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 12
BASE_DIR="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/1.DataProcessing/model_architecture/models"
UPDATE_CSV="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/ModelStatistics.csv"

TOP_MODEL_NAMES_AND_PATHS = {
    #'Alex_FecundityModelMoDataV1': (f'{BASE_DIR}/fecundity_model_mo_data_v1.h5', None),
    #'Alex_4-30_5-1_v0.0':(f'{BASE_DIR}/alex_4-30_5-1_v0.0.h5',None),
    # 'Alex_4-30_v0.0':(f'{BASE_DIR}/alex_4-30_v0.0.h5',None),
    # 'Alex_5-1_v0.0':(f'{BASE_DIR}/alex_5-1_v0.0.h5',None),
    # 'Alex_5-2S_v0.0':(f'{BASE_DIR}/alex_5-2S_v0.0.h5',None),
    # 'Alex_5-2O_v0.0':(f'{BASE_DIR}/alex_5-2O_v0.0.h5',None),
    # 'Alex_BW_4-30_5-1_v0.0':(f'{BASE_DIR}/alex_BW_4-30_5-1_v0.0.h5',None),
    # 'Alex_BW_4-30_v0.0':(f'{BASE_DIR}/alex_BW_4-30_v0.0.h5',None),
    # 'Alex_BW_5-1_v0.0':(f'{BASE_DIR}/alex_BW_5-1_v0.0.h5',None)
    # 'Alex_4-30_5-1_5-2O_v0.0':(f'{BASE_DIR}/alex_4-30_5-1_5-2O_v0.0.h5',None),
    # 'Alex_4-30_5-1_5-2S_v0.0':(f'{BASE_DIR}/alex_4-30_5-1_5-2S_v0.0.h5',None),
    # 'Alex_4-30_5-2O_v0.0':(f'{BASE_DIR}/alex_4-30_5-2O_v0.0.h5',None),
    # 'Alex_4-30_5-2S_v0.0':(f'{BASE_DIR}/alex_4-30_5-2S_v0.0.h5',None),
    # 'Alex_5-1_5-2O_v0.0':(f'{BASE_DIR}/alex_5-1_5-2O_v0.0.h5',None),
    # 'Alex_5-1_5-2S_v0.0':(f'{BASE_DIR}/alex_5-1_5-2S_v0.0.h5',None),
    # 'Alex_CC_A_4-30_v0.0': (f'{BASE_DIR}/alex_CC_A_4-30_v0.0.h5',None),
    'Alex_CC_A_v0.0': (f'{BASE_DIR}/alex_CC_A_v0.0.h5',None),
    'Alex_CC_A_4-30_v0.0': (f'{BASE_DIR}/alex_CC_A_4-30_v0.0.h5',None),
    'Jacob_CC_J_v0.0': (f'{BASE_DIR}/jacob_CC_J_v0.0.h5',None)
}

def predict_egg_count_default(image_path, name, model, model2=None):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])

    return egg_count

# make a csv contianing tile predictions from split CD images
def get_tile_preds(name, model, model2):
    mod1 = tf.keras.models.load_model(model)
    mod2 = None
    if model2 != None:
        mod2 = tf.keras.models.load_model(model2)
    
    csv_name = f'{name}_tile_counts_CD_Complete.csv'
    
    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['CD_RootImage', 'Part', 'Bot',])

    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Bot', 'Human'])
        for img in os.listdir(f"{TESTING_SET}"):
            if 'eggs' not in img or 'unsure' in img:
                continue
            label = int(img.split('eggs')[1].split('count')[0])
            predicted_eggs = predict_egg_count_default(f"{TESTING_SET}/{img}", name, mod1, mod2)
            root_image_a = img.split('count')[1]
            root_image_b = root_image_a.split('pt')
            root_image = root_image_b[0]
            root_image = root_image.strip()
            writer.writerow([img, root_image, predicted_eggs, label])
    return csv_name

# make csv with sums of tile predictions and actual counts
# loop thru CD_IMAGES 

# do calculations and write to ModelStatistics.csv