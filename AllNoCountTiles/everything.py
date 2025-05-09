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

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42

TESTING_SET = "/home/drosophila-lab/Documents/All Lithium Caps-sliced"

TOP_MODEL_NAMES_AND_PATHS = {
    # 'FecundityModelMoDataV3': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_mo_data_v3.h5', None),
    # 'FecundityModelMoDataV4': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_mo_data_v4.h5', None),
    # 'FecundityModelV1': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_v1.h5', None),
    'FecundityModelMoDataV1': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_mo_data_v1.h5', None),
    # 'AugStrV1': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v1.keras', None),
    # 'AugStrV3': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v3.keras', None),
    # 'AugStrV4': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v4.keras', None),
    # 'Regression': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/best_regression_model.keras', None),
    # 'DualZONandAugStrV1': ("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_zero_or_not.keras", '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v1.keras'),
    # 'DualZONandAugStrV4': ("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_zero_or_not.keras", '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v4.keras') ,
    # 'DualZonAndNoZ': ("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_zero_or_not.keras", "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_no_zeros.keras")
}

def predict_egg_count_NoZ(image_path):
    egg_count_model = tf.keras.models.load_model("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_no_zeros.keras")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = egg_count_model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])
   
    return egg_count+1

def predict_egg_count_aug_str_v1(image_path):
    egg_count_model = tf.keras.models.load_model("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v1.keras")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = egg_count_model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

def predict_egg_count_aug_str_v4(image_path):
    egg_count_model = tf.keras.models.load_model("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v4.keras")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = egg_count_model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

def predict_egg_count_default(image_path, name, model, model2=None):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])

    if (model2 != None):
        if egg_count != 0:
            if (name == "DualZonAndNoZ"):
                egg_count = predict_egg_count_NoZ(image_path)
            if (name == "DualZONandAugStrV1"):
                egg_count = predict_egg_count_aug_str_v1(image_path)
            if (name == "DualZONandAugStrV4"):
                egg_count = predict_egg_count_aug_str_v4(image_path)
    return egg_count

def get_tile_preds_data_file(name, model, model2):
    mod1 = tf.keras.models.load_model(model)
    mod2 = None
    if model2 != None:
        mod2 = tf.keras.models.load_model(model2)
    csv_name = f'{name}_tile_counts_all_lith.csv'
    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Bot'])

    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Bot'])
        for img in os.listdir(f"{TESTING_SET}"):
            predicted_eggs = predict_egg_count_default(f"{TESTING_SET}/{img}", name, mod1, mod2)
            root_image = img.split('pt')[0].strip()
            writer.writerow([img, root_image, predicted_eggs])
    return csv_name

def get_actual_total(csv_path, name):
    actual_csv_name = f'{name}_sums_CSV.csv'
    # get all unique names => get the ones with the same names => get the actual counts => sum
    df = pd.read_csv(csv_path)
    root_image_names = np.array(df['RootImage'].unique())
    bot_sums = dict()

    for cap_name in root_image_names:
        bot_sums[cap_name] = 0

    for index, row in df.iterrows():
        bot_sums[row['RootImage']] += row['Bot']
    
    with open(actual_csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RootImage', 'BotSum'])
        for root_img, bot_sum in bot_sums.items():
            writer.writerow([root_img, bot_sum])
    return actual_csv_name

if __name__ == '__main__':
    for name, paths in TOP_MODEL_NAMES_AND_PATHS.items():
        print(f'Getting tiles for {name}')
        tiles_csv_name = get_tile_preds_data_file(name, paths[0], paths[1])
        print(f'Getting sums for {name}')
        sums_csv_name = get_actual_total(tiles_csv_name, name)
