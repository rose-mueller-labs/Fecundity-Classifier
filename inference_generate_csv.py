import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import time
import csv

# v1 is better than v2

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42

# def model 
model = tf.keras.models.load_model('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_mo_data_v3.h5')


# predict egg count in a new image
def predict_egg_count(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

ROOT_DIR = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/Winter 2017 2 21 C pops cap-sliced"
with open("testing_on_winter_mo_v2.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ImagePath', 'Bot', 'Human'])
    for file in os.listdir(ROOT_DIR):
        if 'eggs' not in file or 'unsure' in file:
            continue
        label = file.split('eggs')[1].split('count')[0]
        predicted_eggs = predict_egg_count(f"{ROOT_DIR}/{file}")
        writer.writerow([file, predicted_eggs, label])