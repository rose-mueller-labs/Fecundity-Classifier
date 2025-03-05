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
MAX_EGGS = 9 

# def model 
model = tf.keras.models.load_model('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_v1.h5')

# predict egg count in a new image
def predict_egg_count(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

with open("alex.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ImagePath', 'Predicted', 'Actual'])

# usage
# 0
ROOT_DIR = "/home/drosophila-lab/Documents/Fecundity/AlexanderDataClasses"
with open("alex.csv", "w", newline='') as file:
    for label in os.listdir("/home/drosophila-lab/Documents/Fecundity/AlexanderDataClasses"):
        for img in os.listdir(f"{ROOT_DIR}/{label}"):
            writer = csv.writer(file)
            predicted_eggs = predict_egg_count(f"{ROOT_DIR}/{label}/{img}")
            writer.writerow([img, predicted_eggs, label])