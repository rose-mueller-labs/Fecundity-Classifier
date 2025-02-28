import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import time

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

# usage
# 0
predicted_eggs = predict_egg_count("/home/drosophila-lab/Documents/Fecundity/04-29-cap-800x800-sliced/ACO1 Control 04-29 1 pt1.jpg")
print(f'Predicted number of eggs: {predicted_eggs}')
img = Image.open("/home/drosophila-lab/Documents/Fecundity/04-29-cap-800x800-sliced/ACO1 Control 04-29 1 pt1.jpg")
img.show()
time.sleep(1)

# 1
predicted_eggs = predict_egg_count("/home/drosophila-lab/Documents/Fecundity/04-29-cap-800x800-sliced/ACO1 Control 04-29 1 pt78.jpg")
print(f'Predicted number of eggs: {predicted_eggs}')
img = Image.open("/home/drosophila-lab/Documents/Fecundity/04-29-cap-800x800-sliced/ACO1 Control 04-29 1 pt78.jpg")
img.show()
time.sleep(1)

# i have no idea its 2 or 3?
predicted_eggs = predict_egg_count("/home/drosophila-lab/Documents/Fecundity/04-29-cap-800x800-sliced/nCO1 Lithium 04-29 50 pt82.jpg")
print(f'Predicted number of eggs: {predicted_eggs}')
img = Image.open("/home/drosophila-lab/Documents/Fecundity/04-29-cap-800x800-sliced/nCO1 Lithium 04-29 50 pt82.jpg")
img.show()
time.sleep(1)

# 2
predicted_eggs = predict_egg_count("/home/drosophila-lab/Documents/Fecundity/04-29-cap-800x800-sliced/nCO1 Lithium 04-29 49 pt28.jpg")
print(f'Predicted number of eggs: {predicted_eggs}')
img = Image.open("/home/drosophila-lab/Documents/Fecundity/04-29-cap-800x800-sliced/nCO1 Lithium 04-29 49 pt28.jpg")
img.show()
time.sleep(1)