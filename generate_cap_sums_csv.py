import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import csv
import pandas as pd

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 9 

# def model 
model = tf.keras.models.load_model('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v1.keras')

# predict egg count in a new image
def predict_egg_count(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

# create a csv file with the predicted eggs and the actual
def create_csv_data_file(csv_name, data_path):
    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Actual', 'Expected'])

    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Actual', 'Expected'])
        for label in os.listdir(data_path):
            for img in os.listdir(f"{data_path}/{label}"):
                predicted_eggs = predict_egg_count(f"{data_path}/{label}/{img}")
                root_image_a = img.split('count')[1]
                root_image_b = root_image_a.split('pt')
                root_image = root_image_b[0]
                root_image = root_image.strip()
                writer.writerow([img, root_image, predicted_eggs, label])

def get_actual_total(csv_path, actual_csv_name):
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
    
    with open(actual_csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RootImage', 'ActualSum', 'ExpectedSum'])
        for root_img, actual in actual_counts.items():
            expected = expected_counts[root_img]
            writer.writerow([root_img, actual, expected])



if __name__ == '__main__':
    create_csv_data_file('winter_indivs.csv', "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/Winter")
    get_actual_total('winter_indivs.csv', 'winter_indivs_sums.csv')