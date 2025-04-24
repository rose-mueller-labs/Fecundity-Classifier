'''Testing within one dataset: All Lithium Datasets'''

MODEL = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v1.keras"

ONE_DATASET_CD = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/DATA/AllLithiumData"

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, KFold
from PIL import Image
import time
import csv

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42

# def model 
model = tf.keras.models.load_model(MODEL)

# predict egg count in a new image
def predict_egg_count(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

image_files = []
labels = []
for file_name in os.listdir(ONE_DATASET_CD):
    if 'eggs' in file_name and 'unsure' not in file_name:
        # Extract numeric label between 'eggs' and 'count'
        label_str = file_name.split('eggs')[1].split('count')[0].strip('_')
        label = int(label_str)
        image_files.append(os.path.join(ONE_DATASET_CD, file_name))
        labels.append(label)

# Convert to numpy arrays
image_files = np.array(image_files)
labels = np.array(labels)

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create CSV with headers
with open("lithium_training_vs_lithium_testing_five_fold.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ImagePath', 'Bot', 'Human'])
   
    # Iterate through folds
    for fold, (train_idx, test_idx) in enumerate(kf.split(image_files)):
        print(f"Processing fold {fold+1}/5")
       
        # Get test set for this fold
        test_files = image_files[test_idx]
        test_labels = labels[test_idx]
       
        # Predict and write results
        for file_path, true_label in zip(test_files, test_labels):
            predicted = predict_egg_count(file_path)
            writer.writerow([os.path.basename(file_path), predicted, true_label])

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import curve_fit

df = pd.read_csv('lithium_training_vs_lithium_testing_five_fold.csv')

def get_class_counts():
    counts = dict()
    for index, row in df.iterrows():
        label = int(row['Human'])
        if label not in counts:
            counts[label]=1
        else:
            counts[label]+=1
    
    return counts

total_mse = mean_squared_error(df['Human'], df['Bot'])

print(f'TOTAL MSE: {total_mse}')
print(f'TOTAL RMSE: {np.sqrt(total_mse)}')

mse_by_counts = df.groupby('Human').apply(lambda x: np.mean((x['Human']-x['Bot'])**2))

print("MSE FOR EACH COUNT: ")
print(mse_by_counts)
img_counts = get_class_counts()
print("img counts", img_counts)

## graphing mse & class

plt.figure(figsize=(10,6))
mse_by_counts.plot(kind='bar')
plt.title('Error in Egg Counts per Class')
plt.xlabel('Class/Correct Egg Count')
plt.ylabel('Error in Prediction (Mean Squared Error)')
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.axhline(y=total_mse, color='red', linestyle='--', label='Overall Error (MSE)')
plt.legend()
plt.tight_layout()
plt.plot()
plt.savefig("SJ2_lithium_training_vs_lithium_testing_five_fold")