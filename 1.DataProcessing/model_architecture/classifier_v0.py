import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import csv
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

## TO DO
# - remove data augmentation
# - add k stratified
##

print("Did you update max eggs?")
time.sleep(10)

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42
N_BINS = 10

# load and preprocess ** might need to fix this st all classes are balanced, lots of 0s atm
def load_data(data_dir): # dir has subdirs of classes
    images = []
    labels = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(int(class_name))
    return np.array(images), np.array(labels)

def create_weighted_mse(class_weight_dict):
    def weighted_mse(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bins = np.linspace(0, MAX_EGGS, N_BINS+1)[1:-1]
        bin_indices = tf.cast(tf.histogram_fixed_width_bins(y_true, [0.0, float(MAX_EGGS)], N_BINS), tf.int32)
        class_weights = tf.gather(tf.constant(list(class_weight_dict.values()), dtype=tf.float32), bin_indices)
        return tf.reduce_mean(class_weights * tf.square(y_true - y_pred))
    return weighted_mse

# load 
data_dir = input("Paste partioned data directory path here: ")
which_person = input("'jacob' or 'alex': ")
iteration = int(input("Which iteration is this training on: "))
model_name = f"{which_person}_{iteration}.h5"
X, y = load_data(data_dir)

# normalize 
X = X.astype('float32') / 255.0

# split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create 
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(MAX_EGGS + 1, activation='softmax')  # +1 to include 0 eggs
])

# compile 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# data aug
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

# train 
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)

y_pred = model.predict(X_test).flatten()
test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

# Create bins for evaluation
bins = np.array([0, 8, 16, MAX_EGGS])
bin_indices = np.digitize(y_test, bins)

# eval the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
with open (f'EVAL_{model_name}.txt', 'w') as file:
    file.write(f'test accuracy: {test_acc}\n\n')
    file.write(f'test loss: {test_loss}\n\n')

    file.write(f"Overall MSE: {test_mse:.4f}\n")
    file.write(f"Overall MAE: {test_mae:.4f}\n")
    file.write(f"R² Score: {test_r2:.4f}\n\n")
   
    for bin_idx in range(1, len(bins)):
        mask = bin_indices == bin_idx
        if np.any(mask):
            bin_mse = mean_squared_error(y_test[mask], y_pred[mask])
            file.write(f"Bin {bins[bin_idx-1]}-{bins[bin_idx]} MSE: {bin_mse:.4f}\n")

# save
model.save(model_name)