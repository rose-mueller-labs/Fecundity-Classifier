import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.metrics
import tensorflow.keras.losses
from sklearn.model_selection import train_test_split
from collections import Counter
import time
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from mses_module import get_mse_table_and_plot_and_csvs

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([    
    layers.RandomFlip("horizontal_and_vertical"),  
    layers.RandomRotation(0.2),                    
    layers.RandomZoom(0.1),                        
])

# New class-aware reduction function
def reduce_each_class_fixed(X_train, y_train, reduction_fraction=0.05):
    unique_classes = np.unique(y_train)
    indices_to_keep = []
    for cls in unique_classes:
        cls_indices = np.where(y_train == cls)[0]
        reduce_size = int(len(cls_indices) * reduction_fraction)
        keep_indices = cls_indices[:-reduce_size] if reduce_size > 0 else cls_indices
        indices_to_keep.extend(keep_indices)
    indices_to_keep = np.sort(indices_to_keep)
    return X_train[indices_to_keep], y_train[indices_to_keep]

# load and preprocess
def load_data(data_dir):
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

# load data
data_dir = '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/SilkyJohnson4'
X, y = load_data(data_dir)

# normalize
X = X.astype('float32') / 255.0

# stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize training data
results = []
current_X_train = X_train.copy()
current_y_train = y_train.copy()

while len(current_X_train) > 0:
    # Build fresh model each iteration
    model = models.Sequential([
        data_augmentation,
        layers.Conv2D(32, (3,3), activation='relu',
                     input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(MAX_EGGS+1, activation='softmax')
    ])

    def sparse_mse(y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=MAX_EGGS+1)
        return tf.reduce_mean(tf.square(y_true_onehot - y_pred))
   
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=[sparse_mse])
   
    # Train with silent output
    model.fit(current_X_train, current_y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1)
   
    # Evaluate and store results (now with correct MSE naming)
    test_loss, test_sparse_mse = model.evaluate(X_test, y_test, verbose=0)
    results.append((len(current_X_train), test_loss, test_sparse_mse))

    ### test in winter 2017 CD
    get_mse_table_and_plot_and_csvs(model, f"XTrain{len(current_X_train)}")
    ### end winter testing
   
    # Class-aware reduction with safety check
    prev_size = len(current_X_train)
    current_X_train, current_y_train = reduce_each_class_fixed(
        current_X_train, current_y_train, 0.05
    )
    if len(current_X_train) == prev_size:  # Prevent infinite loop
        break

# Save results with correct MSE labeling
with open('size_reduction_results_v2.txt', 'w') as f:
    for sample_count, loss, mse in results:
        f.write(f"Samples: {sample_count}\tLoss: {loss:.4f}\tMSE: {mse:.4f}\n")