'''
                                NOTES:
Added a bunch of changes that should help with the imbalanced dataset issue.
Taking another stab at a regression x computer vision model.
Using denseweights + stratification
SilkyJohnson5 is the set, included Alex's clusters into this and Jennifer's
04-29 set, and Angela's 5-4.
Eval txt file is bin-based.
Do not run anything else on this computer while the model trains, it should be training
for around an hour.

'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from denseweight import DenseWeight

IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42
N_BINS = 10

# Enhanced data augmentation --> added brightness and contrast, will see how this affects
def create_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2)
    ])

# Custom weighted MSE loss --> i think this is better? google said so
def create_weighted_mse(class_weight_dict):
    def weighted_mse(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bins = np.linspace(0, MAX_EGGS, N_BINS+1)[1:-1]
        bin_indices = tf.cast(tf.histogram_fixed_width_bins(y_true, [0.0, float(MAX_EGGS)], N_BINS), tf.int32)
        class_weights = tf.gather(tf.constant(list(class_weight_dict.values()), dtype=tf.float32), bin_indices)
        return tf.reduce_mean(class_weights * tf.square(y_true - y_pred))
    return weighted_mse

# Load and preprocess data
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

data_dir = '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/SilkyJohnson5'
X, y = load_data(data_dir)
X = X.astype('float32') / 255.0

# Create target bins for stratification
y_bins = np.digitize(y, bins=np.linspace(0, MAX_EGGS, N_BINS+1)[1:-1])

# Train-test split with continuous stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_bins
)

# Calculate class weights
class_counts = np.bincount(np.digitize(y_train, bins=np.linspace(0, MAX_EGGS, N_BINS+1)[1:-1]))
class_weights = 1.0 / (class_counts + 1e-7)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Density-based weighting --> supposed to help with imbalance
dw = DenseWeight(alpha=0.5)
sample_weights = dw.fit(y_train)

# Model architecture
def create_model():
    return models.Sequential([
        create_augmentation(),
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Regression output
    ])

# Stratified K-Fold Cross-Validation --> also helps with imbalance
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, np.digitize(y_train, bins=np.linspace(0, MAX_EGGS, N_BINS+1)[1:-1]))):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    y_train_fold.astype('float32')
    y_val_fold.astype('float32')
    sample_weights_fold = sample_weights[train_idx]

    model = create_model()
    model.compile(
        optimizer='adam',
        loss=create_weighted_mse(class_weight_dict),
        metrics=['mae']
    )

    model.fit(
        X_train_fold, y_train_fold,
        sample_weight=sample_weights_fold,
        validation_data=(X_val_fold, y_val_fold),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2
    )

# Final evaluation
model.save('fecundity_model_aug_str_v5.keras')

# Per-bin evaluation
y_pred = model.predict(X_test).flatten()
test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

# Create bins for evaluation
bins = np.array([0, 8, 16, MAX_EGGS])
bin_indices = np.digitize(y_test, bins)

with open('model_evaluation.txt', 'w') as f:
    f.write(f"Overall MSE: {test_mse:.4f}\n")
    f.write(f"Overall MAE: {test_mae:.4f}\n")
    f.write(f"R² Score: {test_r2:.4f}\n\n")
   
    for bin_idx in range(1, len(bins)):
        mask = bin_indices == bin_idx
        if np.any(mask):
            bin_mse = mean_squared_error(y_test[mask], y_pred[mask])
            f.write(f"Bin {bins[bin_idx-1]}-{bins[bin_idx]} MSE: {bin_mse:.4f}\n")