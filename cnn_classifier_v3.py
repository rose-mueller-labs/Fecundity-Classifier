import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import pandas as pd
from collections import Counter

# Constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42

# Data Loading with Balancing, with each pass we make sure there is at least One sample per class in the training set
def load_data(data_dir):
    images = []
    labels = []
    class_counts = {}
   
    # first pass - count samples
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            count = len(os.listdir(class_dir))
            class_counts[int(class_name)] = count
            print(f"Class {class_name}: {count} samples")

    # second pass - load data
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.listdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = tf.keras.preprocessing.image.load_img(
                    img_path,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    color_mode='rgb' if CHANNELS == 3 else 'grayscale'
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(int(class_name))
   
    return np.array(images), np.array(labels)

# Load and preprocess data
data_dir = '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/SilkyJohnson2'
X, y = load_data(data_dir)
X = X.astype('float32') / 255.0

# Class Balancing
min_samples = 1
unique_classes = np.unique(y)
X_balanced, y_balanced = [], []

for cls in unique_classes:
    class_indices = np.where(y == cls)[0]
    if len(class_indices) < min_samples:
        raise ValueError(f"Class {cls} has less than {min_samples} samples")
    selected_indices = np.random.choice(class_indices, size=max(len(class_indices), min_samples), replace=True)
    X_balanced.append(X[selected_indices])
    y_balanced.append(y[selected_indices])

X = np.concatenate(X_balanced)
y = np.concatenate(y_balanced)

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Data Augmentation (may or may not remove later)
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

# Class Weighting
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model Architecture
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(MAX_EGGS+1, activation='softmax')
    ])
   
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# K-Fold Training
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
]

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    print(f"\nFold {fold+1}")
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
   
    model = create_model()
   
    history = model.fit(
        aug.flow(X_train_fold, y_train_fold, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train_fold) // BATCH_SIZE,
        validation_data=(X_val_fold, y_val_fold),
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

# final Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Per-Class MSE Evaluation (mse_calc here)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import mean_squared_error
for cls in np.unique(y_test):
    cls_mask = y_test == cls
    if sum(cls_mask) > 0:
        cls_mse = mean_squared_error(y_test[cls_mask], y_pred_classes[cls_mask])
        print(f"Class {cls:2d} MSE: {cls_mse:.4f} - {sum(cls_mask)} samples")

model.save('fecundity_model_eq_cls_v1.h5')