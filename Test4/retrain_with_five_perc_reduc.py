import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image
from sklearn.metrics import r2_score

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42

# Data augmentation pipeline (added to see how it chngs may rm later)
data_augmentation = tf.keras.Sequential([     
    layers.RandomFlip("horizontal_and_vertical"),   
    layers.RandomRotation(0.2),                    
    layers.RandomZoom(0.1),                        
])                                                 

# load and preprocess
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

# load
data_dir = '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/SilkyJohnson2'
X, y = load_data(data_dir)

# normalize
X = X.astype('float32') / 255.0

# stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# After stratified train-test split
results = []
current_X_train = X_train.copy()
current_y_train = y_train.copy()

while len(current_X_train) > 0:
    # Build and compile fresh model each iteration
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
   
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
   
    # Train with silent output
    model.fit(current_X_train, current_y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=0)
   
    # Evaluate and store results
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results.append((len(current_X_train), test_loss, test_acc))
   
    # Reduce dataset size
    reduce_size = int(len(current_X_train) * 0.05)
    if reduce_size == 0: break
    current_X_train = current_X_train[:-reduce_size]
    current_y_train = current_y_train[:-reduce_size]

# Save results analysis
with open('size_reduction_results.txt', 'w') as f:
    for sample_count, loss, acc in results:
        f.write(f"Samples: {sample_count}\tLoss: {loss:.4f}\tAccuracy: {acc:.4f}\n")