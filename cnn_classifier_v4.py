import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.utils.class_weight import compute_class_weight

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 100
MAX_EGGS = 42

# Your data loading function with normalization
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

# Load your data
data_dir = '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/SilkyJohnson4'
X, y = load_data(data_dir)

# Stratified train-test split (maintains class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Enhanced data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1, fill_mode='constant'),
    layers.RandomContrast(0.1)
])

# Regression-optimized model architecture
def build_regression_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        data_augmentation,
        layers.Rescaling(1./255),  # Built-in normalization
       
        # Enhanced feature extraction
        layers.Conv2D(64, (5,5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
       
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
       
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
       
        # Regression-specific head
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='linear')  # Continuous output
    ])
   
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model

# Class balancing for imbalanced counts
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

# Training callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

model_checkpoint = callbacks.ModelCheckpoint(
    'best_regression_model.keras',
    save_best_only=True,
    monitor='val_mse'
)

# Cross-validation with per-class tracking
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
per_class_results = {cls: [] for cls in np.unique(y)}

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
   
    model = build_regression_model()
   
    model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
   
    # Load best model weights
    model.load_weights('best_regression_model.keras')
   
    # Per-class MSE evaluation
    y_pred = model.predict(X_test).flatten()
    for cls in np.unique(y_test):
        cls_mask = y_test == cls
        if np.sum(cls_mask) > 0:
            cls_mse = mean_squared_error(y_test[cls_mask], y_pred[cls_mask])
            per_class_results[cls].append(cls_mse)
model.save('best_regression_model.keras')
# Final evaluation
final_model = build_regression_model()
final_model.load_weights('best_regression_model.keras')
test_mse = final_model.evaluate(X_test, y_test, verbose=0)[1]

# Save per-class results
with open('per_class_mse.txt', 'w') as f:
    f.write(f"Overall Test MSE: {test_mse:.4f}\n\n")
    for cls in sorted(per_class_results.keys()):
        avg_mse = np.mean(per_class_results[cls])
        std_mse = np.std(per_class_results[cls])
        f.write(f"Class {cls}:\n")
        f.write(f"  Average MSE: {avg_mse:.4f} ± {std_mse:.4f}\n")
        f.write(f"  Individual Fold MSEs: {[f'{mse:.4f}' for mse in per_class_results[cls]]}\n\n")