import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, StratifiedKFold

IMG_HEIGHT, IMG_WIDTH = 800, 800
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 143

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

def load_data(data_dir):
    images = []
    labels = []
    for class_name in os.listdir(data_dir):
        if not class_name.isdigit():
            continue
        class_idx = int(class_name)
        if class_idx < 1 or class_idx > MAX_EGGS:
            continue
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(class_idx - 1)
    return np.array(images), np.array(labels)

data_dir = '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/FullCapsClean'
X, y = load_data(data_dir)
print('Unique labels:', np.unique(y))
print('Label min:', y.min(), 'Label max:', y.max())
print('X shape:', X.shape)
print('y shape:', y.shape)

X = X.astype('float32') / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    model = models.Sequential([
        data_augmentation,
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(MAX_EGGS, activation='softmax')
    ])
   
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
   
    model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

model_name = 'fecundity_model_full_caps_v1.keras'
model.save(model_name)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

with open(f'eval.txt_{model_name}', 'w') as file:
    file.write(f'test accuracy: {test_acc}\n')
    file.write(f'test loss: {test_loss}\n')
    file.write('---DISREGARD BELOW THIS---\n')
    file.write('y_pred:\n')
    file.write(np.array2string(y_pred_classes))
    file.write('\n\n')
    file.write('y_test:\n')
    file.write(np.array2string(y_test))
    file.write('\n\n')
