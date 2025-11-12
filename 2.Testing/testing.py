import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import time
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit

TESTING_SETS=[
    "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/Winter 2017 2 21 C pops cap-sliced",
    "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/5-4-cap-sliced-Julie"
]

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 12
BASE_DIR="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/1.DataProcessing/model_architecture/models"
UPDATE_CSV="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/ModelStatistics_54.csv"

TOP_MODEL_NAMES_AND_PATHS = {
    # 'Alex_FecundityModelMoDataV1': (f'{BASE_DIR}/fecundity_model_mo_data_v1.h5', None),
    # 'Alex_4-30_5-1_v0.0':(f'{BASE_DIR}/alex_4-30_5-1_v0.0.h5',None), # thursday 4 pm
    # 'Alex_4-30_v0.0':(f'{BASE_DIR}/alex_4-30_v0.0.h5',None),
    # 'Alex_5-1_v0.0':(f'{BASE_DIR}/alex_5-1_v0.0.h5',None),
    # 'Alex_5-2S_v0.0':(f'{BASE_DIR}/alex_5-2S_v0.0.h5',None),
    # 'Alex_5-2O_v0.0':(f'{BASE_DIR}/alex_5-2O_v0.0.h5',None),
    # 'Alex_BW_4-30_5-1_v0.0':(f'{BASE_DIR}/alex_BW_4-30_5-1_v0.0.h5',None),
    # 'Alex_BW_4-30_v0.0':(f'{BASE_DIR}/alex_BW_4-30_v0.0.h5',None),
    # 'Alex_BW_5-1_v0.0':(f'{BASE_DIR}/alex_BW_5-1_v0.0.h5',None),
    # 'Alex_4-30_5-1_5-2O_v0.0':(f'{BASE_DIR}/alex_4-30_5-1_5-2O_v0.0.h5',None),
    # 'Alex_4-30_5-1_5-2S_v0.0':(f'{BASE_DIR}/alex_4-30_5-1_5-2S_v0.0.h5',None),
    # 'Alex_4-30_5-2O_v0.0':(f'{BASE_DIR}/alex_4-30_5-2O_v0.0.h5',None), # actual: 11/09 17:37 (73 hrs) @ 360k
    # 'Alex_4-30_5-2S_v0.0':(f'{BASE_DIR}/alex_4-30_5-2S_v0.0.h5',None),
    # 'Alex_5-1_5-2O_v0.0':(f'{BASE_DIR}/alex_5-1_5-2O_v0.0.h5',None),
    # 'Alex_5-1_5-2S_v0.0':(f'{BASE_DIR}/alex_5-1_5-2S_v0.0.h5',None),
    # 'Alex_CC_A_4-30_v0.0': (f'{BASE_DIR}/alex_CC_A_4-30_v0.0.h5',None),
    # 'Alex_CC_A_v0.0': (f'{BASE_DIR}/alex_CC_A_v0.0.h5',None),
    # 'Alex_CC_A_4-30_v0.0': (f'{BASE_DIR}/alex_CC_A_4-30_v0.0.h5',None),
    # 'Jacob_CC_J_v0.0': (f'{BASE_DIR}/jacob_CC_J_v0.0.h5',None) # expected: 11/11 @ 6 pm
    'Alex_4-30_5-1_5-2O_CC_A_v0.0': (f'{BASE_DIR}/alex_4-30_5-1_5-2O_CC_A_v0.0.h5', None),
    'Alex_4-30_5-1_CC_A_v0.0': (f'{BASE_DIR}/alex_4-30_5-1_CC_A_v0.0.h5', None),
    'Alex_4-30_CD_5-1_CC_A_v0.0': (f'{BASE_DIR}/alex_4-30_CD_5-1_CC_A_v0.0.h5', None),
    'Alex_5-1_5-2S_CC_A_v0.0': (f'{BASE_DIR}/alex_5-1_5-2S_CC_A_v0.0.h5', None),
    'JAlex_4-30_5-1_5-2S_CC_A_CC_J_v0.0': (f'{BASE_DIR}/jalex_4-30_5-1_5-2S_CC_A_CC_J_v0.0.h5', None),
    'JAlex_4-30_5-1_5-2S_CC_A_CC_J_CD_v0.0': (f'{BASE_DIR}/jalex_4-30_5-1_5-2S_CC_A_CC_J_CD_v0.0.h5', None),
    'JAlex_4-30_5-1_CC_A_CC_J_v0.0': (f'{BASE_DIR}/jalex_4-30_5-1_CC_A_CC_J_v0.0.h5', None),
    'Alex_GS_4-30_5-1_5-2O_CC_A_v0.0': (f'{BASE_DIR}/jalex_GS_4-30_5-1_5-2O_CC_A_v0.0.h5', None)
}

def predict_egg_count_NoZ(image_path):
    egg_count_model = tf.keras.models.load_model("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_no_zeros.keras")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = egg_count_model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])
   
    return egg_count+1

def predict_egg_count_aug_str_v1(image_path):
    egg_count_model = tf.keras.models.load_model("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v1.keras")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = egg_count_model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

def predict_egg_count_aug_str_v4(image_path):
    egg_count_model = tf.keras.models.load_model("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v4.keras")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = egg_count_model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

def predict_egg_count_default(image_path, name, model, model2=None):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])

    if (model2 != None):
        if egg_count != 0:
            if (name == "DualZonAndNoZ"):
                egg_count = predict_egg_count_NoZ(image_path)
            if (name == "DualZONandAugStrV1"):
                egg_count = predict_egg_count_aug_str_v1(image_path)
            if (name == "DualZONandAugStrV4"):
                egg_count = predict_egg_count_aug_str_v4(image_path)
    return egg_count

def predict_egg_count_default_nom2(image_path, name, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])

    return egg_count

def get_tile_preds_data_file(name, model, model2):
    mod1 = tf.keras.models.load_model(model)
    mod2 = None
    if model2 != None:
        mod2 = tf.keras.models.load_model(model2)
    csv_name = f'{name}_tile_counts_{'CD' if 'Winter 2017 2 21 C pops cap-sliced' in TESTING_SET else '54'}.csv'
    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Bot', 'Human'])

    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Bot', 'Human'])
        for img in os.listdir(f"{TESTING_SET}"):
            if 'eggs' not in img or 'unsure' in img:
                continue
            label = int(img.split('eggs')[1].split('count')[0])
            predicted_eggs = predict_egg_count_default(f"{TESTING_SET}/{img}", name, mod1, mod2)
            root_image_a = img.split('count')[1]
            root_image_b = root_image_a.split('pt')
            root_image = root_image_b[0]
            root_image = root_image.strip()
            writer.writerow([img, root_image, predicted_eggs, label])
    return csv_name

def get_actual_total(csv_path, name):
    actual_csv_name = f'{name}_sums_{'CD' if 'Winter 2017 2 21 C pops cap-sliced' in TESTING_SET else '54'}_CSV.csv'
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
        actual_counts[row['RootImage']] += row['Bot']
        expected_counts[row['RootImage']] += row['Human']
    
    with open(actual_csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RootImage', 'BotSum', 'HumanSum'])
        for root_img, actual in actual_counts.items():
            expected = expected_counts[root_img]
            writer.writerow([root_img, actual, expected])
    return actual_csv_name

def MSEs_get_class_counts(cap_csvs):
    df = pd.read_csv(cap_csvs)
    counts = dict()
    for index, row in df.iterrows():
        label = int(row['HumanSum'])
        if label not in counts:
            counts[label]=1
        else:
            counts[label]+=1
    
    return counts

# to_write = {'ModelPath': "", "TrainingSet": "", 
#             "CD_MSE": -1, "CD_RMSE": -1, "CD_R2": -1,
#             "54_MSE": -1, "54_RMSE": -1, "54_R2": -1}

def MSEs_metrics_and_graph(caps_csvs, name):
    df = pd.read_csv(caps_csvs)

    total_mse = mean_squared_error(df['HumanSum'], df['BotSum'])
    total_r2 = r2_score(df['HumanSum'], df['BotSum'])

    mse_by_counts = df.groupby('HumanSum').apply(lambda x: np.mean((x['HumanSum']-x['BotSum'])**2))
    std_dev_by_counts = df.groupby('HumanSum').apply(lambda x: np.std(x['BotSum']))
    var_by_counts = df.groupby('HumanSum').apply(lambda x: np.var(x['BotSum']))
    mean_by_counts = df.groupby('HumanSum').apply(lambda x: np.mean(x['BotSum']))

    r2_score_by_counts = df.groupby('HumanSum').apply(lambda x: r2_score(x['HumanSum'], x['BotSum']))

    if "CD" in TESTING_SET:
        TO_WRITE['CD_MSE'] = [total_mse]
        TO_WRITE['CD_R2'] = [total_r2]
        TO_WRITE['CD_RMSE'] = [np.sqrt(total_mse)]
    else:
        TO_WRITE['54_MSE'] = [total_mse]
        TO_WRITE['54_R2'] = [total_r2]
        TO_WRITE['54_RMSE'] = [np.sqrt(total_mse)]
    
    with open(f"{name}_metrics_{'CD' if 'Winter 2017 2 21 C pops cap-sliced' in TESTING_SET else '54'}.txt", "w") as file:
        print("MSE FOR EACH COUNT: \n", file=file)
        print(mse_by_counts, file=file)
        print('\n', file=file)
        img_counts = MSEs_get_class_counts(caps_csvs)
        print(f"img counts {img_counts}", file=file)
        print('\n', file=file)

if __name__ == '__main__':
    for name, paths in TOP_MODEL_NAMES_AND_PATHS.items():
        TO_WRITE = {'ModelPath': [], "TrainingSet": [], "Iteration": [],
            "CD_MSE": [], "CD_RMSE": [], "CD_R2": [],
            "54_MSE": [], "54_RMSE": [], "54_R2": []}
        
        for TESTING_SET in TESTING_SETS:
            print(f'Getting tiles for {name}')
            tiles_csv_name = get_tile_preds_data_file(name, paths[0], paths[1])
            print(f'Getting sums for {name}')
            sums_csv_name = get_actual_total(tiles_csv_name, name)
            print(f'Getting metrics for {name}')
            MSEs_metrics_and_graph(sums_csv_name, name)

        # write results
        TO_WRITE['ModelPath'] = [paths[0]]
        TO_WRITE['TrainingSet'] = ['_'.join(paths[0].split('/')[-1].split('.h5')[0].split('_')[1:-1])]
        TO_WRITE['Iteration'] = [paths[0].split('/')[-1].split('.h5')[0].split('_')[-1].split('.')[-1]]
        
        if not TO_WRITE['CD_MSE']:
            TO_WRITE['CD_MSE'] = [-1]
            TO_WRITE['CD_RMSE'] = [-1]
            TO_WRITE['CD_R2'] = [-1]
        if not TO_WRITE['54_MSE']:
            TO_WRITE['54_MSE'] = [-1]
            TO_WRITE['54_RMSE'] = [-1]
            TO_WRITE['54_R2'] = [-1]
        
        new_df = pd.DataFrame.from_dict(TO_WRITE)
        print(new_df)

        if os.path.exists(UPDATE_CSV):
            new_df.to_csv(UPDATE_CSV, mode='a', index=False, header=False)
        else:
            new_df.to_csv(UPDATE_CSV, mode='a', index=False, header=True)