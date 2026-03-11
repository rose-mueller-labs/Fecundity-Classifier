import argparse
import os
import numpy as np
import tensorflow as tf
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from image_shredder import main as shred

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42
BASE_DIR="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/1.DataProcessing/model_architecture/models"

# add models here as needed
MODELS = {
    'DEFAULT_MODEL': (f'{BASE_DIR}/alex_4-30_5-1_CC_A_v0.0.h5', None),
    'CLUSTERING_MODEL': (f'{BASE_DIR}/alex_5-1_5-2S_v0.0.h5', None)
}

DEFAULT_MODEL = "DEFAULT_MODEL"

# parse CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Count Drosophila eggs in petri dish images."
    )
    parser.add_argument("--n", dest="model_name", default=DEFAULT_MODEL, metavar="MODEL_NAME",
        help=f"Model name/key to use for counting (default: {DEFAULT_MODEL}). "
             f"Available: {list(DEFAULT_MODEL.keys())}")
    parser.add_argument("--D", dest="data_dir", required=True, metavar="DIRECTORY",
        help="Path to directory containing caps images (NOT THE FULL GRID).")
    parser.add_argument("--C", dest="cluster", action="store_true", default=False,
        help="Use CLUSTERING_MODEL for inference instead of the selected model (default: False)")
    return parser.parse_args()

def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

def predict_egg_count_default(image_path, name, model, model2=None):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])

    return egg_count

def get_tile_preds(name, model, model2, set_name, TESTING_SET):
    mod1 = tf.keras.models.load_model(model)
    mod2 = None
    if model2 != None:
        mod2 = tf.keras.models.load_model(model2)

    csv_name = f'{set_name}_{name}_tile_counts_cage_testing.csv'

    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'Part', 'Bot'])
        for img in os.listdir(f"{TESTING_SET}"):
            root_image = img.split("JPG")[0].split(".")[0]
            predicted_eggs = int(predict_egg_count_default(f"{TESTING_SET}/{img}", name, mod1, mod2))
            part = img.split("pt")[-1].split('.')[0]
            writer.writerow([root_image, part, predicted_eggs])
    return csv_name

def get_sums(csv_path, name, set_name):
    actual_csv_name = f'{set_name}_{name}_sums_cage_testing.csv'

    df = pd.read_csv(csv_path)
    root_image_names = np.array(df['ImageName'].unique())

    actual_counts = dict()
    for cap_name in root_image_names:
        actual_counts[cap_name] = 0

    for index, row in df.iterrows():
        actual_counts[row['ImageName']] += row['Bot']
    
    with open(actual_csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'BotSum'])
        for root_img, actual in actual_counts.items():
            writer.writerow([root_img, actual])
    return actual_csv_name

if __name__ == '__main__':
    args = parse_args()

    # if --C is set, override model selection with cluster_model
    if args.cluster:
        name = 'CLUSTERING_MODEL'
    else:
        name = args.model_name

    if name not in MODELS:
        print(f"Model '{name}' not found. Available: {list(MODELS.keys())}")
        exit(1)

    set_name = os.path.basename(args.data_dir.rstrip("/\\")) # derive set_name from directory's name

    # slice images if not already done --> paths now from args.data_dir
    TESTING_SET = f"{args.data_dir}-sliced"
    if not os.path.isdir(TESTING_SET):
        shred(args.data_dir)

    paths = MODELS[name]

    print(f'Getting tiles for {name}')
    print_time()
    tiles_csv_name = get_tile_preds(name, paths[0], paths[1], set_name, TESTING_SET)
    print(f'Getting sums for {name}')
    print_time()
    sums_csv_name = get_sums(tiles_csv_name, name, set_name)
    print(f'Done. Results saved to: {sums_csv_name}')