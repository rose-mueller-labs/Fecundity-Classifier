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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR') 

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42
BASE_DIR="/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/1.DataProcessing/model_architecture/models"

# BASE_DIR="/Volumes/Crucial X9/Fecundity-Classifier/1.DataProcessing/model_architecture/models"

# add models here as needed
MODELS = {
    'CLUSTERING_MODEL': (f'{BASE_DIR}/alex_4-30_5-1_CC_A_v0.0.h5', None),
    'DEFAULT_MODEL': (f'{BASE_DIR}/alex_5-1_5-2S_v0.0.h5', None)
}

# MODELS = {
#     'DEFAULT_MODEL': (f'{BASE_DIR}/alex_4-30_5-1_5-2O_v0.0.h5', None),
#     'CLUSTERING_MODEL': (f'{BASE_DIR}/alex_4-30_5-1_CC_A_v0.0.h5', None)
# }

DEFAULT_MODEL = "DEFAULT_MODEL"

# parse CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "Count our lab images of Drosophila eggs in petri dish images.\n\n"
            "Workflow:\n"
            "  1. Provide a directory of cap images via --D.\n"
            "  2. We then auto-sliced the caps into tiles (saved to <DIRECTORY>-sliced) if not already pre-sliced in the directory given.\n"
            "  3. A model runs inference on each tile and sums the predictions per image which are then saved into 2 CSVs: one with per-tile counts, one with per-image sums.\n"
            "Examples:\n"
            "  python cli.py --D /data/<NAME>\n"
            "  python cli.py --D /data/<NAME> --n CLUSTERING_MODEL\n"
            "  python cli.py --D /data/<NAME> --C"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", dest="model_name", default=DEFAULT_MODEL, metavar="MODEL_NAME",
        help=(
            f"Model name/key to use for inference (default: {DEFAULT_MODEL}). "
            f"Available: {list(MODELS.keys())}. "
            "Ignored if --C is set."
        ))
    parser.add_argument("--D", dest="data_dir", required=True, metavar="DIRECTORY",
        help=(
            "Path to directory containing cap images (NOT the full grid image). "
            "Images will be auto-sliced into tiles if a '<DIRECTORY>-sliced' "
            "folder does not already exist."
        ))
    parser.add_argument("--C", dest="cluster", action="store_true", default=False,
        help=(
            "Select this if you would like to use the clustering model (default: use non-cluster model)."
        ))
    parser.add_argument("--o", dest="output_dir", default=".",
        help="Directory to save CSV output files to (default: current directory).")
    parser.add_argument("--blah", action="store_true", default=False,
        help="blah")
    parser.add_argument("--T", dest="del_tiles", action="store_false", default=True,
        help="Select this if you would like to keep the tiles count output CSV (default: delete the CSV).")
    parser.add_argument("--t", dest="is_tiles", action="store_true", default=False,
        help="Determines if the data directory provided contains tiles or non-tile cap images. Indicate --t if the data directory contains tiles.")
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

def get_tile_preds(name, model, model2, set_name, TESTING_SET, output_dir):
    print(f'Predicting with {model}...')
    mod1 = tf.keras.models.load_model(model)
    mod2 = None
    if model2 != None:
        mod2 = tf.keras.models.load_model(model2)

    csv_name = os.path.join(output_dir, f'{set_name}_{name}_tile_counts.csv')

    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'Part', 'Bot'])
        for img in os.listdir(f"{TESTING_SET}"):
            root_image = img.split("pt")[0]
            # print(img)
            if "jpg" in root_image:
                root_image = root_image.split("jpg")[0][:-1]
            elif "png" in root_image:
                root_image = root_image.split("png")[0][:-1]
            elif 'JPG':
                root_image = root_image.split("JPG")[0][:-1]
            elif 'jpeg':
                root_image = root_image.split("jpeg")[0][:-1]
            # print(root_image)
            predicted_eggs = int(predict_egg_count_default(f"{TESTING_SET}/{img}", name, mod1, mod2))
            part = img.split("pt")[-1].split('.')[0]
            writer.writerow([root_image, part, predicted_eggs])
    return csv_name

def get_sums(csv_path, name, set_name, output_dir):
    actual_csv_name = os.path.join(output_dir, f'{set_name}_{name}_sums.csv')

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
    if args.del_tiles:
        os.remove(csv_path)
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

    os.makedirs(args.output_dir, exist_ok=True) # only if it don't exist

    set_name = os.path.basename(args.data_dir.rstrip("/\\"))

    # slice images if not already done --> paths now from args.data_dir
    TESTING_SET = f"{args.data_dir}-sliced"
    if not os.path.isdir(TESTING_SET) and not args.is_tiles:
        try:
            shred(args.data_dir)
        except ValueError:
            print("ERROR: Tiles have been given instead of cap images. If you meant to give tiles, indicate that with the '--t' option.")
            exit()
    elif args.is_tiles:
        TESTING_SET = args.data_dir

    paths = MODELS[name]

    print(f'Getting tiles for {name}')
    print_time()
    tiles_csv_name = get_tile_preds(name, paths[0], paths[1], set_name, TESTING_SET, args.output_dir)
    print(f'Getting sums for {name}')
    print_time()
    sums_csv_name = get_sums(tiles_csv_name, name, set_name, args.output_dir)
    print(f'Done. Results saved to: {sums_csv_name}')