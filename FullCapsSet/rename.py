import os

path="/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/FullCapsSplit"

for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and item.endswith("-sliced"):
            new_name = item.split('-')[0]
            new_path = os.path.join(path, new_name)
            os.rename(item_path, new_path)

