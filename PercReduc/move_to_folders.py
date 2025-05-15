import os
import shutil

DP = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/PercReduc"

for name in os.listdir(DP):
    datapath=f"{DP}/{name}"
    if "4907" in name or "XTrain" not in name:
        continue
    if name.endswith("png"):
        shutil.move(f'{datapath}', f'{DP}/AlexResults/mses/{name}')
    elif name.endswith("txt"):
        shutil.move(f'{datapath}', f'{DP}/AlexResults/mses/{name}')
    elif name.endswith("csv"):
        if "sums" in name:
            shutil.move(f'{datapath}', f'{DP}/AlexResults/sums/{name}')
        elif "tile" in name:
            shutil.move(f'{datapath}', f'{DP}/AlexResults/tiles/{name}')