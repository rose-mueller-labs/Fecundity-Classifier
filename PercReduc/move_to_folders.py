import os
import shutil

DP = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/PercReduc"

for name in os.listdir(DP):
    datapath=f"{DP}/{name}"
    if name.endswith("png"):
        shutil.move(f'{datapath}', f'{DP}/mses/{name}')
    elif name.endswith("txt"):
        shutil.move(f'{datapath}', f'{DP}/mses/{name}')
    elif name.endswith("csv"):
        if "sums" in name:
            shutil.move(f'{datapath}', f'{DP}/sums/{name}')
        elif "tile" in name:
            shutil.move(f'{datapath}', f'{DP}/tiles/{name}')