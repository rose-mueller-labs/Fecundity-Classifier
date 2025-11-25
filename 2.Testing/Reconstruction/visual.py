# get the csv file with the predicted counts and actual and get the difference and graph it?
# step one:
# put the actual img tg
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import matplotlib.patches as patches
import matplotlib.colors as mcolors

def generate_vis(df_new, images_i_want, ROOT_IMG):
    # df = pd.read_csv("/home/drosophila-lab/Documents/Fecundity/Lithium-Caps-Organization/alex2.csv")
    # color = FF0000 #['red', 'cyan', 'purple', 'orange', 'green', 'yellow', 'blue', 'pink', 'brown', 'gray', 'olive']
    colors = ['purple', 'violet', 'plum', 'thistle', 'mediumorchid', 'darkviolet', 'darkorchid', 'indigo']

    for img_wanted in images_i_want:
        fig, ax = plt.subplots(figsize=(6, 6))
        tiles_cnt = 0
        for index, row in df_new.iterrows():
            if tiles_cnt == 100:
                break
            if row['FileName'] == img_wanted:
                tiles_cnt += 1
                # print(row['FileName'])
                full_path = os.path.join(ROOT_IMG, row['ImageName'])
                # Load the image
                img = mpimg.imread(full_path)
                # Create a figure and axes
                x_coord = row['x']+0.5
                y_coord = 10-row['y']-0.5
                # print(x_coord, y_coord)
                # Create an OffsetImage object with the image data
                imagebox = OffsetImage(img, zoom=0.45)  # Adjust zoom as needed
                # Create an AnnotationBbox to place the image at the specified coordinates
                ab = AnnotationBbox(imagebox, (x_coord, y_coord), frameon=False)
                # print(ab)
                if (int(row['EggCount']) > 0):
                    # rect = patches.Rectangle((x_coord-0.5, y_coord+0.5-1), 1, 1, linewidth=2, edgecolor=f"#ff{int(row['EggCount'])*1000}", facecolor=f"#ff{int(row['EggCount'])*1000}", alpha=0.2)
                    rect = patches.Rectangle((x_coord-0.5, y_coord+0.5-1), 1, 1, linewidth=2, edgecolor=colors[int(row['EggCount'])], facecolor=colors[int(row['EggCount'])], alpha=0.2)
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                    rect.set_zorder(4)
                ax.add_artist(ab)
                # print(ax)
                ax.set_xlim(0, 10) 
                ax.set_ylim(0, 10)
                plt.xticks(np.arange(0, 10, step=1))
                plt.yticks(np.arange(0, 10, step=1))
                plt.plot()
        plt.savefig(f"CD{img_wanted}_plot.png")