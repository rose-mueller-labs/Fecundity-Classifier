import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import matplotlib.patches as patches

cmap = plt.cm.Purples # for max_eggs > 8

def generate_vis(df_new, images_i_want, ROOT_IMG, model_name):  
    for img_wanted in images_i_want:
        # max_eggs = df_new[df_new['FileName'] == img_wanted]['EggCount'].apply(int).max()
        max_eggs = 42
        print('Max Tile Egg Cnt:', max_eggs)
        # colors = ['purple', 'violet', 'plum', 'thistle', 'mediumorchid', 'darkviolet', 'darkorchid', 'indigo']
        fig, ax = plt.subplots(figsize=(6, 6))
        tiles_cnt = 0
        for index, row in df_new.iterrows():
            if tiles_cnt == 100:
                break
            if row['FileName'] == img_wanted:
                tiles_cnt += 1
                full_path = os.path.join(ROOT_IMG, row['ImageName'])
                img = mpimg.imread(full_path)
                x_coord = row['x']+0.5
                y_coord = 10-row['y']-0.5
                imagebox = OffsetImage(img, zoom=0.45)
                ab = AnnotationBbox(imagebox, (x_coord, y_coord), frameon=False)

                if (int(row['EggCount']) > 0):
                    egg_count = int(row['EggCount'])
                    t = egg_count/max_eggs
                    t = 0.3 + 0.7 * t
                    color = cmap(t)
                    rect = patches.Rectangle((x_coord-0.5, y_coord+0.5-1), 
                                             1, 1, linewidth=2, edgecolor=color, 
                                             facecolor=color, alpha=0.4)
                    ax.add_patch(rect)
                    rect.set_zorder(4)
                ax.add_artist(ab)
                ax.set_xlim(0, 10) 
                ax.set_ylim(0, 10)
                plt.xticks(np.arange(0, 10, step=1))
                plt.yticks(np.arange(0, 10, step=1))
                plt.plot()
        plt.savefig(f"{model_name}_CD{img_wanted}_plot.png")